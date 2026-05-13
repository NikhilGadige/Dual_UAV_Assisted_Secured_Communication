import argparse
import csv
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.environment import EnvConfig, UAVEnvironment
from core.config_utils import build_env_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class AdvancedRLConfig:
    episodes: int = 200
    evaluation_episodes: int = 10
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 50000
    min_replay_size: int = 1000
    hidden_dim: int = 128
    seed: int = 42
    device: str = "cpu"
    fading_model: str = "rician"
    rician_k: float = 5.0
    control_mode: str = "velocity"
    role_switching: bool = False
    td3_policy_delay: int = 2
    td3_target_noise: float = 0.2
    td3_noise_clip: float = 0.5
    exploration_noise: float = 0.2
    sac_alpha: float = 0.2
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    user_mobile: bool = False
    use_los_model: bool = False
    observation_mode: str = "full"
    normalize_observations: bool = True
    # NTN / Satellite
    enable_ntn: bool = False
    satellite_altitude_km: float = 500.0
    satellite_horizontal_offset_km: float = 100.0
    ntn_carrier_frequency_hz: float = 2e9
    ntn_atmospheric_loss_db: float = 0.5
    ntn_rician_k_db: float = 10.0


def make_env_config(seed: int, cfg: AdvancedRLConfig) -> EnvConfig:
    return build_env_config(
        seed=seed,
        fading_model=cfg.fading_model,
        rician_k=cfg.rician_k,
        control_mode=cfg.control_mode,
        role_switching=cfg.role_switching,
        user_mobile=cfg.user_mobile,
        use_los_model=cfg.use_los_model,
        observation_mode=cfg.observation_mode,
        normalize_observations=cfg.normalize_observations,
        enable_ntn=cfg.enable_ntn,
        satellite_altitude_km=cfg.satellite_altitude_km,
        satellite_horizontal_offset_km=cfg.satellite_horizontal_offset_km,
        ntn_carrier_frequency_hz=cfg.ntn_carrier_frequency_hz,
        ntn_atmospheric_loss_db=cfg.ntn_atmospheric_loss_db,
        ntn_rician_k_db=cfg.ntn_rician_k_db,
    )


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            np.asarray(r, dtype=np.float32),
            np.asarray(ns, dtype=np.float32),
            np.asarray(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DeterministicActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=1))


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(s)
        return self.mean(h), torch.clamp(self.log_std(h), -5.0, 2.0)

    def sample(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(s)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)

    def deterministic(self, s: torch.Tensor) -> torch.Tensor:
        mean, _ = self(s)
        return torch.tanh(mean)


def soft_update(src: nn.Module, dst: nn.Module, tau: float) -> None:
    for p_src, p_dst in zip(src.parameters(), dst.parameters()):
        p_dst.data.copy_(tau * p_src.data + (1.0 - tau) * p_dst.data)


def split_action(action: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool]:
    role_switch = bool(action.shape[0] > 5 and action[5] > 0.5)
    return action[:2], action[2:4], float(action[4]), role_switch


def rollout_episode(env: UAVEnvironment, actor_fn, device: torch.device) -> dict:
    state = env.reset().astype(np.float32)
    done = False
    total_reward = 0.0
    total_rsec = 0.0
    steps = 0
    with torch.no_grad():
        while not done:
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = actor_fn(s_t).cpu().numpy()[0]
            a_relay, a_jammer, jammer_power, role_switch = split_action(action)
            state, reward, done, info = env.step(a_relay, a_jammer, jammer_power, role_switch)
            state = state.astype(np.float32)
            total_reward += reward
            total_rsec += info["R_sec"]
            steps += 1
    return {
        "avg_rsec_mbps": float((total_rsec / max(steps, 1)) / 1e6),
        "episode_secrecy_mbits": float((total_rsec * env.config.dt) / 1e6),
        "episode_reward_bps_step": float(total_reward),
    }


def evaluate_actor(actor_fn, cfg: AdvancedRLConfig, device: torch.device) -> dict:
    metrics = []
    for i in range(cfg.evaluation_episodes):
        env = UAVEnvironment(make_env_config(cfg.seed + 1000 + i, cfg))
        metrics.append(rollout_episode(env, actor_fn, device))
    return {
        "mean_avg_rsec_mbps": float(np.mean([m["avg_rsec_mbps"] for m in metrics])),
        "mean_episode_secrecy_mbits": float(np.mean([m["episode_secrecy_mbits"] for m in metrics])),
    }


def write_log(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_td3(cfg: AdvancedRLConfig | None = None, output_dir: str = "outputs/training/td3") -> dict:
    cfg = cfg or AdvancedRLConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    env = UAVEnvironment(make_env_config(cfg.seed, cfg))
    state_dim = env.reset().shape[0]
    action_dim = 6 if cfg.role_switching else 5

    actor = DeterministicActor(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_actor = DeterministicActor(state_dim, action_dim, cfg.hidden_dim).to(device)
    critic1 = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    critic2 = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_critic1 = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_critic2 = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())
    actor_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=cfg.critic_lr)
    replay = ReplayBuffer(cfg.replay_size)

    rows = []
    global_step = 0
    rolling = []
    for ep in range(1, cfg.episodes + 1):
        state = env.reset().astype(np.float32)
        done = False
        ep_reward = ep_rsec = 0.0
        steps = 0
        while not done:
            with torch.no_grad():
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = actor(s_t).cpu().numpy()[0]
            action = np.clip(action + np.random.normal(0.0, cfg.exploration_noise, size=action_dim), -1.0, 1.0)
            a_relay, a_jammer, jammer_power, role_switch = split_action(action)
            next_state, reward, done, info = env.step(a_relay, a_jammer, jammer_power, role_switch)
            replay.add(state, action, float(reward), next_state.astype(np.float32), float(done))
            state = next_state.astype(np.float32)
            ep_reward += reward
            ep_rsec += info["R_sec"]
            steps += 1
            global_step += 1

            if len(replay) >= cfg.min_replay_size:
                s, a, r, ns, d = replay.sample(cfg.batch_size)
                s_t = torch.tensor(s, dtype=torch.float32, device=device)
                a_t = torch.tensor(a, dtype=torch.float32, device=device)
                r_t = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
                ns_t = torch.tensor(ns, dtype=torch.float32, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)
                with torch.no_grad():
                    noise = torch.clamp(
                        torch.randn_like(a_t) * cfg.td3_target_noise,
                        -cfg.td3_noise_clip,
                        cfg.td3_noise_clip,
                    )
                    next_a = torch.clamp(target_actor(ns_t) + noise, -1.0, 1.0)
                    q_next = torch.min(target_critic1(ns_t, next_a), target_critic2(ns_t, next_a))
                    q_target = r_t + (1.0 - d_t) * cfg.gamma * q_next
                critic_loss = nn.MSELoss()(critic1(s_t, a_t), q_target) + nn.MSELoss()(critic2(s_t, a_t), q_target)
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()
                if global_step % cfg.td3_policy_delay == 0:
                    actor_loss = -critic1(s_t, actor(s_t)).mean()
                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()
                    soft_update(actor, target_actor, cfg.tau)
                    soft_update(critic1, target_critic1, cfg.tau)
                    soft_update(critic2, target_critic2, cfg.tau)

        avg_rsec = (ep_rsec / max(steps, 1)) / 1e6
        rolling.append(avg_rsec)
        rows.append(_row("TD3", cfg, ep, global_step, ep_reward, ep_rsec, steps, rolling))

    return _save_advanced("td3", actor, rows, cfg, output_dir, lambda s: actor(s))


def train_sac(cfg: AdvancedRLConfig | None = None, output_dir: str = "outputs/training/sac") -> dict:
    cfg = cfg or AdvancedRLConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    env = UAVEnvironment(make_env_config(cfg.seed, cfg))
    state_dim = env.reset().shape[0]
    action_dim = 6 if cfg.role_switching else 5

    actor = GaussianActor(state_dim, action_dim, cfg.hidden_dim).to(device)
    critic1 = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    critic2 = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_critic1 = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_critic2 = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())
    actor_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=cfg.critic_lr)
    replay = ReplayBuffer(cfg.replay_size)

    rows = []
    global_step = 0
    rolling = []
    for ep in range(1, cfg.episodes + 1):
        state = env.reset().astype(np.float32)
        done = False
        ep_reward = ep_rsec = 0.0
        steps = 0
        while not done:
            with torch.no_grad():
                action, _ = actor.sample(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
            action_np = action.cpu().numpy()[0]
            a_relay, a_jammer, jammer_power, role_switch = split_action(action_np)
            next_state, reward, done, info = env.step(a_relay, a_jammer, jammer_power, role_switch)
            replay.add(state, action_np, float(reward), next_state.astype(np.float32), float(done))
            state = next_state.astype(np.float32)
            ep_reward += reward
            ep_rsec += info["R_sec"]
            steps += 1
            global_step += 1

            if len(replay) >= cfg.min_replay_size:
                s, a, r, ns, d = replay.sample(cfg.batch_size)
                s_t = torch.tensor(s, dtype=torch.float32, device=device)
                a_t = torch.tensor(a, dtype=torch.float32, device=device)
                r_t = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
                ns_t = torch.tensor(ns, dtype=torch.float32, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)
                with torch.no_grad():
                    next_a, next_logp = actor.sample(ns_t)
                    q_next = torch.min(target_critic1(ns_t, next_a), target_critic2(ns_t, next_a))
                    q_target = r_t + (1.0 - d_t) * cfg.gamma * (q_next - cfg.sac_alpha * next_logp)
                critic_loss = nn.MSELoss()(critic1(s_t, a_t), q_target) + nn.MSELoss()(critic2(s_t, a_t), q_target)
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()
                new_a, logp = actor.sample(s_t)
                actor_loss = (cfg.sac_alpha * logp - torch.min(critic1(s_t, new_a), critic2(s_t, new_a))).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                soft_update(critic1, target_critic1, cfg.tau)
                soft_update(critic2, target_critic2, cfg.tau)

        avg_rsec = (ep_rsec / max(steps, 1)) / 1e6
        rolling.append(avg_rsec)
        rows.append(_row("SAC", cfg, ep, global_step, ep_reward, ep_rsec, steps, rolling))

    return _save_advanced("sac", actor, rows, cfg, output_dir, lambda s: actor.deterministic(s))


def train_ppo(cfg: AdvancedRLConfig | None = None, output_dir: str = "outputs/training/ppo") -> dict:
    cfg = cfg or AdvancedRLConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    env = UAVEnvironment(make_env_config(cfg.seed, cfg))
    state_dim = env.reset().shape[0]
    action_dim = 6 if cfg.role_switching else 5
    actor = GaussianActor(state_dim, action_dim, cfg.hidden_dim).to(device)
    value = nn.Sequential(
        nn.Linear(state_dim, cfg.hidden_dim),
        nn.ReLU(),
        nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        nn.ReLU(),
        nn.Linear(cfg.hidden_dim, 1),
    ).to(device)
    actor_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    value_opt = optim.Adam(value.parameters(), lr=cfg.critic_lr)

    rows = []
    global_step = 0
    rolling = []
    for ep in range(1, cfg.episodes + 1):
        states = []
        actions = []
        logps = []
        rewards = []
        state = env.reset().astype(np.float32)
        done = False
        ep_reward = ep_rsec = 0.0
        steps = 0
        while not done:
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, logp = actor.sample(s_t)
            action_np = action.cpu().numpy()[0]
            a_relay, a_jammer, jammer_power, role_switch = split_action(action_np)
            next_state, reward, done, info = env.step(a_relay, a_jammer, jammer_power, role_switch)
            states.append(state)
            actions.append(action_np)
            logps.append(float(logp.item()))
            rewards.append(float(reward))
            state = next_state.astype(np.float32)
            ep_reward += reward
            ep_rsec += info["R_sec"]
            steps += 1
            global_step += 1

        returns = []
        ret = 0.0
        for reward in reversed(rewards):
            ret = reward + cfg.gamma * ret
            returns.append(ret)
        returns.reverse()
        s_t = torch.tensor(np.asarray(states), dtype=torch.float32, device=device)
        a_t = torch.tensor(np.asarray(actions), dtype=torch.float32, device=device)
        old_logp_t = torch.tensor(logps, dtype=torch.float32, device=device).unsqueeze(1)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)
        with torch.no_grad():
            advantages = returns_t - value(s_t)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        for _ in range(cfg.ppo_epochs):
            mean, log_std = actor(s_t)
            std = log_std.exp()
            z = torch.atanh(torch.clamp(a_t, -0.999, 0.999))
            normal = torch.distributions.Normal(mean, std)
            logp = (normal.log_prob(z) - torch.log(1.0 - a_t.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
            ratio = torch.exp(logp - old_logp_t)
            clipped = torch.clamp(ratio, 1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip)
            actor_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            value_loss = nn.MSELoss()(value(s_t), returns_t)
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()

        avg_rsec = (ep_rsec / max(steps, 1)) / 1e6
        rolling.append(avg_rsec)
        rows.append(_row("PPO", cfg, ep, global_step, ep_reward, ep_rsec, steps, rolling))

    return _save_advanced("ppo", actor, rows, cfg, output_dir, lambda s: actor.deterministic(s))


def _row(
    algorithm: str,
    cfg: AdvancedRLConfig,
    ep: int,
    global_step: int,
    ep_reward: float,
    ep_rsec: float,
    steps: int,
    rolling: list[float],
) -> dict:
    roll20 = float(np.mean(rolling[-20:]))
    roll100 = float(np.mean(rolling[-100:]))
    return {
        "algorithm": algorithm,
        "episode": ep,
        "global_step": global_step,
        "fading_model": cfg.fading_model,
        "control_mode": cfg.control_mode,
        "role_switching": cfg.role_switching,
        "user_mobile": cfg.user_mobile,
        "use_los_model": cfg.use_los_model,
        "observation_mode": cfg.observation_mode,
        "normalize_observations": cfg.normalize_observations,
        "enable_energy_harvesting": False,
        "observation_has_eh": cfg.observation_mode == "full_eh",
        "enable_ntn": cfg.enable_ntn,
        "satellite_altitude_km": cfg.satellite_altitude_km,
        "episode_reward_bps_step": float(ep_reward),
        # "avg_shaped_reward": float((ep_reward / max(steps, 1)) / 1e6),
        # "avg_shaped_reward": float(ep_reward / max(ep_steps, 1)),
        "avg_shaped_reward": float(ep_reward / max(steps, 1)),
        "avg_R_sec_mbps": float((ep_rsec / max(steps, 1)) / 1e6),
        "episode_secrecy_mbits": float((ep_rsec * 0.1) / 1e6),
        "steps": steps,
        "rolling20_avg_R_sec_mbps": roll20,
        "rolling100_avg_R_sec_mbps": roll100,
        "convergence_gap20_100_mbps": float(abs(roll20 - roll100)),
    }


def _save_advanced(
    algorithm: str,
    actor: nn.Module,
    rows: list[dict],
    cfg: AdvancedRLConfig,
    output_dir: str,
    actor_fn,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{algorithm}_training_log.csv"
    model_path = out_dir / f"{algorithm}_actor.pt"
    write_log(rows, log_path)
    torch.save(actor.state_dict(), model_path)
    eval_summary = evaluate_actor(actor_fn, cfg, torch.device(cfg.device if torch.cuda.is_available() else "cpu"))
    print(
        f"{algorithm.upper()} complete | episodes={cfg.episodes} | "
        f"avg_R_sec={eval_summary['mean_avg_rsec_mbps']:.4f} Mbps"
    )
    return {
        "algorithm": algorithm,
        "training_log_csv": str(log_path.resolve()),
        "model_path": str(model_path.resolve()),
        "mean_avg_rsec_mbps": eval_summary["mean_avg_rsec_mbps"],
        "mean_episode_secrecy_mbits": eval_summary["mean_episode_secrecy_mbits"],
    }


def train_advanced(method: str, cfg: AdvancedRLConfig, output_dir: str) -> dict:
    method = method.lower()
    method_dir = str(Path(output_dir) / method)
    if method == "td3":
        return train_td3(cfg, method_dir)
    if method == "sac":
        return train_sac(cfg, method_dir)
    if method == "ppo":
        return train_ppo(cfg, method_dir)
    raise ValueError(f"Unsupported advanced method: {method}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Train TD3, SAC, or PPO for the dual-UAV secrecy environment.")
    parser.add_argument("--method", type=str, default="td3", choices=["td3", "sac", "ppo"])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channel-model", type=str, default="rician", choices=["rician", "rayleigh"])
    parser.add_argument("--control-mode", type=str, default="velocity", choices=["velocity", "waypoint"])
    parser.add_argument("--role-switching", action="store_true")
    parser.add_argument("--enable-ntn", action="store_true", help="Enable NTN satellite-assisted communication")
    parser.add_argument("--satellite-altitude-km", type=float, default=500.0, help="Satellite altitude (km)")
    parser.add_argument("--satellite-horizontal-offset-km", type=float, default=100.0, help="Satellite horizontal offset (km)")
    parser.add_argument("--ntn-carrier-frequency-hz", type=float, default=2e9, help="NTN carrier frequency (Hz)")
    parser.add_argument("--ntn-atmospheric-loss-db", type=float, default=0.5, help="NTN atmospheric loss (dB)")
    parser.add_argument("--ntn-rician-k-db", type=float, default=10.0, help="NTN Rician K-factor (dB)")
    parser.add_argument("--output-dir", type=str, default="outputs/training")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_advanced(
        args.method,
        AdvancedRLConfig(
            episodes=args.episodes,
            evaluation_episodes=args.eval_episodes,
            seed=args.seed,
            fading_model=args.channel_model,
            control_mode=args.control_mode,
            role_switching=args.role_switching,
            enable_ntn=args.enable_ntn,
            satellite_altitude_km=args.satellite_altitude_km,
            satellite_horizontal_offset_km=args.satellite_horizontal_offset_km,
            ntn_carrier_frequency_hz=args.ntn_carrier_frequency_hz,
            ntn_atmospheric_loss_db=args.ntn_atmospheric_loss_db,
            ntn_rician_k_db=args.ntn_rician_k_db,
        ),
        args.output_dir,
    )
