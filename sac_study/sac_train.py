from __future__ import annotations

import argparse
import csv
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.environment import UAVEnvironment
from sac_study.configs import SACStudyConfig, build_output_dir, make_env_config
from sac_study.plotting import generate_channel_comparison, generate_single_run_plots


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


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

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(state)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self(state)
        return torch.tanh(mean)


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

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=1))


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def split_action(action: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool]:
    role_switch = bool(action.shape[0] > 5 and action[5] > 0.5)
    return action[:2], action[2:4], float(action[4]), role_switch


def evaluate_actor(actor: GaussianActor, cfg: SACStudyConfig, device: torch.device, episodes: int, seed_offset: int) -> dict:
    actor.eval()
    avg_secrecy = []
    payload = []
    with torch.no_grad():
        for idx in range(episodes):
            env = UAVEnvironment(make_env_config(cfg.seed + seed_offset + idx, cfg))
            state = env.reset().astype(np.float32)
            done = False
            total_rsec = 0.0
            steps = 0
            while not done:
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = actor.deterministic(state_t).cpu().numpy()[0]
                a_relay, a_jammer, jammer_power, role_switch = split_action(action)
                state, _, done, info = env.step(a_relay, a_jammer, jammer_power, role_switch)
                state = state.astype(np.float32)
                total_rsec += info["R_sec"]
                steps += 1
            avg_secrecy.append((total_rsec / max(steps, 1)) / 1e6)
            payload.append((total_rsec * env.config.dt) / 1e6)
    actor.train()
    return {
        "mean_avg_rsec_mbps": float(np.mean(avg_secrecy)),
        "mean_episode_secrecy_mbits": float(np.mean(payload)),
    }


def _write_rows(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_sac(cfg: SACStudyConfig | None = None, output_dir: str | None = None) -> dict:
    cfg = cfg or SACStudyConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir or build_output_dir(cfg))
    out_dir.mkdir(parents=True, exist_ok=True)

    env = UAVEnvironment(make_env_config(cfg.seed, cfg))
    state_dim = env.reset().shape[0]
    action_dim = 6 if cfg.role_switching else 5
    target_entropy = -float(action_dim)

    actor = GaussianActor(state_dim, action_dim, cfg.hidden_dim).to(device)
    critic1 = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    critic2 = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    critic1_target = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    critic2_target = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=cfg.critic_lr)
    log_alpha = torch.tensor(np.log(cfg.init_alpha), dtype=torch.float32, device=device, requires_grad=True)
    alpha_opt = optim.Adam([log_alpha], lr=cfg.alpha_lr)
    replay = ReplayBuffer(cfg.replay_size)

    rows: list[dict] = []
    rolling_secrecy: list[float] = []
    global_step = 0
    last_eval: float | str = ""
    last_critic_loss: float | str = ""
    last_actor_loss: float | str = ""
    last_alpha_loss: float | str = ""
    last_entropy: float | str = ""

    for episode in range(1, cfg.episodes + 1):
        state = env.reset().astype(np.float32)
        done = False
        ep_reward = 0.0
        ep_rsec = 0.0
        ep_rlegit = 0.0
        ep_reve = 0.0
        ep_energy = 0.0
        ep_jammer_power = 0.0
        ep_steps = 0
        relay_start = env.relay_position.copy()
        jammer_start = env.jammer_position.copy()
        relay_path = 0.0
        jammer_path = 0.0

        while not done:
            relay_prev = env.relay_position.copy()
            jammer_prev = env.jammer_position.copy()
            with torch.no_grad():
                if len(replay) < cfg.min_replay_size:
                    action = np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
                else:
                    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = actor.sample(state_t)[0].cpu().numpy()[0]
            a_relay, a_jammer, jammer_power, role_switch = split_action(action)
            next_state, reward, done, info = env.step(a_relay, a_jammer, jammer_power, role_switch)
            next_state = next_state.astype(np.float32)
            replay.add(state, action, float(reward), next_state, float(done))
            state = next_state

            ep_reward += reward
            ep_rsec += info["R_sec"]
            ep_rlegit += info["R_legit"]
            ep_reve += info["R_eve"]
            ep_energy += info["total_energy_j"]
            ep_jammer_power += info["jammer_power"]
            relay_path += float(np.linalg.norm(env.relay_position - relay_prev))
            jammer_path += float(np.linalg.norm(env.jammer_position - jammer_prev))
            ep_steps += 1
            global_step += 1

            if len(replay) >= cfg.min_replay_size:
                states, actions, rewards, next_states, dones = replay.sample(cfg.batch_size)
                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
                alpha = log_alpha.exp()

                with torch.no_grad():
                    next_actions, next_log_probs = actor.sample(next_states_t)
                    next_q = torch.min(
                        critic1_target(next_states_t, next_actions),
                        critic2_target(next_states_t, next_actions),
                    )
                    q_target = rewards_t + (1.0 - dones_t) * cfg.gamma * (next_q - alpha * next_log_probs)

                critic_loss = nn.MSELoss()(critic1(states_t, actions_t), q_target)
                critic_loss = critic_loss + nn.MSELoss()(critic2(states_t, actions_t), q_target)
                critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(list(critic1.parameters()) + list(critic2.parameters()), cfg.grad_clip_norm)
                critic_opt.step()
                last_critic_loss = float(critic_loss.item())

                new_actions, log_probs = actor.sample(states_t)
                min_q = torch.min(critic1(states_t, new_actions), critic2(states_t, new_actions))
                actor_loss = (alpha.detach() * log_probs - min_q).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip_norm)
                actor_opt.step()
                last_actor_loss = float(actor_loss.item())
                last_entropy = float((-log_probs).mean().item())

                if cfg.auto_entropy_tuning:
                    alpha_loss = -(log_alpha * (log_probs + target_entropy).detach()).mean()
                    alpha_opt.zero_grad()
                    alpha_loss.backward()
                    alpha_opt.step()
                    last_alpha_loss = float(alpha_loss.item())

                soft_update(critic1, critic1_target, cfg.tau)
                soft_update(critic2, critic2_target, cfg.tau)

        avg_rsec = (ep_rsec / max(ep_steps, 1)) / 1e6
        rolling_secrecy.append(avg_rsec)
        roll20 = float(np.mean(rolling_secrecy[-20:]))
        roll100 = float(np.mean(rolling_secrecy[-100:]))
        eval_rsec: float | str = ""
        if cfg.eval_interval > 0 and episode % cfg.eval_interval == 0:
            eval_summary = evaluate_actor(actor, cfg, device, cfg.train_eval_episodes, seed_offset=5000 + episode)
            last_eval = eval_summary["mean_avg_rsec_mbps"]
            eval_rsec = last_eval
        distances = env.compute_distances()
        rows.append(
            {
                "episode": episode,
                "global_step": global_step,
                "algorithm": "SAC",
                "fading_model": cfg.fading_model,
                "hidden_dim": cfg.hidden_dim,
                "actor_lr": cfg.actor_lr,
                "critic_lr": cfg.critic_lr,
                "alpha_lr": cfg.alpha_lr,
                "batch_size": cfg.batch_size,
                "alpha": float(log_alpha.exp().item()),
                "target_entropy": target_entropy,
                "auto_entropy_tuning": cfg.auto_entropy_tuning,
                "control_mode": cfg.control_mode,
                "role_switching": cfg.role_switching,
                "user_mobile": cfg.user_mobile,
                "observation_mode": cfg.observation_mode,
                "normalize_observations": cfg.normalize_observations,
                "episode_reward_bps_step": float(ep_reward),
                "avg_shaped_reward": float(ep_reward / max(ep_steps, 1)),
                "avg_R_legit_mbps": float((ep_rlegit / max(ep_steps, 1)) / 1e6),
                "avg_R_eve_mbps": float((ep_reve / max(ep_steps, 1)) / 1e6),
                "avg_R_sec_mbps": float(avg_rsec),
                "eval_R_sec_mbps": eval_rsec,
                "last_eval_R_sec_mbps": last_eval,
                "episode_secrecy_mbits": float((ep_rsec * env.config.dt) / 1e6),
                "avg_energy_j": float(ep_energy / max(ep_steps, 1)),
                "avg_jammer_power_w": float(ep_jammer_power / max(ep_steps, 1)),
                "steps": ep_steps,
                "relay_path_m": relay_path,
                "jammer_path_m": jammer_path,
                "relay_displacement_m": float(np.linalg.norm(env.relay_position - relay_start)),
                "jammer_displacement_m": float(np.linalg.norm(env.jammer_position - jammer_start)),
                "final_d_UR_m": float(distances["d_UR"]),
                "final_d_RB_m": float(distances["d_RB"]),
                "final_d_UE_m": float(distances["d_UE"]),
                "final_d_JE_m": float(distances["d_JE"]),
                "rolling20_avg_R_sec_mbps": roll20,
                "rolling100_avg_R_sec_mbps": roll100,
                "convergence_gap20_100_mbps": float(abs(roll20 - roll100)),
                "critic_loss": last_critic_loss,
                "actor_loss": last_actor_loss,
                "alpha_loss": last_alpha_loss,
                "policy_entropy": last_entropy,
                "replay_size": len(replay),
            }
        )

        if episode == 1 or episode % 25 == 0 or episode == cfg.episodes:
            print(
                f"Episode {episode:4d}/{cfg.episodes} | "
                f"alpha={log_alpha.exp().item():.3f} | avg_R_sec={avg_rsec:.3f} Mbps | "
                f"roll100={roll100:.3f} Mbps"
            )

    log_path = out_dir / "sac_training_log.csv"
    actor_path = out_dir / "sac_actor.pt"
    critic1_path = out_dir / "sac_critic1.pt"
    _write_rows(rows, log_path)
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic1.state_dict(), critic1_path)
    plot_paths = generate_single_run_plots(str(log_path), str(out_dir))
    generate_channel_comparison(cfg.output_root)
    final_eval = evaluate_actor(actor, cfg, device, cfg.final_eval_episodes, seed_offset=9000)
    print(
        f"\nSAC {cfg.fading_model} complete | episodes={cfg.episodes} | "
        f"final_eval_R_sec={final_eval['mean_avg_rsec_mbps']:.4f} Mbps"
    )
    print(f"Training log: {log_path.resolve()}")
    print(f"Plots/model:  {out_dir.resolve()}")
    return {
        "training_log_csv": str(log_path.resolve()),
        "actor_path": str(actor_path.resolve()),
        "critic1_path": str(critic1_path.resolve()),
        "plot_paths": plot_paths,
        **final_eval,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAC convergence study for dual-UAV secure uplink.")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--channel-model", choices=["rician", "rayleigh"], default="rician")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-root", type=str, default="sac_study/output")
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--train-eval-episodes", type=int, default=5)
    parser.add_argument("--final-eval-episodes", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_sac(
        SACStudyConfig(
            episodes=args.episodes,
            fading_model=args.channel_model,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
            device=args.device,
            output_root=args.output_root,
            eval_interval=args.eval_interval,
            train_eval_episodes=args.train_eval_episodes,
            final_eval_episodes=args.final_eval_episodes,
        )
    )

