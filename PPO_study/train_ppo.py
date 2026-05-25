import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.config_utils import build_env_config
from core.environment import EnvConfig, UAVEnvironment


@dataclass
class PPOConfig:
    episodes: int = 2500
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    ppo_epochs: int = 8
    minibatch_size: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 8e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 64
    seed: int = 42
    device: str = "cpu"
    fading_model: str = "rician"
    rician_k: float = 5.0
    eval_interval: int = 50
    train_eval_episodes: int = 5
    control_mode: str = "velocity"
    user_mobile: bool = True
    use_los_model: bool = False
    observation_mode: str = "full"
    normalize_observations: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env_config(seed: int, cfg: PPOConfig) -> EnvConfig:
    return build_env_config(
        seed=seed,
        fading_model=cfg.fading_model,
        rician_k=cfg.rician_k,
        control_mode=cfg.control_mode,
        role_switching=False,
        user_mobile=cfg.user_mobile,
        use_los_model=cfg.use_los_model,
        observation_mode=cfg.observation_mode,
        normalize_observations=cfg.normalize_observations,
    )


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def _dist(self, states: torch.Tensor) -> torch.distributions.Normal:
        h = self.net(states)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std, -4.0, 1.0)
        std = log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def sample(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._dist(states)
        z = dist.rsample()
        actions = torch.tanh(z)
        logp = (dist.log_prob(z) - torch.log(1.0 - actions.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
        entropy = dist.entropy().sum(dim=1, keepdim=True)
        return actions, logp, entropy

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        dist = self._dist(states)
        logp = (dist.log_prob(z) - torch.log(1.0 - actions.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
        entropy = dist.entropy().sum(dim=1, keepdim=True)
        return logp, entropy

    def deterministic(self, states: torch.Tensor) -> torch.Tensor:
        h = self.net(states)
        return torch.tanh(self.mean(h))


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


def split_action(action: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    return action[:2], action[2:4], float(action[4])


def evaluate_policy(env: UAVEnvironment, actor: GaussianActor, device: torch.device, episodes: int) -> dict:
    actor.eval()
    episode_avg = []
    with torch.no_grad():
        for _ in range(episodes):
            state = env.reset().astype(np.float32)
            done = False
            total_r_sec = 0.0
            steps = 0
            while not done:
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = actor.deterministic(s).cpu().numpy()[0]
                a_relay, a_jammer, jammer_power = split_action(action)
                next_state, _, done, info = env.step(a_relay, a_jammer, jammer_power)
                total_r_sec += info["R_sec"]
                steps += 1
                state = next_state.astype(np.float32)
            episode_avg.append((total_r_sec / max(steps, 1)) / 1e6)
    return {"mean_avg_rsec_mbps": float(np.mean(episode_avg))}


def _gae_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = last_value if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        adv[t] = gae
    ret = adv + values
    return adv, ret


def train_ppo(cfg: PPOConfig, output_dir: str) -> dict:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    env = UAVEnvironment(make_env_config(cfg.seed, cfg))
    state_dim = env.reset().shape[0]
    action_dim = 5

    actor = GaussianActor(state_dim, action_dim, cfg.hidden_dim).to(device)
    critic = ValueNet(state_dim, cfg.hidden_dim).to(device)
    actor_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "ppo_training_log.csv"
    actor_path = out_dir / "ppo_actor.pt"

    global_step = 0
    rolling = []
    rows = []
    last_eval = ""

    for ep in range(1, cfg.episodes + 1):
        states, actions, old_logps, rewards, dones, values = [], [], [], [], [], []
        state = env.reset().astype(np.float32)
        done = False

        ep_reward = 0.0
        ep_rsec = 0.0
        ep_rlegit = 0.0
        ep_reve = 0.0
        steps = 0

        while not done:
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                v = critic(s_t)
                a_t, logp_t, _ = actor.sample(s_t)
            action = a_t.cpu().numpy()[0]
            a_relay, a_jammer, jammer_power = split_action(action)
            next_state, reward, done, info = env.step(a_relay, a_jammer, jammer_power)

            states.append(state)
            actions.append(action)
            old_logps.append(float(logp_t.item()))
            values.append(float(v.item()))
            rewards.append(float(reward))
            dones.append(float(done))

            state = next_state.astype(np.float32)
            ep_reward += reward
            ep_rsec += info["R_sec"]
            ep_rlegit += info["R_legit"]
            ep_reve += info["R_eve"]
            steps += 1
            global_step += 1

        with torch.no_grad():
            last_v = 0.0 if done else float(critic(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).item())

        rewards_np = np.asarray(rewards, dtype=np.float32)
        values_np = np.asarray(values, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.float32)
        adv_np, ret_np = _gae_returns(rewards_np, values_np, dones_np, last_v, cfg.gamma, cfg.gae_lambda)

        adv_t = torch.tensor(adv_np, dtype=torch.float32, device=device).unsqueeze(1)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)
        ret_t = torch.tensor(ret_np, dtype=torch.float32, device=device).unsqueeze(1)
        s_t = torch.tensor(np.asarray(states), dtype=torch.float32, device=device)
        a_t = torch.tensor(np.asarray(actions), dtype=torch.float32, device=device)
        old_logp_t = torch.tensor(np.asarray(old_logps), dtype=torch.float32, device=device).unsqueeze(1)

        idx = np.arange(len(states))
        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, len(idx), cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                s_mb = s_t[mb]
                a_mb = a_t[mb]
                old_logp_mb = old_logp_t[mb]
                adv_mb = adv_t[mb]
                ret_mb = ret_t[mb]

                logp, entropy = actor.evaluate(s_mb, a_mb)
                ratio = torch.exp(logp - old_logp_mb)
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                actor_loss = -(torch.min(ratio * adv_mb, clipped * adv_mb)).mean() - cfg.entropy_coef * entropy.mean()

                value_pred = critic(s_mb)
                critic_loss = cfg.value_coef * nn.MSELoss()(value_pred, ret_mb)

                actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                actor_opt.step()

                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                critic_opt.step()

        avg_rsec_mbps = (ep_rsec / max(steps, 1)) / 1e6
        avg_shaped_reward = ep_reward / max(steps, 1)
        rolling.append(avg_rsec_mbps)
        roll20 = float(np.mean(rolling[-20:]))
        roll100 = float(np.mean(rolling[-100:]))
        gap = float(abs(roll20 - roll100))
        eval_rsec = ""

        if cfg.eval_interval > 0 and ep % cfg.eval_interval == 0:
            eval_env = UAVEnvironment(make_env_config(cfg.seed + 5000 + ep, cfg))
            eval_summary = evaluate_policy(eval_env, actor, device, episodes=cfg.train_eval_episodes)
            last_eval = eval_summary["mean_avg_rsec_mbps"]
            eval_rsec = last_eval

        rows.append(
            {
                "episode": ep,
                "global_step": global_step,
                "fading_model": cfg.fading_model,
                "hidden_dim": cfg.hidden_dim,
                "avg_shaped_reward": float(avg_shaped_reward),
                "avg_R_legit_mbps": float((ep_rlegit / max(steps, 1)) / 1e6),
                "avg_R_eve_mbps": float((ep_reve / max(steps, 1)) / 1e6),
                "avg_R_sec_mbps": float(avg_rsec_mbps),
                "eval_R_sec_mbps": eval_rsec,
                "last_eval_R_sec_mbps": last_eval,
                "steps": steps,
                "rolling20": roll20,
                "rolling20_avg_R_sec_mbps": roll20,
                "rolling100": roll100,
                "rolling100_avg_R_sec_mbps": roll100,
                "convergence_gap": gap,
                "convergence_gap20_100_mbps": gap,
            }
        )

        if ep % 25 == 0 or ep == 1 or ep == cfg.episodes:
            print(
                f"Episode {ep:4d}/{cfg.episodes} | "
                f"avg_R_sec={avg_rsec_mbps:.3f} Mbps | roll100={roll100:.3f} Mbps"
            )

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    torch.save(actor.state_dict(), actor_path)
    return {
        "training_log_csv": str(log_path.resolve()),
        "model_path": str(actor_path.resolve()),
    }
