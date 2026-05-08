import csv
import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment import EnvConfig, UAVEnvironment
from baselines import distance_greedy_policy, evaluate_policy, random_policy


@dataclass
class DDPGConfig:
    episodes: int = 400
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 50000
    min_replay_size: int = 1000
    hidden_dim: int = 128
    noise_std_start: float = 0.25
    noise_std_end: float = 0.05
    noise_decay_steps: int = 50000
    seed: int = 42
    device: str = "cpu"
    fading_model: str = "rician"
    rician_k: float = 5.0


def make_env_config(seed: int, cfg: DDPGConfig) -> EnvConfig:
    return EnvConfig(seed=seed, fading_model=cfg.fading_model, rician_k=cfg.rician_k)


class Actor(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def noise_by_step(step: int, cfg: DDPGConfig) -> float:
    if step >= cfg.noise_decay_steps:
        return cfg.noise_std_end
    span = cfg.noise_std_start - cfg.noise_std_end
    frac = step / max(cfg.noise_decay_steps, 1)
    return cfg.noise_std_start - span * frac


def soft_update(src: nn.Module, dst: nn.Module, tau: float) -> None:
    for p_src, p_dst in zip(src.parameters(), dst.parameters()):
        p_dst.data.copy_(tau * p_src.data + (1.0 - tau) * p_dst.data)


def split_action(action_5d: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    return action_5d[:2], action_5d[2:4], float(action_5d[4])


def evaluate_ddpg(
    env: UAVEnvironment,
    actor: Actor,
    device: torch.device,
    episodes: int = 20,
) -> dict:
    actor.eval()
    episode_sec_mbits = []
    episode_avg_sec_mbps = []

    with torch.no_grad():
        for _ in range(episodes):
            state = env.reset().astype(np.float32)
            done = False
            total_r_sec = 0.0
            steps = 0
            while not done:
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = actor(s_t).cpu().numpy()[0]
                a_relay, a_jammer, jammer_power = split_action(action)
                next_state, _, done, info = env.step(a_relay, a_jammer, jammer_power)
                total_r_sec += info["R_sec"]
                steps += 1
                state = next_state.astype(np.float32)

            avg_r_sec_bps = total_r_sec / max(steps, 1)
            episode_avg_sec_mbps.append(avg_r_sec_bps / 1e6)
            episode_sec_mbits.append((total_r_sec * env.config.dt) / 1e6)

    return {
        "mean_avg_rsec_mbps": float(np.mean(episode_avg_sec_mbps)),
        "mean_episode_secrecy_mbits": float(np.mean(episode_sec_mbits)),
    }


def train_ddpg(cfg: DDPGConfig | None = None, output_dir: str = "outputs/ddpg") -> dict:
    cfg = cfg or DDPGConfig()
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    env = UAVEnvironment(make_env_config(cfg.seed, cfg))
    state_dim = env.reset().shape[0]
    action_dim = 5

    actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(device)
    critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    replay = ReplayBuffer(cfg.replay_size)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "ddpg_training_log.csv"

    global_step = 0
    rolling_rewards: list[float] = []
    train_rows: list[dict] = []

    for ep in range(1, cfg.episodes + 1):
        state = env.reset().astype(np.float32)
        done = False
        ep_reward = 0.0
        ep_rsec_bps = 0.0
        ep_energy_j = 0.0
        ep_steps = 0

        while not done:
            noise_std = noise_by_step(global_step, cfg)
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = actor(s_t).cpu().numpy()[0]
            action = np.clip(action + np.random.normal(0.0, noise_std, size=action_dim), -1.0, 1.0)

            a_relay, a_jammer, jammer_power = split_action(action)
            next_state, reward, done, info = env.step(a_relay, a_jammer, jammer_power)
            next_state = next_state.astype(np.float32)

            replay.add(state, action, float(reward), next_state, float(done))
            state = next_state
            ep_reward += reward
            ep_rsec_bps += info["R_sec"]
            ep_energy_j += info["total_energy_j"]
            ep_steps += 1
            global_step += 1

            if len(replay) >= cfg.min_replay_size:
                s, a, r, ns, d = replay.sample(cfg.batch_size)
                s_t = torch.tensor(s, dtype=torch.float32, device=device)
                a_t = torch.tensor(a, dtype=torch.float32, device=device)
                r_t = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
                ns_t = torch.tensor(ns, dtype=torch.float32, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)

                with torch.no_grad():
                    next_a = target_actor(ns_t)
                    q_next = target_critic(ns_t, next_a)
                    q_target = r_t + (1.0 - d_t) * cfg.gamma * q_next

                q_pred = critic(s_t, a_t)
                critic_loss = nn.MSELoss()(q_pred, q_target)
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                actor_loss = -critic(s_t, actor(s_t)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                soft_update(actor, target_actor, cfg.tau)
                soft_update(critic, target_critic, cfg.tau)

        avg_rsec_mbps = (ep_rsec_bps / max(ep_steps, 1)) / 1e6
        ep_secrecy_mbits = (ep_rsec_bps * env.config.dt) / 1e6
        rolling_rewards.append(avg_rsec_mbps)
        roll100 = float(np.mean(rolling_rewards[-100:]))
        train_rows.append(
            {
                "episode": ep,
                "noise_std": noise_std,
                "episode_reward_bps_step": float(ep_reward),
                "avg_R_sec_mbps": float(avg_rsec_mbps),
                "episode_secrecy_mbits": float(ep_secrecy_mbits),
                "avg_energy_j": float(ep_energy_j / max(ep_steps, 1)),
                "rolling100_avg_R_sec_mbps": roll100,
            }
        )

        if ep % 25 == 0 or ep == 1 or ep == cfg.episodes:
            print(
                f"Episode {ep:4d}/{cfg.episodes} | noise={noise_std:.3f} | "
                f"avg_R_sec={avg_rsec_mbps:.3f} Mbps | roll100={roll100:.3f} Mbps"
            )

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(train_rows[0].keys()))
        writer.writeheader()
        writer.writerows(train_rows)

    actor_path = out_dir / "ddpg_actor.pt"
    torch.save(actor.state_dict(), actor_path)

    eval_env = UAVEnvironment(make_env_config(cfg.seed + 999, cfg))
    ddpg_eval = evaluate_ddpg(eval_env, actor, device=device, episodes=20)
    random_eval = evaluate_policy(
        "Random Walk",
        random_policy,
        episodes=20,
        seed=cfg.seed + 999,
        env_config=make_env_config(cfg.seed + 999, cfg),
    )
    greedy_eval = evaluate_policy(
        "Distance-Greedy",
        distance_greedy_policy,
        episodes=20,
        seed=cfg.seed + 999,
        env_config=make_env_config(cfg.seed + 999, cfg),
    )

    summary = {
        "fading_model": cfg.fading_model,
        "ddpg_mean_avg_rsec_mbps": ddpg_eval["mean_avg_rsec_mbps"],
        "ddpg_mean_episode_secrecy_mbits": ddpg_eval["mean_episode_secrecy_mbits"],
        "random_mean_avg_rsec_mbps": random_eval["mean_avg_R_sec_mbps"],
        "greedy_mean_avg_rsec_mbps": greedy_eval["mean_avg_R_sec_mbps"],
        "training_log_csv": str(log_path.resolve()),
        "actor_path": str(actor_path.resolve()),
    }

    print("\nFinal comparison (evaluation episodes=20):")
    print(f"  Channel model            : {summary['fading_model']}")
    print(f"  DDPG avg secrecy rate     : {summary['ddpg_mean_avg_rsec_mbps']:.4f} Mbps")
    print(f"  Random avg secrecy rate   : {summary['random_mean_avg_rsec_mbps']:.4f} Mbps")
    print(f"  Greedy avg secrecy rate   : {summary['greedy_mean_avg_rsec_mbps']:.4f} Mbps")
    print(f"  DDPG secrecy payload/ep   : {summary['ddpg_mean_episode_secrecy_mbits']:.4f} Mbits")
    print(f"  Saved training log        : {summary['training_log_csv']}")
    print(f"  Saved actor               : {summary['actor_path']}")
    return summary


def _parse_args():
    parser = argparse.ArgumentParser(description="Train DDPG for dual-UAV secrecy environment.")
    parser.add_argument("--episodes", type=int, default=400, help="Training episodes")
    parser.add_argument(
        "--channel-model",
        type=str,
        default="rician",
        choices=["rician", "rayleigh"],
        help="Fading model to use in the environment",
    )
    parser.add_argument("--rician-k", type=float, default=5.0, help="Rician K-factor")
    parser.add_argument("--output-dir", type=str, default="outputs/ddpg", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_ddpg(
        DDPGConfig(
            episodes=args.episodes,
            seed=args.seed,
            fading_model=args.channel_model,
            rician_k=args.rician_k,
        ),
        output_dir=args.output_dir,
    )
