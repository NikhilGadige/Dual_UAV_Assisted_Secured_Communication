import csv
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.config_utils import build_env_config
from core.environment import EnvConfig, UAVEnvironment


@dataclass
class D3QNConfig:
    episodes: int = 3000
    gamma: float = 0.99
    lr: float = 8e-4
    batch_size: int = 64
    replay_size: int = 50000
    min_replay_size: int = 2000
    hidden_dim: int = 64
    epsilon_start: float = 1.0
    epsilon_end: float = 0.03
    epsilon_decay_steps: int = 3000 * 120
    target_update_tau: float = 0.005
    grad_clip_norm: float = 5.0
    td_target_clip: float = 20.0
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


def make_env_config(seed: int, cfg: D3QNConfig) -> EnvConfig:
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


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feature(x)
        value = self.value_stream(z)
        advantage = self.adv_stream(z)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


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
            np.asarray(a, dtype=np.int64),
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


def make_action_table() -> list[tuple[np.ndarray, np.ndarray, float]]:
    dirs = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([0.0, -1.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([1.0, -1.0], dtype=np.float32),
        np.array([-1.0, 1.0], dtype=np.float32),
        np.array([-1.0, -1.0], dtype=np.float32),
    ]
    speed_levels = [0.0, 0.5, 1.0]
    power_levels = [-1.0, 0.0, 1.0]
    table = []
    for relay_speed in speed_levels:
        relay_commands = [np.zeros(2, dtype=np.float32)] if relay_speed == 0.0 else [
            relay_speed * (direction / np.linalg.norm(direction)) for direction in dirs
        ]
        for jammer_speed in speed_levels:
            jammer_commands = [np.zeros(2, dtype=np.float32)] if jammer_speed == 0.0 else [
                jammer_speed * (direction / np.linalg.norm(direction)) for direction in dirs
            ]
            for a_r in relay_commands:
                for a_j in jammer_commands:
                    for jammer_power in power_levels:
                        table.append((a_r.astype(np.float32), a_j.astype(np.float32), float(jammer_power)))
    return table


def epsilon_by_step(step: int, cfg: D3QNConfig) -> float:
    if step >= cfg.epsilon_decay_steps:
        return cfg.epsilon_end
    span = cfg.epsilon_start - cfg.epsilon_end
    frac = step / max(cfg.epsilon_decay_steps, 1)
    return cfg.epsilon_start - span * frac


def soft_update(src: nn.Module, dst: nn.Module, tau: float) -> None:
    for p_src, p_dst in zip(src.parameters(), dst.parameters()):
        p_dst.data.copy_(tau * p_src.data + (1.0 - tau) * p_dst.data)


def evaluate_d3qn(
    env: UAVEnvironment,
    q_net: DuelingQNetwork,
    action_table: list[tuple[np.ndarray, np.ndarray, float]],
    device: torch.device,
    episodes: int = 5,
) -> dict:
    q_net.eval()
    values = []
    with torch.no_grad():
        for _ in range(episodes):
            state = env.reset().astype(np.float32)
            done = False
            total_r_sec = 0.0
            steps = 0
            while not done:
                action_id = int(torch.argmax(q_net(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)), dim=1).item())
                a_relay, a_jammer, jammer_power = action_table[action_id]
                next_state, _, done, info = env.step(a_relay, a_jammer, jammer_power)
                total_r_sec += info["R_sec"]
                steps += 1
                state = next_state.astype(np.float32)
            values.append((total_r_sec / max(steps, 1)) / 1e6)
    return {"mean_avg_rsec_mbps": float(np.mean(values))}


def train_d3qn(cfg: D3QNConfig, output_dir: str) -> dict:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    env = UAVEnvironment(make_env_config(cfg.seed, cfg))
    state_dim = env.reset().shape[0]
    action_table = make_action_table()
    action_dim = len(action_table)

    q_net = DuelingQNetwork(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_net = DuelingQNetwork(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    replay = ReplayBuffer(cfg.replay_size)
    optimizer = optim.Adam(q_net.parameters(), lr=cfg.lr)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.csv"
    model_path = out_dir / "checkpoints" / "qnet.pt"

    global_step = 0
    rolling_rewards = []
    rows = []
    last_eval = ""

    for ep in range(1, cfg.episodes + 1):
        state = env.reset().astype(np.float32)
        done = False
        ep_reward = 0.0
        ep_rsec_bps = 0.0
        ep_rlegit_bps = 0.0
        ep_reve_bps = 0.0
        ep_steps = 0
        ep_num_eves = 0
        ep_nearest_eve_dist = 0.0
        ep_mean_eve_dist = 0.0
        ep_max_eve_cap = 0.0

        while not done:
            eps = epsilon_by_step(global_step, cfg)
            if random.random() < eps:
                action_id = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action_id = int(torch.argmax(q_net(s_t), dim=1).item())

            a_relay, a_jammer, jammer_power = action_table[action_id]
            next_state, reward, done, info = env.step(a_relay, a_jammer, jammer_power)
            next_state = next_state.astype(np.float32)
            replay.add(state, action_id, float(reward), next_state, float(done))

            state = next_state
            ep_reward += reward
            ep_rlegit_bps += info["R_legit"]
            ep_reve_bps += info["R_eve"]
            ep_rsec_bps += info["R_sec"]
            ep_num_eves += info.get("num_eves", 1)
            ep_nearest_eve_dist += info.get("nearest_eve_distance", 0.0)
            ep_mean_eve_dist += info.get("mean_eve_distance", 0.0)
            ep_max_eve_cap += info.get("max_eve_capacity", 0.0)
            ep_steps += 1
            global_step += 1

            if len(replay) >= cfg.min_replay_size:
                s, a, r, ns, d = replay.sample(cfg.batch_size)
                s_t = torch.tensor(s, dtype=torch.float32, device=device)
                a_t = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
                r_t = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
                ns_t = torch.tensor(ns, dtype=torch.float32, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)

                q_pred = q_net(s_t).gather(1, a_t)
                with torch.no_grad():
                    # Double DQN: online net selects action, target net evaluates it.
                    next_action_online = q_net(ns_t).argmax(dim=1, keepdim=True)
                    q_next = target_net(ns_t).gather(1, next_action_online)
                    q_target = r_t + (1.0 - d_t) * cfg.gamma * q_next
                    q_target = torch.clamp(q_target, -cfg.td_target_clip, cfg.td_target_clip)

                loss = nn.SmoothL1Loss()(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                soft_update(q_net, target_net, cfg.target_update_tau)

        avg_rsec_mbps = (ep_rsec_bps / max(ep_steps, 1)) / 1e6
        avg_shaped_reward = ep_reward / max(ep_steps, 1)
        rolling_rewards.append(avg_rsec_mbps)
        roll20 = float(np.mean(rolling_rewards[-20:]))
        roll100 = float(np.mean(rolling_rewards[-100:]))
        gap = float(abs(roll20 - roll100))
        eval_rsec = ""

        if cfg.eval_interval > 0 and ep % cfg.eval_interval == 0:
            eval_env = UAVEnvironment(make_env_config(cfg.seed + 7000 + ep, cfg))
            eval_summary = evaluate_d3qn(eval_env, q_net, action_table, device, episodes=cfg.train_eval_episodes)
            last_eval = eval_summary["mean_avg_rsec_mbps"]
            eval_rsec = last_eval

        rows.append(
            {
                "episode": ep,
                "global_step": global_step,
                "fading_model": cfg.fading_model,
                "hidden_dim": cfg.hidden_dim,
                "epsilon": eps,
                "avg_shaped_reward": float(avg_shaped_reward),
                "avg_R_legit_mbps": float((ep_rlegit_bps / max(ep_steps, 1)) / 1e6),
                "avg_R_eve_mbps": float((ep_reve_bps / max(ep_steps, 1)) / 1e6),
                "avg_R_sec_mbps": float(avg_rsec_mbps),
                "avg_num_eves": float(ep_num_eves / max(ep_steps, 1)),
                "avg_nearest_eve_distance": float(ep_nearest_eve_dist / max(ep_steps, 1)),
                "avg_mean_eve_distance": float(ep_mean_eve_dist / max(ep_steps, 1)),
                "avg_max_eve_capacity": float((ep_max_eve_cap / max(ep_steps, 1)) / 1e6),
                "eval_R_sec_mbps": eval_rsec,
                "last_eval_R_sec_mbps": last_eval,
                "steps": ep_steps,
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
                f"Episode {ep:4d}/{cfg.episodes} | eps={eps:.3f} | "
                f"avg_R_sec={avg_rsec_mbps:.3f} Mbps | roll100={roll100:.3f} Mbps"
            )

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    torch.save(q_net.state_dict(), model_path)

    plots_dir = out_dir / "plots"
    from d3qn_study.plot_d3qn import generate_single_run_plots as d3qn_plots
    d3qn_plots(str(log_path), str(plots_dir), f"D3QN + {cfg.fading_model.title()}", "#1f77b4")

    return {
        "training_log_csv": str(log_path.resolve()),
        "model_path": str(model_path.resolve()),
    }
