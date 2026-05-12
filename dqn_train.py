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
from config_utils import build_env_config
from baselines import distance_greedy_policy, evaluate_policy, random_policy


@dataclass
class DQNConfig:
    episodes: int = 400
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 50000
    min_replay_size: int = 1000
    target_update_freq: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 40000
    hidden_dim: int = 128
    seed: int = 42
    device: str = "cpu"
    fading_model: str = "rician"
    rician_k: float = 5.0
    evaluation_episodes: int = 20
    control_mode: str = "velocity"
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


def make_env_config(seed: int, cfg: DQNConfig) -> EnvConfig:
    return build_env_config(
        seed=seed,
        fading_model=cfg.fading_model,
        rician_k=cfg.rician_k,
        control_mode=cfg.control_mode,
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


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def epsilon_by_step(step: int, cfg: DQNConfig) -> float:
    if step >= cfg.epsilon_decay_steps:
        return cfg.epsilon_end
    span = cfg.epsilon_start - cfg.epsilon_end
    frac = step / max(cfg.epsilon_decay_steps, 1)
    return cfg.epsilon_start - span * frac


def evaluate_dqn(
    env: UAVEnvironment,
    q_net: QNetwork,
    action_table: list[tuple[np.ndarray, np.ndarray, float]],
    device: torch.device,
    episodes: int = 20,
) -> dict:
    q_net.eval()
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
                action_id = int(torch.argmax(q_net(s_t), dim=1).item())
                a_relay, a_jammer, jammer_power = action_table[action_id]
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


def train_dqn(
    cfg: DQNConfig | None = None,
    output_dir: str = "outputs/dqn",
) -> dict:
    cfg = cfg or DQNConfig()
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    env = UAVEnvironment(make_env_config(cfg.seed, cfg))
    state_dim = env.reset().shape[0]
    action_table = make_action_table()
    action_dim = len(action_table)

    q_net = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_net = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.replay_size)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "dqn_training_log.csv"

    global_step = 0
    rolling_rewards: list[float] = []
    train_rows: list[dict] = []

    for ep in range(1, cfg.episodes + 1):
        state = env.reset().astype(np.float32)
        done = False
        ep_reward = 0.0
        ep_rsec_bps = 0.0
        ep_rlegit_bps = 0.0
        ep_reve_bps = 0.0
        ep_energy_j = 0.0
        ep_jammer_power = 0.0
        ep_steps = 0
        relay_start = env.relay_position.copy()
        jammer_start = env.jammer_position.copy()
        relay_path_m = 0.0
        jammer_path_m = 0.0

        while not done:
            prev_relay_position = env.relay_position.copy()
            prev_jammer_position = env.jammer_position.copy()
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
            ep_energy_j += info["total_energy_j"]
            ep_jammer_power += info["jammer_power"]
            relay_path_m += float(np.linalg.norm(env.relay_position - prev_relay_position))
            jammer_path_m += float(np.linalg.norm(env.jammer_position - prev_jammer_position))
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
                    q_next = target_net(ns_t).max(dim=1, keepdim=True)[0]
                    q_target = r_t + (1.0 - d_t) * cfg.gamma * q_next

                loss = nn.MSELoss()(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % cfg.target_update_freq == 0:
                    target_net.load_state_dict(q_net.state_dict())

        avg_rsec_mbps = (ep_rsec_bps / max(ep_steps, 1)) / 1e6
        # avg_shaped_reward = (ep_reward / max(ep_steps, 1)) / 1e6
        avg_shaped_reward = ep_reward / max(ep_steps, 1)
        ep_secrecy_mbits = (ep_rsec_bps * env.config.dt) / 1e6
        rolling_rewards.append(avg_rsec_mbps)
        roll20 = float(np.mean(rolling_rewards[-20:]))
        roll100 = float(np.mean(rolling_rewards[-100:]))
        convergence_gap_mbps = float(abs(roll20 - roll100))
        distances = env.compute_distances()
        train_rows.append(
            {
                "episode": ep,
                "global_step": global_step,
                "fading_model": cfg.fading_model,
                "control_mode": cfg.control_mode,
                "user_mobile": cfg.user_mobile,
                "use_los_model": cfg.use_los_model,
                "observation_mode": cfg.observation_mode,
                "normalize_observations": cfg.normalize_observations,
                "enable_energy_harvesting": env.config.enable_energy_harvesting,
                "observation_has_eh": cfg.observation_mode == "full_eh",
                "enable_ntn": env.config.enable_ntn,
                "satellite_altitude_km": env.config.satellite_altitude_km,
                "epsilon": eps,
                "episode_reward_bps_step": float(ep_reward),
                "avg_shaped_reward": float(avg_shaped_reward),
                "avg_R_legit_mbps": float((ep_rlegit_bps / max(ep_steps, 1)) / 1e6),
                "avg_R_eve_mbps": float((ep_reve_bps / max(ep_steps, 1)) / 1e6),
                "avg_R_sec_mbps": float(avg_rsec_mbps),
                "episode_secrecy_mbits": float(ep_secrecy_mbits),
                "avg_energy_j": float(ep_energy_j / max(ep_steps, 1)),
                "avg_jammer_power_w": float(ep_jammer_power / max(ep_steps, 1)),
                "steps": ep_steps,
                "relay_path_m": relay_path_m,
                "jammer_path_m": jammer_path_m,
                "relay_displacement_m": float(np.linalg.norm(env.relay_position - relay_start)),
                "jammer_displacement_m": float(np.linalg.norm(env.jammer_position - jammer_start)),
                "final_d_UR_m": float(distances["d_UR"]),
                "final_d_RB_m": float(distances["d_RB"]),
                "final_d_UE_m": float(distances["d_UE"]),
                "final_d_JE_m": float(distances["d_JE"]),
                "rolling20_avg_R_sec_mbps": roll20,
                "rolling100_avg_R_sec_mbps": roll100,
                "convergence_gap20_100_mbps": convergence_gap_mbps,
            }
        )

        if ep % 25 == 0 or ep == 1 or ep == cfg.episodes:
            print(
                f"Episode {ep:4d}/{cfg.episodes} | eps={eps:.3f} | "
                f"avg_R_sec={avg_rsec_mbps:.3f} Mbps | roll100={roll100:.3f} Mbps"
            )

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(train_rows[0].keys()))
        writer.writeheader()
        writer.writerows(train_rows)

    model_path = out_dir / "dqn_qnet.pt"
    torch.save(q_net.state_dict(), model_path)

    eval_env = UAVEnvironment(make_env_config(cfg.seed + 999, cfg))
    dqn_eval = evaluate_dqn(eval_env, q_net, action_table, device=device, episodes=cfg.evaluation_episodes)
    random_eval = evaluate_policy(
        "Random Walk",
        random_policy,
        episodes=cfg.evaluation_episodes,
        seed=cfg.seed + 999,
        env_config=make_env_config(cfg.seed + 999, cfg),
    )
    greedy_eval = evaluate_policy(
        "Distance-Greedy",
        distance_greedy_policy,
        episodes=cfg.evaluation_episodes,
        seed=cfg.seed + 999,
        env_config=make_env_config(cfg.seed + 999, cfg),
    )

    summary = {
        "fading_model": cfg.fading_model,
        "dqn_mean_avg_rsec_mbps": dqn_eval["mean_avg_rsec_mbps"],
        "dqn_mean_episode_secrecy_mbits": dqn_eval["mean_episode_secrecy_mbits"],
        "random_mean_avg_rsec_mbps": random_eval["mean_avg_R_sec_mbps"],
        "greedy_mean_avg_rsec_mbps": greedy_eval["mean_avg_R_sec_mbps"],
        "training_log_csv": str(log_path.resolve()),
        "model_path": str(model_path.resolve()),
    }

    print(f"\nFinal comparison (evaluation episodes={cfg.evaluation_episodes}):")
    print(f"  Channel model            : {summary['fading_model']}")
    print(f"  DQN avg secrecy rate      : {summary['dqn_mean_avg_rsec_mbps']:.4f} Mbps")
    print(f"  Random avg secrecy rate   : {summary['random_mean_avg_rsec_mbps']:.4f} Mbps")
    print(f"  Greedy avg secrecy rate   : {summary['greedy_mean_avg_rsec_mbps']:.4f} Mbps")
    print(f"  DQN secrecy payload/ep    : {summary['dqn_mean_episode_secrecy_mbits']:.4f} Mbits")
    print(f"  Saved training log        : {summary['training_log_csv']}")
    print(f"  Saved model               : {summary['model_path']}")

    return summary


def _parse_args():
    parser = argparse.ArgumentParser(description="Train DQN for dual-UAV secrecy environment.")
    parser.add_argument("--episodes", type=int, default=400, help="Training episodes")
    parser.add_argument(
        "--channel-model",
        type=str,
        default="rician",
        choices=["rician", "rayleigh"],
        help="Fading model to use in the environment",
    )
    parser.add_argument("--rician-k", type=float, default=5.0, help="Rician K-factor")
    parser.add_argument("--output-dir", type=str, default="outputs/dqn", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Evaluation episodes after training")
    parser.add_argument(
        "--control-mode",
        type=str,
        default="velocity",
        choices=["velocity", "waypoint"],
        help="Velocity-vector or normalized waypoint control",
    )
    parser.add_argument("--enable-ntn", action="store_true", help="Enable NTN satellite-assisted communication")
    parser.add_argument("--satellite-altitude-km", type=float, default=500.0, help="Satellite altitude (km)")
    parser.add_argument("--satellite-horizontal-offset-km", type=float, default=100.0, help="Satellite horizontal offset (km)")
    parser.add_argument("--ntn-carrier-frequency-hz", type=float, default=2e9, help="NTN carrier frequency (Hz)")
    parser.add_argument("--ntn-atmospheric-loss-db", type=float, default=0.5, help="NTN atmospheric loss (dB)")
    parser.add_argument("--ntn-rician-k-db", type=float, default=10.0, help="NTN Rician K-factor (dB)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_dqn(
        DQNConfig(
            episodes=args.episodes,
            seed=args.seed,
            fading_model=args.channel_model,
            rician_k=args.rician_k,
            evaluation_episodes=args.eval_episodes,
            control_mode=args.control_mode,
            enable_ntn=args.enable_ntn,
            satellite_altitude_km=args.satellite_altitude_km,
            satellite_horizontal_offset_km=args.satellite_horizontal_offset_km,
            ntn_carrier_frequency_hz=args.ntn_carrier_frequency_hz,
            ntn_atmospheric_loss_db=args.ntn_atmospheric_loss_db,
            ntn_rician_k_db=args.ntn_rician_k_db,
        ),
        output_dir=args.output_dir,
    )
