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

from core.environment import EnvConfig, UAVEnvironment
from core.config_utils import build_env_config
from analysis.baselines import distance_greedy_policy, evaluate_policy, random_policy
from rl.marl_utils import (
    relay_observation,
    jammer_observation,
    relay_obs_dim,
    jammer_obs_dim,
    make_relay_action_table,
    make_jammer_action_table,
    decode_jammer_action,
)

@dataclass
class MarlDQNConfig:
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
    # MARL options
    agent_obs_mode: str = "shared"  # "shared" or "split"
    # NTN / Satellite
    enable_ntn: bool = False
    satellite_altitude_km: float = 500.0
    satellite_horizontal_offset_km: float = 100.0
    ntn_carrier_frequency_hz: float = 2e9
    ntn_atmospheric_loss_db: float = 0.5
    ntn_rician_k_db: float = 10.0

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


def epsilon_by_step(step: int, cfg: MarlDQNConfig) -> float:
    if step >= cfg.epsilon_decay_steps:
        return cfg.epsilon_end
    span = cfg.epsilon_start - cfg.epsilon_end
    frac = step / max(cfg.epsilon_decay_steps, 1)
    return cfg.epsilon_start - span * frac


def make_env_config(seed: int, cfg: MarlDQNConfig) -> EnvConfig:
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


def compute_q_entropy(q_values: torch.Tensor) -> float:
    probs = torch.softmax(q_values, dim=-1)
    ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean().item()
    return ent

def evaluate_marl_dqn(
    env: UAVEnvironment,
    relay_q_net: QNetwork,
    jammer_q_net: QNetwork,
    relay_action_table: list[np.ndarray],
    jammer_action_table: list[np.ndarray],
    device: torch.device,
    agent_obs_mode: str,
    episodes: int = 20,
) -> dict:
    relay_q_net.eval()
    jammer_q_net.eval()
    episode_sec_mbits = []
    episode_avg_sec_mbps = []

    with torch.no_grad():
        for _ in range(episodes):
            state = env.reset().astype(np.float32)
            done = False
            total_r_sec = 0.0
            steps = 0
            while not done:
                if agent_obs_mode == "shared":
                    r_obs = state
                    j_obs = state
                else:
                    r_obs = relay_observation(state)
                    j_obs = jammer_observation(state)

                rs_t = torch.tensor(r_obs, dtype=torch.float32, device=device).unsqueeze(0)
                js_t = torch.tensor(j_obs, dtype=torch.float32, device=device).unsqueeze(0)
                relay_id = int(torch.argmax(relay_q_net(rs_t), dim=1).item())
                jammer_id = int(torch.argmax(jammer_q_net(js_t), dim=1).item())

                a_relay = relay_action_table[relay_id]
                jammer_vec = jammer_action_table[jammer_id]
                a_jammer_vel, a_jammer_power = decode_jammer_action(jammer_vec)

                next_state, _, done, info = env.step(a_relay, a_jammer_vel, a_jammer_power)
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

def train_marl_dqn(
    cfg: MarlDQNConfig | None = None,
    output_dir: str | None = None,
) -> dict:
    cfg = cfg or MarlDQNConfig()
    if output_dir is None:
        output_dir = f"outputs/training/marl_{cfg.agent_obs_mode}"
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    env = UAVEnvironment(make_env_config(cfg.seed, cfg))
    full_obs = env.reset()
    full_dim = full_obs.shape[0]

    # Agent-specific dimensions
    if cfg.agent_obs_mode == "shared":
        relay_dim = full_dim
        jammer_dim = full_dim
    else:
        relay_dim = relay_obs_dim(full_dim)
        jammer_dim = jammer_obs_dim(full_dim)

    relay_action_table = make_relay_action_table()
    jammer_action_table = make_jammer_action_table()
    relay_action_dim = len(relay_action_table)
    jammer_action_dim = len(jammer_action_table)

    # Networks
    relay_q_net = QNetwork(relay_dim, relay_action_dim, cfg.hidden_dim).to(device)
    relay_target = QNetwork(relay_dim, relay_action_dim, cfg.hidden_dim).to(device)
    relay_target.load_state_dict(relay_q_net.state_dict())
    relay_target.eval()

    jammer_q_net = QNetwork(jammer_dim, jammer_action_dim, cfg.hidden_dim).to(device)
    jammer_target = QNetwork(jammer_dim, jammer_action_dim, cfg.hidden_dim).to(device)
    jammer_target.load_state_dict(jammer_q_net.state_dict())
    jammer_target.eval()

    relay_opt = optim.Adam(relay_q_net.parameters(), lr=cfg.lr)
    jammer_opt = optim.Adam(jammer_q_net.parameters(), lr=cfg.lr)

    relay_replay = ReplayBuffer(cfg.replay_size)
    jammer_replay = ReplayBuffer(cfg.replay_size)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "marl_dqn_training_log.csv"

    relay_step = 0  # separate step counters for epsilon decay
    jammer_step = 0
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
        ep_jammer_pwr = 0.0
        ep_steps = 0
        relay_start = env.relay_position.copy()
        jammer_start = env.jammer_position.copy()
        relay_path_m = 0.0
        jammer_path_m = 0.0
        ep_relay_q_entropy = 0.0
        ep_jammer_q_entropy = 0.0
        ep_relay_loss = 0.0
        ep_jammer_loss = 0.0

        while not done:
            prev_relay_position = env.relay_position.copy()
            prev_jammer_position = env.jammer_position.copy()

            # --- Build per-agent observations ---
            if cfg.agent_obs_mode == "shared":
                r_obs = state
                j_obs = state
            else:
                r_obs = relay_observation(state)
                j_obs = jammer_observation(state)

            # --- Relay agent act ---
            relay_eps = epsilon_by_step(relay_step, cfg)
            if random.random() < relay_eps:
                relay_id = random.randrange(relay_action_dim)
            else:
                with torch.no_grad():
                    rs_t = torch.tensor(r_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    relay_id = int(torch.argmax(relay_q_net(rs_t), dim=1).item())
            a_relay = relay_action_table[relay_id]

            # --- Jammer agent act ---
            jammer_eps = epsilon_by_step(jammer_step, cfg)
            if random.random() < jammer_eps:
                jammer_id = random.randrange(jammer_action_dim)
            else:
                with torch.no_grad():
                    js_t = torch.tensor(j_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    jammer_id = int(torch.argmax(jammer_q_net(js_t), dim=1).item())
            jammer_vec = jammer_action_table[jammer_id]
            a_jammer_vel, a_jammer_power = decode_jammer_action(jammer_vec)

            # --- Environment step ---
            next_state, reward, done, info = env.step(a_relay, a_jammer_vel, a_jammer_power)
            next_state = next_state.astype(np.float32)

            # --- Store transitions ---
            if cfg.agent_obs_mode == "shared":
                next_r_obs = next_state
                next_j_obs = next_state
            else:
                next_r_obs = relay_observation(next_state)
                next_j_obs = jammer_observation(next_state)

            relay_replay.add(r_obs, relay_id, float(reward), next_r_obs, float(done))
            jammer_replay.add(j_obs, jammer_id, float(reward), next_j_obs, float(done))
            state = next_state

            # --- Episode tracking ---
            ep_reward += reward
            ep_rlegit_bps += info["R_legit"]
            ep_reve_bps += info["R_eve"]
            ep_rsec_bps += info["R_sec"]
            ep_energy_j += info["total_energy_j"]
            ep_jammer_pwr += info["jammer_power"]
            relay_path_m += float(np.linalg.norm(env.relay_position - prev_relay_position))
            jammer_path_m += float(np.linalg.norm(env.jammer_position - prev_jammer_position))
            ep_steps += 1
            relay_step += 1
            jammer_step += 1

            # --- Q-entropy logging (softmax(Q) sharpness, not policy entropy) ---
            with torch.no_grad():
                rs_t = torch.tensor(r_obs, dtype=torch.float32, device=device).unsqueeze(0)
                js_t = torch.tensor(j_obs, dtype=torch.float32, device=device).unsqueeze(0)
                ep_relay_q_entropy += compute_q_entropy(relay_q_net(rs_t))
                ep_jammer_q_entropy += compute_q_entropy(jammer_q_net(js_t))

            # --- Training step (Relay) ---
            if len(relay_replay) >= cfg.min_replay_size:
                s, a, r, ns, d = relay_replay.sample(cfg.batch_size)
                s_t = torch.tensor(s, dtype=torch.float32, device=device)
                a_t = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
                r_t = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
                ns_t = torch.tensor(ns, dtype=torch.float32, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)

                q_pred = relay_q_net(s_t).gather(1, a_t)
                with torch.no_grad():
                    q_next = relay_target(ns_t).max(dim=1, keepdim=True)[0]
                    q_target = r_t + (1.0 - d_t) * cfg.gamma * q_next

                r_loss = nn.MSELoss()(q_pred, q_target)
                relay_opt.zero_grad()
                r_loss.backward()
                relay_opt.step()
                ep_relay_loss += r_loss.item()

                if relay_step % cfg.target_update_freq == 0:
                    relay_target.load_state_dict(relay_q_net.state_dict())

            # --- Training step (Jammer) ---
            if len(jammer_replay) >= cfg.min_replay_size:
                s, a, r, ns, d = jammer_replay.sample(cfg.batch_size)
                s_t = torch.tensor(s, dtype=torch.float32, device=device)
                a_t = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
                r_t = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
                ns_t = torch.tensor(ns, dtype=torch.float32, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)

                q_pred = jammer_q_net(s_t).gather(1, a_t)
                with torch.no_grad():
                    q_next = jammer_target(ns_t).max(dim=1, keepdim=True)[0]
                    q_target = r_t + (1.0 - d_t) * cfg.gamma * q_next

                j_loss = nn.MSELoss()(q_pred, q_target)
                jammer_opt.zero_grad()
                j_loss.backward()
                jammer_opt.step()
                ep_jammer_loss += j_loss.item()

                if jammer_step % cfg.target_update_freq == 0:
                    jammer_target.load_state_dict(jammer_q_net.state_dict())

        # --- Episode end: log ---
        avg_rsec_mbps = (ep_rsec_bps / max(ep_steps, 1)) / 1e6
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
                "global_step_relay": relay_step,
                "global_step_jammer": jammer_step,
                "fading_model": cfg.fading_model,
                "control_mode": cfg.control_mode,
                "user_mobile": cfg.user_mobile,
                "use_los_model": cfg.use_los_model,
                "observation_mode": cfg.observation_mode,
                "agent_obs_mode": cfg.agent_obs_mode,
                "normalize_observations": cfg.normalize_observations,
                "enable_ntn": env.config.enable_ntn,
                "satellite_altitude_km": env.config.satellite_altitude_km,
                "relay_epsilon": relay_eps,
                "jammer_epsilon": jammer_eps,
                "episode_reward_bps_step": float(ep_reward),
                "avg_shaped_reward": float(avg_shaped_reward),
                "avg_R_legit_mbps": float((ep_rlegit_bps / max(ep_steps, 1)) / 1e6),
                "avg_R_eve_mbps": float((ep_reve_bps / max(ep_steps, 1)) / 1e6),
                "avg_R_sec_mbps": float(avg_rsec_mbps),
                "episode_secrecy_mbits": float(ep_secrecy_mbits),
                "avg_energy_j": float(ep_energy_j / max(ep_steps, 1)),
                "avg_jammer_power_w": float(ep_jammer_pwr / max(ep_steps, 1)),
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
                # Both agents optimise the SAME global cooperative reward.
                "relay_q_entropy": float(ep_relay_q_entropy / max(ep_steps, 1)),
                "jammer_q_entropy": float(ep_jammer_q_entropy / max(ep_steps, 1)),
                "relay_loss": float(ep_relay_loss / max(ep_steps, 1)),
                "jammer_loss": float(ep_jammer_loss / max(ep_steps, 1)),
            }
        )

        if ep % 25 == 0 or ep == 1 or ep == cfg.episodes:
            print(
                f"MARL Ep {ep:4d}/{cfg.episodes} | "
                f"r_eps={relay_eps:.3f} j_eps={jammer_eps:.3f} | "
                f"R_sec={avg_rsec_mbps:.3f} Mbps | "
                f"roll100={roll100:.3f} Mbps"
            )

    # --- Save log ---
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(train_rows[0].keys()))
        writer.writeheader()
        writer.writerows(train_rows)

    # --- Save models ---
    relay_path = out_dir / "marl_relay_qnet.pt"
    jammer_path = out_dir / "marl_jammer_qnet.pt"
    torch.save(relay_q_net.state_dict(), relay_path)
    torch.save(jammer_q_net.state_dict(), jammer_path)

    # --- Evaluation ---
    eval_env = UAVEnvironment(make_env_config(cfg.seed + 999, cfg))
    marl_eval = evaluate_marl_dqn(
        eval_env, relay_q_net, jammer_q_net,
        relay_action_table, jammer_action_table,
        device, cfg.agent_obs_mode,
        episodes=cfg.evaluation_episodes,
    )
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
        "agent_obs_mode": cfg.agent_obs_mode,
        "marl_mean_avg_rsec_mbps": marl_eval["mean_avg_rsec_mbps"],
        "marl_mean_episode_secrecy_mbits": marl_eval["mean_episode_secrecy_mbits"],
        "random_mean_avg_rsec_mbps": random_eval["mean_avg_R_sec_mbps"],
        "greedy_mean_avg_rsec_mbps": greedy_eval["mean_avg_R_sec_mbps"],
        "training_log_csv": str(log_path.resolve()),
        "relay_model_path": str(relay_path.resolve()),
        "jammer_model_path": str(jammer_path.resolve()),
    }

    print(f"\nMARL DQN evaluation (episodes={cfg.evaluation_episodes}):")
    print(f"  Channel model        : {summary['fading_model']}")
    print(f"  Agent obs mode       : {summary['agent_obs_mode']}")
    print(f"  MARL avg secrecy rate : {summary['marl_mean_avg_rsec_mbps']:.4f} Mbps")
    print(f"  Random avg secrecy    : {summary['random_mean_avg_rsec_mbps']:.4f} Mbps")
    print(f"  Greedy avg secrecy    : {summary['greedy_mean_avg_rsec_mbps']:.4f} Mbps")

    return summary

def _parse_args():
    parser = argparse.ArgumentParser(description="Train MARL DQN for dual-UAV secrecy.")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--channel-model", type=str, default="rician", choices=["rician", "rayleigh"])
    parser.add_argument("--rician-k", type=float, default=5.0)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto-derived from agent-obs-mode)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--control-mode", type=str, default="velocity", choices=["velocity", "waypoint"])
    parser.add_argument("--agent-obs-mode", type=str, default="shared", choices=["shared", "split"])
    parser.add_argument("--observation-mode", type=str, default="full", choices=["geometry", "channels", "full", "full_eh"])
    parser.add_argument("--enable-ntn", action="store_true", help="Enable NTN satellite-assisted communication")
    parser.add_argument("--satellite-altitude-km", type=float, default=500.0, help="Satellite altitude (km)")
    parser.add_argument("--satellite-horizontal-offset-km", type=float, default=100.0, help="Satellite horizontal offset (km)")
    parser.add_argument("--ntn-carrier-frequency-hz", type=float, default=2e9, help="NTN carrier frequency (Hz)")
    parser.add_argument("--ntn-atmospheric-loss-db", type=float, default=0.5, help="NTN atmospheric loss (dB)")
    parser.add_argument("--ntn-rician-k-db", type=float, default=10.0, help="NTN Rician K-factor (dB)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"outputs/training/marl_{args.agent_obs_mode}"
    train_marl_dqn(
        MarlDQNConfig(
            episodes=args.episodes,
            seed=args.seed,
            fading_model=args.channel_model,
            rician_k=args.rician_k,
            evaluation_episodes=args.eval_episodes,
            control_mode=args.control_mode,
            observation_mode=args.observation_mode,
            agent_obs_mode=args.agent_obs_mode,
            enable_ntn=args.enable_ntn,
            satellite_altitude_km=args.satellite_altitude_km,
            satellite_horizontal_offset_km=args.satellite_horizontal_offset_km,
            ntn_carrier_frequency_hz=args.ntn_carrier_frequency_hz,
            ntn_atmospheric_loss_db=args.ntn_atmospheric_loss_db,
            ntn_rician_k_db=args.ntn_rician_k_db,
        ),
        output_dir=output_dir,
    )
