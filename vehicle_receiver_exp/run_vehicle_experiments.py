"""Run all vehicle receiver experiments.

6 algorithms x 2 channel models = 12 experiments.
Each experiment uses VehicleUAVEnvironment with configurable mobility.
"""

import argparse
import csv
import os
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.config_utils import build_env_config
from core.environment import EnvConfig, UAVEnvironment
from analysis.baselines import evaluate_policy, random_policy

from vehicle_receiver_exp.vehicle_models import VehicleUAVEnvironment
from vehicle_receiver_exp.configs import (
    build_output_dir,
    build_vehicle_dqn_config,
    build_vehicle_ddpg_config,
    build_vehicle_d3qn_config,
    build_vehicle_ppo_config,
    build_vehicle_sac_config,
    build_vehicle_td3pg_config,
    build_run_name,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_vehicle_env_config(seed: int, fading_model: str, control_mode: str = "velocity") -> EnvConfig:
    return build_env_config(
        seed=seed,
        fading_model=fading_model,
        rician_k=5.0,
        control_mode=control_mode,
        role_switching=False,
        user_mobile=True,
        use_los_model=False,
        observation_mode="full",
        normalize_observations=True,
    )


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def _apply_plot_style() -> None:
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 200, "axes.titlesize": 13, "axes.labelsize": 11})


def _save_plot(episodes, values, title, ylabel, path, color="#1f77b4"):
    import matplotlib.pyplot as plt
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(episodes, values, color=color, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Episode"); ax.set_ylabel(ylabel)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_vehicle_plots(csv_path: str, output_dir: str, run_key: str) -> dict:
    import matplotlib.pyplot as plt
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    episodes = [int(r["episode"]) for r in rows]
    secrecy = [float(r.get("avg_R_sec_mbps", 0)) for r in rows]
    reward = [float(r.get("avg_shaped_reward", 0)) for r in rows]
    roll100 = [float(r.get("rolling100_avg_R_sec_mbps", r.get("rolling100", 0))) for r in rows]

    color = "#2ca02c" if "rician" in run_key else "#d62728"

    reward_p = out / "reward_curve.png"
    _save_plot(episodes, reward, f"{run_key} - Avg Reward", "Avg Shaped Reward", reward_p, color)

    secrecy_p = out / "secrecy_curve.png"
    _save_plot(episodes, secrecy, f"{run_key} - Secrecy Rate", "Secrecy Rate (Mbps)", secrecy_p, color)

    roll100_p = out / "rolling100_curve.png"
    _save_plot(episodes, roll100, f"{run_key} - Rolling 100 Secrecy", "Rolling100 Secrecy (Mbps)", roll100_p, color)

    return {"reward_curve": str(reward_p), "secrecy_curve": str(secrecy_p), "rolling100_curve": str(roll100_p)}


# ---------------------------------------------------------------------------
# Make env factories
# ---------------------------------------------------------------------------

def make_vehicle_env(fading_model: str, seed: int, mobility_mode: str = "straight_road", vehicle_max_speed: float = 10.0):
    cfg = build_vehicle_env_config(seed, fading_model)
    return VehicleUAVEnvironment(cfg, mobility_mode=mobility_mode, vehicle_max_speed=vehicle_max_speed)


# ===================================================================
# DQN Training
# ===================================================================

from rl.dqn_train import QNetwork, make_action_table, ReplayBuffer as DQNReplayBuffer


def train_vehicle_dqn(fading_model: str, output_dir: str, episodes: int = 100, seed: int = 42, mobility_mode: str = "straight_road"):
    cfg = build_vehicle_dqn_config(fading_model, episodes, seed)
    set_seed(cfg.seed)
    device = torch.device("cpu")
    env = make_vehicle_env(fading_model, cfg.seed, mobility_mode)
    state_dim = env.reset().shape[0]
    action_table = make_action_table()
    action_dim = len(action_table)

    q_net = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_net = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay = DQNReplayBuffer(cfg.replay_size)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.csv"

    global_step = 0
    rolling = []
    train_rows = []

    for ep in range(1, cfg.episodes + 1):
        state = env.reset().astype(np.float32)
        done = False
        ep_reward = 0.0
        ep_rsec = 0.0
        ep_rlegit = 0.0
        ep_reve = 0.0
        steps = 0

        while not done:
            eps = max(cfg.epsilon_end, cfg.epsilon_start - (cfg.epsilon_start - cfg.epsilon_end) * global_step / max(cfg.epsilon_decay_steps, 1))
            if random.random() < eps:
                action_id = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    action_id = int(torch.argmax(q_net(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)), dim=1).item())

            a_relay, a_jammer, jammer_power = action_table[action_id]
            next_state, reward, done, info = env.step(a_relay, a_jammer, jammer_power)
            next_state = next_state.astype(np.float32)
            replay.add(state, action_id, float(reward), next_state, float(done))
            state = next_state
            ep_reward += reward
            ep_rsec += info["R_sec"]
            ep_rlegit += info["R_legit"]
            ep_reve += info["R_eve"]
            steps += 1
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
                    online_next = q_net(ns_t).argmax(dim=1, keepdim=True)
                    q_next = target_net(ns_t).gather(1, online_next)
                    q_target = r_t + (1.0 - d_t) * cfg.gamma * q_next
                loss = nn.MSELoss()(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for p_src, p_dst in zip(q_net.parameters(), target_net.parameters()):
                    p_dst.data.copy_(cfg.target_update_tau * p_src.data + (1.0 - cfg.target_update_tau) * p_dst.data)

        avg_rsec = (ep_rsec / max(steps, 1)) / 1e6
        rolling.append(avg_rsec)
        roll20 = float(np.mean(rolling[-20:]))
        roll100 = float(np.mean(rolling[-100:]))
        train_rows.append({
            "episode": ep, "global_step": global_step, "fading_model": cfg.fading_model,
            "avg_shaped_reward": float(ep_reward / max(steps, 1)),
            "avg_R_legit_mbps": float((ep_rlegit / max(steps, 1)) / 1e6),
            "avg_R_eve_mbps": float((ep_reve / max(steps, 1)) / 1e6),
            "avg_R_sec_mbps": float(avg_rsec),
            "eval_R_sec_mbps": "", "last_eval_R_sec_mbps": "",
            "steps": steps,
            "rolling20": roll20, "rolling20_avg_R_sec_mbps": roll20,
            "rolling100": roll100, "rolling100_avg_R_sec_mbps": roll100,
            "convergence_gap": float(abs(roll20 - roll100)),
            "convergence_gap20_100_mbps": float(abs(roll20 - roll100)),
        })
        if ep == 1 or ep % 25 == 0 or ep == cfg.episodes:
            print(f"  [DQN] Ep {ep:4d}/{cfg.episodes} | R_sec={avg_rsec:.3f} Mbps | roll100={roll100:.3f}")

    with log_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(train_rows[0].keys()))
        w.writeheader(); w.writerows(train_rows)

    plots = generate_vehicle_plots(str(log_path), str(out_dir), f"DQN_{fading_model}")
    return {"training_log_csv": str(log_path.resolve()), **plots, "final_roll100": roll100, "best_secrecy": float(np.max(rolling)), "avg_reward": float(np.mean([r["avg_shaped_reward"] for r in train_rows]))}


# ===================================================================
# DDPG Training
# ===================================================================

from rl.ddpg_train import Actor, Critic, OUNoise, ReplayBuffer as DDPGReplayBuffer


def _split_action_5d(action):
    return action[:2], action[2:4], float(action[4]), bool(action.shape[0] > 5 and action[5] > 0.5)


def train_vehicle_ddpg(fading_model: str, output_dir: str, episodes: int = 100, seed: int = 42, mobility_mode: str = "straight_road"):
    cfg = build_vehicle_ddpg_config(fading_model, episodes, seed)
    set_seed(cfg.seed)
    device = torch.device("cpu")
    env = make_vehicle_env(fading_model, cfg.seed, mobility_mode)
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
    replay = DDPGReplayBuffer(cfg.replay_size)
    ou_noise = OUNoise(action_dim, theta=cfg.ou_theta, dt=cfg.ou_dt)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.csv"

    global_step = 0
    rolling = []
    train_rows = []

    for ep in range(1, cfg.episodes + 1):
        state = env.reset().astype(np.float32)
        done = False
        ep_reward = 0.0
        ep_rsec = 0.0
        ep_rlegit = 0.0
        ep_reve = 0.0
        steps = 0
        ou_noise.reset()

        while not done:
            noise_std = cfg.noise_std_end if global_step >= cfg.noise_decay_steps else cfg.noise_std_start - (cfg.noise_std_start - cfg.noise_std_end) * global_step / max(cfg.noise_decay_steps, 1)
            with torch.no_grad():
                action = actor(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).cpu().numpy()[0]
            if cfg.noise_type == "ou":
                noise = ou_noise.sample(noise_std)
            else:
                noise = np.random.normal(0.0, noise_std, size=action_dim)
            action = np.clip(action + noise, -1.0, 1.0)
            a_relay, a_jammer, jammer_power, _ = _split_action_5d(action)
            next_state, reward, done, info = env.step(a_relay, a_jammer, jammer_power)
            next_state = next_state.astype(np.float32)
            replay.add(state, action, float(reward), next_state, float(done))
            state = next_state
            ep_reward += reward; ep_rsec += info["R_sec"]; ep_rlegit += info["R_legit"]; ep_reve += info["R_eve"]
            steps += 1; global_step += 1

            if len(replay) >= cfg.min_replay_size:
                s, a, r, ns, d = replay.sample(cfg.batch_size)
                s_t = torch.tensor(s, dtype=torch.float32, device=device)
                a_t = torch.tensor(a, dtype=torch.float32, device=device)
                r_t = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
                ns_t = torch.tensor(ns, dtype=torch.float32, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)
                with torch.no_grad():
                    q_next = target_critic(ns_t, target_actor(ns_t))
                    q_target = r_t + (1.0 - d_t) * cfg.gamma * q_next
                c_loss = nn.MSELoss()(critic(s_t, a_t), q_target)
                critic_opt.zero_grad(); c_loss.backward(); critic_opt.step()
                a_loss = -critic(s_t, actor(s_t)).mean()
                actor_opt.zero_grad(); a_loss.backward(); actor_opt.step()
                for ps, pd in zip(actor.parameters(), target_actor.parameters()):
                    pd.data.copy_(cfg.tau * ps.data + (1.0 - cfg.tau) * pd.data)
                for ps, pd in zip(critic.parameters(), target_critic.parameters()):
                    pd.data.copy_(cfg.tau * ps.data + (1.0 - cfg.tau) * pd.data)

        avg_rsec = (ep_rsec / max(steps, 1)) / 1e6
        rolling.append(avg_rsec)
        roll20 = float(np.mean(rolling[-20:])); roll100 = float(np.mean(rolling[-100:]))
        train_rows.append({
            "episode": ep, "global_step": global_step, "fading_model": cfg.fading_model,
            "avg_shaped_reward": float(ep_reward / max(steps, 1)),
            "avg_R_legit_mbps": float((ep_rlegit / max(steps, 1)) / 1e6),
            "avg_R_eve_mbps": float((ep_reve / max(steps, 1)) / 1e6),
            "avg_R_sec_mbps": float(avg_rsec),
            "eval_R_sec_mbps": "", "last_eval_R_sec_mbps": "", "steps": steps,
            "rolling20": roll20, "rolling20_avg_R_sec_mbps": roll20,
            "rolling100": roll100, "rolling100_avg_R_sec_mbps": roll100,
            "convergence_gap": float(abs(roll20 - roll100)),
            "convergence_gap20_100_mbps": float(abs(roll20 - roll100)),
        })
        if ep == 1 or ep % 25 == 0 or ep == cfg.episodes:
            print(f"  [DDPG] Ep {ep:4d}/{cfg.episodes} | R_sec={avg_rsec:.3f} Mbps | roll100={roll100:.3f}")

    with log_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(train_rows[0].keys()))
        w.writeheader(); w.writerows(train_rows)
    plots = generate_vehicle_plots(str(log_path), str(out_dir), f"DDPG_{fading_model}")
    return {"training_log_csv": str(log_path.resolve()), **plots, "final_roll100": roll100, "best_secrecy": float(np.max(rolling)), "avg_reward": float(np.mean([r["avg_shaped_reward"] for r in train_rows]))}


# ===================================================================
# D3QN Training
# ===================================================================

from d3qn_study.train_d3qn import DuelingQNetwork, ReplayBuffer as D3QNReplayBuffer


def train_vehicle_d3qn(fading_model: str, output_dir: str, episodes: int = 100, seed: int = 42, mobility_mode: str = "straight_road"):
    cfg = build_vehicle_d3qn_config(fading_model, episodes, seed)
    set_seed(cfg.seed)
    device = torch.device("cpu")
    env = make_vehicle_env(fading_model, cfg.seed, mobility_mode)
    state_dim = env.reset().shape[0]
    action_table = make_action_table()
    action_dim = len(action_table)

    q_net = DuelingQNetwork(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_net = DuelingQNetwork(state_dim, action_dim, cfg.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay = D3QNReplayBuffer(cfg.replay_size)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.csv"

    global_step = 0
    rolling = []
    train_rows = []

    for ep in range(1, cfg.episodes + 1):
        state = env.reset().astype(np.float32)
        done = False
        ep_reward = 0.0; ep_rsec = 0.0; ep_rlegit = 0.0; ep_reve = 0.0; steps = 0

        while not done:
            eps = max(cfg.epsilon_end, cfg.epsilon_start - (cfg.epsilon_start - cfg.epsilon_end) * global_step / max(cfg.epsilon_decay_steps, 1))
            if random.random() < eps:
                action_id = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    action_id = int(torch.argmax(q_net(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)), dim=1).item())
            a_relay, a_jammer, jammer_power = action_table[action_id]
            next_state, reward, done, info = env.step(a_relay, a_jammer, jammer_power)
            next_state = next_state.astype(np.float32)
            replay.add(state, action_id, float(reward), next_state, float(done))
            state = next_state
            ep_reward += reward; ep_rsec += info["R_sec"]; ep_rlegit += info["R_legit"]; ep_reve += info["R_eve"]
            steps += 1; global_step += 1

            if len(replay) >= cfg.min_replay_size:
                s, a, r, ns, d = replay.sample(cfg.batch_size)
                s_t = torch.tensor(s, dtype=torch.float32, device=device)
                a_t = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
                r_t = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
                ns_t = torch.tensor(ns, dtype=torch.float32, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)
                q_pred = q_net(s_t).gather(1, a_t)
                with torch.no_grad():
                    next_online = q_net(ns_t).argmax(dim=1, keepdim=True)
                    q_next = target_net(ns_t).gather(1, next_online)
                    q_target = r_t + (1.0 - d_t) * cfg.gamma * q_next
                loss = nn.SmoothL1Loss()(q_pred, q_target)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                for ps, pd in zip(q_net.parameters(), target_net.parameters()):
                    pd.data.copy_(cfg.target_update_tau * ps.data + (1.0 - cfg.target_update_tau) * pd.data)

        avg_rsec = (ep_rsec / max(steps, 1)) / 1e6
        rolling.append(avg_rsec)
        roll20 = float(np.mean(rolling[-20:])); roll100 = float(np.mean(rolling[-100:]))
        train_rows.append({
            "episode": ep, "global_step": global_step, "fading_model": cfg.fading_model,
            "avg_shaped_reward": float(ep_reward / max(steps, 1)),
            "avg_R_legit_mbps": float((ep_rlegit / max(steps, 1)) / 1e6),
            "avg_R_eve_mbps": float((ep_reve / max(steps, 1)) / 1e6),
            "avg_R_sec_mbps": float(avg_rsec),
            "eval_R_sec_mbps": "", "last_eval_R_sec_mbps": "", "steps": steps,
            "rolling20": roll20, "rolling20_avg_R_sec_mbps": roll20,
            "rolling100": roll100, "rolling100_avg_R_sec_mbps": roll100,
            "convergence_gap": float(abs(roll20 - roll100)),
            "convergence_gap20_100_mbps": float(abs(roll20 - roll100)),
        })
        if ep == 1 or ep % 25 == 0 or ep == cfg.episodes:
            print(f"  [D3QN] Ep {ep:4d}/{cfg.episodes} | R_sec={avg_rsec:.3f} Mbps | roll100={roll100:.3f}")

    with log_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(train_rows[0].keys()))
        w.writeheader(); w.writerows(train_rows)
    plots = generate_vehicle_plots(str(log_path), str(out_dir), f"D3QN_{fading_model}")
    return {"training_log_csv": str(log_path.resolve()), **plots, "final_roll100": roll100, "best_secrecy": float(np.max(rolling)), "avg_reward": float(np.mean([r["avg_shaped_reward"] for r in train_rows]))}


# ===================================================================
# PPO Training
# ===================================================================

from rl.advanced_rl_train import GaussianActor as PPOActor


def train_vehicle_ppo(fading_model: str, output_dir: str, episodes: int = 100, seed: int = 42, mobility_mode: str = "straight_road"):
    cfg = build_vehicle_ppo_config(fading_model, episodes, seed)
    set_seed(cfg.seed)
    device = torch.device("cpu")
    env = make_vehicle_env(fading_model, cfg.seed, mobility_mode)
    state_dim = env.reset().shape[0]
    action_dim = 5

    actor = PPOActor(state_dim, action_dim, cfg.hidden_dim).to(device)
    value = nn.Sequential(
        nn.Linear(state_dim, cfg.hidden_dim), nn.ReLU(),
        nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(),
        nn.Linear(cfg.hidden_dim, 1),
    ).to(device)
    actor_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    value_opt = optim.Adam(value.parameters(), lr=cfg.critic_lr)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.csv"

    global_step = 0
    rolling = []
    train_rows = []

    for ep in range(1, cfg.episodes + 1):
        states = []; actions = []; logps = []; rewards = []
        state = env.reset().astype(np.float32)
        done = False
        ep_reward = 0.0; ep_rsec = 0.0; ep_rlegit = 0.0; ep_reve = 0.0; steps = 0

        while not done:
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, logp = actor.sample(s_t)
            action_np = action.cpu().numpy()[0]
            a_relay, a_jammer, jammer_power, _ = _split_action_5d(action_np)
            next_state, reward, done, info = env.step(a_relay, a_jammer, jammer_power)
            states.append(state); actions.append(action_np); logps.append(float(logp.item())); rewards.append(float(reward))
            state = next_state.astype(np.float32)
            ep_reward += reward; ep_rsec += info["R_sec"]; ep_rlegit += info["R_legit"]; ep_reve += info["R_eve"]
            steps += 1; global_step += 1

        returns = []
        ret = 0.0
        for r in reversed(rewards):
            ret = r + cfg.gamma * ret
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
            a_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            v_loss = nn.MSELoss()(value(s_t), returns_t)
            actor_opt.zero_grad(); a_loss.backward(); actor_opt.step()
            value_opt.zero_grad(); v_loss.backward(); value_opt.step()

        avg_rsec = (ep_rsec / max(steps, 1)) / 1e6
        rolling.append(avg_rsec)
        roll20 = float(np.mean(rolling[-20:])); roll100 = float(np.mean(rolling[-100:]))
        train_rows.append({
            "episode": ep, "global_step": global_step, "fading_model": cfg.fading_model,
            "avg_shaped_reward": float(ep_reward / max(steps, 1)),
            "avg_R_legit_mbps": float((ep_rlegit / max(steps, 1)) / 1e6),
            "avg_R_eve_mbps": float((ep_reve / max(steps, 1)) / 1e6),
            "avg_R_sec_mbps": float(avg_rsec),
            "eval_R_sec_mbps": "", "last_eval_R_sec_mbps": "", "steps": steps,
            "rolling20": roll20, "rolling20_avg_R_sec_mbps": roll20,
            "rolling100": roll100, "rolling100_avg_R_sec_mbps": roll100,
            "convergence_gap": float(abs(roll20 - roll100)),
            "convergence_gap20_100_mbps": float(abs(roll20 - roll100)),
        })
        if ep == 1 or ep % 25 == 0 or ep == cfg.episodes:
            print(f"  [PPO] Ep {ep:4d}/{cfg.episodes} | R_sec={avg_rsec:.3f} Mbps | roll100={roll100:.3f}")

    with log_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(train_rows[0].keys()))
        w.writeheader(); w.writerows(train_rows)
    plots = generate_vehicle_plots(str(log_path), str(out_dir), f"PPO_{fading_model}")
    return {"training_log_csv": str(log_path.resolve()), **plots, "final_roll100": roll100, "best_secrecy": float(np.max(rolling)), "avg_reward": float(np.mean([r["avg_shaped_reward"] for r in train_rows]))}


# ===================================================================
# SAC Training
# ===================================================================

from sac_study.sac_train import GaussianActor as SACActor, Critic as SACCritic


def train_vehicle_sac(fading_model: str, output_dir: str, episodes: int = 100, seed: int = 42, mobility_mode: str = "straight_road"):
    cfg = build_vehicle_sac_config(fading_model, episodes, seed)
    set_seed(cfg.seed)
    device = torch.device("cpu")
    env = make_vehicle_env(fading_model, cfg.seed, mobility_mode)
    state_dim = env.reset().shape[0]
    action_dim = 5
    target_entropy = -float(action_dim)

    actor = SACActor(state_dim, action_dim, cfg.hidden_dim).to(device)
    critic1 = SACCritic(state_dim, action_dim, cfg.hidden_dim).to(device)
    critic2 = SACCritic(state_dim, action_dim, cfg.hidden_dim).to(device)
    c1_target = SACCritic(state_dim, action_dim, cfg.hidden_dim).to(device)
    c2_target = SACCritic(state_dim, action_dim, cfg.hidden_dim).to(device)
    c1_target.load_state_dict(critic1.state_dict())
    c2_target.load_state_dict(critic2.state_dict())
    actor_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=cfg.critic_lr)
    log_alpha = torch.tensor(np.log(cfg.init_alpha), dtype=torch.float32, device=device, requires_grad=True)
    alpha_opt = optim.Adam([log_alpha], lr=cfg.alpha_lr)
    replay = deque(maxlen=cfg.replay_size)

    def replay_add(s, a, r, ns, d):
        replay.append((s, a, r, ns, d))

    def replay_sample(batch_size):
        batch = random.sample(replay, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.asarray(s, dtype=np.float32), np.asarray(a, dtype=np.float32),
                np.asarray(r, dtype=np.float32), np.asarray(ns, dtype=np.float32), np.asarray(d, dtype=np.float32))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.csv"

    global_step = 0
    rolling = []
    train_rows = []

    for ep in range(1, cfg.episodes + 1):
        state = env.reset().astype(np.float32)
        done = False
        ep_reward = 0.0; ep_rsec = 0.0; ep_rlegit = 0.0; ep_reve = 0.0; steps = 0

        while not done:
            with torch.no_grad():
                if len(replay) < cfg.min_replay_size:
                    action = np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
                else:
                    action = actor.sample(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))[0].cpu().numpy()[0]
            a_relay, a_jammer, jammer_power, _ = _split_action_5d(action)
            next_state, reward, done, info = env.step(a_relay, a_jammer, jammer_power)
            next_state = next_state.astype(np.float32)
            replay_add(state, action, float(reward), next_state, float(done))
            state = next_state
            ep_reward += reward; ep_rsec += info["R_sec"]; ep_rlegit += info["R_legit"]; ep_reve += info["R_eve"]
            steps += 1; global_step += 1

            if len(replay) >= cfg.min_replay_size:
                s, a, r, ns, d = replay_sample(cfg.batch_size)
                s_t = torch.tensor(s, dtype=torch.float32, device=device)
                a_t = torch.tensor(a, dtype=torch.float32, device=device)
                r_t = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
                ns_t = torch.tensor(ns, dtype=torch.float32, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)
                alpha = log_alpha.exp()
                with torch.no_grad():
                    next_a, next_lp = actor.sample(ns_t)
                    next_q = torch.min(c1_target(ns_t, next_a), c2_target(ns_t, next_a))
                    q_target = r_t + (1.0 - d_t) * cfg.gamma * (next_q - alpha * next_lp)
                c_loss = nn.MSELoss()(critic1(s_t, a_t), q_target) + nn.MSELoss()(critic2(s_t, a_t), q_target)
                critic_opt.zero_grad(); c_loss.backward(); critic_opt.step()
                new_a, lp = actor.sample(s_t)
                a_loss = (alpha.detach() * lp - torch.min(critic1(s_t, new_a), critic2(s_t, new_a))).mean()
                actor_opt.zero_grad(); a_loss.backward(); actor_opt.step()
                if cfg.auto_entropy_tuning:
                    alpha_loss = -(log_alpha * (lp + target_entropy).detach()).mean()
                    alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
                for ps, pd in zip(critic1.parameters(), c1_target.parameters()):
                    pd.data.copy_(cfg.tau * ps.data + (1.0 - cfg.tau) * pd.data)
                for ps, pd in zip(critic2.parameters(), c2_target.parameters()):
                    pd.data.copy_(cfg.tau * ps.data + (1.0 - cfg.tau) * pd.data)

        avg_rsec = (ep_rsec / max(steps, 1)) / 1e6
        rolling.append(avg_rsec)
        roll20 = float(np.mean(rolling[-20:])); roll100 = float(np.mean(rolling[-100:]))
        train_rows.append({
            "episode": ep, "global_step": global_step, "fading_model": cfg.fading_model,
            "avg_shaped_reward": float(ep_reward / max(steps, 1)),
            "avg_R_legit_mbps": float((ep_rlegit / max(steps, 1)) / 1e6),
            "avg_R_eve_mbps": float((ep_reve / max(steps, 1)) / 1e6),
            "avg_R_sec_mbps": float(avg_rsec),
            "eval_R_sec_mbps": "", "last_eval_R_sec_mbps": "", "steps": steps,
            "rolling20": roll20, "rolling20_avg_R_sec_mbps": roll20,
            "rolling100": roll100, "rolling100_avg_R_sec_mbps": roll100,
            "convergence_gap": float(abs(roll20 - roll100)),
            "convergence_gap20_100_mbps": float(abs(roll20 - roll100)),
        })
        if ep == 1 or ep % 25 == 0 or ep == cfg.episodes:
            print(f"  [SAC] Ep {ep:4d}/{cfg.episodes} | R_sec={avg_rsec:.3f} Mbps | roll100={roll100:.3f}")

    with log_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(train_rows[0].keys()))
        w.writeheader(); w.writerows(train_rows)
    plots = generate_vehicle_plots(str(log_path), str(out_dir), f"SAC_{fading_model}")
    return {"training_log_csv": str(log_path.resolve()), **plots, "final_roll100": roll100, "best_secrecy": float(np.max(rolling)), "avg_reward": float(np.mean([r["avg_shaped_reward"] for r in train_rows]))}


# ===================================================================
# TD3PG Training
# ===================================================================

from td3pg_study.td3pg_train import Actor as TD3Actor, Critic as TD3Critic


def train_vehicle_td3pg(fading_model: str, output_dir: str, episodes: int = 100, seed: int = 42, mobility_mode: str = "straight_road"):
    cfg = build_vehicle_td3pg_config(fading_model, episodes, seed)
    set_seed(cfg.seed)
    device = torch.device("cpu")
    env = make_vehicle_env(fading_model, cfg.seed, mobility_mode)
    state_dim = env.reset().shape[0]
    action_dim = 5

    actor = TD3Actor(state_dim, action_dim, cfg.hidden_dim).to(device)
    actor_target = TD3Actor(state_dim, action_dim, cfg.hidden_dim).to(device)
    critic1 = TD3Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    critic2 = TD3Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    c1_target = TD3Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    c2_target = TD3Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
    actor_target.load_state_dict(actor.state_dict())
    c1_target.load_state_dict(critic1.state_dict())
    c2_target.load_state_dict(critic2.state_dict())
    actor_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=cfg.critic_lr)
    replay = deque(maxlen=cfg.replay_size)

    def replay_add(s, a, r, ns, d):
        replay.append((s, a, r, ns, d))

    def replay_sample(batch_size):
        batch = random.sample(replay, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.asarray(s, dtype=np.float32), np.asarray(a, dtype=np.float32),
                np.asarray(r, dtype=np.float32), np.asarray(ns, dtype=np.float32), np.asarray(d, dtype=np.float32))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.csv"

    global_step = 0
    rolling = []
    train_rows = []

    for ep in range(1, cfg.episodes + 1):
        state = env.reset().astype(np.float32)
        done = False
        ep_reward = 0.0; ep_rsec = 0.0; ep_rlegit = 0.0; ep_reve = 0.0; steps = 0

        while not done:
            noise_std = cfg.exploration_noise_end if global_step >= cfg.exploration_noise_decay_steps else cfg.exploration_noise_start + (cfg.exploration_noise_end - cfg.exploration_noise_start) * global_step / max(cfg.exploration_noise_decay_steps, 1)
            with torch.no_grad():
                action = actor(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).cpu().numpy()[0]
            action = np.clip(action + np.random.normal(0.0, noise_std, size=action_dim), -1.0, 1.0)
            a_relay, a_jammer, jammer_power, _ = _split_action_5d(action)
            next_state, reward, done, info = env.step(a_relay, a_jammer, jammer_power)
            next_state = next_state.astype(np.float32)
            replay_add(state, action, float(reward), next_state, float(done))
            state = next_state
            ep_reward += reward; ep_rsec += info["R_sec"]; ep_rlegit += info["R_legit"]; ep_reve += info["R_eve"]
            steps += 1; global_step += 1

            if len(replay) >= cfg.min_replay_size:
                s, a, r, ns, d = replay_sample(cfg.batch_size)
                s_t = torch.tensor(s, dtype=torch.float32, device=device)
                a_t = torch.tensor(a, dtype=torch.float32, device=device)
                r_t = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
                ns_t = torch.tensor(ns, dtype=torch.float32, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)
                with torch.no_grad():
                    noise = torch.clamp(torch.randn_like(a_t) * cfg.target_policy_noise, -cfg.target_noise_clip, cfg.target_noise_clip)
                    next_a = torch.clamp(actor_target(ns_t) + noise, -1.0, 1.0)
                    target_q = torch.min(c1_target(ns_t, next_a), c2_target(ns_t, next_a))
                    q_target = r_t + (1.0 - d_t) * cfg.gamma * target_q
                c_loss = nn.MSELoss()(critic1(s_t, a_t), q_target) + nn.MSELoss()(critic2(s_t, a_t), q_target)
                critic_opt.zero_grad(); c_loss.backward(); critic_opt.step()
                if global_step % cfg.policy_delay == 0:
                    a_loss = -critic1(s_t, actor(s_t)).mean()
                    actor_opt.zero_grad(); a_loss.backward(); actor_opt.step()
                    for ps, pd in zip(actor.parameters(), actor_target.parameters()):
                        pd.data.copy_(cfg.tau * ps.data + (1.0 - cfg.tau) * pd.data)
                    for ps, pd in zip(critic1.parameters(), c1_target.parameters()):
                        pd.data.copy_(cfg.tau * ps.data + (1.0 - cfg.tau) * pd.data)
                    for ps, pd in zip(critic2.parameters(), c2_target.parameters()):
                        pd.data.copy_(cfg.tau * ps.data + (1.0 - cfg.tau) * pd.data)

        avg_rsec = (ep_rsec / max(steps, 1)) / 1e6
        rolling.append(avg_rsec)
        roll20 = float(np.mean(rolling[-20:])); roll100 = float(np.mean(rolling[-100:]))
        train_rows.append({
            "episode": ep, "global_step": global_step, "fading_model": cfg.fading_model,
            "avg_shaped_reward": float(ep_reward / max(steps, 1)),
            "avg_R_legit_mbps": float((ep_rlegit / max(steps, 1)) / 1e6),
            "avg_R_eve_mbps": float((ep_reve / max(steps, 1)) / 1e6),
            "avg_R_sec_mbps": float(avg_rsec),
            "eval_R_sec_mbps": "", "last_eval_R_sec_mbps": "", "steps": steps,
            "rolling20": roll20, "rolling20_avg_R_sec_mbps": roll20,
            "rolling100": roll100, "rolling100_avg_R_sec_mbps": roll100,
            "convergence_gap": float(abs(roll20 - roll100)),
            "convergence_gap20_100_mbps": float(abs(roll20 - roll100)),
        })
        if ep == 1 or ep % 25 == 0 or ep == cfg.episodes:
            print(f"  [TD3PG] Ep {ep:4d}/{cfg.episodes} | R_sec={avg_rsec:.3f} Mbps | roll100={roll100:.3f}")

    with log_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(train_rows[0].keys()))
        w.writeheader(); w.writerows(train_rows)
    plots = generate_vehicle_plots(str(log_path), str(out_dir), f"TD3PG_{fading_model}")
    return {"training_log_csv": str(log_path.resolve()), **plots, "final_roll100": roll100, "best_secrecy": float(np.max(rolling)), "avg_reward": float(np.mean([r["avg_shaped_reward"] for r in train_rows]))}


# ===================================================================
# IoT baseline training (for comparison)
# ===================================================================

def train_iot_comparison(algorithm: str, fading_model: str, output_dir: str, episodes: int = 100, seed: int = 42):
    """Train on standard UAVEnvironment (IoT receiver) for comparison."""
    from rl.dqn_train import train_dqn
    from rl.ddpg_train import train_ddpg
    from rl.advanced_rl_train import train_advanced, AdvancedRLConfig

    if algorithm == "dqn":
        cfg = build_vehicle_dqn_config(fading_model, episodes, seed)
        result = train_dqn(cfg, output_dir=output_dir)
    elif algorithm == "ddpg":
        cfg = build_vehicle_ddpg_config(fading_model, episodes, seed)
        result = train_ddpg(cfg, output_dir=output_dir)
    elif algorithm == "d3qn":
        from d3qn_study.train_d3qn import train_d3qn
        result = train_d3qn(build_vehicle_d3qn_config(fading_model, episodes, seed), output_dir=output_dir)
    elif algorithm == "ppo":
        result = train_advanced("ppo", build_vehicle_ppo_config(fading_model, episodes, seed), output_dir=output_dir)
    elif algorithm == "sac":
        from sac_study.sac_train import train_sac
        result = train_sac(build_vehicle_sac_config(fading_model, episodes, seed), output_dir=output_dir)
    elif algorithm == "td3pg":
        from td3pg_study.td3pg_train import train_td3pg
        result = train_td3pg(build_vehicle_td3pg_config(fading_model, episodes, seed), output_dir=output_dir)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return result


# ===================================================================
# Summary generation
# ===================================================================

def _read_final_metrics(csv_path: str) -> dict:
    """Read final metrics from training log CSV."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"final_roll100": 0.0, "best_secrecy": 0.0, "avg_reward": 0.0, "convergence_ep": 0}
    secrecy = [float(r.get("avg_R_sec_mbps", 0)) for r in rows]
    roll100_vals = [float(r.get("rolling100_avg_R_sec_mbps", r.get("rolling100", 0))) for r in rows]
    rewards = [float(r.get("avg_shaped_reward", 0)) for r in rows]
    final_roll100 = roll100_vals[-1] if roll100_vals else 0.0
    best_secrecy = float(np.max(secrecy)) if secrecy else 0.0
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    # Convergence episode: where roll100 first reaches 90% of final
    target = 0.9 * final_roll100
    conv_ep = next((i + 1 for i, v in enumerate(roll100_vals) if v >= target), len(rows))
    return {"final_roll100": final_roll100, "best_secrecy": best_secrecy, "avg_reward": avg_reward, "convergence_ep": conv_ep}


def generate_vehicle_summary(results: dict, output_dir: str) -> str:
    """Generate vehicle_summary.csv."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "vehicle_summary.csv"
    rows_data = []
    for key, data in results.items():
        algo, chan = key.split("_", 1)
        rows_data.append({
            "Algorithm": algo.upper(),
            "Channel": chan.capitalize(),
            "Final_Rolling100_Secrecy": f"{data.get('final_roll100', 0):.4f}",
            "Best_Secrecy": f"{data.get('best_secrecy', 0):.4f}",
            "Average_Reward": f"{data.get('avg_reward', 0):.4f}",
            "Convergence_Episode": data.get("convergence_ep", 0),
        })
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Algorithm", "Channel", "Final_Rolling100_Secrecy", "Best_Secrecy", "Average_Reward", "Convergence_Episode"])
        w.writeheader(); w.writerows(rows_data)
    return str(path.resolve())


def generate_iot_vs_vehicle_summary(vehicle_results: dict, iot_results: dict, output_dir: str) -> str:
    """Generate IoT vs Vehicle comparison CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "iot_vs_vehicle_summary.csv"
    rows_data = []
    all_keys = set(list(vehicle_results.keys()) + list(iot_results.keys()))
    for key in sorted(all_keys):
        algo, chan = key.split("_", 1)
        v = vehicle_results.get(key, {})
        i = iot_results.get(key, {})
        v_final = float(v.get("final_roll100", 0)) if v else 0.0
        i_final = float(i.get("final_roll100", 0)) if i else 0.0
        rows_data.append({
            "Algorithm": algo.upper(),
            "Channel": chan.capitalize(),
            "IoT_Final_Rolling100": f"{i_final:.4f}",
            "Vehicle_Final_Rolling100": f"{v_final:.4f}",
            "IoT_Best_Secrecy": f"{float(i.get('best_secrecy', 0)):.4f}" if i else "N/A",
            "Vehicle_Best_Secrecy": f"{float(v.get('best_secrecy', 0)):.4f}" if v else "N/A",
            "Difference_Rolling100": f"{v_final - i_final:.4f}",
        })
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Algorithm", "Channel", "IoT_Final_Rolling100", "Vehicle_Final_Rolling100", "IoT_Best_Secrecy", "Vehicle_Best_Secrecy", "Difference_Rolling100"])
        w.writeheader(); w.writerows(rows_data)
    return str(path.resolve())


# ===================================================================
# Main orchestrator
# ===================================================================

ALGORITHMS = ["dqn", "ddpg", "d3qn", "ppo", "sac", "td3pg"]
CHANNELS = ["rician", "rayleigh"]

TRAIN_FN_MAP = {
    "dqn": train_vehicle_dqn,
    "ddpg": train_vehicle_ddpg,
    "d3qn": train_vehicle_d3qn,
    "ppo": train_vehicle_ppo,
    "sac": train_vehicle_sac,
    "td3pg": train_vehicle_td3pg,
}


def run_vehicle_experiments(
    episodes: int = 100,
    seed: int = 42,
    mobility_mode: str = "straight_road",
    vehicle_max_speed: float = 10.0,
    output_root: str = "outputs/vehicle_receiver",
    algorithms: list[str] | None = None,
    channels: list[str] | None = None,
    run_iot: bool = False,
) -> dict:
    """Run vehicle receiver experiments for all algorithm/channel pairs."""
    algos = algorithms or ALGORITHMS
    chans = channels or CHANNELS
    vehicle_results = {}
    iot_results = {}

    print(f"\n{'='*60}")
    print(f"VEHICLE RECEIVER EXPERIMENTS")
    print(f"Mobility mode: {mobility_mode}")
    print(f"Episodes: {episodes}")
    print(f"{'='*60}\n")

    for algo in algos:
        for chan in chans:
            key = f"{algo}_{chan}"
            odir = build_output_dir(algo, chan, output_root)
            print(f"\n--- {algo.upper()} + {chan.title()} (Vehicle) ---")
            train_fn = TRAIN_FN_MAP[algo]
            result = train_fn(chan, odir, episodes=episodes, seed=seed, mobility_mode=mobility_mode)
            result["convergence_ep"] = _read_final_metrics(result["training_log_csv"])["convergence_ep"]
            vehicle_results[key] = result

    if run_iot:
        print(f"\n{'='*60}")
        print(f"IOT BASELINE EXPERIMENTS (for comparison)")
        print(f"{'='*60}\n")
        iot_out_root = f"{output_root}/iot_comparison"
        for algo in algos:
            for chan in chans:
                key = f"{algo}_{chan}"
                odir = f"{iot_out_root}/{algo}/{chan}"
                print(f"\n--- {algo.upper()} + {chan.title()} (IoT) ---")
                try:
                    train_iot_comparison(algo, chan, odir, episodes=episodes, seed=seed)
                except Exception as e:
                    print(f"  IoT {algo} {chan} failed: {e}")
                    continue
                iot_csv = str(Path(odir) / "training_log.csv")
                if os.path.exists(iot_csv):
                    iot_results[key] = _read_final_metrics(iot_csv)

    # Generate summaries
    vs = generate_vehicle_summary(vehicle_results, output_root)
    print(f"\nVehicle summary CSV: {vs}")

    if iot_results:
        ivs = generate_iot_vs_vehicle_summary(vehicle_results, iot_results, output_root)
        print(f"IoT vs Vehicle summary CSV: {ivs}")

    return {"vehicle_results": vehicle_results, "iot_results": iot_results, "vehicle_summary": vs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vehicle receiver experiments.")
    parser.add_argument("--episodes", type=int, default=100, help="Training episodes per run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mobility", type=str, default="straight_road", choices=["straight_road", "lane_change", "urban_grid"])
    parser.add_argument("--max-speed", type=float, default=10.0, help="Vehicle max speed (m/s)")
    parser.add_argument("--output-root", type=str, default="outputs/vehicle_receiver")
    parser.add_argument("--run-iot", action="store_true", help="Also run IoT baseline for comparison")
    parser.add_argument("--algorithms", type=str, nargs="+", default=None, choices=ALGORITHMS + ["all"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    algos = ALGORITHMS if args.algorithms is None or "all" in args.algorithms else args.algorithms
    run_vehicle_experiments(
        episodes=args.episodes,
        seed=args.seed,
        mobility_mode=args.mobility,
        vehicle_max_speed=args.max_speed,
        output_root=args.output_root,
        algorithms=algos,
        run_iot=args.run_iot,
    )
