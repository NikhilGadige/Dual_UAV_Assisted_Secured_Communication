"""MADRL evaluation against baselines — 3-agent version."""

from __future__ import annotations

import os

import numpy as np

from madrl_updated_exp.configs import MADRLConfig, EnvConfig
from madrl_updated_exp.environment import ISACMultiAgentEnv


N_EVE = 3
N_VEH = 3


def random_feasible_baseline(cfg: EnvConfig, n_episodes: int = 10) -> dict:
    """Random feasible actions baseline."""
    env = ISACMultiAgentEnv(cfg, seed=42)
    secrecies, sensings, violations = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(cfg.N_time):
            actions = {name: env.action_spaces[name].sample()
                       for name in env.agent_names}
            obs, rewards, terminated, truncated, info = env.step(actions)
        secrecies.append(float(info.get("secrecy", 0.0)))
        sensings.append(float(info.get("sensing", 0.0)))
        violations.append(float(info.get("violation", 0.0)))

    return {
        "avg_reward": 0.0,
        "std_reward": 0.0,
        "avg_secrecy": float(np.mean(secrecies)),
        "std_secrecy": float(np.std(secrecies)),
        "avg_sensing": float(np.mean(sensings)),
        "std_sensing": float(np.std(sensings)),
        "avg_violation": float(np.mean(violations)),
    }


def sca_bcd_baseline(cfg: EnvConfig, n_episodes: int = 3) -> dict:
    """Full SCA-BCD optimization baseline."""
    from sca_bcd_exp.run_sca_bcd import run_sca_bcd
    from sca_bcd_exp.configs import SCABCDConfig

    secrecies, sensings, violations = [], [], []

    for ep in range(n_episodes):
        sca_cfg = SCABCDConfig(
            M_bs=cfg.M_bs, N_ris=cfg.N_ris, N_j=cfg.N_j, N_time=cfg.N_time,
            seed=cfg.seed + ep,
        )
        result = run_sca_bcd(sca_cfg)
        secrecies.append(result.get("final_secrecy", 0.0))
        sensings.append(result.get("final_sensing", 0.0))
        viol_sum = sum(result.get("final_violations", {}).values())
        violations.append(viol_sum)

    return {
        "avg_secrecy": float(np.mean(secrecies)),
        "std_secrecy": float(np.std(secrecies)),
        "avg_sensing": float(np.mean(sensings)),
        "std_sensing": float(np.std(sensings)),
        "avg_violation": float(np.mean(violations)),
    }


def compute_constraint_violations(q_uav, w_bs, v_jammer, cfg):
    viol = 0.0
    N_t = cfg.N_time
    bs_power = [float(np.linalg.norm(w_bs[n]) ** 2) for n in range(N_t)]
    viol += float(np.max(np.maximum(np.array(bs_power) - cfg.P_bs_max, 0.0)))
    j_power = [float(np.linalg.norm(v_jammer[n]) ** 2) for n in range(N_t)]
    viol += float(np.max(np.maximum(np.array(j_power) - cfg.P_j_max, 0.0)))
    speeds = [0.0]
    for n in range(1, N_t):
        dist = float(np.linalg.norm(q_uav[n] - q_uav[n-1]))
        speeds.append(dist / cfg.dt)
    viol += float(np.max(np.maximum(np.array(speeds) - cfg.v_max, 0.0)))
    return viol


def compare(
    trained_dir: str,
    cfg: MADRLConfig,
    n_episodes: int = 10,
) -> dict:
    """Compare trained agents against baselines."""
    results = {}

    print("Running random feasible baseline...")
    results["random_feasible"] = random_feasible_baseline(cfg.env, n_episodes)

    print("Running SCA-BCD baseline...")
    results["sca_bcd"] = sca_bcd_baseline(cfg.env, n_episodes // 2)

    from madrl_updated_exp.agents.mappo import MAPPOAgent
    from madrl_updated_exp.agents.matd3 import MATD3Agent

    env = ISACMultiAgentEnv(cfg.env, cfg.reward, seed=42)
    agents = {}
    for ac in cfg.agents:
        name = ac.name
        obs_dim = env.observation_spaces[name].shape[0]
        act_dim = env.action_spaces[name].shape[0]
        if cfg.training.algorithm == "mappo":
            agent = MAPPOAgent(obs_dim, act_dim, name, device="cpu")
        else:
            agent = MATD3Agent(obs_dim, act_dim, name, device="cpu")
        ckpt_path = os.path.join(trained_dir, "checkpoints", f"{name}_final.pt")
        if os.path.exists(ckpt_path):
            agent.load(ckpt_path)
        else:
            print(f"Warning: {ckpt_path} not found, using untrained")
        agents[name] = agent

    for agent in agents.values():
        agent.eval_mode()

    secrecies, sensings, violations = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(cfg.training.max_steps_per_episode):
            actions = {n: a.act(obs[n], deterministic=True) for n, a in agents.items()}
            obs, rewards, terminated, truncated, info = env.step(actions)
        secrecies.append(float(info.get("secrecy", 0.0)))
        sensings.append(float(info.get("sensing", 0.0)))
        violations.append(float(info.get("violation", 0.0)))

    results["trained_madrl"] = {
        "avg_secrecy": float(np.mean(secrecies)),
        "std_secrecy": float(np.std(secrecies)),
        "avg_sensing": float(np.mean(sensings)),
        "std_sensing": float(np.std(sensings)),
        "avg_violation": float(np.mean(violations)),
    }

    return results
