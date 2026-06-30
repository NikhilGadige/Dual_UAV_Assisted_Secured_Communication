"""
Reward and Policy Audit

Determines why trained policies achieve lower secrecy than random feasible
despite higher reward.

Parts:
  1. Reward decomposition — log reward components, compute correlations
  2. Policy behaviour audit — statistics vs baselines
  3. Action saturation — fraction clipped/bounded
  4. Observation importance — ablation study
  5. Reward weight sweep — alpha sweep

Outputs: outputs/reinforcement_learning/policy_diagnostics/
"""

from __future__ import annotations

import csv
import os
import sys
import json
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from madrl_exp.configs import MADRLConfig, EnvConfig, AgentConfig, TrainingConfig, RewardConfig
from madrl_exp.training.trainer import MARLTrainer
from madrl_exp.environment import ISACMultiAgentEnv, SCENE_MIN, SCENE_MAX
from madrl_exp.agents.mappo import MAPPOAgent
from madrl_exp.agents.matd3 import MATD3Agent
from madrl_exp.evaluation import compute_constraint_violations

OUTPUT_ROOT = os.path.join("outputs", "reinforcement_learning", "policy_diagnostics")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Latest checkpoint paths
MAPPO_CKPT_DIR = os.path.join("outputs", "reinforcement_learning", "training_runs", "three_agent_training", "mappo_20260629_152202", "checkpoints")
MATD3_CKPT_DIR = os.path.join("outputs", "reinforcement_learning", "training_runs", "three_agent_training", "matd3_20260629_152626", "checkpoints")

N_EVAL = 10  # episodes per evaluation


def make_env_cfg(alpha: float = 0.5, seed: int = 42) -> EnvConfig:
    return EnvConfig(alpha=alpha, seed=seed)


def make_agents_cfg(env_cfg: EnvConfig) -> list[AgentConfig]:
    return [
        AgentConfig(name="bs_beamformer", act_dim=2 * env_cfg.N_time * env_cfg.M_bs),
        AgentConfig(name="uav_trajectory", act_dim=3 * env_cfg.N_time),
        AgentConfig(name="jammer_beamformer", act_dim=2 * env_cfg.N_time * env_cfg.N_j),
    ]


def load_agent(name: str, obs_dim: int, act_dim: int, ckpt_dir: str,
               algo: str = "mappo") -> MAPPOAgent | MATD3Agent:
    if algo == "mappo":
        agent = MAPPOAgent(obs_dim, act_dim, name, device="cpu")
    else:
        agent = MATD3Agent(obs_dim, act_dim, name, device="cpu")
    ckpt_path = os.path.join(ckpt_dir, f"{name}_ep100.pt")
    if os.path.exists(ckpt_path):
        agent.load(ckpt_path)
        agent.eval_mode()
    else:
        print(f"  WARNING: {ckpt_path} not found")
    return agent


def load_trained_agents(env: ISACMultiAgentEnv, ckpt_dir: str,
                        algo: str = "mappo") -> dict:
    agents = {}
    for name in env.agent_names:
        obs_dim = env.observation_spaces[name].shape[0]
        act_dim = env.action_spaces[name].shape[0]
        agents[name] = load_agent(name, obs_dim, act_dim, ckpt_dir, algo)
    return agents


# ═══════════════════════════════════════════════════════════════
#  PART 1 — REWARD DECOMPOSITION
# ═══════════════════════════════════════════════════════════════

def run_reward_decomposition(algo: str, ckpt_dir: str, n_episodes: int = N_EVAL) -> list[dict]:
    print(f"\n--- Part 1: Reward Decomposition ({algo}) ---")
    env_cfg = make_env_cfg()
    env = ISACMultiAgentEnv(env_cfg, seed=42)
    agents = load_trained_agents(env, ckpt_dir, algo)

    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        ep_rows = []
        for step in range(env.cfg.N_time):
            actions = {n: a.act(obs[n], deterministic=True) for n, a in agents.items()}
            obs, rewards, terminated, truncated, info = env.step(actions)

            # Compute penalty components
            cons_penalty = info.get("violation", 0.0) * 0.1
            R_norm = info.get("R_norm", 0.0)
            U_norm = info.get("U_norm", 0.0)
            secrecy = info.get("secrecy", 0.0)
            sensing = info.get("sensing", 0.0)

            row = {
                "episode": ep,
                "step": step,
                "reward_total": float(rewards[env.agent_names[0]]),
                "reward_secrecy": float(R_norm * 0.5),
                "reward_sensing": float(U_norm * 0.5),
                "reward_constraint_penalty": float(-cons_penalty),
                "secrecy_raw": float(secrecy),
                "sensing_raw": float(sensing),
                "violation": float(info.get("violation", 0.0)),
                "f": float(info.get("f", 0.0)),
            }
            ep_rows.append(row)
            rows.append(row)

        if (ep + 1) % 5 == 0:
            print(f"  {algo} ep {ep + 1}/{n_episodes} (steps: {len(ep_rows)})")

    return rows


def compute_reward_correlations(rows: list[dict]) -> dict:
    totals = np.array([r["reward_total"] for r in rows])
    secrecies = np.array([r["secrecy_raw"] for r in rows])
    sensings = np.array([r["sensing_raw"] for r in rows])
    penalties = np.array([r["reward_constraint_penalty"] for r in rows])

    def corr(x, y):
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    return {
        "corr_total_secrecy": corr(totals, secrecies),
        "corr_total_sensing": corr(totals, sensings),
        "corr_total_penalties": corr(totals, penalties),
        "mean_reward": float(np.mean(totals)),
        "mean_secrecy": float(np.mean(secrecies)),
        "mean_sensing": float(np.mean(sensings)),
    }


def save_reward_decomposition(rows: list[dict], correlations: dict, label: str):
    path_csv = os.path.join(OUTPUT_ROOT, f"reward_components_{label}.csv")
    with open(path_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"  Saved {path_csv}")

    path_corr = os.path.join(OUTPUT_ROOT, f"reward_correlations_{label}.json")
    with open(path_corr, "w") as f:
        json.dump(correlations, f, indent=2)
    print(f"  Saved {path_corr}")


# ═══════════════════════════════════════════════════════════════
#  PART 2 — POLICY BEHAVIOUR AUDIT
# ═══════════════════════════════════════════════════════════════

def evaluate_policy_stats(agents: dict, env: ISACMultiAgentEnv,
                          n_episodes: int = N_EVAL, label: str = "") -> list[dict]:
    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        for step in range(env.cfg.N_time):
            actions = {n: a.act(obs[n], deterministic=True) for n, a in agents.items()}
            obs, rewards, terminated, truncated, info = env.step(actions)

            q_uav = env.q_uav
            w_bs = env.w_bs
            v_jammer = env.v_jammer

            # Compute statistics
            bs_powers = [float(np.linalg.norm(w_bs[n]) ** 2) for n in range(env.cfg.N_time)]
            j_powers = [float(np.linalg.norm(v_jammer[n]) ** 2) for n in range(env.cfg.N_time)]
            speeds = [0.0]
            for n in range(1, env.cfg.N_time):
                speeds.append(float(np.linalg.norm(q_uav[n] - q_uav[n-1])) / env.cfg.dt)

            row = {
                "label": label,
                "episode": ep,
                "step": step,
                "mean_bs_power": float(np.mean(bs_powers)),
                "mean_jammer_power": float(np.mean(j_powers)),
                "mean_uav_speed": float(np.mean(speeds)),
                "secrecy": float(info.get("secrecy", 0.0)),
                "sensing": float(info.get("sensing", 0.0)),
                "reward": float(rewards[env.agent_names[0]]),
                "violation": float(info.get("violation", 0.0)),
            }
            rows.append(row)
    return rows


def random_feasible_stats(env: ISACMultiAgentEnv,
                          n_episodes: int = N_EVAL) -> list[dict]:
    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        for step in range(env.cfg.N_time):
            actions = {n: env.action_spaces[n].sample() for n in env.agent_names}
            obs, rewards, terminated, truncated, info = env.step(actions)

            q_uav = env.q_uav
            w_bs = env.w_bs
            v_jammer = env.v_jammer

            bs_powers = [float(np.linalg.norm(w_bs[n]) ** 2) for n in range(env.cfg.N_time)]
            j_powers = [float(np.linalg.norm(v_jammer[n]) ** 2) for n in range(env.cfg.N_time)]
            speeds = [0.0]
            for n in range(1, env.cfg.N_time):
                speeds.append(float(np.linalg.norm(q_uav[n] - q_uav[n-1])) / env.cfg.dt)

            rows.append({
                "label": "random_feasible",
                "episode": ep,
                "step": step,
                "mean_bs_power": float(np.mean(bs_powers)),
                "mean_jammer_power": float(np.mean(j_powers)),
                "mean_uav_speed": float(np.mean(speeds)),
                "secrecy": float(info.get("secrecy", 0.0)),
                "sensing": float(info.get("sensing", 0.0)),
                "reward": float(rewards[env.agent_names[0]]),
                "violation": float(info.get("violation", 0.0)),
            })
    return rows


def sca_bcd_stats(n_episodes: int = 3) -> list[dict]:
    try:
        from optimization_problem_exp.environments.optimization_problem_env import (
            OptimizationProblemEnv, OptimizationConfig,
        )
        from sca_bcd_exp.environments.sca_environment import SCAEnvironment
        from sca_bcd_exp.configs import SCABCDConfig

        rows = []
        for ep in range(n_episodes):
            cfg = EnvConfig(seed=ep)
            opt_cfg = OptimizationConfig(
                N_ris=cfg.N_ris, N_j=cfg.N_j, N_time=cfg.N_time,
                N_tx_sense=cfg.N_tx_sense, N_rx_sense=cfg.N_rx_sense,
                L_pilot=cfg.L_pilot, P_bs_max=cfg.P_bs_max, P_j_max=cfg.P_j_max,
                sigma2=cfg.sigma2, noise_power_sense=cfg.noise_power_sense,
                v_max=cfg.v_max, dt=cfg.dt, d_ant=cfg.d_ant, wavelength=cfg.wavelength,
                eta_ris=cfg.eta_ris, seed=ep, sensing_utility_mode=cfg.sensing_utility_mode,
                M_bs=cfg.M_bs,
            )
            env = OptimizationProblemEnv(opt_cfg)
            sca_cfg = SCABCDConfig(M_bs=cfg.M_bs)
            sca_env = SCAEnvironment(env, sca_cfg)
            sca_env.reset()
            for _ in range(50):
                _, _, term, trunc, _ = sca_env.step()
                if term or trunc:
                    break

            q_uav = sca_env.env.q_uav
            w_bs = sca_env.env.w_bs
            v_jammer = sca_env.env.v_jammer

            bs_powers = [float(np.linalg.norm(w_bs[n]) ** 2) for n in range(cfg.N_time)]
            j_powers = [float(np.linalg.norm(v_jammer[n]) ** 2) for n in range(cfg.N_time)]
            speeds = [0.0]
            for n in range(1, cfg.N_time):
                speeds.append(float(np.linalg.norm(q_uav[n] - q_uav[n-1])) / cfg.dt)

            from optimization_problem_exp.optimization.problem_formulation import (
                compute_secrecy_rate, compute_sensing_utility,
            )
            env2 = ISACMultiAgentEnv(cfg, seed=ep)
            sec = compute_secrecy_rate(
                q_bs=sca_env.env.q_bs, q_user=sca_env.env.q_user,
                q_eves=sca_env.env.q_eves, q_jammer=sca_env.env.q_jammer,
                N_ris=cfg.N_ris, N_j=cfg.N_j, Phi=None,
                q_uav=q_uav, w_bs=w_bs, v_jammer=v_jammer,
                P_bs_max=cfg.P_bs_max, P_j_max=cfg.P_j_max,
                sigma2=cfg.sigma2, seed=ep,
                jammer_mode="mixed", jammer_mix_alpha=cfg.alpha,
                jammer_power_factor=max(0.01, cfg.alpha),
                include_direct_links=cfg.include_direct_links,
                eta_ris=cfg.eta_ris, ris_alignment_alpha=cfg.alpha,
                M_bs=cfg.M_bs,
            )
            sense = compute_sensing_utility(
                q_uav=q_uav, q_vehicles=sca_env.env.q_vehicles,
                rcs_list=sca_env.env.rcs_list,
                N_tx=cfg.N_tx_sense, N_rx=cfg.N_rx_sense,
                L_pilot=cfg.L_pilot,
                noise_power=cfg.noise_power_sense,
                d_ant=cfg.d_ant, wavelength=cfg.wavelength,
                seed=ep, mode=cfg.sensing_utility_mode,
            )
            viol = compute_constraint_violations(q_uav, w_bs, v_jammer, cfg)

            rows.append({
                "label": "sca_bcd",
                "episode": ep,
                "step": 0,
                "mean_bs_power": float(np.mean(bs_powers)),
                "mean_jammer_power": float(np.mean(j_powers)),
                "mean_uav_speed": float(np.mean(speeds)),
                "secrecy": float(sec["R_s_total"]),
                "sensing": float(sense["U_sense_total"]),
                "reward": 0.0,
                "violation": float(viol),
            })
        return rows
    except Exception as e:
        print(f"  SCA-BCD baseline unavailable: {e}")
        return []


def save_policy_stats(all_stats: dict, label: str):
    path = os.path.join(OUTPUT_ROOT, f"policy_statistics_{label}.csv")
    all_rows = []
    for k, v in all_stats.items():
        all_rows.extend(v)
    if all_rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
#  PART 3 — ACTION SATURATION
# ═══════════════════════════════════════════════════════════════

def compute_action_saturation(algo: str, ckpt_dir: str,
                              n_episodes: int = N_EVAL) -> list[dict]:
    print(f"\n--- Part 3: Action Saturation ({algo}) ---")
    env_cfg = make_env_cfg()
    env = ISACMultiAgentEnv(env_cfg, seed=42)
    agents = load_trained_agents(env, ckpt_dir, algo)

    rows = []
    for agent_name in env.agent_names:
        lb_count = 0
        ub_count = 0
        total = 0
        n_clipped = 0
        all_actions = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=42 + ep)
            for step in range(env.cfg.N_time):
                act = agents[agent_name].act(obs[agent_name], deterministic=True)
                all_actions.extend(act.tolist())
                total += len(act)
                lb_count += int(np.sum(act <= -0.99))
                ub_count += int(np.sum(act >= 0.99))
                # Count how many would have been clipped if outside [-1, 1]
                raw = np.array(act)
                n_clipped += int(np.sum(np.abs(raw) > 1.0))
                obs, _, _, _, _ = env.step(
                    {n: agents[n].act(obs[n], deterministic=True)
                     if n != agent_name else act
                     for n in env.agent_names}
                )

        lb_frac = lb_count / max(total, 1)
        ub_frac = ub_count / max(total, 1)
        sat_frac = (lb_count + ub_count) / max(total, 1)

        row = {
            "algo": algo,
            "agent": agent_name,
            "total_dims": total,
            "lb_count": lb_count,
            "ub_count": ub_count,
            "lb_fraction": round(lb_frac, 4),
            "ub_fraction": round(ub_frac, 4),
            "saturation_fraction": round(sat_frac, 4),
            "clipped_fraction": round(n_clipped / max(total, 1), 4),
            "mean_value": round(float(np.mean(all_actions)), 4),
            "std_value": round(float(np.std(all_actions)), 4),
        }
        rows.append(row)
        print(f"  {agent_name}: sat={sat_frac:.2%} "
              f"(lb={lb_frac:.2%} ub={ub_frac:.2%}) clip={row['clipped_fraction']:.2%}")

    return rows


def save_action_saturation(rows: list[dict], label: str):
    path = os.path.join(OUTPUT_ROOT, f"action_saturation_{label}.csv")
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
#  PART 4 — OBSERVATION IMPORTANCE (ABLATION)
# ═══════════════════════════════════════════════════════════════

def train_and_eval_ablation(ablation_name: str, obs_mod_fn,
                            n_episodes: int = 50) -> dict:
    """Train MAPPO with modified observations and evaluate."""
    print(f"\n--- Part 4: Ablation — {ablation_name} ---")

    class AblationEnv(ISACMultiAgentEnv):
        def _build_obs_bs(self, q_uav, w_bs, v_jammer, sec, sense, cons_viol):
            obs = super()._build_obs_bs(q_uav, w_bs, v_jammer, sec, sense, cons_viol)
            return obs_mod_fn("bs_beamformer", obs, sec, sense)
        def _build_obs_uav_trajectory(self, q_uav, w_bs, v_jammer, sec, sense, cons_viol):
            obs = super()._build_obs_uav_trajectory(q_uav, w_bs, v_jammer, sec, sense, cons_viol)
            return obs_mod_fn("uav_trajectory", obs, sec, sense)
        def _build_obs_jammer(self, q_uav, w_bs, v_jammer, sec, sense, cons_viol):
            obs = super()._build_obs_jammer(q_uav, w_bs, v_jammer, sec, sense, cons_viol)
            return obs_mod_fn("jammer_beamformer", obs, sec, sense)

    env_cfg = make_env_cfg()
    agents_cfg = make_agents_cfg(env_cfg)
    train_cfg = TrainingConfig(algorithm="mappo", n_episodes=n_episodes,
                               max_steps_per_episode=50,
                               eval_interval=25, save_interval=50,
                               log_interval=10, seed=42,
                               output_root=os.path.join(OUTPUT_ROOT, "ablation_runs"))

    cfg = MADRLConfig(env=env_cfg, agents=agents_cfg, training=train_cfg,
                      reward=RewardConfig(), output_root=train_cfg.output_root)

    class AblationTrainer(MARLTrainer):
        def __init__(self, cfg):
            self.cfg = cfg
            self.env = AblationEnv(cfg.env, cfg.reward, cfg.training.seed)
            self.agents = {}
            self._init_agents()
            self._make_dirs()
            self.history = defaultdict(list)
            self.buffer = {name: [] for name in self.env.agent_names}
            self.loss_history_rows = []
            np.random.seed(cfg.training.seed)

    trainer = AblationTrainer(cfg)
    trainer.train()

    # Evaluate
    eval_results = evaluate_policy_stats(trainer.agents, trainer.env,
                                          n_episodes=5, label=ablation_name)
    avg_sec = float(np.mean([r["secrecy"] for r in eval_results]))
    avg_sen = float(np.mean([r["sensing"] for r in eval_results]))
    avg_rew = float(np.mean([r["reward"] for r in eval_results]))

    return {
        "ablation": ablation_name,
        "avg_secrecy": avg_sec,
        "avg_sensing": avg_sen,
        "avg_reward": avg_rew,
        "n_episodes": n_episodes,
        "secrecy_degradation": 0.0,
    }


def run_ablation_study(n_episodes: int = 50) -> list[dict]:
    print(f"\n{'=' * 60}")
    print("PART 4: OBSERVATION IMPORTANCE (ABLATION)")
    print(f"{'=' * 60}")

    # Train full-obs baseline
    def no_mod(agent_name, obs, sec, sense):
        return obs

    baseline = train_and_eval_ablation("full_obs", no_mod, n_episodes)

    # Ablation 1: remove secrecy info
    SEC_IDX = {"bs_beamformer": 72, "uav_trajectory": 25, "jammer_beamformer": 44}
    SEN_IDX = {"bs_beamformer": 73, "uav_trajectory": 26, "jammer_beamformer": 45}

    def remove_secrecy(agent_name, obs, sec, sense):
        o = obs.copy()
        idx = SEC_IDX.get(agent_name)
        if idx is not None and idx < len(o):
            o[idx] = 0.0
        return o

    ablate_sec = train_and_eval_ablation("no_secrecy", remove_secrecy, n_episodes)

    # Ablation 2: remove sensing info
    def remove_sensing(agent_name, obs, sec, sense):
        o = obs.copy()
        idx = SEN_IDX.get(agent_name)
        if idx is not None and idx < len(o):
            o[idx] = 0.0
        return o

    ablate_sen = train_and_eval_ablation("no_sensing", remove_sensing, n_episodes)

    # Ablation 3: remove channel info (zero out channel-related obs)
    def remove_channels(agent_name, obs, sec, sense):
        o = obs.copy()
        if agent_name == "bs_beamformer":
            o[:8 + 24] = 0.0
        elif agent_name == "jammer_beamformer":
            o[:40 + 3] = 0.0
        return o

    ablate_chan = train_and_eval_ablation("no_channels", remove_channels, n_episodes)

    # Compute degradation
    base_sec = baseline["avg_secrecy"]
    for r in [ablate_sec, ablate_sen, ablate_chan]:
        r["secrecy_degradation"] = (base_sec - r["avg_secrecy"]) / max(abs(base_sec), 1e-10) * 100

    results = [baseline, ablate_sec, ablate_sen, ablate_chan]
    path = os.path.join(OUTPUT_ROOT, "observation_ablation.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"  Saved {path}")
    return results


# ═══════════════════════════════════════════════════════════════
#  PART 5 — REWARD WEIGHT SWEEP
# ═══════════════════════════════════════════════════════════════

def run_alpha_sweep(n_episodes: int = 50) -> list[dict]:
    print(f"\n{'=' * 60}")
    print("PART 5: REWARD WEIGHT SWEEP (ALPHA)")
    print(f"{'=' * 60}")

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    for alpha in alphas:
        print(f"\n  Training MAPPO with alpha={alpha} for {n_episodes} episodes...")
        env_cfg = make_env_cfg(alpha=alpha)
        agents_cfg = make_agents_cfg(env_cfg)
        train_cfg = TrainingConfig(algorithm="mappo", n_episodes=n_episodes,
                                   max_steps_per_episode=50,
                                   eval_interval=25, save_interval=50,
                                   log_interval=10, seed=42,
                                   output_root=os.path.join(OUTPUT_ROOT, "alpha_sweep_runs"))
        cfg = MADRLConfig(env=env_cfg, agents=agents_cfg, training=train_cfg,
                          reward=RewardConfig(), output_root=train_cfg.output_root)

        trainer = MARLTrainer(cfg)
        trainer.train()

        # Evaluate
        eval_rows = evaluate_policy_stats(trainer.agents, trainer.env,
                                           n_episodes=5, label=f"alpha={alpha}")
        avg_sec = float(np.mean([r["secrecy"] for r in eval_rows]))
        avg_sen = float(np.mean([r["sensing"] for r in eval_rows]))
        avg_rew = float(np.mean([r["reward"] for r in eval_rows]))

        results.append({
            "alpha": alpha,
            "avg_secrecy": avg_sec,
            "avg_sensing": avg_sen,
            "avg_reward": avg_rew,
        })
        print(f"    alpha={alpha}: secrecy={avg_sec:.4f}, sensing={avg_sen:.4f}, reward={avg_rew:.4f}")

    path = os.path.join(OUTPUT_ROOT, "alpha_sweep.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"  Saved {path}")
    return results


# ═══════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_reward_components(all_rows: dict[str, list[dict]]):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {"mappo": "tab:blue", "matd3": "tab:green"}

    for idx, (label, rows) in enumerate(all_rows.items()):
        color = colors.get(label, "gray")
        steps = np.arange(len(rows))
        axes[0, 0].plot(steps, [r["reward_total"] for r in rows],
                        label=label, color=color, alpha=0.7)
        axes[0, 1].plot(steps, [r["reward_secrecy"] for r in rows],
                        label=label, color=color, alpha=0.7)
        axes[1, 0].plot(steps, [r["reward_sensing"] for r in rows],
                        label=label, color=color, alpha=0.7)
        axes[1, 1].plot(steps, [r["reward_constraint_penalty"] for r in rows],
                        label=label, color=color, alpha=0.7)

    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 1].set_ylabel("Secrecy Reward")
    axes[1, 0].set_ylabel("Sensing Reward")
    axes[1, 1].set_ylabel("Constraint Penalty")
    for ax in axes.flat:
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Reward Decomposition", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUTPUT_ROOT, "reward_components.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved reward_components.png")


def plot_policy_statistics(all_stats: dict[str, list[dict]]):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {"random_feasible": "tab:gray", "sca_bcd": "tab:red",
              "mappo": "tab:blue", "matd3": "tab:green"}

    metrics = [("secrecy", "Secrecy Rate"),
               ("sensing", "Sensing Utility"),
               ("mean_bs_power", "Mean BS Power"),
               ("mean_jammer_power", "Mean Jammer Power")]

    for idx, (key, ylabel) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        for label, rows in all_stats.items():
            vals = [r[key] for r in rows if key in r]
            if not vals:
                continue
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            color = colors.get(label, "gray")
            ax.bar(label, mean_v, yerr=std_v, capsize=5,
                   color=color, alpha=0.7, label=label)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Policy Statistics Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUTPUT_ROOT, "policy_statistics.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved policy_statistics.png")


def plot_action_saturation(all_rows: dict[str, list]):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(3)
    width = 0.3
    for idx, (label, rows) in enumerate(all_rows.items()):
        sats = [r["saturation_fraction"] for r in rows]
        offset = (idx - 0.5) * width
        ax.bar(x + offset, sats, width, label=label, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["bs_beamformer", "uav_trajectory", "jammer_beamformer"])
    ax.set_ylabel("Saturation Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Action Saturation by Agent", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_ROOT, "action_saturation.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved action_saturation.png")


def plot_ablation(ablation_results: list[dict]):
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [r["ablation"] for r in ablation_results]
    secrecies = [r["avg_secrecy"] for r in ablation_results]
    sensings = [r["avg_sensing"] for r in ablation_results]
    colors_seed = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    x = np.arange(len(names))
    ax.bar(x - 0.2, secrecies, 0.35, label="Secrecy", color="tab:blue", alpha=0.7)
    ax.bar(x + 0.2, sensings, 0.35, label="Sensing", color="tab:green", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Observation Ablation Study", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_ROOT, "observation_ablation.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved observation_ablation.png")


def plot_alpha_sweep(alpha_results: list[dict]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    alphas = [r["alpha"] for r in alpha_results]
    secrecies = [r["avg_secrecy"] for r in alpha_results]
    sensings = [r["avg_sensing"] for r in alpha_results]

    ax1.plot(alphas, secrecies, "o-", color="tab:blue", linewidth=2, markersize=8)
    ax1.set_xlabel("Alpha (secrecy weight)")
    ax1.set_ylabel("Avg Secrecy Rate")
    ax1.grid(True, alpha=0.3)

    ax2.plot(alphas, sensings, "o-", color="tab:green", linewidth=2, markersize=8)
    ax2.set_xlabel("Alpha (secrecy weight)")
    ax2.set_ylabel("Avg Sensing Utility")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Alpha Sweep — Secrecy vs Sensing Trade-off", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUTPUT_ROOT, "alpha_sweep.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved alpha_sweep.png")

    # Pareto front
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(secrecies, sensings, "o-", color="tab:purple", linewidth=2, markersize=8)
    for i, a in enumerate(alphas):
        ax.annotate(f"\u03b1={a}", (secrecies[i], sensings[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Secrecy Rate")
    ax.set_ylabel("Sensing Utility")
    ax.grid(True, alpha=0.3)
    fig.suptitle("Pareto Front \u2014 Secrecy vs Sensing", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_ROOT, "pareto_front.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved pareto_front.png")


def write_report(all_correlations: dict, all_action_sat: dict,
                 ablation_results: list[dict], alpha_results: list[dict],
                 all_policy_stats: dict[str, list[dict]]):
    path = os.path.join(OUTPUT_ROOT, "reward_audit_report.md")
    with open(path, "w") as f:
        f.write("# \u2014 Reward and Policy Audit Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Part 1
        f.write("## Part 1: Reward Decomposition\n\n")
        for label, corr in all_correlations.items():
            f.write(f"### {label}\n")
            f.write(f"- corr(total, secrecy): {corr['corr_total_secrecy']:.4f}\n")
            f.write(f"- corr(total, sensing): {corr['corr_total_sensing']:.4f}\n")
            f.write(f"- corr(total, penalty): {corr['corr_total_penalties']:.4f}\n")
            f.write(f"- mean reward: {corr['mean_reward']:.4f}\n")
            f.write(f"- mean secrecy: {corr['mean_secrecy']:.4f}\n")
            f.write(f"- mean sensing: {corr['mean_sensing']:.4f}\n\n")

        # Part 2
        f.write("## Part 2: Policy Behaviour Audit\n\n")
        f.write("| Policy | Secrecy | Sensing | BS Power | Jammer Power | UAV Speed |\n")
        f.write("|--------|---------|---------|----------|--------------|-----------|\n")
        for label, rows in all_policy_stats.items():
            if not rows:
                continue
            sec = np.mean([r["secrecy"] for r in rows])
            sen = np.mean([r["sensing"] for r in rows])
            bp = np.mean([r["mean_bs_power"] for r in rows])
            jp = np.mean([r["mean_jammer_power"] for r in rows])
            sp = np.mean([r["mean_uav_speed"] for r in rows])
            f.write(f"| {label} | {sec:.4f} | {sen:.4f} | {bp:.4f} | {jp:.4f} | {sp:.4f} |\n")

        # Part 3
        f.write("\n## Part 3: Action Saturation\n\n")
        for label, rows in all_action_sat.items():
            f.write(f"### {label}\n")
            f.write("| Agent | Saturation | LB Fraction | UB Fraction | Clipped | Mean |\n")
            f.write("|-------|------------|-------------|-------------|---------|------|\n")
            for r in rows:
                f.write(f"| {r['agent']} | {r['saturation_fraction']:.2%} | "
                        f"{r['lb_fraction']:.2%} | {r['ub_fraction']:.2%} | "
                        f"{r['clipped_fraction']:.2%} | {r['mean_value']:.4f} |\n")

        # Part 4
        f.write("\n## Part 4: Observation Importance (Ablation)\n\n")
        f.write("| Condition | Secrecy | Sensing | Degradation |\n")
        f.write("|-----------|---------|---------|-------------|\n")
        baseline_sec = ablation_results[0]["avg_secrecy"] if ablation_results else 0
        for r in ablation_results:
            f.write(f"| {r['ablation']} | {r['avg_secrecy']:.4f} | "
                    f"{r['avg_sensing']:.4f} | {r.get('secrecy_degradation', 0):.1f}% |\n")

        # Part 5
        f.write("\n## Part 5: Reward Weight Sweep\n\n")
        f.write("| Alpha | Secrecy | Sensing | Reward |\n")
        f.write("|-------|---------|---------|--------|\n")
        for r in alpha_results:
            f.write(f"| {r['alpha']} | {r['avg_secrecy']:.4f} | "
                    f"{r['avg_sensing']:.4f} | {r['avg_reward']:.4f} |\n")

        # Acceptance criteria
        f.write("\n## Acceptance Criteria\n\n")

        # C1: corr(total_reward, secrecy) > 0.3
        c1 = any(c.get("corr_total_secrecy", 0) > 0.3
                 for c in all_correlations.values())
        f.write(f"- C1 (corr reward-secrecy > 0.3): {'PASS' if c1 else 'FAIL'}\n")
        for label, corr in all_correlations.items():
            f.write(f"  - {label}: {corr['corr_total_secrecy']:.4f}\n")

        # C2: trained secrecy >= random secrecy
        rand_sec = 0.0
        if "random_feasible" in all_policy_stats and all_policy_stats["random_feasible"]:
            rand_sec = float(np.mean([r["secrecy"] for r in all_policy_stats["random_feasible"]]))
        trained_max_sec = 0.0
        for label in ["mappo", "matd3"]:
            if label in all_policy_stats and all_policy_stats[label]:
                sec = float(np.mean([r["secrecy"] for r in all_policy_stats[label]]))
                trained_max_sec = max(trained_max_sec, sec)
        c2 = trained_max_sec >= rand_sec
        f.write(f"- C2 (trained secrecy >= random): {'PASS' if c2 else 'FAIL'}\n")
        f.write(f"  - Random secrecy: {rand_sec:.4f}, Trained max: {trained_max_sec:.4f}\n")

        # C3: less than 50% actions clipped
        max_sat = 0.0
        for label, rows in all_action_sat.items():
            for r in rows:
                max_sat = max(max_sat, r["saturation_fraction"])
        c3 = max_sat < 0.5
        f.write(f"- C3 (<50% actions saturated): {'PASS' if c3 else 'FAIL'}\n")
        f.write(f"  - Max saturation: {max_sat:.2%}\n")

        # C4: removing secrecy observations decreases performance
        c4 = False
        if len(ablation_results) >= 2:
            base_sec = ablation_results[0]["avg_secrecy"]
            no_sec_sec = ablation_results[1]["avg_secrecy"]
            c4 = no_sec_sec < base_sec
            f.write(f"- C4 (no-secrecy ablation degrades): {'PASS' if c4 else 'FAIL'}\n")
            f.write(f"  - Full obs secrecy: {base_sec:.4f}, No secrecy obs: {no_sec_sec:.4f}\n")

        # C5: meaningful Pareto trade-off
        c5 = len(alpha_results) >= 3
        if c5:
            sec_vals = [r["avg_secrecy"] for r in alpha_results]
            sen_vals = [r["avg_sensing"] for r in alpha_results]
            sec_range = max(sec_vals) - min(sec_vals)
            sen_range = max(sen_vals) - min(sen_vals)
            c5 = sec_range > 0.1 and sen_range > 0.1
            f.write(f"- C5 (Pareto trade-off): {'PASS' if c5 else 'FAIL'}\n")
            f.write(f"  - Secrecy range: {sec_range:.4f}, Sensing range: {sen_range:.4f}\n")

        all_pass = all([c1, c2, c3, c4, c5])
        decision = "REWARD_DESIGN_VALIDATED" if all_pass else "REWARD_DESIGN_BROKEN"
        f.write(f"\n## Decision: {decision}\n")

    print(f"  Saved {path}")
    with open(os.path.join(OUTPUT_ROOT, "decision.txt"), "w") as f:
        f.write(decision)
    return decision


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("REWARD AND POLICY AUDIT")
    print("=" * 70)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    all_correlations = {}
    all_action_sat = {}
    all_policy_stats = {}

    # ── Part 1: Reward Decomposition ──
    print(f"\n{'=' * 60}")
    print("PART 1: REWARD DECOMPOSITION")
    print(f"{'=' * 60}")

    for algo, ckpt_dir in [("mappo", MAPPO_CKPT_DIR), ("matd3", MATD3_CKPT_DIR)]:
        rows = run_reward_decomposition(algo, ckpt_dir, n_episodes=N_EVAL)
        corr = compute_reward_correlations(rows)
        all_correlations[algo] = corr
        save_reward_decomposition(rows, corr, algo)
        print(f"  {algo}: corr(total,secrecy)={corr['corr_total_secrecy']:.4f}, "
              f"corr(total,sensing)={corr['corr_total_sensing']:.4f}")

    # Combine for plotting
    combined_rew = {}
    for algo in ["mappo", "matd3"]:
        path = os.path.join(OUTPUT_ROOT, f"reward_components_{algo}.csv")
        if os.path.exists(path):
            with open(path) as f:
                reader = csv.DictReader(f)
                combined_rew[algo] = list(reader)
    if combined_rew:
        plot_reward_components(combined_rew)

    # ── Part 2: Policy Behaviour Audit ──
    print(f"\n{'=' * 60}")
    print("PART 2: POLICY BEHAVIOUR AUDIT")
    print(f"{'=' * 60}")

    env_cfg = make_env_cfg()
    env = ISACMultiAgentEnv(env_cfg, seed=42)

    # Random baseline
    rand_stats = random_feasible_stats(env, n_episodes=N_EVAL)
    all_policy_stats["random_feasible"] = rand_stats
    print(f"  random_feasible: secrecy={np.mean([r['secrecy'] for r in rand_stats]):.4f}")

    # MAPPO evaluation
    if os.path.exists(MAPPO_CKPT_DIR):
        mappo_agents = load_trained_agents(env, MAPPO_CKPT_DIR, "mappo")
        mappo_stats = evaluate_policy_stats(mappo_agents, env, n_episodes=N_EVAL, label="mappo")
        all_policy_stats["mappo"] = mappo_stats
        print(f"  mappo: secrecy={np.mean([r['secrecy'] for r in mappo_stats]):.4f}")

    # MATD3 evaluation
    if os.path.exists(MATD3_CKPT_DIR):
        matd3_agents = load_trained_agents(env, MATD3_CKPT_DIR, "matd3")
        matd3_stats = evaluate_policy_stats(matd3_agents, env, n_episodes=N_EVAL, label="matd3")
        all_policy_stats["matd3"] = matd3_stats
        print(f"  matd3: secrecy={np.mean([r['secrecy'] for r in matd3_stats]):.4f}")

    # SCA-BCD baseline
    sca_stats = sca_bcd_stats(n_episodes=3)
    if sca_stats:
        all_policy_stats["sca_bcd"] = sca_stats
        print(f"  sca_bcd: secrecy={np.mean([r['secrecy'] for r in sca_stats]):.4f}")

    save_policy_stats(all_policy_stats, "all")
    plot_policy_statistics(all_policy_stats)

    # ── Part 3: Action Saturation ──
    for algo, ckpt_dir in [("mappo", MAPPO_CKPT_DIR), ("matd3", MATD3_CKPT_DIR)]:
        if os.path.exists(ckpt_dir):
            sat_rows = compute_action_saturation(algo, ckpt_dir, n_episodes=N_EVAL)
            all_action_sat[algo] = sat_rows
            save_action_saturation(sat_rows, algo)

    plot_action_saturation(all_action_sat)

    # ── Part 4: Observation Importance (Ablation) ──
    ablation_results = run_ablation_study(n_episodes=50)
    plot_ablation(ablation_results)

    # ── Part 5: Reward Weight Sweep ──
    alpha_results = run_alpha_sweep(n_episodes=50)
    plot_alpha_sweep(alpha_results)

    # ── Report and Decision ──
    print(f"\n{'=' * 60}")
    print("GENERATING REPORT")
    print(f"{'=' * 60}")

    decision = write_report(all_correlations, all_action_sat,
                            ablation_results, alpha_results, all_policy_stats)
    print(f"\nDecision: {decision}")
    print(f"\nAll outputs in {OUTPUT_ROOT}")
    return decision


if __name__ == "__main__":
    decision = main()
    sys.exit(0 if "VALIDATED" in decision else 1)
