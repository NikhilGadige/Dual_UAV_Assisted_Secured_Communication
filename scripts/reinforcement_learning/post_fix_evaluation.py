"""
Post-Fix Evaluation

Re-evaluate MARL after fixing actor initialization (orthogonal init,
actor output gain = 0.01).

Trains MAPPO and MATD3 for 1000 episodes each, compares against
random feasible, previous MARL results, and SCA-BCD.

Outputs: outputs/reinforcement_learning/post_fix_evaluation/
"""

from __future__ import annotations

import csv
import gc
import json
import os
import sys
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

from madrl_exp.configs import (
    MADRLConfig, EnvConfig, AgentConfig, TrainingConfig, RewardConfig,
)
from madrl_exp.training.trainer import MARLTrainer
from madrl_exp.environment import ISACMultiAgentEnv
from madrl_exp.evaluation import random_feasible_baseline, sca_bcd_baseline

OUTPUT_ROOT = os.path.join("outputs", "madrl_post_fix")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ── Baselines from previous runs ────────────────────────────
PREVIOUS_MAPPO = {"secrecy": 2.7884, "sensing": 42.2637}
PREVIOUS_MATD3 = {"secrecy": 0.0500, "sensing": 41.8938}
PREVIOUS_SCA_BCD = {"secrecy": 9.4397, "sensing": 43.6146}


def make_agents_cfg(env_cfg: EnvConfig, algo: str,
                    lr: float = 3e-4, temperature: float = 1.0,
                    clip_logits: bool = False) -> list[AgentConfig]:
    bs_act = 2 * env_cfg.N_time * env_cfg.M_bs
    uav_act = 3 * env_cfg.N_time
    jam_act = 2 * env_cfg.N_time * env_cfg.N_j
    return [
        AgentConfig(name="bs_beamformer", act_dim=bs_act,
                    lr=lr, temperature=temperature, clip_logits=clip_logits),
        AgentConfig(name="uav_trajectory", act_dim=uav_act,
                    lr=lr, temperature=temperature, clip_logits=clip_logits),
        AgentConfig(name="jammer_beamformer", act_dim=jam_act,
                    lr=lr, temperature=temperature, clip_logits=clip_logits),
    ]


def train_and_evaluate(algo: str, env_cfg: EnvConfig, reward_cfg: RewardConfig,
                       n_episodes: int = 1000) -> dict:
    """Train MARL for n_episodes and return evaluation results."""
    label = algo.upper()
    print(f"\n{'=' * 60}")
    print(f"  Training {label} for {n_episodes} episodes")
    print(f"{'=' * 60}")

    agents_cfg = make_agents_cfg(env_cfg, algo)
    train_cfg = TrainingConfig(
        algorithm=algo, n_episodes=n_episodes,
        max_steps_per_episode=50,
        eval_interval=100 if n_episodes >= 200 else 25,
        save_interval=200,
        log_interval=10, seed=42,
        output_root=os.path.join(OUTPUT_ROOT, f"run_{algo}"),
    )
    cfg = MADRLConfig(
        env=env_cfg, agents=agents_cfg, training=train_cfg,
        reward=reward_cfg,
        output_root=train_cfg.output_root,
    )

    trainer = MARLTrainer(cfg)
    trainer.train()

    env = trainer.env
    agents = trainer.agents

    # ── Evaluate ────────────────────────────────────────────
    for agent in agents.values():
        agent.clear_action_log()

    eval_secrecies, eval_sensings, eval_rewards = [], [], []
    for ep in range(20):
        obs, _ = env.reset(seed=42 + ep)
        for step in range(env_cfg.N_time):
            actions = {n: a.act(obs[n], deterministic=True)
                       for n, a in agents.items()}
            obs, rew_dict, _, _, info = env.step(actions)
        eval_secrecies.append(float(info.get("secrecy", 0.0)))
        eval_sensings.append(float(info.get("sensing", 0.0)))
        eval_rewards.append(float(rew_dict[env.agent_names[0]]))

    # Action saturation
    agent_sats = {}
    for name, agent in agents.items():
        sat = agent.compute_saturation()
        agent_sats[name] = sat

    bs_sat = agent_sats.get("bs_beamformer", {}).get("fraction_saturated", 1.0)
    max_sat = max(s["fraction_saturated"] for s in agent_sats.values())

    # Correlation
    if np.std(eval_rewards) > 1e-12 and np.std(eval_secrecies) > 1e-12:
        corr_rs = float(np.corrcoef(eval_rewards, eval_secrecies)[0, 1])
    else:
        corr_rs = 0.0

    result = {
        "algo": label,
        "avg_secrecy": float(np.mean(eval_secrecies)),
        "std_secrecy": float(np.std(eval_secrecies)),
        "avg_sensing": float(np.mean(eval_sensings)),
        "std_sensing": float(np.std(eval_sensings)),
        "avg_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "bs_saturation": bs_sat,
        "max_saturation": max_sat,
        "corr_reward_secrecy": corr_rs,
        "agent_saturations": agent_sats,
        "history": dict(trainer.history),
    }

    # Save trained model for generalization eval
    ckpt_dir = os.path.join(OUTPUT_ROOT, "checkpoints", algo)
    os.makedirs(ckpt_dir, exist_ok=True)
    for name, agent in agents.items():
        path = os.path.join(ckpt_dir, f"{name}_final.pt")
        agent.save(path)

    print(f"\n  {label} results: secrecy={result['avg_secrecy']:.4f} +/- "
          f"{result['std_secrecy']:.4f}, "
          f"reward={result['avg_reward']:.4f}, "
          f"BS sat={result['bs_saturation']:.2%}, "
          f"corr={result['corr_reward_secrecy']:.4f}")

    del trainer
    gc.collect()
    return result


def evaluate_generalization(algo: str, env_cfg: EnvConfig,
                            reward_cfg: RewardConfig,
                            seeds: list[int] = None) -> dict:
    """Evaluate trained policy on different random seeds."""
    if seeds is None:
        seeds = [0, 10, 20, 30, 50, 100, 200, 500]

    from madrl_exp.agents.mappo import MAPPOAgent
    from madrl_exp.agents.matd3 import MATD3Agent
    AgentClass = MAPPOAgent if algo == "mappo" else MATD3Agent

    env = ISACMultiAgentEnv(env_cfg, reward_cfg, seed=42)
    agents = {}
    for ac in make_agents_cfg(env_cfg, algo):
        name = ac.name
        obs_dim = env.observation_spaces[name].shape[0]
        act_dim = env.action_spaces[name].shape[0]
        agent = AgentClass(obs_dim, act_dim, name, device="cpu")
        ckpt_path = os.path.join(OUTPUT_ROOT, "checkpoints", algo, f"{name}_final.pt")
        if os.path.exists(ckpt_path):
            agent.load(ckpt_path)
        else:
            print(f"  WARNING: {ckpt_path} not found")
            return {}
        agents[name] = agent

    for agent in agents.values():
        agent.eval_mode()

    results = {}
    for seed in seeds:
        secrecies, sensings, rewards = [], [], []
        for ep in range(5):
            obs, _ = env.reset(seed=seed + ep)
            for step in range(env_cfg.N_time):
                actions = {n: a.act(obs[n], deterministic=True)
                           for n, a in agents.items()}
                obs, rew_dict, _, _, info = env.step(actions)
            secrecies.append(float(info.get("secrecy", 0.0)))
            sensings.append(float(info.get("sensing", 0.0)))
            rewards.append(float(rew_dict[env.agent_names[0]]))
        results[seed] = {
            "secrecy": float(np.mean(secrecies)),
            "sensing": float(np.mean(sensings)),
            "reward": float(np.mean(rewards)),
        }

    return results


def plot_learning_curves(mappo_result: dict, matd3_result: dict):
    """Plot training curves for MAPPO and MATD3."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (label, result) in enumerate([("MAPPO", mappo_result),
                                            ("MATD3", matd3_result)]):
        hist = result.get("history", {})
        names = ["bs_beamformer", "uav_trajectory", "jammer_beamformer"]

        # Reward
        ax = axes[0, 0]
        for n in names:
            vals = hist.get(f"{n}/reward", [])
            if vals:
                ax.plot(vals, alpha=0.6, label=f"{label} {n}" if idx == 0 else None)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        ax.set_title("Training Reward")

        # Secrecy
        ax = axes[0, 1]
        vals = hist.get("secrecy", [])
        if vals:
            ax.plot(vals, label=label, alpha=0.8)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Secrecy Rate")
        ax.grid(True, alpha=0.3)

        # Policy loss
        ax = axes[1, 0]
        for n in names:
            vals = hist.get(f"{n}/policy_loss", [])
            if vals:
                ax.plot(vals, alpha=0.4, label=f"{label} {n}" if idx == 0 else None)
        ax.set_xlabel("Update")
        ax.set_ylabel("Policy Loss")
        ax.grid(True, alpha=0.3)

        # Pre-tanh mean
        ax = axes[1, 1]
        for n in names:
            vals = hist.get(f"{n}/pre_tanh_mean", [])
            if vals:
                ax.plot(vals, alpha=0.6, label=f"{label} {n}" if idx == 0 else None)
        ax.axhline(y=5.0, color="gray", linestyle=":", label="Threshold (5.0)")
        ax.set_xlabel("Update")
        ax.set_ylabel("Mean |pre_tanh|")
        ax.grid(True, alpha=0.3)

    axes[0, 1].legend()
    axes[1, 1].legend()
    fig.suptitle("Post-Fix Learning Curves", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_ROOT, "learning_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved learning_curves.png")


def plot_secrecy_comparison(mappo_result: dict, matd3_result: dict,
                            random_base: dict, sca_base: dict):
    """Bar chart comparing secrecy across methods."""
    names = ["Random\nFeasible", "SCA-BCD", "MAPPO\n(new init)",
             "MATD3\n(new init)", "MAPPO\n(previous)", "MATD3\n(previous)"]
    secrecies = [
        random_base["avg_secrecy"],
        sca_base["avg_secrecy"],
        mappo_result["avg_secrecy"],
        matd3_result["avg_secrecy"],
        PREVIOUS_MAPPO["secrecy"],
        PREVIOUS_MATD3["secrecy"],
    ]
    stds = [
        random_base.get("std_secrecy", 0),
        sca_base.get("std_secrecy", 0),
        mappo_result.get("std_secrecy", 0),
        matd3_result.get("std_secrecy", 0),
        0, 0,
    ]
    colors = ["gray", "tab:blue", "tab:green", "tab:orange", "lightgreen", "peachpuff"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    bars = ax.bar(x, secrecies, 0.6, color=colors, alpha=0.85, yerr=stds,
                  capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Secrecy Rate")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Secrecy Comparison — Post-Fix Re-evaluation")

    for bar, val in zip(bars, secrecies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_ROOT, "secrecy_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved secrecy_curves.png")


def write_saturation_report(mappo_result: dict, matd3_result: dict) -> str:
    path = os.path.join(OUTPUT_ROOT, "action_saturation_report.md")
    with open(path, "w") as f:
        f.write("# Action Saturation Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for label, result in [("MAPPO", mappo_result), ("MATD3", matd3_result)]:
            f.write(f"## {label}\n\n")
            f.write("| Agent | Saturation | Pre-Tanh | Post-Tanh |\n")
            f.write("|-------|------------|----------|-----------|\n")
            for aname, sat in result.get("agent_saturations", {}).items():
                f.write(f"| {aname} | {sat['fraction_saturated']:.2%} | "
                        f"{sat['mean_pre_tanh']:.4f} | {sat['mean_post_tanh']:.4f} |\n")
            f.write("\n")

        f.write("## Acceptance\n\n")
        f.write("C1: BS saturation < 50%\n\n")
        for label, result in [("MAPPO", mappo_result), ("MATD3", matd3_result)]:
            bs_sat = result.get("bs_saturation", 1.0)
            s = "PASS" if bs_sat < 0.5 else "FAIL"
            f.write(f"- **{label}**: BS saturation = {bs_sat:.2%} ({s})\n")

    print(f"Saved {path}")
    return path


def write_generalization_report(mappo_gen: dict, matd3_gen: dict) -> str:
    path = os.path.join(OUTPUT_ROOT, "generalization_report.md")
    with open(path, "w") as f:
        f.write("# Generalization Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Evaluation of trained policies across different random seeds.\n\n")

        for label, gen in [("MAPPO", mappo_gen), ("MATD3", matd3_gen)]:
            f.write(f"## {label}\n\n")
            f.write("| Seed | Secrecy | Sensing | Reward |\n")
            f.write("|------|---------|---------|--------|\n")
            secrecies = []
            for seed in sorted(gen.keys()):
                r = gen[seed]
                secrecies.append(r["secrecy"])
                f.write(f"| {seed} | {r['secrecy']:.4f} | {r['sensing']:.4f} | {r['reward']:.4f} |\n")
            f.write(f"\n**Mean across seeds**: {float(np.mean(secrecies)):.4f} +/- "
                    f"{float(np.std(secrecies)):.4f}\n\n")

    print(f"Saved {path}")
    return path


def write_final_report(mappo_result: dict, matd3_result: dict,
                       random_base: dict, sca_base: dict,
                       mappo_gen: dict, matd3_gen: dict) -> str:
    path = os.path.join(OUTPUT_ROOT, "post_fix_report.md")
    with open(path, "w") as f:
        f.write("# Post-Fix Re-evaluation Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Comparison Table\n\n")
        f.write("| Method | Secrecy | Sensing | Reward | BS Sat | corr(R,S) |\n")
        f.write("|--------|---------|---------|--------|--------|-----------|\n")

        entries = [
            ("Random Feasible", random_base["avg_secrecy"],
             random_base["avg_sensing"], 0.0, None, None),
            ("SCA-BCD", sca_base["avg_secrecy"],
             sca_base["avg_sensing"], 0.0, None, None),
            ("MAPPO (new init)", mappo_result["avg_secrecy"],
             mappo_result["avg_sensing"], mappo_result["avg_reward"],
             mappo_result["bs_saturation"], mappo_result["corr_reward_secrecy"]),
            ("MATD3 (new init)", matd3_result["avg_secrecy"],
             matd3_result["avg_sensing"], matd3_result["avg_reward"],
             matd3_result["bs_saturation"], matd3_result["corr_reward_secrecy"]),
            ("MAPPO (previous)", PREVIOUS_MAPPO["secrecy"],
             PREVIOUS_MAPPO["sensing"], None, None, None),
            ("MATD3 (previous)", PREVIOUS_MATD3["secrecy"],
             PREVIOUS_MATD3["sensing"], None, None, None),
        ]
        for name, sec, sen, rew, sat, corr in entries:
            rew_s = f"{rew:.4f}" if rew is not None else "N/A"
            sat_s = f"{sat:.2%}" if sat is not None else "N/A"
            corr_s = f"{corr:.4f}" if corr is not None else "N/A"
            f.write(f"| {name} | {sec:.4f} | {sen:.4f} | {rew_s} | {sat_s} | {corr_s} |\n")

        f.write("\n## Acceptance Criteria\n\n")

        rand_sec = random_base["avg_secrecy"]

        # C1: BS saturation < 50%
        c1_mappo = mappo_result.get("bs_saturation", 1.0) < 0.5
        c1_matd3 = matd3_result.get("bs_saturation", 1.0) < 0.5
        f.write(f"- **C1 (BS sat < 50%)**: MAPPO={'PASS' if c1_mappo else 'FAIL'} "
                f"({mappo_result.get('bs_saturation', 1.0):.2%}), "
                f"MATD3={'PASS' if c1_matd3 else 'FAIL'} "
                f"({matd3_result.get('bs_saturation', 1.0):.2%})\n")

        # C2: trained secrecy > previous trained secrecy
        c2_mappo = mappo_result["avg_secrecy"] > PREVIOUS_MAPPO["secrecy"]
        c2_matd3 = matd3_result["avg_secrecy"] > PREVIOUS_MATD3["secrecy"]
        f.write(f"- **C2 (secrecy > previous)**: MAPPO={'PASS' if c2_mappo else 'FAIL'} "
                f"({mappo_result['avg_secrecy']:.4f} > {PREVIOUS_MAPPO['secrecy']:.4f}), "
                f"MATD3={'PASS' if c2_matd3 else 'FAIL'} "
                f"({matd3_result['avg_secrecy']:.4f} > {PREVIOUS_MATD3['secrecy']:.4f})\n")

        # C3: trained secrecy > random secrecy
        c3_mappo = mappo_result["avg_secrecy"] > rand_sec
        c3_matd3 = matd3_result["avg_secrecy"] > rand_sec
        f.write(f"- **C3 (secrecy > random)**: MAPPO={'PASS' if c3_mappo else 'FAIL'} "
                f"({mappo_result['avg_secrecy']:.4f} > {rand_sec:.4f}), "
                f"MATD3={'PASS' if c3_matd3 else 'FAIL'} "
                f"({matd3_result['avg_secrecy']:.4f} > {rand_sec:.4f})\n")

        # C4: corr(reward,secrecy) > 0.5
        c4_mappo = mappo_result.get("corr_reward_secrecy", 0) > 0.5
        c4_matd3 = matd3_result.get("corr_reward_secrecy", 0) > 0.5
        f.write(f"- **C4 (corr > 0.5)**: MAPPO={'PASS' if c4_mappo else 'FAIL'} "
                f"({mappo_result.get('corr_reward_secrecy', 0):.4f}), "
                f"MATD3={'PASS' if c4_matd3 else 'FAIL'} "
                f"({matd3_result.get('corr_reward_secrecy', 0):.4f})\n")

        all_pass = all([c1_mappo, c2_mappo, c3_mappo, c4_mappo,
                        c1_matd3, c2_matd3, c3_matd3, c4_matd3])
        decision = "POST_FIX_MARL_VALIDATED" if all_pass else "FURTHER_TUNING_REQUIRED"
        f.write(f"\n## Decision: {decision}\n")

    print(f"Saved {path}")
    return decision


def save_comparison_csv(mappo_result: dict, matd3_result: dict,
                        random_base: dict, sca_base: dict):
    path = os.path.join(OUTPUT_ROOT, "comparison_table.csv")
    with open(path, "w", newline="") as f:
        fields = ["method", "avg_secrecy", "std_secrecy", "avg_sensing",
                   "std_sensing", "avg_reward", "std_reward",
                   "bs_saturation", "max_saturation", "corr_reward_secrecy"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        w.writerow({
            "method": "random_feasible", "avg_secrecy": random_base["avg_secrecy"],
            "std_secrecy": random_base.get("std_secrecy", 0),
            "avg_sensing": random_base["avg_sensing"],
            "std_sensing": random_base.get("std_sensing", 0),
        })
        w.writerow({
            "method": "sca_bcd", "avg_secrecy": sca_base["avg_secrecy"],
            "std_secrecy": sca_base.get("std_secrecy", 0),
            "avg_sensing": sca_base["avg_sensing"],
            "std_sensing": sca_base.get("std_sensing", 0),
        })
        for name, result in [("mappo", mappo_result), ("matd3", matd3_result)]:
            w.writerow({
                "method": name,
                "avg_secrecy": result["avg_secrecy"],
                "std_secrecy": result.get("std_secrecy", 0),
                "avg_sensing": result["avg_sensing"],
                "std_sensing": result.get("std_sensing", 0),
                "avg_reward": result["avg_reward"],
                "std_reward": result.get("std_reward", 0),
                "bs_saturation": result.get("bs_saturation", 0),
                "max_saturation": result.get("max_saturation", 0),
                "corr_reward_secrecy": result.get("corr_reward_secrecy", 0),
            })

    print(f"Saved {path}")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    env_cfg = EnvConfig(seed=42, action_range=1.0, beamform_mode="reim")
    reward_cfg = RewardConfig(
        reward_mode="normalized",
        lambda_constraint=0.1,
        lambda_outage=0.5,
        lambda_secret=1.0,
        R_target=2.5,
        lambda_action=0.0,
        obs_clip=0.0,
    )

    # ── Baselines ────────────────────────────────────────────
    print("Computing random feasible baseline...")
    random_base = random_feasible_baseline(env_cfg, n_episodes=20)
    print(f"  Random: secrecy={random_base['avg_secrecy']:.4f}, "
          f"sensing={random_base['avg_sensing']:.4f}")

    print("Computing SCA-BCD baseline...")
    sca_base = sca_bcd_baseline(env_cfg, n_episodes=5)
    print(f"  SCA-BCD: secrecy={sca_base['avg_secrecy']:.4f}, "
          f"sensing={sca_base['avg_sensing']:.4f}")

    # ── Train MAPPO ──────────────────────────────────────────
    mappo_result = train_and_evaluate("mappo", env_cfg, reward_cfg, n_episodes=1000)

    # ── Train MATD3 ──────────────────────────────────────────
    matd3_result = train_and_evaluate("matd3", env_cfg, reward_cfg, n_episodes=1000)

    # ── Generalization ───────────────────────────────────────
    print("\nEvaluating generalization...")
    mappo_gen = evaluate_generalization("mappo", env_cfg, reward_cfg)
    matd3_gen = evaluate_generalization("matd3", env_cfg, reward_cfg)

    # ── Outputs ──────────────────────────────────────────────
    save_comparison_csv(mappo_result, matd3_result, random_base, sca_base)
    plot_learning_curves(mappo_result, matd3_result)
    plot_secrecy_comparison(mappo_result, matd3_result, random_base, sca_base)
    write_saturation_report(mappo_result, matd3_result)
    write_generalization_report(mappo_gen, matd3_gen)
    decision = write_final_report(mappo_result, matd3_result, random_base, sca_base,
                                   mappo_gen, matd3_gen)

    with open(os.path.join(OUTPUT_ROOT, "decision.txt"), "w") as f:
        f.write(decision)
    print(f"\nDecision: {decision}")
    print(f"All outputs in {OUTPUT_ROOT}")
    return decision


if __name__ == "__main__":
    decision = main()
    sys.exit(0 if "VALIDATED" in decision else 1)
