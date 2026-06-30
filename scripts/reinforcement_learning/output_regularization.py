"""
Output Layer Regularization

Prevent actor output-layer weight growth that causes tanh saturation.

Configs (MAPPO, 500 episodes each):
  Baseline: lr=3e-4, wd=0
  A: lr=1e-4, wd=0
  B: lr=3e-4, wd=1e-4
  C: lr=1e-4, wd=1e-4
  D: lr=5e-5, wd=5e-4

Outputs: outputs/reinforcement_learning/action_saturation/regularization/
"""

from __future__ import annotations

import csv
import gc
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

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
from madrl_exp.evaluation import random_feasible_baseline


OUTPUT_ROOT = os.path.join("outputs", "reinforcement_learning", "action_saturation", "regularization")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


@dataclass
class CfgEntry:
    name: str
    lr: float
    weight_decay: float


CONFIGS = [
    CfgEntry("Baseline", lr=3e-4, weight_decay=0.0),
    CfgEntry("A", lr=1e-4, weight_decay=0.0),
    CfgEntry("B", lr=3e-4, weight_decay=1e-4),
    CfgEntry("C", lr=1e-4, weight_decay=1e-4),
    CfgEntry("D", lr=5e-5, weight_decay=5e-4),
]


def make_agents_cfg(env_cfg: EnvConfig, lr: float, wd: float) -> list[AgentConfig]:
    bs_act = 2 * env_cfg.N_time * env_cfg.M_bs
    uav_act = 3 * env_cfg.N_time
    jam_act = 2 * env_cfg.N_time * env_cfg.N_j
    return [
        AgentConfig(name="bs_beamformer", act_dim=bs_act, lr=lr, weight_decay=wd),
        AgentConfig(name="uav_trajectory", act_dim=uav_act, lr=lr, weight_decay=wd),
        AgentConfig(name="jammer_beamformer", act_dim=jam_act, lr=lr, weight_decay=wd),
    ]


def run_config(entry: CfgEntry, env_cfg: EnvConfig,
               reward_cfg: RewardConfig) -> dict:
    """Train MAPPO for 500 episodes with given config and evaluate."""
    print(f"\n{'=' * 60}")
    print(f"  Config {entry.name}: lr={entry.lr}, wd={entry.weight_decay}")
    print(f"{'=' * 60}")

    agents_cfg = make_agents_cfg(env_cfg, entry.lr, entry.weight_decay)
    train_cfg = TrainingConfig(
        algorithm="mappo", n_episodes=500,
        max_steps_per_episode=50,
        eval_interval=100,
        save_interval=999999,
        log_interval=10, seed=42,
        output_root=os.path.join(OUTPUT_ROOT, f"config_{entry.name}"),
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

    # Evaluate
    for a in agents.values():
        a.clear_action_log()

    eval_secrecies, eval_sensings, eval_rewards = [], [], []
    for ep in range(20):
        obs, _ = env.reset(seed=42 + ep)
        for _ in range(env_cfg.N_time):
            actions = {n: a.act(obs[n], deterministic=True)
                       for n, a in agents.items()}
            obs, rew_dict, _, _, info = env.step(actions)
        eval_secrecies.append(float(info.get("secrecy", 0.0)))
        eval_sensings.append(float(info.get("sensing", 0.0)))
        eval_rewards.append(float(rew_dict[env.agent_names[0]]))

    # Saturation
    agent_sats = {}
    for name, agent in agents.items():
        sat = agent.compute_saturation()
        agent_sats[name] = sat

    bs_sat = agent_sats.get("bs_beamformer", {}).get("fraction_saturated", 1.0)

    # Correlation
    if np.std(eval_rewards) > 1e-12 and np.std(eval_secrecies) > 1e-12:
        corr_rs = float(np.corrcoef(eval_rewards, eval_secrecies)[0, 1])
    else:
        corr_rs = 0.0

    # Pre-tanh stats from training history (last update)
    last_pre_tanh = 0.0
    for name in agents:
        vals = trainer.history.get(f"{name}/pre_tanh_mean", [])
        if vals:
            last_pre_tanh = max(last_pre_tanh, float(np.mean(vals[-10:])))

    # Weight norm history
    weight_norm_history = {}
    for name in agents:
        vals = trainer.history.get(f"{name}/output_weight_norm", [])
        weight_norm_history[name] = vals
    # Also collect per-update
    wnorm_by_update = defaultdict(list)
    for name in agents:
        vals = trainer.history.get(f"{name}/output_weight_norm", [])
        for i, v in enumerate(vals):
            wnorm_by_update[name].append(v)

    # Build per-update weight norm data across all agents
    first_key = list(weight_norm_history.keys())[0] if weight_norm_history else None
    n_updates = len(weight_norm_history.get(first_key, [])) if first_key else 0

    pre_tanh_by_agent = {}
    for name in agents:
        pre_tanh_by_agent[name] = trainer.history.get(f"{name}/pre_tanh_mean", [])

    result = {
        "name": entry.name,
        "lr": entry.lr,
        "weight_decay": entry.weight_decay,
        "avg_secrecy": float(np.mean(eval_secrecies)),
        "std_secrecy": float(np.std(eval_secrecies)),
        "avg_sensing": float(np.mean(eval_sensings)),
        "std_sensing": float(np.std(eval_sensings)),
        "avg_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "bs_saturation": bs_sat,
        "max_saturation": max(s["fraction_saturated"] for s in agent_sats.values()),
        "corr_reward_secrecy": corr_rs,
        "last_pre_tanh_mean": last_pre_tanh,
        "final_weight_norm": float(np.mean([
            weight_norm_history[n][-1] if weight_norm_history.get(n) else 0
            for n in agents
        ])),
        "agent_saturations": agent_sats,
        "weight_norm_history": dict(weight_norm_history),
        "pre_tanh_history": pre_tanh_by_agent,
        "n_updates": n_updates,
    }

    print(f"  Result: secrecy={result['avg_secrecy']:.4f}, "
          f"BS sat={result['bs_saturation']:.2%}, "
          f"corr={result['corr_reward_secrecy']:.4f}, "
          f"pre_tanh={result['last_pre_tanh_mean']:.4f}, "
          f"wnorm={result['final_weight_norm']:.4f}")

    del trainer
    gc.collect()
    return result


def plot_weight_norm_curves(all_results: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = ["bs_beamformer", "uav_trajectory", "jammer_beamformer"]

    # Per-agent weight norm
    ax = axes[0]
    for res in all_results:
        hist = res["weight_norm_history"]
        for n in names:
            vals = hist.get(n, [])
            if vals:
                x = np.arange(len(vals))
                ax.plot(x, vals, alpha=0.6,
                        label=f"{res['name']} {n}" if n == names[0] else None)
    ax.set_xlabel("Update")
    ax.set_ylabel("Output Weight Norm")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    # Mean weight norm across agents per config
    ax = axes[1]
    for res in all_results:
        hist = res["weight_norm_history"]
        means = []
        for i in range(res["n_updates"]):
            vals_at_i = [hist[n][i] for n in names if i < len(hist.get(n, []))]
            means.append(float(np.mean(vals_at_i)) if vals_at_i else 0)
        if means:
            ax.plot(means, label=f"{res['name']} (lr={res['lr']}, wd={res['weight_decay']})")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Update")
    ax.set_ylabel("Mean Output Weight Norm")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Output Weight Norm Curves", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUTPUT_ROOT, "weight_norm_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved weight_norm_curves.png")


def plot_logit_statistics(all_results: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = ["bs_beamformer", "uav_trajectory", "jammer_beamformer"]

    ax = axes[0]
    for res in all_results:
        pre_hist = res["pre_tanh_history"]
        for n in names:
            vals = pre_hist.get(n, [])
            if vals:
                ax.plot(vals, alpha=0.6,
                        label=f"{res['name']} {n}" if n == names[0] else None)
    ax.axhline(y=5.0, color="gray", linestyle=":", label="Threshold (5.0)")
    ax.set_xlabel("Update")
    ax.set_ylabel("Mean |pre_tanh|")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    # Final pre_tanh by agent (bar chart)
    ax = axes[1]
    x = np.arange(len(all_results))
    width = 0.25
    for i, n in enumerate(names):
        vals = [res.get("agent_saturations", {}).get(n, {}).get("mean_pre_tanh", 0)
                for res in all_results]
        ax.bar(x + i * width, vals, width, alpha=0.7, label=n)
    ax.axhline(y=5.0, color="gray", linestyle=":")
    ax.set_xticks(x + width)
    ax.set_xticklabels([r["name"] for r in all_results])
    ax.set_ylabel("Final Mean |pre_tanh|")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    fig.suptitle("Logit Statistics", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUTPUT_ROOT, "logit_statistics.png"), dpi=150)
    plt.close(fig)
    print("Saved logit_statistics.png")


def plot_saturation_curves(all_results: list[dict]):
    names = ["bs_beamformer", "uav_trajectory", "jammer_beamformer"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, n in enumerate(names):
        ax = axes[idx]
        cfg_names = []
        sats = []
        for res in all_results:
            sat = res.get("agent_saturations", {}).get(n, {}).get("fraction_saturated", 0)
            cfg_names.append(res["name"])
            sats.append(sat)
        bars = ax.bar(range(len(sats)), sats, 0.6, alpha=0.7)
        ax.axhline(y=0.7, color="red", linestyle="--", label="70% threshold")
        ax.set_xticks(range(len(sats)))
        ax.set_xticklabels(cfg_names, rotation=20, ha="right")
        ax.set_ylabel("Fraction Saturated")
        ax.set_title(n)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, sats):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Action Saturation by Agent", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUTPUT_ROOT, "saturation_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved saturation_curves.png")


def write_comparison_csv(all_results: list[dict], random_base: dict):
    path = os.path.join(OUTPUT_ROOT, "comparison_table.csv")
    with open(path, "w", newline="") as f:
        fields = [
            "config", "lr", "weight_decay",
            "avg_secrecy", "std_secrecy", "avg_sensing",
            "avg_reward", "bs_saturation", "max_saturation",
            "corr_reward_secrecy", "last_pre_tanh_mean", "final_weight_norm",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        w.writerow({
            "config": "random_feasible",
            "avg_secrecy": random_base["avg_secrecy"],
            "std_secrecy": random_base.get("std_secrecy", 0),
            "avg_sensing": random_base["avg_sensing"],
        })

        for res in all_results:
            w.writerow({
                "config": res["name"],
                "lr": res["lr"],
                "weight_decay": res["weight_decay"],
                "avg_secrecy": res["avg_secrecy"],
                "std_secrecy": res["std_secrecy"],
                "avg_sensing": res["avg_sensing"],
                "avg_reward": res["avg_reward"],
                "bs_saturation": res["bs_saturation"],
                "max_saturation": res["max_saturation"],
                "corr_reward_secrecy": res["corr_reward_secrecy"],
                "last_pre_tanh_mean": res["last_pre_tanh_mean"],
                "final_weight_norm": res["final_weight_norm"],
            })
    print(f"Saved {path}")


def write_report(all_results: list[dict], random_base: dict) -> str:
    path = os.path.join(OUTPUT_ROOT, "regularization_report.md")
    with open(path, "w") as f:
        f.write("# Output Layer Regularization Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Comparison Table\n\n")
        f.write("| Config | LR | WD | Secrecy | Sensing | Reward | BS Sat | "
                "Max Sat | corr(R,S) | |pre_tanh| | WNorm |\n")
        f.write("|--------|-----|------|---------|---------|--------|--------|"
                "---------|-----------|----------|-------|\n")

        f.write(f"| Random Feasible | - | - | {random_base['avg_secrecy']:.4f} | "
                f"{random_base['avg_sensing']:.4f} | - | - | - | - | - | - |\n")

        for res in all_results:
            f.write(f"| {res['name']} | {res['lr']:.0e} | {res['weight_decay']:.0e} | "
                    f"{res['avg_secrecy']:.4f} | {res['avg_sensing']:.4f} | "
                    f"{res['avg_reward']:.4f} | {res['bs_saturation']:.2%} | "
                    f"{res['max_saturation']:.2%} | {res['corr_reward_secrecy']:.4f} | "
                    f"{res['last_pre_tanh_mean']:.4f} | {res['final_weight_norm']:.4f} |\n")

        f.write("\n## Acceptance Criteria\n\n")
        rand_sec = random_base["avg_secrecy"]

        for res in all_results:
            f.write(f"### {res['name']} (lr={res['lr']:.0e}, wd={res['weight_decay']:.0e})\n\n")

            c1 = res["bs_saturation"] < 0.7
            f.write(f"- **C1 (BS sat < 70%)**: {'PASS' if c1 else 'FAIL'} "
                    f"({res['bs_saturation']:.2%})\n")

            wnorm = res["final_weight_norm"]
            c2 = wnorm < 2.0
            f.write(f"- **C2 (weight norm stabilizes < 2)**: "
                    f"{'PASS' if c2 else 'FAIL'} ({wnorm:.4f})\n")

            c3 = res["last_pre_tanh_mean"] < 5.0
            f.write(f"- **C3 (|pre_tanh| < 5)**: {'PASS' if c3 else 'FAIL'} "
                    f"({res['last_pre_tanh_mean']:.4f})\n")

            c4 = res["avg_secrecy"] >= rand_sec
            f.write(f"- **C4 (secrecy >= random {rand_sec:.4f})**: "
                    f"{'PASS' if c4 else 'FAIL'} ({res['avg_secrecy']:.4f})\n")

            c5 = res["corr_reward_secrecy"] > 0.5
            f.write(f"- **C5 (corr > 0.5)**: {'PASS' if c5 else 'FAIL'} "
                    f"({res['corr_reward_secrecy']:.4f})\n")

            all_pass = all([c1, c2, c3, c4, c5])
            f.write(f"\n**Overall**: {'PASS' if all_pass else 'FAIL'}\n\n")

        # Decision
        any_pass = all(
            all([
                r["bs_saturation"] < 0.7,
                r["final_weight_norm"] < 2.0,
                r["last_pre_tanh_mean"] < 5.0,
                r["avg_secrecy"] >= rand_sec,
                r["corr_reward_secrecy"] > 0.5,
            ])
            for r in all_results
        )
        f.write(f"## Decision: {'OUTPUT_REGULARIZATION_SUCCESS' if any_pass else 'SATURATION_PERSISTS'}\n")

    print(f"Saved {path}")
    return path


def main():
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

    # Random baseline
    print("Computing random feasible baseline...")
    random_base = random_feasible_baseline(env_cfg, n_episodes=20)
    print(f"  Random: secrecy={random_base['avg_secrecy']:.4f}, "
          f"sensing={random_base['avg_sensing']:.4f}")

    # Run configs sequentially
    all_results = []
    for entry in CONFIGS:
        result = run_config(entry, env_cfg, reward_cfg)
        all_results.append(result)

    # Generate outputs
    write_comparison_csv(all_results, random_base)
    plot_weight_norm_curves(all_results)
    plot_logit_statistics(all_results)
    plot_saturation_curves(all_results)
    report_path = write_report(all_results, random_base)

    # Read decision
    with open(report_path) as f:
        content = f.read()
    decision = None
    for line in content.splitlines():
        if line.startswith("## Decision:"):
            decision = line.replace("## Decision:", "").strip()

    with open(os.path.join(OUTPUT_ROOT, "decision.txt"), "w") as f:
        f.write(decision or "UNKNOWN")
    print(f"\nDecision: {decision}")
    print(f"All outputs in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
    sys.exit(0)
