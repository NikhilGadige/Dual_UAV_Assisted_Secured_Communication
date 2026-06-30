"""
Reward Design Repair Experiment

Redesign reward so maximizing reward also improves secrecy.

Parts:
  1. Reward normalization (z-score)
  2. Secrecy floor penalty
  3. Action regularization
  4. Reward ablation study (4 variants, 100ep each)
  5. Acceptance criteria

Outputs: outputs/reinforcement_learning/reward_design/repair/
"""

from __future__ import annotations

import csv
import os
import sys
import json
import math
from collections import defaultdict
from datetime import datetime
from copy import deepcopy

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

OUTPUT_ROOT = os.path.join("outputs", "reinforcement_learning", "reward_design", "repair")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def make_env_cfg(alpha: float = 0.5, seed: int = 42) -> EnvConfig:
    return EnvConfig(alpha=alpha, seed=seed)


def make_agents_cfg(env_cfg: EnvConfig) -> list[AgentConfig]:
    return [
        AgentConfig(name="bs_beamformer", act_dim=2 * env_cfg.N_time * env_cfg.M_bs),
        AgentConfig(name="uav_trajectory", act_dim=3 * env_cfg.N_time),
        AgentConfig(name="jammer_beamformer", act_dim=2 * env_cfg.N_time * env_cfg.N_j),
    ]


def compute_random_secrecy_baseline(env_cfg: EnvConfig, n_episodes: int = 10) -> float:
    """Compute average secrecy of random feasible actions."""
    env = ISACMultiAgentEnv(env_cfg, seed=42)
    secrecies = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        for _ in range(env_cfg.N_time):
            actions = {name: env.action_spaces[name].sample()
                       for name in env.agent_names}
            obs, _, _, _, info = env.step(actions)
        secrecies.append(float(info.get("secrecy", 0.0)))
    return float(np.mean(secrecies))


def evaluate_policy(agents: dict, env: ISACMultiAgentEnv,
                    n_episodes: int = 10) -> list[dict]:
    """Run evaluation episodes and return per-step metrics."""
    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        for step in range(env.cfg.N_time):
            actions = {n: a.act(obs[n], deterministic=True) for n, a in agents.items()}
            obs, rewards, terminated, truncated, info = env.step(actions)
            rows.append({
                "episode": ep,
                "step": step,
                "secrecy": float(info.get("secrecy", 0.0)),
                "sensing": float(info.get("sensing", 0.0)),
                "reward": float(rewards[env.agent_names[0]]),
                "violation": float(info.get("violation", 0.0)),
                "f": float(info.get("f", 0.0)),
                "R_norm": float(info.get("R_norm", 0.0)),
                "U_norm": float(info.get("U_norm", 0.0)),
            })
    return rows


def compute_action_saturation(agents: dict, env: ISACMultiAgentEnv,
                               n_episodes: int = 10) -> dict:
    """Compute action saturation statistics per agent."""
    results = {}
    for agent_name in env.agent_names:
        lb_count, ub_count, total = 0, 0, 0
        all_actions = []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=42 + ep)
            for step in range(env.cfg.N_time):
                act = agents[agent_name].act(obs[agent_name], deterministic=True)
                all_actions.extend(act.tolist())
                total += len(act)
                lb_count += int(np.sum(act <= -0.99))
                ub_count += int(np.sum(act >= 0.99))
                # Step all agents to advance env
                all_acts = {n: agents[n].act(obs[n], deterministic=True)
                            for n in env.agent_names}
                obs, _, _, _, _ = env.step(all_acts)

        sat_frac = (lb_count + ub_count) / max(total, 1)
        results[agent_name] = {
            "saturation_fraction": sat_frac,
            "lb_fraction": lb_count / max(total, 1),
            "ub_fraction": ub_count / max(total, 1),
            "mean_value": float(np.mean(all_actions)),
        }
    return results


def get_training_rewards(trainer: MARLTrainer) -> list[float]:
    """Extract training reward history from trainer."""
    names = list(trainer.agents.keys())
    if not names:
        return []
    key = f"{names[0]}/reward"
    return [float(x) for x in trainer.history.get(key, [])]


def run_reward_ablation() -> dict:
    """Run 4 reward variants and collect results."""
    print("=" * 60)
    print("REWARD REPAIR EXPERIMENT")
    print("=" * 60)

    base_env_cfg = make_env_cfg()
    print(f"\nComputing random secrecy baseline...")
    random_secrecy = compute_random_secrecy_baseline(base_env_cfg, n_episodes=10)
    print(f"  Random secrecy: {random_secrecy:.4f}")

    # Define 4 reward variants
    variants = [
        {
            "name": "original",
            "label": "Original Reward",
            "reward_cfg": RewardConfig(reward_mode="original", R_target=random_secrecy),
        },
        {
            "name": "normalized",
            "label": "Normalized Reward",
            "reward_cfg": RewardConfig(reward_mode="normalized", R_target=random_secrecy),
        },
        {
            "name": "normalized_penalty",
            "label": "Norm + Secrecy Penalty",
            "reward_cfg": RewardConfig(
                reward_mode="normalized_penalty", R_target=random_secrecy,
                lambda_secret=1.0,
            ),
        },
        {
            "name": "full",
            "label": "Full (Norm + Penalty + Action Reg)",
            "reward_cfg": RewardConfig(
                reward_mode="full", R_target=random_secrecy,
                lambda_secret=1.0, lambda_action=1e-3,
            ),
        },
    ]

    results = []
    normalization_stats = []

    for variant in variants:
        name = variant["name"]
        label = variant["label"]
        reward_cfg = variant["reward_cfg"]

        print(f"\n{'=' * 50}")
        print(f"Training: {label}")
        print(f"{'=' * 50}")

        env_cfg = make_env_cfg()
        agents_cfg = make_agents_cfg(env_cfg)
        train_cfg = TrainingConfig(
            algorithm="mappo", n_episodes=100,
            max_steps_per_episode=50,
            eval_interval=25, save_interval=50,
            log_interval=10, seed=42,
            output_root=os.path.join(OUTPUT_ROOT, f"run_{name}"),
        )
        cfg = MADRLConfig(
            env=env_cfg, agents=agents_cfg, training=train_cfg,
            reward=reward_cfg,
            output_root=train_cfg.output_root,
        )

        trainer = MARLTrainer(cfg)
        trainer.train()

        # Extract normalization statistics from env
        env = trainer.env
        norm_stats = {
            "variant": name,
            "norm_R_count": env._norm_R.count,
            "norm_R_mean": round(env._norm_R.mean, 6),
            "norm_R_std": round(env._norm_R.std, 6),
            "norm_U_count": env._norm_U.count,
            "norm_U_mean": round(env._norm_U.mean, 6),
            "norm_U_std": round(env._norm_U.std, 6),
        }
        normalization_stats.append(norm_stats)

        # Evaluate
        print(f"  Evaluating {label}...")
        eval_rows = evaluate_policy(trainer.agents, env, n_episodes=10)

        # Compute action saturation
        sat = compute_action_saturation(trainer.agents, env, n_episodes=10)

        # Training reward progression for C5
        train_rewards = get_training_rewards(trainer)
        first_10_avg = float(np.mean(train_rewards[:10])) if len(train_rewards) >= 10 else 0.0
        last_10_avg = float(np.mean(train_rewards[-10:])) if len(train_rewards) >= 10 else 0.0

        # Compute correlations
        r_list = [r["reward"] for r in eval_rows]
        s_list = [r["secrecy"] for r in eval_rows]
        if np.std(r_list) > 1e-12 and np.std(s_list) > 1e-12:
            corr_rew_sec = float(np.corrcoef(r_list, s_list)[0, 1])
        else:
            corr_rew_sec = 0.0

        # Per-step averages
        avg_secrecy = float(np.mean([r["secrecy"] for r in eval_rows]))
        avg_sensing = float(np.mean([r["sensing"] for r in eval_rows]))
        avg_reward = float(np.mean([r["reward"] for r in eval_rows]))

        result = {
            "variant": name,
            "label": label,
            "avg_secrecy": avg_secrecy,
            "avg_sensing": avg_sensing,
            "avg_reward": avg_reward,
            "corr_reward_secrecy": corr_rew_sec,
            "secrecy_improvement": avg_secrecy - random_secrecy,
            "first_10_avg_reward": first_10_avg,
            "last_10_avg_reward": last_10_avg,
            "reward_improvement": last_10_avg - first_10_avg,
            "max_saturation": max(sat.values(), key=lambda x: x["saturation_fraction"])["saturation_fraction"],
            "action_saturation": sat,
            "random_secrecy": random_secrecy,
        }
        results.append(result)

        print(f"  secrecy={avg_secrecy:.4f}, sensing={avg_sensing:.4f}, "
              f"reward={avg_reward:.4f}, corr={corr_rew_sec:.4f}, "
              f"max_sat={result['max_saturation']:.2%}")

        # Clean up to free memory
        import gc
        del trainer
        gc.collect()

    # Save normalization statistics
    norm_path = os.path.join(OUTPUT_ROOT, "reward_normalization_statistics.csv")
    with open(norm_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "variant", "norm_R_count", "norm_R_mean", "norm_R_std",
            "norm_U_count", "norm_U_mean", "norm_U_std",
        ])
        w.writeheader()
        w.writerows(normalization_stats)
    print(f"\nSaved {norm_path}")

    return {
        "results": results,
        "normalization_stats": normalization_stats,
        "random_secrecy": random_secrecy,
    }


def plot_results(data: dict):
    """Generate ablation plot."""
    results = data["results"]
    names = [r["label"] for r in results]
    secrecies = [r["avg_secrecy"] for r in results]
    sensings = [r["avg_sensing"] for r in results]
    rewards = [r["avg_reward"] for r in results]
    max_sats = [r["max_saturation"] for r in results]
    corrs = [r["corr_reward_secrecy"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    x = np.arange(len(names))

    # Secrecy
    ax = axes[0, 0]
    colors_sec = ["tab:blue" if s >= data["random_secrecy"] else "tab:red" for s in secrecies]
    ax.bar(x, secrecies, 0.5, color=colors_sec, alpha=0.8)
    ax.axhline(y=data["random_secrecy"], color="gray", linestyle="--",
               label=f"Random: {data['random_secrecy']:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Avg Secrecy Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Secrecy Rate by Reward Variant")

    # Sensing
    ax = axes[0, 1]
    ax.bar(x, sensings, 0.5, color="tab:green", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Avg Sensing Utility")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Sensing Utility by Reward Variant")

    # Reward
    ax = axes[1, 0]
    ax.bar(x, rewards, 0.5, color="tab:orange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Avg Reward")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Total Reward by Reward Variant")

    # Correlation and Saturation
    ax = axes[1, 1]
    width = 0.35
    ax.bar(x - width/2, corrs, width, label="corr(reward, secrecy)", color="tab:purple", alpha=0.7)
    ax.bar(x + width/2, max_sats, width, label="max saturation", color="tab:red", alpha=0.5)
    ax.axhline(y=0.3, color="purple", linestyle=":", alpha=0.5)
    ax.axhline(y=0.7, color="red", linestyle=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Correlation and Saturation")

    fig.suptitle("Reward Ablation Study \u2014 ", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_ROOT, "reward_ablation.png"), dpi=150)
    plt.close(fig)
    print(f"Saved reward_ablation.png")


def save_csv(data: dict):
    """Save ablation results CSV."""
    results = data["results"]
    path = os.path.join(OUTPUT_ROOT, "reward_ablation.csv")
    with open(path, "w", newline="") as f:
        fieldnames = [
            "variant", "label", "avg_secrecy", "avg_sensing", "avg_reward",
            "corr_reward_secrecy", "secrecy_improvement",
            "first_10_avg_reward", "last_10_avg_reward", "reward_improvement",
            "max_saturation", "random_secrecy",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"Saved {path}")


def write_report(data: dict) -> str:
    """Generate report with acceptance criteria."""
    results = data["results"]
    random_secrecy = data["random_secrecy"]

    path = os.path.join(OUTPUT_ROOT, "reward_repair_report.md")
    with open(path, "w") as f:
        f.write("# \u2014 Reward Repair Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Random secrecy baseline**: {random_secrecy:.4f}\n\n")

        f.write("## Reward Ablation Results\n\n")
        f.write("| Variant | Secrecy | Sensing | Reward | corr(R,sec) | Max Sat | Sec Impr |\n")
        f.write("|---------|---------|---------|--------|-------------|---------|----------|\n")
        for r in results:
            f.write(f"| {r['label']} | {r['avg_secrecy']:.4f} | {r['avg_sensing']:.4f} | "
                    f"{r['avg_reward']:.4f} | {r['corr_reward_secrecy']:.4f} | "
                    f"{r['max_saturation']:.2%} | {r['secrecy_improvement']:+.4f} |\n")

        f.write("\n## Action Saturation Details\n\n")
        for r in results:
            f.write(f"### {r['label']}\n")
            f.write("| Agent | Saturation | LB Fraction | UB Fraction | Mean |\n")
            f.write("|-------|------------|-------------|-------------|------|\n")
            for agent_name, sat in r["action_saturation"].items():
                f.write(f"| {agent_name} | {sat['saturation_fraction']:.2%} | "
                        f"{sat['lb_fraction']:.2%} | {sat['ub_fraction']:.2%} | "
                        f"{sat['mean_value']:.4f} |\n")

        f.write("\n## Normalization Statistics\n\n")
        f.write("| Variant | R count | R mean | R std | U count | U mean | U std |\n")
        f.write("|---------|---------|--------|-------|---------|--------|-------|\n")
        for ns in data["normalization_stats"]:
            f.write(f"| {ns['variant']} | {ns['norm_R_count']} | {ns['norm_R_mean']:.4f} | "
                    f"{ns['norm_R_std']:.4f} | {ns['norm_U_count']} | {ns['norm_U_mean']:.4f} | "
                    f"{ns['norm_U_std']:.4f} |\n")

        f.write("\n## Acceptance Criteria\n\n")

        # C1: corr(reward, secrecy) > 0.3 for any variant
        c1 = any(r["corr_reward_secrecy"] > 0.3 for r in results)
        f.write(f"- **C1 (corr reward-secrecy > 0.3)**: {'**PASS**' if c1 else '**FAIL**'}\n")
        for r in results:
            f.write(f"  - {r['label']}: {r['corr_reward_secrecy']:.4f}\n")

        # C2: trained secrecy >= random secrecy for any variant
        c2 = any(r["avg_secrecy"] >= random_secrecy for r in results)
        best_sec = max(r["avg_secrecy"] for r in results) if results else 0
        f.write(f"- **C2 (trained secrecy >= random)**: {'**PASS**' if c2 else '**FAIL**'}\n")
        f.write(f"  - Random: {random_secrecy:.4f}, Best trained: {best_sec:.4f}\n")
        for r in results:
            status = "PASS" if r["avg_secrecy"] >= random_secrecy else "FAIL"
            f.write(f"  - {r['label']}: {r['avg_secrecy']:.4f} ({status})\n")

        # C3: action saturation < 70% for any variant
        c3 = any(r["max_saturation"] < 0.7 for r in results)
        min_sat = min(r["max_saturation"] for r in results) if results else 1.0
        f.write(f"- **C3 (<70% action saturated)**: {'**PASS**' if c3 else '**FAIL**'}\n")
        f.write(f"  - Min max-saturation: {min_sat:.2%}\n")
        for r in results:
            status = "PASS" if r["max_saturation"] < 0.7 else "FAIL"
            f.write(f"  - {r['label']}: {r['max_saturation']:.2%} ({status})\n")

        # C4: reward and secrecy positively correlated (> 0) for any variant
        c4 = any(r["corr_reward_secrecy"] > 0.0 for r in results)
        best_corr = max(r["corr_reward_secrecy"] for r in results) if results else 0
        f.write(f"- **C4 (reward-secrecy positive corr)**: {'**PASS**' if c4 else '**FAIL**'}\n")
        f.write(f"  - Best correlation: {best_corr:.4f}\n")
        for r in results:
            status = "PASS" if r["corr_reward_secrecy"] > 0.0 else "FAIL"
            f.write(f"  - {r['label']}: {r['corr_reward_secrecy']:.4f} ({status})\n")

        # C5: reward improvement (last 10 ep avg > first 10 ep avg) for any variant
        c5 = any(r["reward_improvement"] > 0 for r in results)
        best_imp = max(r["reward_improvement"] for r in results) if results else 0
        f.write(f"- **C5 (reward improvement during training)**: {'**PASS**' if c5 else '**FAIL**'}\n")
        f.write(f"  - Best improvement: {best_imp:+.4f}\n")
        for r in results:
            status = "PASS" if r["reward_improvement"] > 0 else "FAIL"
            f.write(f"  - {r['label']}: {r['first_10_avg_reward']:.4f} -> {r['last_10_avg_reward']:.4f} ({r['reward_improvement']:+.4f}) ({status})\n")

        all_pass = all([c1, c2, c3, c4, c5])
        decision = "REWARD_REPAIRED" if all_pass else "REWARD_STILL_BROKEN"
        f.write(f"\n## Decision: {decision}\n")

    print(f"Saved {path}")
    with open(os.path.join(OUTPUT_ROOT, "decision.txt"), "w") as f:
        f.write(decision)
    print(f"Decision: {decision}")
    return decision


def main() -> str:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    data = run_reward_ablation()
    save_csv(data)
    plot_results(data)
    decision = write_report(data)

    return decision


if __name__ == "__main__":
    decision = main()
    sys.exit(0 if "REPAIRED" in decision else 1)
