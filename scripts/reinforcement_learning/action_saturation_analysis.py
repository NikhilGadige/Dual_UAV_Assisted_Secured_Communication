"""
Action Saturation Analysis

Find config that reduces saturation while preserving repaired reward.

Outputs: outputs/reinforcement_learning/action_saturation/
"""

from __future__ import annotations

import csv
import os
import sys
import gc
import math
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

OUTPUT_ROOT = os.path.join("outputs", "reinforcement_learning", "action_saturation")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def make_agents_cfg(env_cfg: EnvConfig, entropy_coef: float = 0.01) -> list[AgentConfig]:
    return [
        AgentConfig(name="bs_beamformer", act_dim=2 * env_cfg.N_time * env_cfg.M_bs, entropy_coef=entropy_coef),
        AgentConfig(name="uav_trajectory", act_dim=3 * env_cfg.N_time, entropy_coef=entropy_coef),
        AgentConfig(name="jammer_beamformer", act_dim=2 * env_cfg.N_time * env_cfg.N_j, entropy_coef=entropy_coef),
    ]


def compute_random_baseline(env_cfg: EnvConfig, n_episodes: int = 10) -> dict:
    env = ISACMultiAgentEnv(env_cfg, seed=42)
    secrecies, sensings, rewards = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        for _ in range(env_cfg.N_time):
            actions = {n: env.action_spaces[n].sample() for n in env.agent_names}
            obs, rew_dict, _, _, info = env.step(actions)
        secrecies.append(float(info.get("secrecy", 0.0)))
        sensings.append(float(info.get("sensing", 0.0)))
        rewards.append(float(rew_dict[env.agent_names[0]]))
    return {
        "avg_secrecy": float(np.mean(secrecies)),
        "avg_sensing": float(np.mean(sensings)),
        "avg_reward": float(np.mean(rewards)),
    }


def evaluate_config(env_cfg: EnvConfig, reward_cfg: RewardConfig,
                    entropy_coef: float, label: str, n_episodes: int = 100) -> dict:
    """Train MAPPO and evaluate."""
    print(f"\n{'=' * 50}")
    print(f"  Config: {label}")
    print(f"  action_range={env_cfg.action_range}, lambda_action={reward_cfg.lambda_action}, "
          f"entropy={entropy_coef}, obs_clip={reward_cfg.obs_clip}")
    print(f"{'=' * 50}")

    agents_cfg = make_agents_cfg(env_cfg, entropy_coef)
    train_cfg = TrainingConfig(
        algorithm="mappo", n_episodes=n_episodes,
        max_steps_per_episode=50,
        eval_interval=25, save_interval=50,
        log_interval=10, seed=42,
        output_root=os.path.join(OUTPUT_ROOT, f"run_{label}"),
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

    # 1) Evaluate policy performance
    eval_rows = []
    for ep in range(10):
        obs, _ = env.reset(seed=42 + ep)
        for step in range(env_cfg.N_time):
            actions = {n: a.act(obs[n], deterministic=True) for n, a in agents.items()}
            obs, rew_dict, _, _, info = env.step(actions)
            eval_rows.append({
                "episode": ep, "step": step,
                "secrecy": float(info.get("secrecy", 0.0)),
                "sensing": float(info.get("sensing", 0.0)),
                "reward": float(rew_dict[env.agent_names[0]]),
            })

    avg_secrecy = float(np.mean([r["secrecy"] for r in eval_rows]))
    avg_sensing = float(np.mean([r["sensing"] for r in eval_rows]))
    avg_reward = float(np.mean([r["reward"] for r in eval_rows]))

    # Correlation
    r_vec = [r["reward"] for r in eval_rows]
    s_vec = [r["secrecy"] for r in eval_rows]
    if np.std(r_vec) > 1e-12 and np.std(s_vec) > 1e-12:
        corr_rs = float(np.corrcoef(r_vec, s_vec)[0, 1])
    else:
        corr_rs = 0.0

    # 2) Action saturation (evaluate with action logging)
    for agent in agents.values():
        agent.clear_action_log()

    for ep in range(10):
        obs, _ = env.reset(seed=42 + ep)
        for step in range(env_cfg.N_time):
            actions = {n: agents[n].act(obs[n], deterministic=True) for n in agents}
            obs, _, _, _, _ = env.step(actions)

    agent_sats = {}
    for name, agent in agents.items():
        sat = agent.compute_saturation()
        agent_sats[name] = sat
        print(f"  {name}: sat={sat['fraction_saturated']:.2%}, "
              f"pre_tanh={sat['mean_pre_tanh']:.4f}, "
              f"post_tanh={sat['mean_post_tanh']:.4f}")

    max_sat = max(s["fraction_saturated"] for s in agent_sats.values())

    # 3) Training reward trend (C5)
    train_rewards = []
    names_list = list(agents.keys())
    if names_list:
        key = f"{names_list[0]}/reward"
        train_rewards = [float(x) for x in trainer.history.get(key, [])]
    first_10 = float(np.mean(train_rewards[:10])) if len(train_rewards) >= 10 else 0.0
    last_10 = float(np.mean(train_rewards[-10:])) if len(train_rewards) >= 10 else 0.0
    reward_improvement = last_10 - first_10

    result = {
        "label": label,
        "action_range": env_cfg.action_range,
        "lambda_action": reward_cfg.lambda_action,
        "entropy_coef": entropy_coef,
        "obs_clip": reward_cfg.obs_clip,
        "avg_secrecy": avg_secrecy,
        "avg_sensing": avg_sensing,
        "avg_reward": avg_reward,
        "corr_reward_secrecy": corr_rs,
        "max_saturation": max_sat,
        "reward_improvement": reward_improvement,
        "agent_saturations": agent_sats,
    }

    print(f"  secrecy={avg_secrecy:.4f}, reward={avg_reward:.4f}, "
          f"corr={corr_rs:.4f}, max_sat={max_sat:.2%}, "
          f"rew_impr={reward_improvement:+.4f}")

    # Cleanup
    del trainer
    gc.collect()

    return result


def plot_results(all_results: list[dict], random_baseline: dict):
    names = [r["label"] for r in all_results]
    secrecies = [r["avg_secrecy"] for r in all_results]
    rewards = [r["avg_reward"] for r in all_results]
    corrs = [r["corr_reward_secrecy"] for r in all_results]
    max_sats = [r["max_saturation"] for r in all_results]
    imprs = [r["reward_improvement"] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(names))

    # Secrecy
    ax = axes[0, 0]
    rand_sec = random_baseline["avg_secrecy"]
    colors = ["tab:green" if s >= rand_sec else "tab:red" for s in secrecies]
    ax.bar(x, secrecies, 0.5, color=colors, alpha=0.8)
    ax.axhline(y=rand_sec, color="gray", linestyle="--", label=f"Random: {rand_sec:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Secrecy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Secrecy Rate")

    # Reward
    ax = axes[0, 1]
    ax.bar(x, rewards, 0.5, color="tab:orange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Total Reward")

    # Correlation
    ax = axes[1, 0]
    colors_corr = ["tab:green" if c > 0.5 else "tab:red" for c in corrs]
    ax.bar(x, corrs, 0.5, color=colors_corr, alpha=0.8)
    ax.axhline(y=0.5, color="purple", linestyle=":", label="C1 threshold (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("corr(reward, secrecy)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Reward-Secrecy Correlation")

    # Saturation
    ax = axes[1, 1]
    colors_sat = ["tab:green" if s < 0.7 else "tab:red" for s in max_sats]
    ax.bar(x, max_sats, 0.5, color=colors_sat, alpha=0.8)
    ax.axhline(y=0.7, color="red", linestyle=":", label="C3 threshold (70%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Max Saturation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Action Saturation")

    fig.suptitle("\u2014 Action Saturation Repair", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_ROOT, "saturation_ablation.png"), dpi=150)
    plt.close(fig)
    print(f"Saved saturation_ablation.png")


def write_report(all_results: list[dict], random_baseline: dict) -> str:
    path = os.path.join(OUTPUT_ROOT, "action_repair_report.md")
    with open(path, "w") as f:
        f.write("# \u2014 Action Saturation Repair Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Random baseline**: secrecy={random_baseline['avg_secrecy']:.4f}, "
                f"sensing={random_baseline['avg_sensing']:.4f}, "
                f"reward={random_baseline['avg_reward']:.4f}\n\n")

        f.write("## Configuration Results\n\n")
        f.write("| Config | Action R | la | Entropy | ObsClip | Secrecy | Reward | "
                "corr(R,S) | Max Sat | Rew Impr |\n")
        f.write("|--------|----------|----|---------|---------|---------|--------|"
                "-----------|---------|----------|\n")
        for r in all_results:
            f.write(f"| {r['label']} | {r['action_range']} | {r['lambda_action']} | "
                    f"{r['entropy_coef']} | {r['obs_clip']} | "
                    f"{r['avg_secrecy']:.4f} | {r['avg_reward']:.4f} | "
                    f"{r['corr_reward_secrecy']:.4f} | {r['max_saturation']:.2%} | "
                    f"{r['reward_improvement']:+.4f} |\n")

        f.write("\n## Per-Agent Saturation Details\n\n")
        for r in all_results:
            f.write(f"### {r['label']}\n")
            f.write("| Agent | Saturation | Pre-Tanh | Post-Tanh |\n")
            f.write("|-------|------------|----------|-----------|\n")
            for aname, sat in r["agent_saturations"].items():
                f.write(f"| {aname} | {sat['fraction_saturated']:.2%} | "
                        f"{sat['mean_pre_tanh']:.4f} | {sat['mean_post_tanh']:.4f} |\n")

        f.write("\n## Acceptance Criteria\n\n")

        # C1: max saturation < 70%
        c1 = any(r["max_saturation"] < 0.7 for r in all_results)
        best_label_sat = ""
        best_sat = 1.0
        for r in all_results:
            if r["max_saturation"] < best_sat:
                best_sat = r["max_saturation"]
                best_label_sat = r["label"]
        f.write(f"- **C1 (max saturation < 70%)**: {'**PASS**' if c1 else '**FAIL**'}\n")
        f.write(f"  - Best: {best_label_sat} ({best_sat:.2%})\n")
        for r in all_results:
            s = "PASS" if r["max_saturation"] < 0.7 else "FAIL"
            f.write(f"  - {r['label']}: {r['max_saturation']:.2%} ({s})\n")

        # C2: corr(reward, secrecy) > 0.5
        c2 = any(r["corr_reward_secrecy"] > 0.5 for r in all_results)
        best_corr = max(r["corr_reward_secrecy"] for r in all_results)
        best_label_corr = ""
        for r in all_results:
            if r["corr_reward_secrecy"] == best_corr:
                best_label_corr = r["label"]
        f.write(f"- **C2 (corr reward-secrecy > 0.5)**: {'**PASS**' if c2 else '**FAIL**'}\n")
        f.write(f"  - Best: {best_label_corr} ({best_corr:.4f})\n")
        for r in all_results:
            s = "PASS" if r["corr_reward_secrecy"] > 0.5 else "FAIL"
            f.write(f"  - {r['label']}: {r['corr_reward_secrecy']:.4f} ({s})\n")

        # C3: trained secrecy >= random secrecy
        rand_sec = random_baseline["avg_secrecy"]
        c3 = any(r["avg_secrecy"] >= rand_sec for r in all_results)
        best_sec = max(r["avg_secrecy"] for r in all_results)
        best_label_sec = ""
        for r in all_results:
            if r["avg_secrecy"] == best_sec:
                best_label_sec = r["label"]
        f.write(f"- **C3 (trained secrecy >= random)**: {'**PASS**' if c3 else '**FAIL**'}\n")
        f.write(f"  - Random: {rand_sec:.4f}, Best: {best_label_sec} ({best_sec:.4f})\n")
        for r in all_results:
            s = "PASS" if r["avg_secrecy"] >= rand_sec else "FAIL"
            f.write(f"  - {r['label']}: {r['avg_secrecy']:.4f} ({s})\n")

        # Final
        all_pass = all([c1, c2, c3])
        decision = "ACTION_POLICY_FIXED" if all_pass else "ACTION_POLICY_STILL_SATURATED"
        f.write(f"\n## Decision: {decision}\n")

    print(f"Saved {path}")
    with open(os.path.join(OUTPUT_ROOT, "decision.txt"), "w") as f:
        f.write(decision)
    print(f"Decision: {decision}")
    return decision


def save_csv(all_results: list[dict]):
    path = os.path.join(OUTPUT_ROOT, "saturation_ablation.csv")
    with open(path, "w", newline="") as f:
        fields = [
            "label", "action_range", "lambda_action", "entropy_coef", "obs_clip",
            "avg_secrecy", "avg_sensing", "avg_reward",
            "corr_reward_secrecy", "max_saturation", "reward_improvement",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"Saved {path}")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Random baseline
    base_env = EnvConfig(seed=42, action_range=1.0)
    print("Computing random baseline...")
    random_baseline = compute_random_baseline(base_env, n_episodes=10)
    print(f"  Random: secrecy={random_baseline['avg_secrecy']:.4f}, "
          f"sensing={random_baseline['avg_sensing']:.4f}")

    rand_sec = random_baseline["avg_secrecy"]

    # Sequential tuning configs
    configs = [
        # (label, action_range, lambda_action, entropy_coef, obs_clip)
        ("baseline",      1.0,  0.0,    0.01, 0.0),
        ("ar0.5",         0.5,  0.0,    0.01, 0.0),
        ("ar0.5_la1e-2",  0.5,  1e-2,   0.01, 0.0),
        ("ar0.5_la5e-2",  0.5,  5e-2,   0.01, 0.0),
        ("ar0.5_la5e-2_e0.05", 0.5, 5e-2, 0.05, 0.0),
        ("ar0.25_la1e-2", 0.25, 1e-2,   0.01, 0.0),
        ("ar0.25_la5e-2_e0.05", 0.25, 5e-2, 0.05, 0.0),
        ("ar0.5_la1e-1_e0.10_oc5", 0.5, 1e-1, 0.10, 5.0),
    ]

    all_results = []
    for label, ar, la, ec, oc in configs:
        env_cfg = EnvConfig(seed=42, alpha=0.5, action_range=ar)
        reward_cfg = RewardConfig(
            reward_mode="normalized",
            lambda_constraint=0.1,
            lambda_outage=0.5,
            lambda_secret=1.0,
            R_target=rand_sec,
            lambda_action=la,
            obs_clip=oc,
        )
        result = evaluate_config(env_cfg, reward_cfg, ec, label, n_episodes=100)
        all_results.append(result)

        # Early stopping if all criteria met
        c1 = result["max_saturation"] < 0.7
        c2 = result["corr_reward_secrecy"] > 0.5
        c3 = result["avg_secrecy"] >= rand_sec
        if c1 and c2 and c3:
            print(f"\n>>> All criteria met by {label}! Stopping early.")
            break

    save_csv(all_results)
    plot_results(all_results, random_baseline)
    decision = write_report(all_results, random_baseline)

    print(f"\nAll outputs in {OUTPUT_ROOT}")
    return decision


if __name__ == "__main__":
    decision = main()
    sys.exit(0 if "FIXED" in decision else 1)
