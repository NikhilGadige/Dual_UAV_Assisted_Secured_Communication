"""
Actor Stability Analysis

Reduce tanh saturation by:
  1. Orthogonal init with actor_out gain=0.01
  2. Logit temperature (T=1,2,5,10)
  3. Logit clipping (clip to [-10,10])
  4. Reduced actor learning rate (3e-4, 1e-4, 5e-5)

Acceptance: BS saturation < 80% OR pre_tanh_mean < 5 OR pre_tanh_max < 10

Outputs: outputs/reinforcement_learning/actor_stability/
"""

from __future__ import annotations

import csv
import os
import sys
import gc
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

OUTPUT_ROOT = os.path.join("outputs", "reinforcement_learning", "actor_stability")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


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
                    temperature: float, clip_logits: bool, lr: float,
                    label: str, n_episodes: int = 100) -> dict:
    """Train MAPPO with given actor stabilization settings."""
    print(f"\n{'=' * 50}")
    print(f"  Config: {label}")
    print(f"  temperature={temperature}, clip_logits={clip_logits}, lr={lr}")
    print(f"{'=' * 50}")

    bs_act = 2 * env_cfg.N_time * env_cfg.M_bs
    agents_cfg = [
        AgentConfig(name="bs_beamformer", act_dim=bs_act,
                    entropy_coef=0.01, lr=lr,
                    temperature=temperature, clip_logits=clip_logits),
        AgentConfig(name="uav_trajectory",
                    act_dim=3 * env_cfg.N_time, entropy_coef=0.01,
                    lr=lr, temperature=temperature, clip_logits=clip_logits),
        AgentConfig(name="jammer_beamformer",
                    act_dim=2 * env_cfg.N_time * env_cfg.N_j, entropy_coef=0.01,
                    lr=lr, temperature=temperature, clip_logits=clip_logits),
    ]

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

    # 1) Evaluate policy performance (10 episodes)
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

    r_vec = [r["reward"] for r in eval_rows]
    s_vec = [r["secrecy"] for r in eval_rows]
    if np.std(r_vec) > 1e-12 and np.std(s_vec) > 1e-12:
        corr_rs = float(np.corrcoef(r_vec, s_vec)[0, 1])
    else:
        corr_rs = 0.0

    # 2) Action saturation
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

    bs_sat = agent_sats.get("bs_beamformer", {}).get("fraction_saturated", 1.0)
    max_sat = max(s["fraction_saturated"] for s in agent_sats.values())

    # 3) Pre-tanh stats from last update (logged in trainer.history)
    pre_tanh_mean = 0.0
    pre_tanh_max = 0.0
    names_list = list(agents.keys())
    if names_list:
        key_mean = f"{names_list[0]}/pre_tanh_mean"
        key_max = f"{names_list[0]}/pre_tanh_max"
        vals_mean = trainer.history.get(key_mean, [])
        vals_max = trainer.history.get(key_max, [])
        if vals_mean:
            pre_tanh_mean = float(np.mean(vals_mean))
        if vals_max:
            pre_tanh_max = float(np.max(vals_max))

    # 4) Training reward trend
    train_rewards = []
    if names_list:
        key = f"{names_list[0]}/reward"
        train_rewards = [float(x) for x in trainer.history.get(key, [])]
    first_10 = float(np.mean(train_rewards[:10])) if len(train_rewards) >= 10 else 0.0
    last_10 = float(np.mean(train_rewards[-10:])) if len(train_rewards) >= 10 else 0.0
    reward_improvement = last_10 - first_10

    result = {
        "label": label,
        "temperature": temperature,
        "clip_logits": clip_logits,
        "lr": lr,
        "avg_secrecy": avg_secrecy,
        "avg_sensing": avg_sensing,
        "avg_reward": avg_reward,
        "corr_reward_secrecy": corr_rs,
        "bs_saturation": bs_sat,
        "max_saturation": max_sat,
        "pre_tanh_mean": pre_tanh_mean,
        "pre_tanh_max": pre_tanh_max,
        "reward_improvement": reward_improvement,
        "agent_saturations": agent_sats,
        "history": trainer.history,
    }

    print(f"  secrecy={avg_secrecy:.4f}, reward={avg_reward:.4f}, "
          f"corr={corr_rs:.4f}, bs_sat={bs_sat:.2%}, "
          f"pre_tanh_mean={pre_tanh_mean:.4f}, pre_tanh_max={pre_tanh_max:.4f}")

    del trainer
    gc.collect()

    return result


def plot_results(all_results: list[dict], random_baseline: dict):
    names = [r["label"] for r in all_results]
    secrecies = [r["avg_secrecy"] for r in all_results]
    rewards = [r["avg_reward"] for r in all_results]
    bs_sats = [r["bs_saturation"] for r in all_results]
    pt_means = [r["pre_tanh_mean"] for r in all_results]
    pt_maxes = [r["pre_tanh_max"] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(names))

    # Secrecy
    ax = axes[0, 0]
    rand_sec = random_baseline["avg_secrecy"]
    colors = ["tab:green" if s >= rand_sec else "tab:red" for s in secrecies]
    ax.bar(x, secrecies, 0.5, color=colors, alpha=0.8)
    ax.axhline(y=rand_sec, color="gray", linestyle="--", label=f"Random: {rand_sec:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Secrecy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Secrecy Rate")

    # BS saturation
    ax = axes[0, 1]
    colors_sat = ["tab:green" if s < 0.8 else "tab:red" for s in bs_sats]
    ax.bar(x, bs_sats, 0.5, color=colors_sat, alpha=0.8)
    ax.axhline(y=0.8, color="red", linestyle=":", label="BS threshold (80%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("BS Saturation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("BS Action Saturation")

    # Pre-tanh mean
    ax = axes[1, 0]
    colors_pt = ["tab:green" if m < 5 else "tab:red" for m in pt_means]
    ax.bar(x, pt_means, 0.5, color=colors_pt, alpha=0.8)
    ax.axhline(y=5.0, color="green", linestyle=":", label="Threshold (5.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Mean |pre_tanh|")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Pre-Tanh Mean (per update)")

    # Pre-tanh max
    ax = axes[1, 1]
    colors_ptm = ["tab:green" if m < 10 else "tab:red" for m in pt_maxes]
    ax.bar(x, pt_maxes, 0.5, color=colors_ptm, alpha=0.8)
    ax.axhline(y=10.0, color="green", linestyle=":", label="Threshold (10.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Max |pre_tanh|")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Pre-Tanh Max (per update)")

    fig.suptitle("Actor Stabilization", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_ROOT, "logit_histograms.png"), dpi=150)
    plt.close(fig)
    print(f"Saved logit_histograms.png")


def write_report(all_results: list[dict], random_baseline: dict) -> str:
    path = os.path.join(OUTPUT_ROOT, "actor_stability_report.md")
    with open(path, "w") as f:
        f.write("# Actor Stabilization Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Random baseline**: secrecy={random_baseline['avg_secrecy']:.4f}\n\n")

        f.write("## Configuration Results\n\n")
        f.write("| Config | T | Clip | LR | Secrecy | Reward | corr(R,S) | BS Sat | "
                "PT Mean | PT Max |\n")
        f.write("|--------|---|------|----|---------|--------|-----------|--------|"
                "---------|--------|\n")
        for r in all_results:
            f.write(f"| {r['label']} | {r['temperature']} | {r['clip_logits']} | "
                    f"{r['lr']} | {r['avg_secrecy']:.4f} | {r['avg_reward']:.4f} | "
                    f"{r['corr_reward_secrecy']:.4f} | {r['bs_saturation']:.2%} | "
                    f"{r['pre_tanh_mean']:.4f} | {r['pre_tanh_max']:.4f} |\n")

        f.write("\n## Per-Agent Saturation Details\n\n")
        for r in all_results:
            f.write(f"### {r['label']}\n")
            f.write("| Agent | Saturation | Pre-Tanh | Post-Tanh |\n")
            f.write("|-------|------------|----------|-----------|\n")
            for aname, sat in r["agent_saturations"].items():
                f.write(f"| {aname} | {sat['fraction_saturated']:.2%} | "
                        f"{sat['mean_pre_tanh']:.4f} | {sat['mean_post_tanh']:.4f} |\n")

        f.write("\n## Acceptance Criteria\n\n")
        f.write("Accept if ANY of: BS saturation < 80% OR pre_tanh_mean < 5 OR pre_tanh_max < 10\n\n")

        c1 = any(r["bs_saturation"] < 0.8 for r in all_results)
        c2 = any(r["pre_tanh_mean"] < 5.0 for r in all_results)
        c3 = any(r["pre_tanh_max"] < 10.0 for r in all_results)

        f.write(f"- **C1 (BS sat < 80%)**: {'**PASS**' if c1 else '**FAIL**'}\n")
        for r in all_results:
            s = "PASS" if r["bs_saturation"] < 0.8 else "FAIL"
            f.write(f"  - {r['label']}: {r['bs_saturation']:.2%} ({s})\n")

        f.write(f"- **C2 (pre_tanh_mean < 5)**: {'**PASS**' if c2 else '**FAIL**'}\n")
        for r in all_results:
            s = "PASS" if r["pre_tanh_mean"] < 5.0 else "FAIL"
            f.write(f"  - {r['label']}: {r['pre_tanh_mean']:.4f} ({s})\n")

        f.write(f"- **C3 (pre_tanh_max < 10)**: {'**PASS**' if c3 else '**FAIL**'}\n")
        for r in all_results:
            s = "PASS" if r["pre_tanh_max"] < 10.0 else "FAIL"
            f.write(f"  - {r['label']}: {r['pre_tanh_max']:.4f} ({s})\n")

        all_pass = c1 or c2 or c3
        decision = "ACTOR_STABILIZED" if all_pass else "ACTOR_STILL_SATURATED"
        f.write(f"\n## Decision: {decision}\n")

    print(f"Saved {path}")
    with open(os.path.join(OUTPUT_ROOT, "decision.txt"), "w") as f:
        f.write(decision)
    print(f"Decision: {decision}")
    return decision


def save_csv(all_results: list[dict]):
    path = os.path.join(OUTPUT_ROOT, "actor_stability.csv")
    with open(path, "w", newline="") as f:
        fields = [
            "label", "temperature", "clip_logits", "lr",
            "avg_secrecy", "avg_sensing", "avg_reward",
            "corr_reward_secrecy", "bs_saturation", "max_saturation",
            "pre_tanh_mean", "pre_tanh_max", "reward_improvement",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"Saved {path}")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    base_env = EnvConfig(seed=42, action_range=1.0, beamform_mode="reim")
    print("Computing random baseline...")
    random_baseline = compute_random_baseline(base_env, n_episodes=10)
    print(f"  Random: secrecy={random_baseline['avg_secrecy']:.4f}, "
          f"sensing={random_baseline['avg_sensing']:.4f}")

    rand_sec = random_baseline["avg_secrecy"]

    configs = [
        # (label, temperature, clip_logits, lr)
        # Baseline with new orthogonal init
        ("baseline",          1.0, False, 3e-4),
        # Temperature sweep
        ("T2",                2.0, False, 3e-4),
        ("T5",                5.0, False, 3e-4),
        ("T10",              10.0, False, 3e-4),
        # Logit clipping
        ("clip",              1.0, True,  3e-4),
        # Reduced learning rate
        ("lr1e-4",            1.0, False, 1e-4),
        ("lr5e-5",            1.0, False, 5e-5),
    ]

    env_cfg = EnvConfig(seed=42, action_range=1.0, beamform_mode="reim")

    all_results = []
    for label, temperature, clip_logits, lr in configs:
        reward_cfg = RewardConfig(
            reward_mode="normalized",
            lambda_constraint=0.1,
            lambda_outage=0.5,
            lambda_secret=1.0,
            R_target=rand_sec,
            lambda_action=0.0,
            obs_clip=0.0,
        )
        result = evaluate_config(
            env_cfg, reward_cfg, temperature, clip_logits, lr,
            label, n_episodes=100,
        )
        all_results.append(result)

        # Early stopping
        c_acc = (result["bs_saturation"] < 0.8 or
                 result["pre_tanh_mean"] < 5.0 or
                 result["pre_tanh_max"] < 10.0)
        if c_acc:
            print(f"\n>>> Acceptance criteria met by {label}! Stopping early.")
            break

    save_csv(all_results)
    plot_results(all_results, random_baseline)
    decision = write_report(all_results, random_baseline)

    print(f"\nAll outputs in {OUTPUT_ROOT}")
    return decision


if __name__ == "__main__":
    decision = main()
    sys.exit(0 if "STABILIZED" in decision else 1)
