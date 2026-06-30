"""
Beamforming Parameterization Study

Investigate whether BS action saturation is caused by the current
beamforming parameterization (Re/Im weights).

Three variants:
  A (reim):         current — Re(w), Im(w) with power clipping
  B (direction):    u only — w = sqrt(P_bs_max) * u / ||u||
  C (power_direction):  p + u — w = sqrt(p * P_bs_max) * u / ||u||

Outputs: outputs/reinforcement_learning/ablation_studies/
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

OUTPUT_ROOT = os.path.join("outputs", "reinforcement_learning", "ablation_studies", "parameterization")
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
                    label: str, n_episodes: int = 100) -> dict:
    """Train MAPPO with given config and evaluate BS saturation."""
    print(f"\n{'=' * 50}")
    print(f"  Config: {label}")
    print(f"  beamform_mode={env_cfg.beamform_mode}, action_range={env_cfg.action_range}")
    print(f"{'=' * 50}")

    # Agent configs — BS dim depends on beamform_mode
    bs_act = 2 * env_cfg.N_time * env_cfg.M_bs
    if env_cfg.beamform_mode == "power_direction":
        bs_act += env_cfg.N_time
    agents_cfg = [
        AgentConfig(name="bs_beamformer", act_dim=bs_act,
                    entropy_coef=0.01),
        AgentConfig(name="uav_trajectory",
                    act_dim=3 * env_cfg.N_time, entropy_coef=0.01),
        AgentConfig(name="jammer_beamformer",
                    act_dim=2 * env_cfg.N_time * env_cfg.N_j, entropy_coef=0.01),
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
        env.clear_bs_projection_log()
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

    # 2) Agent action saturation (evaluate with action logging)
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

    # 3) BS projection stats (separate eval to capture clean projection log)
    env.clear_bs_projection_log()
    proj_records = []
    for ep in range(10):
        obs, _ = env.reset(seed=42 + ep)
        for step in range(env_cfg.N_time):
            actions = {n: agents[n].act(obs[n], deterministic=True) for n in agents}
            obs, _, _, _, _ = env.step(actions)
        proj_records.extend(env.get_bs_projection_stats())
        env.clear_bs_projection_log()

    if proj_records:
        pre_powers = [r["pre_projection_power"] for r in proj_records]
        post_powers = [r["post_projection_power"] for r in proj_records]
        distances = [r["projection_distance"] for r in proj_records]
        bs_proj_stats = {
            "mean_pre_projection_power": float(np.mean(pre_powers)),
            "mean_post_projection_power": float(np.mean(post_powers)),
            "mean_projection_distance": float(np.mean(distances)),
            "max_projection_distance": float(np.max(distances)),
            "fraction_clipped": float(np.mean(
                [p != q for p, q in zip(pre_powers, post_powers)]
            )),
        }
    else:
        bs_proj_stats = {}

    print(f"  BS proj: pre_power={bs_proj_stats.get('mean_pre_projection_power', 0):.4f}, "
          f"post_power={bs_proj_stats.get('mean_post_projection_power', 0):.4f}, "
          f"clip_frac={bs_proj_stats.get('fraction_clipped', 0):.2%}")

    # Save projection stats to CSV
    proj_path = os.path.join(OUTPUT_ROOT, f"bs_projection_stats_{label}.csv")
    with open(proj_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slot", "pre_projection_power",
                                          "post_projection_power",
                                          "projection_distance"])
        w.writeheader()
        w.writerows(proj_records)
    print(f"  Saved {proj_path}")

    # 4) Training reward trend
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
        "beamform_mode": env_cfg.beamform_mode,
        "action_range": env_cfg.action_range,
        "avg_secrecy": avg_secrecy,
        "avg_sensing": avg_sensing,
        "avg_reward": avg_reward,
        "corr_reward_secrecy": corr_rs,
        "bs_saturation": bs_sat,
        "max_saturation": max_sat,
        "reward_improvement": reward_improvement,
        "agent_saturations": agent_sats,
        "bs_projection_stats": bs_proj_stats,
    }

    print(f"  secrecy={avg_secrecy:.4f}, reward={avg_reward:.4f}, "
          f"corr={corr_rs:.4f}, bs_sat={bs_sat:.2%}, max_sat={max_sat:.2%}")

    del trainer
    gc.collect()

    return result


def plot_results(all_results: list[dict], random_baseline: dict):
    names = [r["label"] for r in all_results]
    secrecies = [r["avg_secrecy"] for r in all_results]
    rewards = [r["avg_reward"] for r in all_results]
    corrs = [r["corr_reward_secrecy"] for r in all_results]
    bs_sats = [r["bs_saturation"] for r in all_results]
    max_sats = [r["max_saturation"] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    x = np.arange(len(names))

    # Secrecy
    ax = axes[0, 0]
    rand_sec = random_baseline["avg_secrecy"]
    colors = ["tab:green" if s >= rand_sec else "tab:red" for s in secrecies]
    ax.bar(x, secrecies, 0.5, color=colors, alpha=0.8)
    ax.axhline(y=rand_sec, color="gray", linestyle="--", label=f"Random: {rand_sec:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Secrecy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Secrecy Rate")

    # Reward
    ax = axes[0, 1]
    ax.bar(x, rewards, 0.5, color="tab:orange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Total Reward")

    # Correlation
    ax = axes[1, 0]
    colors_corr = ["tab:green" if c > 0.5 else "tab:red" for c in corrs]
    ax.bar(x, corrs, 0.5, color=colors_corr, alpha=0.8)
    ax.axhline(y=0.5, color="purple", linestyle=":", label="C3 threshold (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("corr(reward, secrecy)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Reward-Secrecy Correlation")

    # BS Saturation
    ax = axes[1, 1]
    colors_sat = ["tab:green" if s < 0.7 else "tab:red" for s in max_sats]
    ax.bar(x, max_sats, 0.5, color=colors_sat, alpha=0.8, label="Max")
    ax.bar(x, bs_sats, 0.3, color="tab:blue", alpha=0.6, label="BS only")
    ax.axhline(y=0.7, color="red", linestyle=":", label="C1 threshold (70%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Saturation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Action Saturation")

    fig.suptitle("Beamforming Reparameterization", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_ROOT, "beamforming_saturation.png"), dpi=150)
    plt.close(fig)
    print(f"Saved beamforming_saturation.png")


def write_report(all_results: list[dict], random_baseline: dict) -> str:
    path = os.path.join(OUTPUT_ROOT, "parameterization_report.md")
    with open(path, "w") as f:
        f.write("# Beamforming Reparameterization Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Random baseline**: secrecy={random_baseline['avg_secrecy']:.4f}, "
                f"sensing={random_baseline['avg_sensing']:.4f}\n\n")

        # Summary table
        f.write("## Configuration Results\n\n")
        f.write("| Config | Mode | Secrecy | Reward | corr(R,S) | BS Sat | Max Sat | Rew Impr |\n")
        f.write("|--------|------|---------|--------|-----------|--------|---------|----------|\n")
        for r in all_results:
            f.write(f"| {r['label']} | {r['beamform_mode']} | "
                    f"{r['avg_secrecy']:.4f} | {r['avg_reward']:.4f} | "
                    f"{r['corr_reward_secrecy']:.4f} | {r['bs_saturation']:.2%} | "
                    f"{r['max_saturation']:.2%} | {r['reward_improvement']:+.4f} |\n")

        # BS projection stats
        f.write("\n## BS Projection Statistics\n\n")
        f.write("| Config | Mode | Pre Power | Post Power | Proj Dist | Clip Frac |\n")
        f.write("|--------|------|-----------|------------|-----------|-----------|\n")
        for r in all_results:
            ps = r.get("bs_projection_stats", {})
            f.write(f"| {r['label']} | {r['beamform_mode']} | "
                    f"{ps.get('mean_pre_projection_power', 0):.4f} | "
                    f"{ps.get('mean_post_projection_power', 0):.4f} | "
                    f"{ps.get('mean_projection_distance', 0):.4f} | "
                    f"{ps.get('fraction_clipped', 0):.2%} |\n")

        # Per-agent saturation
        f.write("\n## Per-Agent Saturation Details\n\n")
        for r in all_results:
            f.write(f"### {r['label']}\n")
            f.write("| Agent | Saturation | Pre-Tanh | Post-Tanh |\n")
            f.write("|-------|------------|----------|-----------|\n")
            for aname, sat in r["agent_saturations"].items():
                f.write(f"| {aname} | {sat['fraction_saturated']:.2%} | "
                        f"{sat['mean_pre_tanh']:.4f} | {sat['mean_post_tanh']:.4f} |\n")

        # Acceptance criteria
        f.write("\n## Acceptance Criteria\n\n")

        rand_sec = random_baseline["avg_secrecy"]

        # C1: BS saturation < 70%
        c1 = any(r["bs_saturation"] < 0.7 for r in all_results)
        best_sat = min(r["bs_saturation"] for r in all_results)
        best_label_sat = min(all_results, key=lambda x: x["bs_saturation"])["label"]
        f.write(f"- **C1 (BS saturation < 70%)**: {'**PASS**' if c1 else '**FAIL**'}\n")
        f.write(f"  - Best: {best_label_sat} ({best_sat:.2%})\n")
        for r in all_results:
            s = "PASS" if r["bs_saturation"] < 0.7 else "FAIL"
            f.write(f"  - {r['label']}: {r['bs_saturation']:.2%} ({s})\n")

        # C2: trained secrecy >= random
        c2 = any(r["avg_secrecy"] >= rand_sec for r in all_results)
        best_sec = max(r["avg_secrecy"] for r in all_results)
        best_label_sec = max(all_results, key=lambda x: x["avg_secrecy"])["label"]
        f.write(f"- **C2 (trained secrecy >= random)**: {'**PASS**' if c2 else '**FAIL**'}\n")
        f.write(f"  - Random: {rand_sec:.4f}, Best: {best_label_sec} ({best_sec:.4f})\n")
        for r in all_results:
            s = "PASS" if r["avg_secrecy"] >= rand_sec else "FAIL"
            f.write(f"  - {r['label']}: {r['avg_secrecy']:.4f} ({s})\n")

        # C3: corr(reward, secrecy) > 0.5
        c3 = any(r["corr_reward_secrecy"] > 0.5 for r in all_results)
        best_corr = max(r["corr_reward_secrecy"] for r in all_results)
        best_label_corr = max(all_results, key=lambda x: x["corr_reward_secrecy"])["label"]
        f.write(f"- **C3 (corr reward-secrecy > 0.5)**: {'**PASS**' if c3 else '**FAIL**'}\n")
        f.write(f"  - Best: {best_label_corr} ({best_corr:.4f})\n")
        for r in all_results:
            s = "PASS" if r["corr_reward_secrecy"] > 0.5 else "FAIL"
            f.write(f"  - {r['label']}: {r['corr_reward_secrecy']:.4f} ({s})\n")

        all_pass = all([c1, c2, c3])
        decision = "BEAMFORMING_PARAMETERIZATION_FIXED" if all_pass else "BEAMFORMING_PARAMETERIZATION_STILL_BROKEN"
        f.write(f"\n## Decision: {decision}\n")

    print(f"Saved {path}")
    with open(os.path.join(OUTPUT_ROOT, "decision.txt"), "w") as f:
        f.write(decision)
    print(f"Decision: {decision}")
    return decision


def save_csv(all_results: list[dict]):
    path = os.path.join(OUTPUT_ROOT, "parameterization_comparison.csv")
    with open(path, "w", newline="") as f:
        fields = [
            "label", "beamform_mode", "action_range",
            "avg_secrecy", "avg_sensing", "avg_reward",
            "corr_reward_secrecy", "bs_saturation", "max_saturation",
            "reward_improvement",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"Saved {path}")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Random baseline uses default reim mode
    base_env = EnvConfig(seed=42, action_range=1.0, beamform_mode="reim")
    print("Computing random baseline...")
    random_baseline = compute_random_baseline(base_env, n_episodes=10)
    print(f"  Random: secrecy={random_baseline['avg_secrecy']:.4f}, "
          f"sensing={random_baseline['avg_sensing']:.4f}")

    rand_sec = random_baseline["avg_secrecy"]

    configs = [
        ("reim",       EnvConfig(seed=42, action_range=1.0, beamform_mode="reim")),
        ("direction",  EnvConfig(seed=42, action_range=1.0, beamform_mode="direction")),
        ("power_dir",  EnvConfig(seed=42, action_range=1.0, beamform_mode="power_direction")),
    ]

    all_results = []
    for label, env_cfg in configs:
        reward_cfg = RewardConfig(
            reward_mode="normalized",
            lambda_constraint=0.1,
            lambda_outage=0.5,
            lambda_secret=1.0,
            R_target=rand_sec,
            lambda_action=0.0,
            obs_clip=0.0,
        )
        result = evaluate_config(env_cfg, reward_cfg, label, n_episodes=100)
        all_results.append(result)

    save_csv(all_results)
    plot_results(all_results, random_baseline)
    decision = write_report(all_results, random_baseline)

    print(f"\nAll outputs in {OUTPUT_ROOT}")
    return decision


if __name__ == "__main__":
    decision = main()
    sys.exit(0 if "FIXED" in decision else 1)
