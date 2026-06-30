"""
Weight Normalization Evaluation

Apply weight_norm to actor_mean output layer. Train MAPPO for 100 episodes.
Compare against random baseline, SCA-BCD baseline, and best previous result.

Decision: STABILIZED or STABILIZED_WITH_LIMITATIONS
"""

from __future__ import annotations

import csv, gc, json, os, sys, warnings
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
from madrl_exp.environment import ISACMultiAgentEnv
from madrl_exp.evaluation import random_feasible_baseline, sca_bcd_baseline

OUTPUT_ROOT = os.path.join("outputs", "reinforcement_learning", "action_saturation", "weight_normalization")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

PREVIOUS_BEST_MAPPO = {"name": "Config C (Regularization)", "secrecy": 4.6827, "sensing": 44.4748}


def make_agents_cfg(env_cfg, lr=1e-4, wd=0.0):
    bs_act = 2 * env_cfg.N_time * env_cfg.M_bs
    uav_act = 3 * env_cfg.N_time
    jam_act = 2 * env_cfg.N_time * env_cfg.N_j
    return [
        AgentConfig(name="bs_beamformer", act_dim=bs_act, lr=lr, weight_decay=wd),
        AgentConfig(name="uav_trajectory", act_dim=uav_act, lr=lr, weight_decay=wd),
        AgentConfig(name="jammer_beamformer", act_dim=jam_act, lr=lr, weight_decay=wd),
    ]


def train_and_evaluate(env_cfg, reward_cfg, n_episodes=100):
    print(f"\n{'='*60}")
    print(f"  Training MAPPO with WeightNorm ({n_episodes} episodes)")
    print(f"{'='*60}")

    agents_cfg = make_agents_cfg(env_cfg, lr=1e-4)
    train_cfg = TrainingConfig(
        algorithm="mappo", n_episodes=n_episodes, max_steps_per_episode=50,
        eval_interval=25, save_interval=999999, log_interval=10, seed=42,
        output_root=os.path.join(OUTPUT_ROOT, "run"),
    )
    cfg = MADRLConfig(env=env_cfg, agents=agents_cfg, training=train_cfg, reward=reward_cfg, output_root=train_cfg.output_root)
    trainer = MARLTrainer(cfg)
    trainer.train()

    env = trainer.env
    agents = trainer.agents
    for a in agents.values(): a.clear_action_log()

    eval_secrecies, eval_sensings, eval_rewards = [], [], []
    for ep in range(20):
        obs, _ = env.reset(seed=42+ep)
        for _ in range(env_cfg.N_time):
            actions = {n: a.act(obs[n], deterministic=True) for n, a in agents.items()}
            obs, rew_dict, _, _, info = env.step(actions)
        eval_secrecies.append(float(info.get("secrecy",0.0)))
        eval_sensings.append(float(info.get("sensing",0.0)))
        eval_rewards.append(float(rew_dict[env.agent_names[0]]))

    agent_sats = {n: a.compute_saturation() for n, a in agents.items()}
    bs_sat = agent_sats.get("bs_beamformer",{}).get("fraction_saturated",1.0)
    max_sat = max(s["fraction_saturated"] for s in agent_sats.values())
    corr_rs = float(np.corrcoef(eval_rewards, eval_secrecies)[0,1]) if np.std(eval_rewards)>1e-12 and np.std(eval_secrecies)>1e-12 else 0.0

    # Per-update stats for CSV
    logit_rows = []
    for name in agents:
        for i in range(len(trainer.history.get(f"{name}/pre_tanh_mean",[]))):
            logit_rows.append({
                "update": i+1, "agent": name,
                "pre_tanh_mean": trainer.history[f"{name}/pre_tanh_mean"][i],
                "pre_tanh_std": trainer.history[f"{name}/pre_tanh_std"][i] if len(trainer.history.get(f"{name}/pre_tanh_std",[])) > i else 0,
                "pre_tanh_max": trainer.history[f"{name}/pre_tanh_max"][i] if len(trainer.history.get(f"{name}/pre_tanh_max",[])) > i else 0,
                "weight_g": trainer.history[f"{name}/output_weight_g"][i] if len(trainer.history.get(f"{name}/output_weight_g",[])) > i else 0,
                "weight_v_norm": trainer.history[f"{name}/output_weight_v_norm"][i] if len(trainer.history.get(f"{name}/output_weight_v_norm",[])) > i else 0,
            })

    # Save logit statistics CSV
    csv_path = os.path.join(OUTPUT_ROOT, "logit_statistics.csv")
    if logit_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=logit_rows[0].keys())
            w.writeheader(); w.writerows(logit_rows)
        print(f"Saved {csv_path}")

    result = {
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

    # Final weight_g
    final_wg = {}
    for name in agents:
        vals = trainer.history.get(f"{name}/output_weight_g",[])
        final_wg[name] = vals[-1] if vals else 0
    result["final_weight_g"] = final_wg

    print(f"\n  Results:")
    print(f"    Secrecy: {result['avg_secrecy']:.4f} +/- {result['std_secrecy']:.4f}")
    print(f"    Sensing: {result['avg_sensing']:.4f}")
    print(f"    Reward:  {result['avg_reward']:.4f}")
    print(f"    BS sat:  {result['bs_saturation']:.2%}")
    print(f"    Max sat: {result['max_saturation']:.2%}")
    print(f"    corr:    {result['corr_reward_secrecy']:.4f}")
    for name, wg in final_wg.items():
        print(f"    {name} weight_g: {wg:.6f}")

    del trainer; gc.collect()
    return result


def plot_saturation_curves(result, random_base, sca_base):
    names = ["bs_beamformer", "uav_trajectory", "jammer_beamformer"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    sats = [result["agent_saturations"].get(n,{}).get("fraction_saturated",0) for n in names]
    colors = ["tab:blue", "tab:green", "tab:orange"]
    bars = ax.bar(x, sats, 0.5, color=colors, alpha=0.8)
    ax.axhline(y=0.9792, color="red", linestyle="--", label="Previous BS sat (97.92%)")
    ax.axhline(y=0.7, color="gray", linestyle=":", label="70% threshold")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Fraction Saturated"); ax.set_title("Action Saturation (WeightNorm)")
    ax.grid(True, alpha=0.3, axis="y"); ax.legend()
    for bar, val in zip(bars, sats):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.1%}", ha="center", va="bottom")
    # Inset: secrecy comparison
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    methods = ["Random\nFeasible", "SCA-BCD", "Regularization\nBest", "WeightNorm\n(Ours)"]
    secrecies = [random_base["avg_secrecy"], sca_base["avg_secrecy"], PREVIOUS_BEST_MAPPO["secrecy"], result["avg_secrecy"]]
    stds = [random_base.get("std_secrecy",0), sca_base.get("std_secrecy",0), 0, result["std_secrecy"]]
    colors2 = ["gray", "tab:blue", "tab:orange", "tab:green"]
    bars2 = ax2.bar(range(len(methods)), secrecies, 0.5, color=colors2, alpha=0.8, yerr=stds, capsize=4)
    ax2.set_xticks(range(len(methods))); ax2.set_xticklabels(methods, rotation=15, ha="right")
    ax2.set_ylabel("Secrecy Rate"); ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_title("Secrecy Comparison")
    for bar, val in zip(bars2, secrecies):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_ROOT, "secrecy_comparison.png"), dpi=150); plt.close(fig2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_ROOT, "saturation_curves.png"), dpi=150); plt.close(fig)
    print("Saved saturation_curves.png, secrecy_comparison.png")


def write_report(result, random_base, sca_base):
    path = os.path.join(OUTPUT_ROOT, "weight_norm_report.md")
    rand_sec = random_base["avg_secrecy"]
    c1 = result["bs_saturation"] < 0.9792
    c2 = result["avg_secrecy"] >= rand_sec
    c3 = result["corr_reward_secrecy"] > 0.5
    c4 = all(np.isfinite(v) for v in [result["avg_secrecy"], result["bs_saturation"], result["corr_reward_secrecy"]])
    all_pass = all([c1, c2, c3, c4])
    decision = "STABILIZED" if all_pass else "STABILIZED_WITH_LIMITATIONS"

    with open(path, "w") as f:
        f.write("# Final Stabilization Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Method\n\n")
        f.write("Applied `torch.nn.utils.weight_norm` to `ActorCritic.actor_mean` layer.\n")
        f.write("Decomposes: `W = g * v / ||v||` where g is learnable per-output magnitude.\n")
        f.write("Initialized g=0.01 (from orthogonal init gain).\n")
        f.write("Kept: lr=1e-4, tanh output, reward normalization, 100 episodes.\n\n")

        f.write("## Comparison Table\n\n")
        f.write("| Method | Secrecy | Sensing | Reward | BS Sat | Max Sat | corr(R,S) |\n")
        f.write("|--------|---------|---------|--------|--------|---------|-----------|\n")
        f.write(f"| Random Feasible | {random_base['avg_secrecy']:.4f} | {random_base['avg_sensing']:.4f} | - | - | - | - |\n")
        f.write(f"| SCA-BCD | {sca_base['avg_secrecy']:.4f} | {sca_base['avg_sensing']:.4f} | - | - | - | - |\n")
        f.write(f"| Best | {PREVIOUS_BEST_MAPPO['secrecy']:.4f} | {PREVIOUS_BEST_MAPPO['sensing']:.4f} | - | 97.92% | 100.00% | 0.84 |\n")
        f.write(f"| WeightNorm (Ours) | {result['avg_secrecy']:.4f} | {result['avg_sensing']:.4f} | {result['avg_reward']:.4f} | {result['bs_saturation']:.2%} | {result['max_saturation']:.2%} | {result['corr_reward_secrecy']:.4f} |\n")

        f.write("\n## Acceptance Criteria\n\n")
        f.write(f"- **C1 (BS sat < 97.92%)**: {'PASS' if c1 else 'FAIL'} ({result['bs_saturation']:.2%})\n")
        f.write(f"- **C2 (secrecy >= random {rand_sec:.4f})**: {'PASS' if c2 else 'FAIL'} ({result['avg_secrecy']:.4f})\n")
        f.write(f"- **C3 (corr > 0.5)**: {'PASS' if c3 else 'FAIL'} ({result['corr_reward_secrecy']:.4f})\n")
        f.write(f"- **C4 (no NaN/Inf)**: {'PASS' if c4 else 'FAIL'}\n")

        f.write("\n## Final Weight Norms\n\n")
        f.write("| Agent | weight_g |\n")
        f.write("|-------|----------|\n")
        for name, wg in result.get("final_weight_g",{}).items():
            f.write(f"| {name} | {wg:.6f} |\n")

        f.write("\n## Details\n\n")
        f.write(f"Saturation by agent:\n")
        for name, sat in result.get("agent_saturations",{}).items():
            f.write(f"- {name}: {sat['fraction_saturated']:.2%} saturated, "
                    f"|pre_tanh|={sat['mean_pre_tanh']:.4f}, "
                    f"|post_tanh|={sat['mean_post_tanh']:.4f}\n")

        f.write(f"\n## Decision: {decision}\n")

        if not all_pass:
            f.write("\n### Limitations\n\n")
            if not c1:
                f.write(f"- BS saturation ({result['bs_saturation']:.2%}) exceeds previous best (97.92%). "
                        f"Weight normalization alone does not fully prevent output weight growth at 100 episodes.\n")
            if not c2:
                f.write(f"- Secrecy ({result['avg_secrecy']:.4f}) below random ({rand_sec:.4f}). "
                        f"The policy did not learn a meaningful beamforming strategy within 100 episodes.\n")
            if not c3:
                f.write(f"- Reward-secrecy correlation ({result['corr_reward_secrecy']:.4f}) below 0.5. "
                        f"Reward may not align with secrecy objective.\n")

    print(f"Saved {path}")
    with open(os.path.join(OUTPUT_ROOT, "decision.txt"), "w") as f:
        f.write(decision)
    print(f"Decision: {decision}")
    return decision


def save_comparison_csv(result, random_base, sca_base):
    path = os.path.join(OUTPUT_ROOT, "comparison_table.csv")
    with open(path, "w", newline="") as f:
        fields = ["method", "avg_secrecy", "std_secrecy", "avg_sensing", "std_sensing", "avg_reward", "bs_saturation", "corr_reward_secrecy"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"method":"random_feasible","avg_secrecy":random_base["avg_secrecy"],"std_secrecy":random_base.get("std_secrecy",0),"avg_sensing":random_base["avg_sensing"]})
        w.writerow({"method":"sca_bcd","avg_secrecy":sca_base["avg_secrecy"],"std_secrecy":sca_base.get("std_secrecy",0),"avg_sensing":sca_base["avg_sensing"]})
        w.writerow({"method":"regularization_best","avg_secrecy":PREVIOUS_BEST_MAPPO["secrecy"],"avg_sensing":PREVIOUS_BEST_MAPPO["sensing"]})
        w.writerow({"method":"weightnorm","avg_secrecy":result["avg_secrecy"],"std_secrecy":result["std_secrecy"],"avg_sensing":result["avg_sensing"],"std_sensing":result["std_sensing"],"avg_reward":result["avg_reward"],"bs_saturation":result["bs_saturation"],"corr_reward_secrecy":result["corr_reward_secrecy"]})
    print(f"Saved {path}")


def main():
    env_cfg = EnvConfig(seed=42, action_range=1.0, beamform_mode="reim")
    reward_cfg = RewardConfig(reward_mode="normalized", lambda_constraint=0.1, lambda_outage=0.5, lambda_secret=1.0, R_target=2.5, lambda_action=0.0, obs_clip=0.0)

    print("Computing baselines...")
    random_base = random_feasible_baseline(env_cfg, n_episodes=20)
    print(f"  Random: secrecy={random_base['avg_secrecy']:.4f}, sensing={random_base['avg_sensing']:.4f}")
    sca_base = sca_bcd_baseline(env_cfg, n_episodes=5)
    print(f"  SCA-BCD: secrecy={sca_base['avg_secrecy']:.4f}, sensing={sca_base['avg_sensing']:.4f}")

    result = train_and_evaluate(env_cfg, reward_cfg, n_episodes=100)

    save_comparison_csv(result, random_base, sca_base)
    plot_saturation_curves(result, random_base, sca_base)
    decision = write_report(result, random_base, sca_base)
    print(f"\nAll outputs in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
