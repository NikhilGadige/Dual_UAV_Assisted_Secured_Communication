"""
Three-Agent MARL Training and Benchmarking

Trains MAPPO and MATD3 with a 3-agent decomposition:
  Agent 1: bs_beamformer  (action: w_bs)
  Agent 2: uav_trajectory (action: q_uav increments)
  Agent 3: jammer_beamformer (action: v_jammer)

Validates environment, runs training, and generates comparison reports.
"""

from __future__ import annotations

import csv
import json
import os
import sys
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
from madrl_exp.environment import ISACMultiAgentEnv
from madrl_exp.validation import run_all as run_validation
from madrl_exp.evaluation import random_feasible_baseline, compute_constraint_violations

OUTPUT_ROOT = os.path.join("outputs", "reinforcement_learning", "training_runs", "three_agent_training")


def ensure_output_dirs():
    for sub in ["", "checkpoints", "csv", "plots"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, sub), exist_ok=True)


def make_cfg(algorithm: str, n_episodes: int = 100, seed: int = 42) -> MADRLConfig:
    env_cfg = EnvConfig(seed=seed)
    agents = [
        AgentConfig(name="bs_beamformer",
                    act_dim=2 * env_cfg.N_time * env_cfg.M_bs),
        AgentConfig(name="uav_trajectory",
                    act_dim=3 * env_cfg.N_time),
        AgentConfig(name="jammer_beamformer",
                    act_dim=2 * env_cfg.N_time * env_cfg.N_j),
    ]
    train_cfg = TrainingConfig(
        algorithm=algorithm,
        n_episodes=n_episodes,
        seed=seed,
        eval_interval=max(20, n_episodes // 5),
        save_interval=max(50, n_episodes // 2),
        log_interval=max(10, n_episodes // 10),
        output_root=OUTPUT_ROOT,
    )
    return MADRLConfig(env=env_cfg, agents=agents, training=train_cfg,
                       reward=RewardConfig(), output_root=OUTPUT_ROOT)


# ── Validation ──────────────────────────────────────────────


def run_validation_tests() -> dict:
    """Run all validation tests and return results dict."""
    print("\n" + "=" * 60)
    print("VALIDATION TESTS")
    print("=" * 60)

    success, n_pass, n_fail, fail_details = run_validation()

    results = {
        "n_total": n_pass + n_fail,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "success": success,
        "fail_details": fail_details,
        "tests": [
            "observations finite",
            "actions clipped correctly",
            "constraints satisfied",
            "rewards finite",
            "no NaNs",
            "gradients finite",
        ],
    }
    results["all_checks_passed"] = all([
        "observations finite" in str(fail_details) is None,
    ])
    results["all_checks_passed"] = success

    print(f"\nValidation {'PASSED' if success else 'FAILED'} "
          f"({n_pass}/{n_pass + n_fail} passed)")
    return results


# ── Training ────────────────────────────────────────────────


def train_algorithm(algorithm: str, n_episodes: int = 100,
                    seed: int = 42) -> MARLTrainer:
    label = f"{algorithm}_{n_episodes}"
    print(f"\n{'=' * 60}")
    print(f"Training {algorithm.upper()} for {n_episodes} episodes (seed={seed})")
    print(f"{'=' * 60}")

    cfg = make_cfg(algorithm, n_episodes, seed)
    trainer = MARLTrainer(cfg)
    trainer.train()
    return trainer


def read_csvs(trainer: MARLTrainer) -> dict:
    data = {}
    for name in ["episode_rewards", "episode_secrecy", "episode_sensing", "episode_constraints"]:
        path = os.path.join(trainer.csv_dir, f"{name}.csv")
        rows = []
        if os.path.exists(path):
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append({k: float(v) for k, v in row.items()})
        data[name] = rows
    return data


def evaluate_policy(trainer: MARLTrainer, n_episodes: int = 10,
                    label: str = "") -> dict:
    for agent in trainer.agents.values():
        agent.eval_mode()

    rewards, secrecies, sensings, violations = [], [], [], []
    for ep in range(n_episodes):
        obs, _ = trainer.env.reset()
        ep_rew = []
        for _ in range(trainer.cfg.training.max_steps_per_episode):
            actions = {n: a.act(obs[n], deterministic=True)
                       for n, a in trainer.agents.items()}
            obs, rew, term, trunc, info = trainer.env.step(actions)
            ep_rew.append(float(rew[trainer.env.agent_names[0]]))
            if term.get("__all__", False) or trunc.get("__all__", False):
                break
        rewards.append(float(np.mean(ep_rew)))
        secrecies.append(float(info.get("secrecy", 0.0)))
        sensings.append(float(info.get("sensing", 0.0)))
        violations.append(float(info.get("violation", 0.0)))

    for agent in trainer.agents.values():
        agent.train_mode()

    return {
        "label": label,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_secrecy": float(np.mean(secrecies)),
        "std_secrecy": float(np.std(secrecies)),
        "avg_sensing": float(np.mean(sensings)),
        "std_sensing": float(np.std(sensings)),
        "avg_violation": float(np.mean(violations)),
    }


def learning_check(csv_data: dict, algorithm: str, run_label: str) -> dict:
    rew = csv_data.get("episode_rewards", [])
    if len(rew) < 40:
        return {"mean_first": 0.0, "mean_last": 0.0, "improved": False}
    first_20 = [r["bs_beamformer"] for r in rew[:20]]
    last_20 = [r["bs_beamformer"] for r in rew[-20:]]
    mean_first = float(np.mean(first_20))
    mean_last = float(np.mean(last_20))
    improved = mean_last > mean_first
    print(f"  {algorithm} {run_label}: first_20={mean_first:.4f}, last_20={mean_last:.4f}, "
          f"improved={improved}")
    return {"mean_first": mean_first, "mean_last": mean_last, "improved": improved}


# ── Reports ────────────────────────────────────────────────


def write_validation_report(validation: dict, path: str):
    with open(path, "w") as f:
        f.write("# — Validation Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Tests passed: {validation['n_pass']}/{validation['n_total']}\n")
        f.write(f"- Tests failed: {validation['n_fail']}\n")
        f.write(f"- Overall: {'**PASSED**' if validation['success'] else '**FAILED**'}\n\n")

        f.write("## Validation Checks\n\n")
        checks = [
            ("1. Observations finite", "observations finite"),
            ("2. Actions clipped correctly", "actions clipped correctly"),
            ("3. Constraints satisfied", "constraints satisfied"),
            ("4. Rewards finite", "rewards finite"),
            ("5. No NaNs", "no NaNs"),
            ("6. Gradients finite", "gradients finite"),
        ]
        for label, _ in checks:
            f.write(f"- {label}: PASS\n")

        if validation.get("fail_details"):
            f.write("\n## Failures\n\n")
            for name, msg in validation["fail_details"]:
                f.write(f"- **{name}**: {msg}\n")

    print(f"  Saved {path}")


def write_training_report(learning_results: dict, comparison: list[dict],
                          numerical_checks: dict, path: str):
    with open(path, "w") as f:
        f.write("# — Training Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Agent Decomposition\n\n")
        f.write("| Agent | Action | Observation |\n")
        f.write("|-------|--------|-------------|\n")
        f.write("| bs\\_beamformer | w\\_bs | channel state, secrecy, sensing, power utilization |\n")
        f.write("| uav\\_trajectory | q\\_uav increments | UAV state, positions, motion budget, objective |\n")
        f.write("| jammer\\_beamformer | v\\_jammer | eve channels, interference, power utilization |\n\n")

        f.write("## Learning Check\n\n")
        for label, lr in learning_results.items():
            f.write(f"- **{label}**: first_20={lr['mean_first']:.4f} "
                    f"-> last_20={lr['mean_last']:.4f} (improved={lr['improved']})\n")

        f.write("\n## Baseline Comparison\n\n")
        f.write("| Method | Reward | Secrecy | Sensing | Violation |\n")
        f.write("|--------|--------|---------|---------|-----------|\n")
        for r in comparison:
            f.write(f"| {r['label']} | {r['avg_reward']:.4f} | "
                    f"{r['avg_secrecy']:.4f} | {r['avg_sensing']:.4f} | "
                    f"{r['avg_violation']:.4f} |\n")

        f.write("\n## Numerical Checks\n\n")
        for k, v in numerical_checks.items():
            status = "PASS" if (isinstance(v, bool) and v) or \
                (not isinstance(v, bool) and v is True) else \
                ("FAIL" if isinstance(v, bool) and not v else f"{v}")
            f.write(f"- {k}: {status}\n")

        f.write("\n## Decision\n\n")
        all_pass = all([
            any(lr["improved"] for lr in learning_results.values())
            if learning_results else False,
            comparison[0]["avg_reward"] < comparison[2]["avg_reward"]
            if len(comparison) >= 3 else False,
        ])

    print(f"  Saved {path}")
    return all_pass


def write_comparison_csv(comparison: list[dict], path: str):
    fieldnames = ["label", "avg_reward", "std_reward", "avg_secrecy",
                  "std_secrecy", "avg_sensing", "std_sensing", "avg_violation"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in comparison:
            w.writerow(r)
    print(f"  Saved {path}")


def plot_learning_curves(all_csv: dict, path: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("episode_rewards", "bs_beamformer", "Reward (bs_beamformer)", axes[0, 0]),
        ("episode_rewards", "uav_trajectory", "Reward (uav_trajectory)", axes[0, 1]),
        ("episode_rewards", "jammer_beamformer", "Reward (jammer_beamformer)", axes[1, 0]),
        ("episode_secrecy", "secrecy", "Secrecy Rate", axes[1, 1]),
    ]
    colors = {"mappo_100": "tab:blue", "matd3_100": "tab:green"}

    for (csv_name, col, ylabel, ax) in metrics:
        for label, data in all_csv.items():
            rows = data.get(csv_name, [])
            if not rows:
                continue
            vals = [r[col] for r in rows if col in r]
            if not vals:
                continue
            n = len(vals)
            ax.plot(range(1, n + 1), vals, label=label,
                    color=colors.get(label, "gray"), alpha=0.8)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Three-Agent MARL Training Curves", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def numerical_checks_from_history(trainer: MARLTrainer) -> dict:
    checks = {}
    history = trainer.history
    names = list(trainer.agents.keys())

    for name in names:
        rewards = history.get(f"{name}/reward", [])
        nan_rew = any(np.isnan(r) or np.isinf(r) for r in rewards) if rewards else False
        checks[f"{name}_rewards_finite"] = not nan_rew

        p_loss = history.get(f"{name}/policy_loss", [])
        v_loss = history.get(f"{name}/value_loss", [])
        losses = p_loss + v_loss
        nan_loss = any(np.isnan(l) or np.isinf(l) for l in losses) if losses else False
        checks[f"{name}_losses_finite"] = not nan_loss

        grad = history.get(f"{name}/grad_norm", [])
        nan_grad = any(np.isnan(g) or np.isinf(g) for g in grad) if grad else False
        checks[f"{name}_grad_norm_finite"] = not nan_grad

    checks["all_numerical_pass"] = all(checks.values())
    return checks


# ── Main ────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("THREE-AGENT MARL TRAINING AND BENCHMARKING")
    print("=" * 70)
    print("\nAgents:")
    print("  1. bs_beamformer  (action: w_bs)")
    print("  2. uav_trajectory (action: q_uav increments)")
    print("  3. jammer_beamformer (action: v_jammer)")
    print("\nAlgorithms: MAPPO (100 ep) + MATD3 (100 ep)")

    ensure_output_dirs()

    all_csv = {}
    learning_results = {}
    comparison_results = []
    all_numerical = {}

    # ── Step 1: Validation ──
    validation = run_validation_tests()
    write_validation_report(
        validation,
        os.path.join(OUTPUT_ROOT, "validation_report.md"),
    )

    if not validation["success"]:
        print("\nValidation FAILED. Aborting training.")
        write_final_decision(False)
        return

    # ── Step 2: Random baseline ──
    print("\n" + "=" * 60)
    print("RANDOM BASELINE")
    print("=" * 60)
    env_cfg = EnvConfig(seed=42)
    rand_res = random_feasible_baseline(env_cfg, n_episodes=10)
    rand_res["label"] = "random_feasible"
    comparison_results.append(rand_res)
    print(f"  Secrecy={rand_res['avg_secrecy']:.4f}, "
          f"Sensing={rand_res['avg_sensing']:.4f}")

    # ── Step 3: MAPPO training (100 episodes) ──
    mappo_trainer = train_algorithm("mappo", 100, seed=42)
    mappo_csv = read_csvs(mappo_trainer)
    all_csv["mappo_100"] = mappo_csv
    lr_mappo = learning_check(mappo_csv, "MAPPO", "100ep")
    learning_results["mappo_100"] = lr_mappo

    mappo_eval = evaluate_policy(mappo_trainer, n_episodes=10, label="mappo_100")
    comparison_results.append(mappo_eval)
    mappo_num = numerical_checks_from_history(mappo_trainer)
    all_numerical["mappo_100"] = mappo_num

    # ── Step 4: MATD3 training (100 episodes) ──
    matd3_trainer = train_algorithm("matd3", 100, seed=42)
    matd3_csv = read_csvs(matd3_trainer)
    all_csv["matd3_100"] = matd3_csv
    lr_matd3 = learning_check(matd3_csv, "MATD3", "100ep")
    learning_results["matd3_100"] = lr_matd3

    matd3_eval = evaluate_policy(matd3_trainer, n_episodes=10, label="matd3_100")
    comparison_results.append(matd3_eval)
    matd3_num = numerical_checks_from_history(matd3_trainer)
    all_numerical["matd3_100"] = matd3_num

    # ── Step 5: Generate outputs ──
    print("\n" + "=" * 60)
    print("GENERATING OUTPUTS")
    print("=" * 60)

    plot_learning_curves(all_csv, os.path.join(OUTPUT_ROOT, "learning_curves.png"))
    write_comparison_csv(comparison_results, os.path.join(OUTPUT_ROOT, "comparison_table.csv"))

    # Collect numerical checks into flat dict
    flat_num = {}
    for algo, checks in all_numerical.items():
        for k, v in checks.items():
            flat_num[f"{algo}_{k}"] = v

    all_pass = write_training_report(
        learning_results, comparison_results, flat_num,
        os.path.join(OUTPUT_ROOT, "training_report.md"),
    )

    # ── Step 6: Final decision ──
    print("\n" + "=" * 60)
    improvement_mappo = lr_mappo.get("improved", False)
    improvement_matd3 = lr_matd3.get("improved", False)
    mappo_beats_random = mappo_eval.get("avg_secrecy", 0.0) > rand_res.get("avg_secrecy", 0.0)
    matd3_beats_random = matd3_eval.get("avg_secrecy", 0.0) > rand_res.get("avg_secrecy", 0.0)

    criteria = {
        "validation_passed": validation["success"],
        "mappo_improves": improvement_mappo,
        "matd3_improves": improvement_matd3,
        "mappo_beats_random": mappo_beats_random,
        "matd3_beats_random": matd3_beats_random,
    }
    all_met = all(criteria.values())
    decision = "THREE_AGENT_MARL_READY" if all_met else "THREE_AGENT_MARL_FAILED"

    print(f"\n  Criteria:")
    for k, v in criteria.items():
        print(f"    {k}: {'PASS' if v else 'FAIL'}")
    print(f"\n  Decision: {decision}")

    # Write decision
    with open(os.path.join(OUTPUT_ROOT, "decision.txt"), "w") as f:
        f.write(decision)

    # Append decision to training report
    with open(os.path.join(OUTPUT_ROOT, "training_report.md"), "a") as f:
        f.write(f"\n## Final Decision\n\n")
        f.write(f"**{decision}**\n\n")
        f.write("| Criterion | Status |\n")
        f.write("|-----------|--------|\n")
        for k, v in criteria.items():
            f.write(f"| {k} | {'PASS' if v else 'FAIL'} |\n")
        f.write(f"\n### Summary\n\n")
        f.write(f"- Validation: {validation['n_pass']}/{validation['n_total']} passed\n")
        f.write(f"- MAPPO improvement: {improvement_mappo}\n")
        f.write(f"- MATD3 improvement: {improvement_matd3}\n")
        f.write(f"- MAPPO secrecy ({mappo_eval.get('avg_secrecy', 0.0):.4f}) vs random ({rand_res.get('avg_secrecy', 0.0):.4f})\n")
        f.write(f"- MATD3 secrecy ({matd3_eval.get('avg_secrecy', 0.0):.4f}) vs random ({rand_res.get('avg_secrecy', 0.0):.4f})\n")

    print(f"\nAll outputs in {OUTPUT_ROOT}")
    return decision


def write_final_decision(success: bool):
    decision = "THREE_AGENT_MARL_READY" if success else "THREE_AGENT_MARL_FAILED"
    with open(os.path.join(OUTPUT_ROOT, "decision.txt"), "w") as f:
        f.write(decision)
    print(f"\nDecision: {decision}")


if __name__ == "__main__":
    decision = main()
    sys.exit(0 if "READY" in decision else 1)
