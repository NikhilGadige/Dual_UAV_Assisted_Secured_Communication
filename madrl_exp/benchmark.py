"""
MADRL Stage 1 Benchmark.

Runs all training, evaluation, checks, and generates reports.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np

# Agg backend for headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from madrl_exp.configs import MADRLConfig, EnvConfig, AgentConfig, TrainingConfig, RewardConfig
from madrl_exp.training.trainer import MARLTrainer
from madrl_exp.environment import ISACMultiAgentEnv

OUTPUT_ROOT = os.path.join("outputs", "madrl")


def make_cfg(algorithm: str, n_episodes: int, seed: int = 42) -> MADRLConfig:
    env_cfg = EnvConfig(seed=seed)
    n_time = env_cfg.N_time
    m_bs = env_cfg.M_bs
    n_j = env_cfg.N_j
    agents = [
        AgentConfig(name="bs_beamformer", act_dim=2 * n_time * m_bs),
        AgentConfig(name="uav_jammer", act_dim=3 * n_time + 2 * n_time * n_j),
    ]
    train_cfg = TrainingConfig(
        algorithm=algorithm, n_episodes=n_episodes, seed=seed,
        eval_interval=max(20, n_episodes // 5),
        save_interval=max(50, n_episodes // 2),
        log_interval=max(10, n_episodes // 10),
    )
    return MADRLConfig(env=env_cfg, agents=agents, training=train_cfg, reward=RewardConfig(),
                        output_root=OUTPUT_ROOT)


def run_training(algorithm: str, n_episodes: int, seed: int = 42) -> MARLTrainer:
    print(f"\n{'='*60}")
    print(f"Training {algorithm.upper()} for {n_episodes} episodes (seed={seed})")
    print(f"{'='*60}")
    cfg = make_cfg(algorithm, n_episodes, seed)
    trainer = MARLTrainer(cfg)
    trainer.train()
    return trainer


def read_csvs(trainer: MARLTrainer) -> dict:
    data = {}
    for name in ["episode_rewards", "episode_secrecy", "episode_sensing", "episode_constraints"]:
        path = os.path.join(trainer.csv_dir, f"{name}.csv")
        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: float(v) for k, v in row.items()})
        data[name] = rows
    return data


# ── Part 2: Learning Check ─────────────────────────────────────

def learning_check(csv_data: dict, algorithm: str, run_label: str) -> dict:
    rew = csv_data["episode_rewards"]
    n = len(rew)
    first_20 = [r["bs_beamformer"] for r in rew[:20]]
    last_20 = [r["bs_beamformer"] for r in rew[-20:]]
    mean_first = float(np.mean(first_20))
    mean_last = float(np.mean(last_20))
    improved = mean_last > mean_first
    print(f"  {algorithm} {run_label}: first_20={mean_first:.4f}, last_20={mean_last:.4f}, "
          f"improved={improved}")
    return {"mean_first": mean_first, "mean_last": mean_last, "improved": improved}


# ── Part 3: Baseline Comparison ────────────────────────────────

def evaluate_policy(trainer: MARLTrainer, n_episodes: int = 20, label: str = "") -> dict:
    for agent in trainer.agents.values():
        agent.eval_mode()

    rewards, secrecies, sensings, violations, runtimes = [], [], [], [], []
    for ep in range(n_episodes):
        obs, _ = trainer.env.reset()
        ep_rew, ep_start = [], time.time()
        for _ in range(trainer.cfg.training.max_steps_per_episode):
            actions = {n: a.act(obs[n], deterministic=True) for n, a in trainer.agents.items()}
            obs, rew, term, trunc, info = trainer.env.step(actions)
            ep_rew.append(float(rew[trainer.env.agent_names[0]]))
            if term.get("__all__", False) or trunc.get("__all__", False):
                break
        runtimes.append(time.time() - ep_start)
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
        "avg_runtime": float(np.mean(runtimes)),
    }


def random_feasible_baseline(cfg: MADRLConfig, n_episodes: int = 20) -> dict:
    env = ISACMultiAgentEnv(cfg.env, cfg.reward)
    rewards, secrecies, sensings, violations, runtimes = [], [], [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_rew, ep_start = [], time.time()
        for _ in range(cfg.training.max_steps_per_episode):
            actions = {n: env.action_spaces[n].sample() for n in env.agent_names}
            obs, rew, term, trunc, info = env.step(actions)
            ep_rew.append(float(rew[env.agent_names[0]]))
            if term.get("__all__", False) or trunc.get("__all__", False):
                break
        runtimes.append(time.time() - ep_start)
        rewards.append(float(np.mean(ep_rew)))
        secrecies.append(float(info.get("secrecy", 0.0)))
        sensings.append(float(info.get("sensing", 0.0)))
        violations.append(float(info.get("violation", 0.0)))
    return {
        "label": "random_feasible",
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_secrecy": float(np.mean(secrecies)),
        "std_secrecy": float(np.std(secrecies)),
        "avg_sensing": float(np.mean(sensings)),
        "std_sensing": float(np.std(sensings)),
        "avg_violation": float(np.mean(violations)),
        "avg_runtime": float(np.mean(runtimes)),
    }


def sca_bcd_baseline(n_episodes: int = 5) -> dict:
    from sca_bcd_exp.run_sca_bcd import run_sca_bcd
    from sca_bcd_exp.configs import SCABCDConfig

    secrecies, sensings, violations, runtimes = [], [], [], []
    for ep in range(n_episodes):
        sca_cfg = SCABCDConfig(M_bs=4, seed=ep)
        ep_start = time.time()
        result = run_sca_bcd(sca_cfg)
        runtimes.append(time.time() - ep_start)
        secrecies.append(result.get("final_secrecy", 0.0))
        sensings.append(result.get("final_sensing", 0.0))
        viol_sum = sum(result.get("final_violations", {}).values())
        violations.append(viol_sum)

    return {
        "label": "sca_bcd",
        "avg_reward": 0.0,
        "std_reward": 0.0,
        "avg_secrecy": float(np.mean(secrecies)),
        "std_secrecy": float(np.std(secrecies)),
        "avg_sensing": float(np.mean(sensings)),
        "std_sensing": float(np.std(sensings)),
        "avg_violation": float(np.mean(violations)),
        "avg_runtime": float(np.mean(runtimes)),
    }


def baseline_comparison(trainers: dict, n_episodes: int = 20) -> list[dict]:
    print("\n--- Part 3: Baseline Comparison ---")
    results = []

    random_res = random_feasible_baseline(trainers["mappo_1000"].cfg, n_episodes)
    results.append(random_res)
    print(f"  random_feasible: secrecy={random_res['avg_secrecy']:.4f}, "
          f"sensing={random_res['avg_sensing']:.4f}, runtime={random_res['avg_runtime']:.3f}s")

    sca_res = sca_bcd_baseline(max(2, n_episodes // 4))
    results.append(sca_res)
    print(f"  sca_bcd: secrecy={sca_res['avg_secrecy']:.4f}, "
          f"sensing={sca_res['avg_sensing']:.4f}, runtime={sca_res['avg_runtime']:.3f}s")

    for label in ["mappo_1000", "matd3_1000"]:
        tr = trainers[label]
        res = evaluate_policy(tr, n_episodes, label=label)
        results.append(res)
        print(f"  {label}: reward={res['avg_reward']:.4f}, secrecy={res['avg_secrecy']:.4f}, "
              f"sensing={res['avg_sensing']:.4f}, runtime={res['avg_runtime']:.3f}s")

    return results


def compute_weighted_objective(secrecy, sensing, alpha=0.5):
    from optimization_problem_exp.optimization.problem_formulation import (
        get_u_ref, R_S_REF,
    )
    u_ref = get_u_ref("log")
    r_norm = secrecy / R_S_REF
    u_norm = sensing / u_ref if u_ref > 0 else sensing
    return alpha * r_norm + (1.0 - alpha) * u_norm


# ── Part 4: Policy Stability ───────────────────────────────────

def policy_stability(trainer: MARLTrainer, n_seeds: int = 5, n_episodes: int = 10) -> dict:
    print("\n--- Part 4: Policy Stability ---")
    objectives = []
    for seed in range(1, n_seeds + 1):
        env = ISACMultiAgentEnv(trainer.cfg.env, trainer.cfg.reward, seed=seed)
        for agent in trainer.agents.values():
            agent.eval_mode()
        ep_objs = []
        for _ in range(n_episodes):
            obs, _ = env.reset(seed=seed * 100 + _)
            for _ in range(trainer.cfg.training.max_steps_per_episode):
                actions = {n: a.act(obs[n], deterministic=True) for n, a in trainer.agents.items()}
                obs, rew, term, trunc, info = env.step(actions)
                if term.get("__all__", False) or trunc.get("__all__", False):
                    break
            obj = compute_weighted_objective(info.get("secrecy", 0.0), info.get("sensing", 0.0), trainer.cfg.env.alpha)
            ep_objs.append(obj)
        objectives.append(float(np.mean(ep_objs)))
        print(f"  seed={seed}: objective={objectives[-1]:.4f}")

    for agent in trainer.agents.values():
        agent.train_mode()

    mean_obj = float(np.mean(objectives))
    std_obj = float(np.std(objectives))
    cv = std_obj / max(abs(mean_obj), 1e-10) * 100.0
    print(f"  mean={mean_obj:.4f}, std={std_obj:.4f}, CV={cv:.2f}%")
    return {"mean": mean_obj, "std": std_obj, "cv": cv, "per_seed": objectives}


# ── Part 5: Numerical Checks ────────────────────────────────────

def numerical_checks(trainer: MARLTrainer) -> dict:
    print("\n--- Part 5: Numerical Checks ---")
    checks = {"nan_rewards": False, "nan_obs": False, "exploding_actions": False,
              "exploding_gradients": False, "critic_losses_finite": True,
              "actor_losses_finite": True}

    history = trainer.history
    names = list(trainer.agents.keys())

    for name in names:
        rewards = history.get(f"{name}/reward", [])
        if len(rewards) > 0 and (any(np.isnan(r) or np.isinf(r) for r in rewards)):
            checks["nan_rewards"] = True
            print(f"  FAIL: NaN/inf in {name} rewards")
        else:
            print(f"  PASS: {name} rewards all finite")

    for name in names:
        p_vals = history.get(f"{name}/policy_loss", [])
        v_vals = history.get(f"{name}/value_loss", [])
        vals = p_vals + v_vals
        if len(vals) > 0 and any(np.isnan(v) or np.isinf(v) for v in vals):
            checks["exploding_gradients"] = True
            print(f"  FAIL: NaN/inf losses for {name}")
        else:
            print(f"  PASS: {name} losses all finite")

    for name in names:
        algo = trainer.cfg.training.algorithm
        critic_key = f"{name}/value_loss" if algo == "mappo" else f"{name}/critic_loss"
        critic_vals = history.get(critic_key, [])
        if len(critic_vals) > 0 and any(np.isnan(v) or np.isinf(v) for v in critic_vals):
            checks["critic_losses_finite"] = False
            print(f"  FAIL: NaN/inf critic loss for {name}")
        else:
            print(f"  PASS: {name} critic losses finite")

        actor_key = f"{name}/policy_loss" if algo == "mappo" else f"{name}/actor_loss"
        actor_vals = history.get(actor_key, [])
        if len(actor_vals) > 0 and any(np.isnan(v) or np.isinf(v) for v in actor_vals):
            checks["actor_losses_finite"] = False
            print(f"  FAIL: NaN/inf actor loss for {name}")
        else:
            print(f"  PASS: {name} actor losses finite")

    all_rewards = []
    for name in names:
        all_rewards.extend(history.get(f"{name}/reward", []))
    if len(all_rewards) > 0 and max(abs(np.array(all_rewards))) > 1e6:
        checks["exploding_actions"] = True
        print("  FAIL: exploding actions detected")
    else:
        print("  PASS: no exploding actions")

    problems = [k for k, v in checks.items() if (k.startswith("nan_") or k.startswith("exploding_")) and v]
    problems += [k for k, v in checks.items() if k.endswith("_finite") and not v]
    all_pass = len(problems) == 0
    result = "ALL PASS" if all_pass else "SOME FAILURES"
    print(f"  Numerical checks: {result}")
    checks["all_pass"] = all_pass
    return checks


# ── Part 6: Generalization ─────────────────────────────────────

def generalization_check(n_episodes_train: int = 100, n_eval: int = 10) -> dict:
    print("\n--- Part 6: Generalization ---")
    trainer = run_training("mappo", n_episodes_train, seed=1)

    train_perf = evaluate_policy(trainer, n_eval, label="train_seed1")

    eval_seeds = [2, 3, 4, 5]
    perf_drops = []
    for seed in eval_seeds:
        env = ISACMultiAgentEnv(trainer.cfg.env, trainer.cfg.reward, seed=seed)
        for agent in trainer.agents.values():
            agent.eval_mode()
        objs = []
        for _ in range(n_eval):
            obs, _ = env.reset()
            for _ in range(trainer.cfg.training.max_steps_per_episode):
                actions = {n: a.act(obs[n], deterministic=True) for n, a in trainer.agents.items()}
                obs, rew, term, trunc, info = env.step(actions)
                if term.get("__all__", False) or trunc.get("__all__", False):
                    break
            obj = compute_weighted_objective(info["secrecy"], info["sensing"], trainer.cfg.env.alpha)
            objs.append(obj)
        mean_obj = float(np.mean(objs))
        train_obj = compute_weighted_objective(train_perf["avg_secrecy"], train_perf["avg_sensing"], trainer.cfg.env.alpha)
        diff = mean_obj - train_obj
        drop = diff / max(abs(train_obj), 1e-10) * 100
        perf_drops.append({"seed": seed, "objective": mean_obj, "drop_pct": drop})
        print(f"  seed {seed}: objective={mean_obj:.4f}, drop={drop:.2f}%")

    for agent in trainer.agents.values():
        agent.train_mode()

    return {"train_perf": train_perf, "train_obj": train_obj, "perf_drops": perf_drops}


# ── Part 7: Outputs ────────────────────────────────────────────

def ensure_dirs():
    for d in ["training_curves", "evaluation", "benchmark"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, d), exist_ok=True)


def plot_learning_curves(all_csv: dict, output_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("episode_rewards", "bs_beamformer", "Reward", axes[0, 0]),
        ("episode_secrecy", "secrecy", "Secrecy Rate", axes[0, 1]),
        ("episode_sensing", "sensing", "Sensing Utility", axes[1, 0]),
        ("episode_constraints", "violation", "Constraint Violation", axes[1, 1]),
    ]
    colors = {"mappo_100": "tab:blue", "mappo_1000": "tab:orange",
              "matd3_100": "tab:green", "matd3_1000": "tab:red"}
    for (csv_name, col, ylabel, ax) in metrics:
        for label, data in all_csv.items():
            rows = data.get(csv_name, [])
            if not rows:
                continue
            vals = [r[col] for r in rows]
            n = len(vals)
            ax.plot(range(1, n + 1), vals, label=label, color=colors.get(label, "gray"), alpha=0.8)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("MADRL Training Curves", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved learning_curves.png")


def plot_reward_distribution(all_csv: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    data_sets = []
    labels = []
    colors = {"mappo_100": "tab:blue", "mappo_1000": "tab:orange",
              "matd3_100": "tab:green", "matd3_1000": "tab:red"}
    for label in ["mappo_100", "mappo_1000", "matd3_100", "matd3_1000"]:
        rows = all_csv.get(label, {}).get("episode_rewards", [])
        if rows:
            vals = [r["bs_beamformer"] for r in rows]
            data_sets.append(vals)
            labels.append(label)

    if data_sets:
        bp = ax.boxplot(data_sets, labels=labels, patch_artist=True)
        for patch, (label, _) in zip(bp["boxes"], zip(labels, data_sets)):
            patch.set_facecolor(colors.get(label, "gray"))
        ax.set_ylabel("Reward")
        ax.set_title("Reward Distribution Across Episodes")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "reward_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved reward_distribution.png")


def write_comparison_csv(results: list[dict], output_dir: str):
    path = os.path.join(output_dir, "comparison_table.csv")
    fieldnames = ["label", "avg_reward", "std_reward", "avg_secrecy", "std_secrecy",
                  "avg_sensing", "std_sensing", "avg_violation", "avg_runtime"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"  Saved comparison_table.csv")


def write_comparison_report(results: list[dict], learning_results: dict, output_dir: str):
    path = os.path.join(output_dir, "comparison_report.md")
    r_rand = next(r for r in results if r["label"] == "random_feasible")
    r_sca = next(r for r in results if r["label"] == "sca_bcd")
    r_mappo = next(r for r in results if r["label"] == "mappo_1000")
    r_matd3 = next(r for r in results if r["label"] == "matd3_1000")

    with open(path, "w") as f:
        f.write("# MADRL Baseline Comparison Report\n\n")
        f.write("## Metrics\n\n")
        f.write("| Method | Reward | Secrecy | Sensing | Runtime |\n")
        f.write("|--------|--------|---------|---------|--------|\n")
        for r in results:
            f.write(f"| {r['label']} | {r['avg_reward']:.4f} | {r['avg_secrecy']:.4f} | "
                    f"{r['avg_sensing']:.4f} | {r['avg_runtime']:.3f}s |\n")

        f.write("\n## Learning Check\n\n")
        for label, lr in learning_results.items():
            f.write(f"- **{label}**: first_20={lr['mean_first']:.4f}, last_20={lr['mean_last']:.4f}, "
                    f"improved={lr['improved']}\n")

        f.write("\n## Summary\n\n")
        f.write(f"- Random feasible baseline: secrecy={r_rand['avg_secrecy']:.4f}, "
                f"sensing={r_rand['avg_sensing']:.4f}\n")
        f.write(f"- SCA-BCD baseline: secrecy={r_sca['avg_secrecy']:.4f}, "
                f"sensing={r_sca['avg_sensing']:.4f}\n")
        f.write(f"- MAPPO (1000 ep): secrecy={r_mappo['avg_secrecy']:.4f}, "
                f"sensing={r_mappo['avg_sensing']:.4f}\n")
        f.write(f"- MATD3 (1000 ep): secrecy={r_matd3['avg_secrecy']:.4f}, "
                f"sensing={r_matd3['avg_sensing']:.4f}\n")
    print(f"  Saved comparison_report.md")


def write_training_report(learning_results: dict, stability: dict, numerical: dict,
                          generalization: dict, acceptance: dict, output_dir: str):
    path = os.path.join(output_dir, "training_report.md")
    with open(path, "w") as f:
        f.write("# MADRL Stage 1 Training Report\n\n")
        f.write("## Learning Check\n\n")
        for label, lr in learning_results.items():
            f.write(f"- **{label}**: first_20={lr['mean_first']:.4f} "
                    f"-> last_20={lr['mean_last']:.4f} (improved={lr['improved']})\n")

        f.write("\n## Policy Stability\n\n")
        f.write(f"- Mean objective: {stability['mean']:.4f}\n")
        f.write(f"- Std objective: {stability['std']:.4f}\n")
        f.write(f"- CV: {stability['cv']:.2f}%\n")
        f.write(f"- Per seed: {stability['per_seed']}\n")

        f.write("\n## Numerical Checks\n\n")
        for k, v in numerical.items():
            if k == "all_pass":
                continue
            is_ok = (not v) if (k.startswith("nan_") or k.startswith("exploding_")) else v
            f.write(f"- {k}: {'PASS' if is_ok else 'FAIL'}\n")

        f.write("\n## Generalization\n\n")
        for pd in generalization["perf_drops"]:
            f.write(f"- seed {pd['seed']}: objective={pd['objective']:.4f}, "
                    f"drop={pd['drop_pct']:.2f}%\n")

        f.write("\n## Acceptance Criteria\n\n")
        for k, v in acceptance.items():
            status = "PASS" if v else "FAIL"
            f.write(f"- {k}: {status}\n")

        decision = "MADRL_STAGE1_COMPLETE" if all(acceptance.values()) else "MADRL_STAGE1_PARTIAL"
        f.write(f"\n## Decision: {decision}\n")
    print(f"  Saved training_report.md")


def write_generalization_report(gen: dict, output_dir: str):
    path = os.path.join(output_dir, "generalization_report.md")
    with open(path, "w") as f:
        f.write("# MADRL Generalization Report\n\n")
        f.write("## Training\n\n")
        f.write(f"- Train seed: 1\n")
        f.write(f"- Train performance (weighted obj): {gen.get('train_obj', 0.0):.4f}\n\n")
        f.write("## Evaluation on Unseen Seeds\n\n")
        f.write("| Seed | Objective | Drop (%) |\n")
        f.write("|------|-----------|----------|\n")
        for pd in gen["perf_drops"]:
            f.write(f"| {pd['seed']} | {pd['objective']:.4f} | {pd['drop_pct']:.2f} |\n")
    print(f"  Saved generalization_report.md")


# ── Part 8: Acceptance Criteria ────────────────────────────────

def evaluate_acceptance(learning_results: dict, baseline_results: list[dict],
                        numerical: dict, generalization: dict) -> dict:
    print("\n--- Part 8: Acceptance Criteria ---")
    c1 = any(lr["improved"] for lr in learning_results.values())
    print(f"  C1 (reward improves): {c1}")

    r_mappo = next((r for r in baseline_results if r["label"] == "mappo_1000"), None)
    r_rand = next((r for r in baseline_results if r["label"] == "random_feasible"), None)
    c2 = False
    if r_mappo is not None and r_rand is not None:
        c2 = r_mappo["avg_reward"] > r_rand["avg_reward"]
        print(f"  C2 (RL beats random): {c2} (RL={r_mappo['avg_reward']:.4f} vs "
              f"random={r_rand['avg_reward']:.4f})")

    c3 = numerical.get("all_pass", False)
    print(f"  C3 (no numerical instability): {c3}")

    c4 = all(pd["drop_pct"] < 50.0 for pd in generalization["perf_drops"])
    print(f"  C4 (generalizes): {c4}")
    if generalization["perf_drops"]:
        max_drop = max(pd["drop_pct"] for pd in generalization["perf_drops"])
        print(f"      max drop: {max_drop:.2f}%")

    all_pass = all([c1, c2, c3, c4])
    decision = "MADRL_STAGE1_COMPLETE" if all_pass else "MADRL_STAGE1_PARTIAL"
    print(f"\n  Decision: {decision}")
    return {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "all_pass": all_pass,
            "decision": decision}


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MADRL Stage 1 Benchmark")
    print("=" * 60)

    ensure_dirs()
    benchmark_dir = os.path.join(OUTPUT_ROOT, "benchmark")
    curves_dir = os.path.join(OUTPUT_ROOT, "training_curves")
    eval_dir = os.path.join(OUTPUT_ROOT, "evaluation")
    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # ── Part 1: Training Runs ──
    print("\n" + "=" * 60)
    print("PART 1: TRAINING RUNS")
    print("=" * 60)

    runs = [
        ("mappo", 100),
        ("mappo", 1000),
        ("matd3", 100),
        ("matd3", 1000),
    ]

    trainers = {}
    all_csv = {}
    learning_results = {}

    for algo, eps in runs:
        label = f"{algo}_{eps}"
        tr = run_training(algo, eps)
        trainers[label] = tr
        csv_data = read_csvs(tr)
        all_csv[label] = csv_data

        # Part 2: Learning check
        lr = learning_check(csv_data, algo, f"{eps}ep")
        learning_results[label] = lr

        # Copy CSVs to training_curves
        import shutil
        dst = os.path.join(curves_dir, label)
        shutil.copytree(tr.csv_dir, dst, dirs_exist_ok=True)

    # ── Part 3: Baseline Comparison ──
    baseline_results = baseline_comparison(trainers, n_episodes=20)
    write_comparison_csv(baseline_results, benchmark_dir)

    # ── Part 4: Policy Stability ──
    stability = policy_stability(trainers["mappo_1000"], n_seeds=5, n_episodes=10)

    # ── Part 5: Numerical Checks ──
    numerical = numerical_checks(trainers["mappo_1000"])

    # ── Part 6: Generalization ──
    gen = generalization_check(n_episodes_train=100, n_eval=10)

    # ── Part 7: Outputs ──
    print("\n--- Part 7: Outputs ---")
    plot_learning_curves(all_csv, benchmark_dir)
    plot_reward_distribution(all_csv, benchmark_dir)
    write_comparison_report(baseline_results, learning_results, benchmark_dir)
    write_training_report(learning_results, stability, numerical, gen, {}, benchmark_dir)
    write_generalization_report(gen, benchmark_dir)

    # ── Part 8: Acceptance ──
    acceptance = evaluate_acceptance(learning_results, baseline_results, numerical, gen)

    # Re-write training report with acceptance
    write_training_report(learning_results, stability, numerical, gen, acceptance, benchmark_dir)

    print("\n" + "=" * 60)
    print(f"DECISION: {acceptance['decision']}")
    print("=" * 60)

    # Print final acceptance to stdout for easy capture
    with open(os.path.join(benchmark_dir, "decision.txt"), "w") as f:
        f.write(acceptance["decision"])
    print(f"\nAll outputs in {benchmark_dir}")


if __name__ == "__main__":
    main()
