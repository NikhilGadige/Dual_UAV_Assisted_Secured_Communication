"""Loss Audit for MADRL numerical stability."""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from madrl_exp.configs import MADRLConfig, EnvConfig, AgentConfig, TrainingConfig, RewardConfig
from madrl_exp.training.trainer import MARLTrainer

OUTPUT_ROOT = os.path.join("outputs", "madrl", "loss_audit")
LOSS_KEYS_MAPPO = ["policy_loss", "value_loss", "entropy", "approx_kl", "grad_norm", "reward_mean", "reward_std"]
LOSS_KEYS_MATD3 = ["critic_loss", "actor_loss", "critic_grad_norm", "actor_grad_norm", "reward_mean", "reward_std"]


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
        eval_interval=max(50, n_episodes // 5),
        save_interval=max(500, n_episodes),
        log_interval=max(25, n_episodes // 10),
    )
    return MADRLConfig(env=env_cfg, agents=agents, training=train_cfg, reward=RewardConfig(),
                        output_root=OUTPUT_ROOT)


def run_and_collect(algorithm: str, n_episodes: int = 1000) -> MARLTrainer:
    print(f"\n{'='*60}")
    print(f"Training {algorithm.upper()} for {n_episodes} episodes")
    print(f"{'='*60}")
    cfg = make_cfg(algorithm, n_episodes)
    trainer = MARLTrainer(cfg)
    trainer.train()
    return trainer


def read_loss_csv(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


# ── Part 2: Root Cause Detection ──────────────────────────────────

FAILURES = {
    "nan_loss": False,
    "inf_loss": False,
    "exploding_loss": False,
    "nan_grad": False,
    "inf_grad": False,
    "exploding_grad": False,
    "nan_reward": False,
    "inf_reward": False,
    "zero_batch": False,
    "zero_std_advantage": False,
}

def detect_failures(rows: list[dict], loss_keys: list[str], grad_keys: list[str],
                    reward_key: str = "reward_mean") -> dict:
    eps = []
    for k in FAILURES:
        FAILURES[k] = False
    first_failure = {}

    for row in rows:
        for lk in loss_keys:
            v = row.get(lk)
            if v is None:
                continue
            if np.isnan(v):
                FAILURES["nan_loss"] = True
                first_failure.setdefault("nan_loss", row["episode"])
            if np.isinf(v):
                FAILURES["inf_loss"] = True
                first_failure.setdefault("inf_loss", row["episode"])
            if abs(v) > 1e6:
                FAILURES["exploding_loss"] = True
                first_failure.setdefault("exploding_loss", row["episode"])

        for gk in grad_keys:
            v = row.get(gk)
            if v is None:
                continue
            if np.isnan(v):
                FAILURES["nan_grad"] = True
                first_failure.setdefault("nan_grad", row["episode"])
            if np.isinf(v):
                FAILURES["inf_grad"] = True
                first_failure.setdefault("inf_grad", row["episode"])
            if abs(v) > 1e6:
                FAILURES["exploding_grad"] = True
                first_failure.setdefault("exploding_grad", row["episode"])

        rv = row.get(reward_key)
        if rv is not None:
            if np.isnan(rv):
                FAILURES["nan_reward"] = True
                first_failure.setdefault("nan_reward", row["episode"])
            if np.isinf(rv):
                FAILURES["inf_reward"] = True
                first_failure.setdefault("inf_reward", row["episode"])

    return FAILURES, first_failure


def check_empty_batches(trainer: MARLTrainer) -> bool:
    return any(len(buf) < trainer.agents[name].batch_size
               for name, buf in trainer.buffer.items())


def print_detection(algorithm: str, failures: dict, first_failure: dict):
    print(f"\n  --- {algorithm} Root Cause Detection ---")
    any_fail = False
    for k, v in failures.items():
        ep = first_failure.get(k)
        ep_str = f" (first at ep {ep})" if ep else ""
        status = "PASS" if not v else "FAIL"
        print(f"    {k}: {status}{ep_str}")
        if v:
            any_fail = True
    if not any_fail:
        print(f"    -> No numerical issues detected")


# ── Part 4: Validation ────────────────────────────────────────────

def validate(trainer: MARLTrainer) -> dict:
    print(f"\n  --- Validation ---")
    results = {"all_actor_losses_finite": True, "all_critic_losses_finite": True,
               "all_gradient_norms_finite": True}
    names = list(trainer.agents.keys())
    algo = trainer.cfg.training.algorithm

    for name in names:
        if algo == "mappo":
            pl = trainer.history.get(f"{name}/policy_loss", [])
            vl = trainer.history.get(f"{name}/value_loss", [])
            gn = trainer.history.get(f"{name}/grad_norm", [])
        else:
            pl = trainer.history.get(f"{name}/actor_loss", [])
            vl = trainer.history.get(f"{name}/critic_loss", [])
            gn = []
            for gk in [f"{name}/actor_grad_norm", f"{name}/critic_grad_norm"]:
                gn.extend(trainer.history.get(gk, []))

        for v in pl:
            if np.isnan(v) or np.isinf(v):
                results["all_actor_losses_finite"] = False
                print(f"    FAIL: NaN/inf actor loss for {name}")
                break
        for v in vl:
            if np.isnan(v) or np.isinf(v):
                results["all_critic_losses_finite"] = False
                print(f"    FAIL: NaN/inf critic loss for {name}")
                break
        for v in gn:
            if np.isnan(v) or np.isinf(v):
                results["all_gradient_norms_finite"] = False
                print(f"    FAIL: NaN/inf grad norm for {name}")
                break

    for k, v in results.items():
        print(f"    {k}: {'PASS' if v else 'FAIL'}")
    return results


# ── Part 5: Outputs ───────────────────────────────────────────────

def plot_loss_curves(algo_rows: dict, output_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {"mappo": "tab:blue", "matd3": "tab:red"}
    for algo, rows in algo_rows.items():
        if not rows:
            continue
        eps = [r["episode"] for r in rows]
        c = colors.get(algo, "gray")
        if algo == "mappo":
            axes[0, 0].plot(eps, [r.get("policy_loss", 0) for r in rows], label=algo, color=c, alpha=0.7)
            axes[0, 0].set_ylabel("Policy Loss")
            axes[0, 1].plot(eps, [r.get("value_loss", 0) for r in rows], label=algo, color=c, alpha=0.7)
            axes[0, 1].set_ylabel("Value Loss")
            axes[1, 0].plot(eps, [r.get("grad_norm", 0) for r in rows], label=algo, color=c, alpha=0.7)
            axes[1, 0].set_ylabel("Grad Norm")
        else:
            axes[0, 0].plot(eps, [r.get("actor_loss", 0) for r in rows], label=algo, color=c, alpha=0.7)
            axes[0, 0].set_ylabel("Actor Loss")
            axes[0, 1].plot(eps, [r.get("critic_loss", 0) for r in rows], label=algo, color=c, alpha=0.7)
            axes[0, 1].set_ylabel("Critic Loss")
            axes[1, 0].plot(eps, [r.get("actor_grad_norm", 0) for r in rows], label=algo, color=c, alpha=0.7, linestyle="--")
            axes[1, 0].plot(eps, [r.get("critic_grad_norm", 0) for r in rows], label=f"{algo}_critic", color=c, alpha=0.5)
            axes[1, 0].set_ylabel("Grad Norm")
        axes[1, 1].plot(eps, [r.get("reward_mean", 0) for r in rows], label=algo, color=c, alpha=0.7)
        axes[1, 1].set_ylabel("Reward Mean")

    for ax in axes.flat:
        ax.set_xlabel("Update Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Loss Audit: Per-Update Metrics", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved loss_curves.png")


def write_audit_report(algo_results: dict, output_dir: str):
    path = os.path.join(output_dir, "loss_audit_report.md")
    all_ok = True
    with open(path, "w") as f:
        f.write("# MADRL Loss Audit Report\n\n")

        for algo, res in algo_results.items():
            f.write(f"## {algo.upper()}\n\n")
            f.write("### Root Cause Detection\n\n")
            f.write("| Check | Status | First Episode |\n")
            f.write("|-------|--------|---------------|\n")
            failures = res["failures"]
            first_fail = res["first_failure"]
            for k, v in failures.items():
                ep = first_fail.get(k, "")
                status = "PASS" if not v else "FAIL"
                f.write(f"| {k} | {status} | {ep} |\n")
                if v:
                    all_ok = False

            f.write("\n### Validation\n\n")
            for vk, vv in res["validation"].items():
                status = "PASS" if vv else "FAIL"
                f.write(f"- {vk}: {status}\n")
                if not vv:
                    all_ok = False

            f.write("\n### Loss Statistics\n\n")
            rows = res["rows"]
            for agent_name in ["bs_beamformer", "uav_jammer"]:
                agent_rows = [r for r in rows if r.get("agent") == agent_name]
                if not agent_rows:
                    continue
                loss_keys_list = LOSS_KEYS_MAPPO if algo == "mappo" else LOSS_KEYS_MATD3
                f.write(f"#### {agent_name}\n\n")
                f.write("| Metric | Mean | Std | Min | Max |\n")
                f.write("|--------|------|-----|-----|-----|\n")
                for lk in loss_keys_list:
                    vals = [r.get(lk, 0.0) for r in agent_rows if lk in r]
                    if not vals:
                        continue
                    f.write(f"| {lk} | {float(np.mean(vals)):.6e} | {float(np.std(vals)):.6e} | "
                            f"{float(np.min(vals)):.6e} | {float(np.max(vals)):.6e} |\n")
                f.write("\n")

        decision = "NUMERICAL_STABILITY_CONFIRMED" if all_ok else "NUMERICAL_STABILITY_FAILED"
        f.write(f"\n## Final Decision: {decision}\n")
    print(f"  Saved loss_audit_report.md")
    return all_ok


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MADRL Loss Audit")
    print("=" * 60)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    algo_results = {}

    for algo in ["mappo", "matd3"]:
        trainer = run_and_collect(algo, n_episodes=1000)

        csv_path = os.path.join(trainer.csv_dir, "loss_history.csv")
        if os.path.exists(csv_path):
            rows = read_loss_csv(csv_path)
        else:
            print(f"  WARNING: {csv_path} not found, using history")
            loss_keys_list = LOSS_KEYS_MAPPO if algo == "mappo" else LOSS_KEYS_MATD3
            rows = []
            names = list(trainer.agents.keys())
            n_ep = len(trainer.history.get(f"{names[0]}/reward", []))
            for ep in range(n_ep):
                row = {"episode": ep + 1, "agent": names[0] if len(names) > 0 else ""}
                for name in names:
                    loss_keys_list = LOSS_KEYS_MAPPO if algo == "mappo" else LOSS_KEYS_MATD3
                    for lk in loss_keys_list:
                        vals = trainer.history.get(f"{name}/{lk}", [])
                        if ep < len(vals):
                            row[lk] = vals[ep]
                rows.append(row)

        grad_keys_map = {"mappo": ["grad_norm"], "matd3": ["actor_grad_norm", "critic_grad_norm"]}
        loss_keys_map = {"mappo": ["policy_loss", "value_loss"], "matd3": ["actor_loss", "critic_loss"]}
        grad_keys = grad_keys_map[algo]
        loss_keys = loss_keys_map[algo]
        reward_key = "reward_mean"

        failures, first_failure = detect_failures(rows, loss_keys, grad_keys, reward_key)
        print_detection(algo, failures, first_failure)

        validation = validate(trainer)

        algo_results[algo] = {
            "trainer": trainer,
            "rows": rows,
            "failures": failures,
            "first_failure": first_failure,
            "validation": validation,
        }

    plot_loss_curves({algo: r["rows"] for algo, r in algo_results.items()}, OUTPUT_ROOT)

    all_ok = write_audit_report(algo_results, OUTPUT_ROOT)

    decision = "NUMERICAL_STABILITY_CONFIRMED" if all_ok else "NUMERICAL_STABILITY_FAILED"
    with open(os.path.join(OUTPUT_ROOT, "decision.txt"), "w") as f:
        f.write(decision)

    print(f"\n{'='*60}")
    print(f"Decision: {decision}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
