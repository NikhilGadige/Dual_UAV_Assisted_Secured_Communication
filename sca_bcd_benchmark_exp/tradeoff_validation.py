"""Validate that log sensing utility creates a genuine bi-objective trade-off."""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization_problem_exp.environments.optimization_problem_env import (
    OptimizationProblemEnv,
)


OUTPUT_DIR = "outputs/optimization/tradeoff_analysis/final_tradeoff_validation"


def run_tradeoff_validation() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Part 1: mode=log is default
    env = OptimizationProblemEnv()
    assert env.config.sensing_utility_mode == "log", \
        f"Expected log, got {env.config.sensing_utility_mode}"

    # Part 2: Sweep alpha 0..1 step 0.05
    alphas = np.arange(0.0, 1.01, 0.05)
    rows = []
    for a in alphas:
        dv = env._design_alpha_vars(alpha=float(a), rng_seed=42)
        result = env.evaluate(dv, jammer_mode="mixed", alpha=float(a))
        obj = result["objective"]
        sec = result["secrecy"]
        sen = result["sensing"]
        rows.append({
            "alpha": float(a),
            "objective": obj["f"],
            "secrecy_rate": sec["R_s_total"],
            "sensing_utility": sen["U_sense_total"],
            "U_original": sen["U_original_total"],
            "U_log": sen["U_log_total"],
            "U_inverse": sen["U_inverse_total"],
            "U_normalized": sen["U_normalized_total"],
            "U_exponential": sen["U_exponential_total"],
            "U_norm": obj["U_sense_norm"],
            "R_norm": obj["R_s_norm"],
        })

    sec_vals = np.array([r["secrecy_rate"] for r in rows])
    sens_vals = np.array([r["sensing_utility"] for r in rows])

    # Part 3: correlation & trade-off steps
    corr = float(np.corrcoef(sec_vals, sens_vals)[0, 1])

    n_tradeoff = 0
    n_steps = len(rows) - 1
    for i in range(1, len(rows)):
        if rows[i]["secrecy_rate"] > rows[i-1]["secrecy_rate"] and \
           rows[i]["sensing_utility"] < rows[i-1]["sensing_utility"]:
            n_tradeoff += 1

    tradeoff_pct = n_tradeoff / max(n_steps, 1)

    # Part 4: Acceptance
    acc_corr = corr < -0.3
    acc_tradeoff = tradeoff_pct >= 0.6
    accepted = acc_corr or acc_tradeoff
    decision = "TRADE_OFF_ESTABLISHED" if accepted else "TRADE_OFF_NOT_ESTABLISHED"

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "tradeoff_statistics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ── Plots ──────────────────────────────────────────────

    # 1. Pareto front: secrecy vs sensing
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(sec_vals, sens_vals, "bo-", markersize=4)
    ax1.set_xlabel("Secrecy Rate (bps/Hz)")
    ax1.set_ylabel("Sensing Utility (log)")
    ax1.set_title("Pareto Front: Secrecy vs Sensing (log utility)")
    ax1.grid(True, alpha=0.3)
    for r in rows:
        ax1.annotate(
            f"a={r['alpha']:.2f}",
            (r["secrecy_rate"], r["sensing_utility"]),
            fontsize=7, alpha=0.7,
        )
    pareto_path = os.path.join(OUTPUT_DIR, "pareto_front.png")
    fig1.savefig(pareto_path, dpi=150)
    plt.close(fig1)

    # 2. Objective + components vs alpha
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    alpha_arr = np.array([r["alpha"] for r in rows])
    obj_arr = np.array([r["objective"] for r in rows])
    ax2.plot(alpha_arr, obj_arr, "k-o", markersize=4, label="Objective f")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(alpha_arr, sec_vals, "r-s", markersize=3, label="Secrecy R_s")
    ax2_twin.plot(alpha_arr, sens_vals, "g-^", markersize=3, label="Sensing U")
    ax2.set_xlabel("Trade-off weight alpha")
    ax2.set_ylabel("Objective f")
    ax2_twin.set_ylabel("Raw values")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper center")
    ax2.grid(True, alpha=0.3)
    obj_path = os.path.join(OUTPUT_DIR, "objective_vs_alpha.png")
    fig2.savefig(obj_path, dpi=150)
    plt.close(fig2)

    # ── Report ─────────────────────────────────────────────

    lines = [
        "# Bi-Objective Trade-off Validation",
        "",
        f"Mode: log  |  alpha sweep: 0.00 to 1.00 step 0.05 ({len(rows)} steps)",
        f"U_ref: {rows[0]['U_log']:.4f} (at alpha=0)",
        "",
        "## Acceptance Criteria",
        "",
        f"1. corr(R_s, U) = {corr:.4f}  {'< -0.3' if acc_corr else '>= -0.3'}  =>  {'PASS' if acc_corr else 'FAIL'}",
        f"2. Trade-off steps: {n_tradeoff}/{n_steps} ({tradeoff_pct*100:.1f}%)  {'>= 60%' if acc_tradeoff else '< 60%'}  =>  {'PASS' if acc_tradeoff else 'FAIL'}",
        "",
        f"**Decision: {decision}**",
        "",
        "## Alpha sweep data",
        "",
        "| alpha | Objective | Secrecy | Sensing (log) | U_norm | R_norm |",
        "|-------|-----------|---------|---------------|--------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['alpha']:.2f} | {r['objective']:.4f} | {r['secrecy_rate']:.4f} "
            f"| {r['sensing_utility']:.4f} | {r['U_norm']:.4f} | {r['R_norm']:.4f} |"
        )

    lines.extend([
        "",
        "## Step-by-step changes",
        "| Transition | dSecrecy | dSensing | Direction |",
        "|------------|----------|----------|-----------|",
    ])
    for i in range(1, len(rows)):
        ds = rows[i]["secrecy_rate"] - rows[i-1]["secrecy_rate"]
        du = rows[i]["sensing_utility"] - rows[i-1]["sensing_utility"]
        if ds > 0 and du < 0:
            direction = "TRADE-OFF"
        elif ds * du > 0:
            direction = "SAME DIR"
        else:
            direction = "MIXED"
        lines.append(
            f"| {rows[i-1]['alpha']:.2f} -> {rows[i]['alpha']:.2f} | "
            f"{ds:+.4f} | {du:+.4f} | {direction} |"
        )

    lines.extend([
        "",
        "## Summary statistics",
        f"Secrecy:  {float(np.min(sec_vals)):.4f} to {float(np.max(sec_vals)):.4f}  "
        f"(range {float(np.max(sec_vals)-np.min(sec_vals)):.4f})",
        f"Sensing:  {float(np.min(sens_vals)):.4f} to {float(np.max(sens_vals)):.4f}  "
        f"(range {float(np.max(sens_vals)-np.min(sens_vals)):.4f})",
        f"Objective: {float(np.min(obj_arr)):.4f} to {float(np.max(obj_arr)):.4f}  "
        f"(range {float(np.max(obj_arr)-np.min(obj_arr)):.4f})",
        "",
        "## All utility modes at alpha=0..1",
        "| alpha | original | log | inverse | normalized | exponential |",
        "|-------|----------|-----|---------|------------|-------------|",
    ])
    for r in rows:
        lines.append(
            f"| {r['alpha']:.2f} | {r['U_original']:.4f} | {r['U_log']:.4f} "
            f"| {r['U_inverse']:.2e} | {r['U_normalized']:.4f} | {r['U_exponential']:.4f} |"
        )

    report_path = os.path.join(OUTPUT_DIR, "tradeoff_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n=== {decision} ===")
    print(f"  corr(R_s, U) = {corr:.4f}  {acc_corr}")
    print(f"  Trade-off    = {n_tradeoff}/{n_steps} ({tradeoff_pct*100:.1f}%)  {acc_tradeoff}")
    print(f"  Report: {report_path}")
    print(f"  CSV:    {csv_path}")
    print(f"  Pareto: {pareto_path}")
    print(f"  Obj:    {obj_path}")
    return decision


if __name__ == "__main__":
    run_tradeoff_validation()
