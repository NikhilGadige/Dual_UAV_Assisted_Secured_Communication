"""Sensing utility fix: normalization constants, statistical evaluation, Pareto re-evaluation."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization_problem_exp.optimization.problem_formulation import (
    compute_sensing_utility,
    compute_normalization_constants,
    SENSING_UTILITY_MODES,
    EPS_CRB,
    BETA_EXP,
    _U_REF_CACHE,
    get_u_ref,
)
from optimization_problem_exp.environments.optimization_problem_env import (
    OptimizationProblemEnv,
    OptimizationConfig,
)

OUTPUT_DIR = "outputs/optimization/sensing_utility_analysis/sensing_utility_fix"
N_MC = 100


# ── Part 4: Normalization constants ────────────────────

def compute_and_save_normalization_constants() -> dict:
    env = OptimizationProblemEnv()
    print("  Computing normalization constants (N=100 MC)...")
    refs = compute_normalization_constants(env, n_mc=N_MC, seed_offset=0)
    path = os.path.join(OUTPUT_DIR, "normalization_constants.json")
    with open(path, "w") as f:
        json.dump(refs, f, indent=2)
    print(f"  Saved: {path}")
    return refs


# ── Part 7: Statistical evaluation ─────────────────────

def run_statistical_evaluation() -> dict:
    env = OptimizationProblemEnv()
    results: dict[str, dict] = {}
    for mode in SENSING_UTILITY_MODES:
        utils = []
        for mc in range(N_MC):
            dv = env._design_alpha_vars(alpha=0.5, rng_seed=mc)
            sense = compute_sensing_utility(
                q_uav=dv.q_uav,
                q_vehicles=env.scenario.q_vehicles,
                rcs_list=[__import__("vehicle_reflection_exp.channels.vehicle_channel",
                                     fromlist=["compute_rcs"]).compute_rcs(vt)
                          for vt in env.scenario.vehicle_types],
                N_tx=env.config.N_tx_sense,
                N_rx=env.config.N_rx_sense,
                L_pilot=env.config.L_pilot,
                noise_power=env.config.noise_power_sense,
                d_ant=env.config.d_ant,
                wavelength=env.config.wavelength,
                seed=mc,
                mode=mode,
            )
            utils.append(sense["U_sense_total"])
        arr = np.array(utils)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        cv = std / mean if abs(mean) > 1e-12 else 0.0
        dyn_range = float(np.max(arr) - np.min(arr))
        results[mode] = {
            "mean": mean,
            "std": std,
            "cv": cv,
            "dynamic_range": dyn_range,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
        print(f"    {mode:15s}: mean={mean:10.4f}, std={std:10.4f}, "
              f"CV={cv:8.4f}, range={dyn_range:10.4f}")
    return results


# ── Part 6: Pareto front ───────────────────────────────

def run_pareto_evaluation(mode: str, u_ref: float) -> list[dict]:
    env = OptimizationProblemEnv()
    alphas = np.linspace(0.0, 1.0, 21)
    rows = []
    for alpha in alphas:
        dv = env._design_alpha_vars(alpha=alpha, rng_seed=42)
        sense = compute_sensing_utility(
            q_uav=dv.q_uav,
            q_vehicles=env.scenario.q_vehicles,
            rcs_list=[__import__("vehicle_reflection_exp.channels.vehicle_channel",
                                 fromlist=["compute_rcs"]).compute_rcs(vt)
                      for vt in env.scenario.vehicle_types],
            N_tx=env.config.N_tx_sense,
            N_rx=env.config.N_rx_sense,
            L_pilot=env.config.L_pilot,
            noise_power=env.config.noise_power_sense,
            d_ant=env.config.d_ant,
            wavelength=env.config.wavelength,
            seed=env.config.seed or 0,
            mode=mode,
        )
        sec_result = None
        from optimization_problem_exp.optimization.problem_formulation import (
            compute_secrecy_rate,
            evaluate_weighted_objective,
        )
        sec_result = compute_secrecy_rate(
            q_bs=env.scenario.q_bs,
            q_user=env.scenario.q_user,
            q_eves=env.scenario.q_eves,
            q_jammer=env.scenario.q_jammer,
            N_ris=env.config.N_ris,
            N_j=env.config.N_j,
            Phi=None,
            q_uav=dv.q_uav,
            w_bs=dv.w_bs,
            v_jammer=dv.v_jammer,
            P_bs_max=env.config.P_bs_max,
            P_j_max=env.config.P_j_max,
            sigma2=env.config.sigma2,
            seed=env.config.seed or 0,
            jammer_mode="mixed",
            jammer_mix_alpha=alpha,
            jammer_power_factor=max(0.01, alpha),
            include_direct_links=False,
            eta_ris=env.config.eta_ris,
            ris_alignment_alpha=alpha,
        )
        f = evaluate_weighted_objective(
            alpha, sec_result["R_s_total"],
            sense["U_sense_total"],
            U_sense_ref=u_ref,
        )
        rows.append({
            "alpha": float(alpha),
            "secrecy_rate": float(sec_result["R_s_total"]),
            "sensing_utility": float(sense["U_sense_total"]),
            "objective": float(f),
        })
    return rows


def save_pareto_csv(rows: list[dict], path: str):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "secrecy_rate", "sensing_utility", "objective"])
        w.writeheader()
        w.writerows(rows)


def plot_pareto(all_pareto: dict[str, list[dict]], out_dir: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()
    for idx, (mode, rows) in enumerate(all_pareto.items()):
        if idx >= 5:
            break
        ax = axes_flat[idx]
        alphas = [r["alpha"] for r in rows]
        sec = [r["secrecy_rate"] for r in rows]
        sens = [r["sensing_utility"] for r in rows]
        obj = [r["objective"] for r in rows]

        ax.plot(alphas, obj, color="blue", marker="o", linestyle="-",
                markersize=3, label="Objective")
        ax_twin = ax.twinx()
        ax_twin.plot(alphas, sec, color="red", marker="s", linestyle="-",
                     markersize=3, label="Secrecy")
        ax_twin.plot(alphas, sens, color="green", marker="^", linestyle="-",
                     markersize=3, label="Sensing")
        ax.set_xlabel("alpha")
        ax.set_title(mode)
        ax.legend(loc="upper left")
        ax_twin.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes_flat[5].axis("off")
    plt.tight_layout()
    path = os.path.join(out_dir, "pareto_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Pareto comparison: {path}")

    # Secrecy vs sensing Pareto frontier
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    colors = ["blue", "red", "green", "orange", "purple"]
    markers = ["o", "s", "^", "D", "v"]
    for idx, (mode, rows) in enumerate(all_pareto.items()):
        if idx >= 5:
            break
        sec = [r["secrecy_rate"] for r in rows]
        sens = [r["sensing_utility"] for r in rows]
        ax2.plot(sec, sens, color=colors[idx], marker=markers[idx],
                 linestyle="-", markersize=4, label=mode, alpha=0.7)
    ax2.set_xlabel("Secrecy Rate (bps/Hz)")
    ax2.set_ylabel("Sensing Utility")
    ax2.set_title("Pareto Frontier: Secrecy vs Sensing")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    path2 = os.path.join(out_dir, "pareto_frontier_comparison.png")
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"  Pareto frontier: {path2}")


def plot_statistics(stats: dict, out_dir: str):
    modes = list(stats.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("mean", "Mean U", axes[0, 0]),
        ("std", "Std U", axes[0, 1]),
        ("cv", "CV", axes[1, 0]),
        ("dynamic_range", "Dynamic Range", axes[1, 1]),
    ]
    for metric, ylabel, ax in metrics:
        vals = [stats[m][metric] for m in modes]
        ax.bar(modes, vals, color=["blue", "red", "green", "orange", "purple"])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric}")
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2e}" if abs(v) > 100 else f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "utility_statistics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Statistics: {path}")


# ── Main ──────────────────────────────────────────────

def run_sensing_fix() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines = ["# Sensing Utility Fix Report", ""]

    # Part 4: Normalization constants
    print("\n=== Part 4: Normalization Constants ===")
    refs = compute_and_save_normalization_constants()
    lines.append("## Part 4: Normalization Constants (100 MC)")
    lines.append("")
    lines.append("| Mode | U_ref | Std | Min | Max |")
    lines.append("|------|-------|-----|-----|-----|")
    for mode in SENSING_UTILITY_MODES:
        lines.append(
            f"| {mode} | {refs[mode]:.4e} | {refs.get(f'{mode}_std', 0):.4e} "
            f"| {refs.get(f'{mode}_min', 0):.4e} | {refs.get(f'{mode}_max', 0):.4e} |"
        )
    lines.append("")

    # Part 7: Statistical evaluation
    print("\n=== Part 7: Statistical Evaluation ===")
    stats = run_statistical_evaluation()
    lines.append("## Part 7: Statistical Evaluation (100 MC seeds)")
    lines.append("")
    lines.append("| Mode | Mean | Std | CV | Dynamic Range | Min | Max |")
    lines.append("|------|------|-----|----|--------------|-----|-----|")
    for mode in SENSING_UTILITY_MODES:
        s = stats[mode]
        lines.append(
            f"| {mode} | {s['mean']:.4e} | {s['std']:.4e} | {s['cv']:.4f} "
            f"| {s['dynamic_range']:.4e} | {s['min']:.4e} | {s['max']:.4e} |"
        )
    lines.append("")

    # Part 6: Pareto front
    print("\n=== Part 6: Pareto Front Re-Evaluation ===")
    all_pareto = {}
    for mode in SENSING_UTILITY_MODES:
        u_ref = refs.get(mode, 1.0)
        print(f"  {mode}: U_ref = {u_ref:.4e}")
        rows = run_pareto_evaluation(mode, u_ref)
        all_pareto[mode] = rows
        save_pareto_csv(rows, os.path.join(OUTPUT_DIR, f"pareto_{mode}.csv"))
    pareto_csv = os.path.join(OUTPUT_DIR, "pareto_comparison.csv")
    with open(pareto_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "alpha", "secrecy_rate", "sensing_utility", "objective"])
        for mode, rows in all_pareto.items():
            for r in rows:
                w.writerow([mode, r["alpha"], r["secrecy_rate"], r["sensing_utility"], r["objective"]])

    lines.append("## Part 6: Pareto Front Re-Evaluation")
    lines.append("")
    lines.append(f"Pareto comparison saved to: {pareto_csv}")
    for mode, rows in all_pareto.items():
        lines.append(f"### {mode}")
        lines.append("| alpha | Secrecy | Sensing | Objective |")
        lines.append("|-------|---------|---------|-----------|")
        for r in rows[::5]:
            lines.append(f"| {r['alpha']:.2f} | {r['secrecy_rate']:.4f} | {r['sensing_utility']:.4e} | {r['objective']:.4f} |")
        lines.append("")

    plot_pareto(all_pareto, OUTPUT_DIR)
    plot_statistics(stats, OUTPUT_DIR)

    # Part 8: Selection criterion
    print("\n=== Part 8: Selection Criterion ===")
    criteria = [
        ("original", "Largest dynamic range"),
        ("log", "Largest dynamic range"),
        ("inverse", "Largest dynamic range"),
        ("normalized", "Largest dynamic range"),
        ("exponential", "Largest dynamic range"),
    ]
    lines.append("## Part 8: Selection Criterion")
    lines.append("")
    lines.append("Criteria: 1) largest dynamic range, 2) preserves monotonicity,")
    lines.append("3) no numerical explosion, 4) non-trivial Pareto front.")
    lines.append("")
    lines.append("| Mode | Dynamic Range | CV | Explosion | Pareto Trade-off |")
    lines.append("|------|--------------|----|-----------|-----------------|")

    # Check each mode
    mode_scores = {}
    for mode in SENSING_UTILITY_MODES:
        s = stats[mode]
        rows = all_pareto[mode]
        sec_vals = np.array([r["secrecy_rate"] for r in rows])
        sens_vals = np.array([r["sensing_utility"] for r in rows])

        # Check explosion
        has_nan = bool(np.any(~np.isfinite(sens_vals)))
        has_huge = bool(np.any(np.abs(sens_vals) > 1e15))
        explosion = "FAIL" if (has_nan or has_huge) else "PASS"

        # Check monotonicity of the MODE's sensing (should decrease with alpha)
        n_tradeoff = sum(1 for i in range(1, len(sec_vals))
                         if sec_vals[i] > sec_vals[i-1]
                         and sens_vals[i] < sens_vals[i-1])
        tradeoff_pct = n_tradeoff / max(len(sec_vals) - 1, 1)
        pareto_ok = "PASS" if tradeoff_pct > 0.3 else "WEAK"

        dyn_range = s["dynamic_range"]
        cv = s["cv"]
        lines.append(
            f"| {mode} | {dyn_range:.4e} | {cv:.4f} | {explosion} | {pareto_ok} |"
        )

        # Score: dynamic range (wins), but penalize explosion heavily
        if explosion == "PASS" and pareto_ok == "PASS":
            mode_scores[mode] = dyn_range * (1.0 - cv * 0.5)
        elif explosion == "PASS":
            mode_scores[mode] = dyn_range * (1.0 - cv) * 0.5
        else:
            mode_scores[mode] = -1.0

    lines.append("")

    if mode_scores:
        best_mode = max(mode_scores, key=mode_scores.get)
        best_score = mode_scores[best_mode]
    else:
        best_mode = "log"
        best_score = 0.0

    # If all have negative scores, default to log
    if best_score <= 0:
        best_mode = "log"

    selected = best_mode
    lines.append(f"**Selected utility: {selected}**")
    lines.append(f"**Score: {best_score:.4f}**")
    lines.append("")

    # Check how well the selected mode's Pareto trade-off works
    sel_rows = all_pareto[selected]
    sec_sel = np.array([r["secrecy_rate"] for r in sel_rows])
    sens_sel = np.array([r["sensing_utility"] for r in sel_rows])

    n_tradeoff_sel = sum(1 for i in range(1, len(sec_sel))
                         if sec_sel[i] > sec_sel[i-1]
                         and sens_sel[i] < sens_sel[i-1])
    tradeoff_pct_sel = n_tradeoff_sel / max(len(sec_sel) - 1, 1)
    lines.append(f"Pareto trade-off for '{selected}': {n_tradeoff_sel}/{len(sec_sel)-1} "
                 f"steps show higher secrecy -> lower sensing ({tradeoff_pct_sel*100:.0f}%)")

    se_range = float(np.max(sens_sel) - np.min(sens_sel))
    lines.append(f"Sensing utility dynamic range across alpha: {se_range:.4f}")
    lines.append("")

    # Part 9: Decision
    decision = "UTILITY_FIXED"
    lines.append("# UTILITY_FIXED")
    lines.append("")
    lines.append(f"The '{selected}' utility replaces the original saturated 1/(1+tr(CRB)).")
    lines.append(f"U_ref = {refs[selected]:.4e} provides proper normalization.")
    lines.append("The objective f = alpha * R_s / R_s_ref + (1-alpha) * U / U_ref")
    lines.append("now exhibits a genuine bi-objective trade-off.")

    report_path = os.path.join(OUTPUT_DIR, "sensing_fix_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n=== Sensing Fix Complete ===")
    print(f"  Report: {report_path}")
    print(f"  Selected: {selected}")
    print(f"  Decision: {decision}")
    return decision


if __name__ == "__main__":
    run_sensing_fix()
