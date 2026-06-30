#!/usr/bin/env python3
"""SCA-BCD benchmark runner.

Performs:
  1. Monte-Carlo evaluation of all baselines
  2. Ablation bar plots
  3. Complexity scaling study
  4. Pareto-front sweep
  5. Validation tests
  6. Markdown report
"""

from __future__ import annotations

import csv
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sca_bcd_benchmark_exp.baselines import BaselineMethod
from sca_bcd_benchmark_exp.configs import BenchmarkConfig
from sca_bcd_benchmark_exp.evaluation import run_mc_evaluation, aggregate
from sca_bcd_benchmark_exp.plotting import save_ablation_plots
from sca_bcd_benchmark_exp.complexity import run_complexity_study
from sca_bcd_benchmark_exp.pareto import run_pareto_sweep
from sca_bcd_benchmark_exp.validate import run_all_validations


def main():
    cfg = BenchmarkConfig(
        channel_model="rician",
        seed=0,
        max_bcd_iters=30,
        max_sca_iters=10,
        N_mc=100,
    )
    dirs = cfg.ensure_output_dirs()
    root = dirs["root"]
    data_dir = dirs["data"]
    plots_dir = dirs["plots"]

    print("=" * 60)
    print("SCA-BCD Benchmark Evaluation")
    print("=" * 60)
    print(f"Output root: {root}")
    print()

    # ── Part 1-3: Monte-Carlo evaluation of all baselines ────────
    print("\n--- MC Evaluation ---")
    summaries = run_mc_evaluation(cfg, quiet=False)

    # Write benchmark_results.csv
    csv_path = data_dir / "benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "method", "N_mc", "objective_mean", "objective_std", "objective_median",
            "objective_p5", "objective_p95", "secrecy_mean", "secrecy_std",
            "secrecy_median", "secrecy_p5", "secrecy_p95",
            "sensing_mean", "sensing_std", "sensing_median", "sensing_p5", "sensing_p95",
            "runtime_mean", "runtime_std", "n_iters_mean", "violation_mean",
            "converged_fraction",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for s in summaries.values():
            d = s.__dict__.copy()
            for k in list(d.keys()):
                if k.startswith("raw_"):
                    del d[k]
            w.writerow(d)

    print(f"  Results written to {csv_path}")

    # ── Part 4: Ablation bar plots ───────────────────────────────
    print("\n--- Ablation Plots ---")
    plot_paths = save_ablation_plots(summaries, str(plots_dir))
    for name, p in plot_paths.items():
        print(f"  {name}: {p}")

    # ── Part 5: Complexity analysis ──────────────────────────────
    print("\n--- Complexity Analysis ---")
    complexity_res = run_complexity_study(cfg, str(data_dir), seed=0)
    print(f"  CSV: {complexity_res['csv_path']}")
    print("  Power-law fits:")
    for param, fit in complexity_res["power_law_fits"].items():
        print(f"    {param}: T = {fit['c']:.4e} * N^{fit['p']:.4f}")

    # ── Part 6: Pareto front ─────────────────────────────────────
    print("\n--- Pareto Front ---")
    pareto_res = run_pareto_sweep(cfg, str(data_dir), seed=0)
    print(f"  CSV: {pareto_res['csv_path']}")

    # ── Part 8: Validation tests ─────────────────────────────────
    print("\n--- Validation Tests ---")
    val_results = run_all_validations(quiet=False)
    n_pass = sum(1 for v in val_results.values() if v)
    n_total = len(val_results)
    print(f"\n  {n_pass}/{n_total} tests PASSED")

    # ── Part 7: Write benchmark report ───────────────────────────
    print("\n--- Writing Report ---")
    report_path = root / "benchmark_report.md"
    _write_report(report_path, summaries, complexity_res, pareto_res,
                  val_results, n_pass, n_total)
    print(f"  Report: {report_path}")
    print("\nDone.")


def _write_report(
    path: Path,
    summaries,
    complexity_res,
    pareto_res,
    val_results,
    n_pass,
    n_total,
):
    lines = [
        "# SCA-BCD Benchmark Report",
        "",
        "## 1. Baseline Comparison (MC, N=100)",
        "",
        "| Method | Objective (mean±std) | Secrecy rate (mean±std) | "
        "Sensing utility (mean±std) | Runtime (s) | Converged |",
        "|--------|---------------------|------------------------|"
        "-------------------------|-------------|-----------|",
    ]
    for m in [
        "random_feasible", "power_only", "trajectory_only", "jammer_only",
        "no_ris", "no_jammer", "no_secrecy", "no_sensing", "sca_bcd_full",
    ]:
        s = summaries.get(m)
        if s is None:
            continue
        lines.append(
            f"| {s.method} | {s.objective_mean:.4f}±{s.objective_std:.4f} | "
            f"{s.secrecy_mean:.4f}±{s.secrecy_std:.4f} | "
            f"{s.sensing_mean:.4f}±{s.sensing_std:.4f} | "
            f"{s.runtime_mean:.4f} | {s.converged_fraction:.0%} |"
        )

    lines += [
        "",
        "## 2. Ablation Studies",
        "",
        "Bar plots with error bars are available in the plots directory.",
        "",
        "## 3. Complexity Analysis",
        "",
        "| Parameter | Power-law fit |",
        "|-----------|---------------|",
    ]
    for param, fit in complexity_res["power_law_fits"].items():
        lines.append(f"| {param} | T = {fit['c']:.4e} × N^{fit['p']:.4f} |")

    lines += [
        "",
        "## 4. Pareto Front",
        "",
        "| alpha | Objective | Secrecy rate | Sensing utility |",
        "|-------|-----------|-------------|-----------------|",
    ]
    for row in pareto_res["rows"]:
        lines.append(
            f"| {row['alpha']:.1f} | {row['objective']:.4f} | "
            f"{row['secrecy_rate']:.4f} | {row['sensing_utility']:.4f} |"
        )

    lines += [
        "",
        "## 5. Validation Tests",
        "",
        f"**{n_pass}/{n_total} tests PASSED**",
        "",
    ]
    for name, ok in val_results.items():
        status = "PASS" if ok else "FAIL"
        lines.append(f"- {status}: {name}")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
