"""Quick end-to-end test of the benchmark pipeline (N_mc=5)."""
from __future__ import annotations

import csv
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sca_bcd_benchmark_exp.configs import BenchmarkConfig
from sca_bcd_benchmark_exp.evaluation import run_mc_evaluation
from sca_bcd_benchmark_exp.plotting import save_ablation_plots
from sca_bcd_benchmark_exp.complexity import run_complexity_study
from sca_bcd_benchmark_exp.pareto import run_pareto_sweep
from sca_bcd_benchmark_exp.validate import run_all_validations


def main():
    cfg = BenchmarkConfig(N_mc=5, max_bcd_iters=10, max_sca_iters=5)
    dirs = cfg.ensure_output_dirs()
    root = dirs["root"]
    data_dir = dirs["data"]
    plots_dir = dirs["plots"]

    print("=== MC Evaluation (N=5) ===")
    summaries = run_mc_evaluation(cfg, quiet=False)

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
    print(f"CSV: {csv_path}")

    print("\n=== Ablation Plots ===")
    plot_paths = save_ablation_plots(summaries, str(plots_dir))
    for name, p in plot_paths.items():
        print(f"  {name}: {p}")

    print("\n=== Complexity ===")
    complexity_res = run_complexity_study(cfg, str(data_dir), seed=0)
    print(f"  CSV: {complexity_res['csv_path']}")
    for param, fit in complexity_res["power_law_fits"].items():
        print(f"    {param}: T = {fit['c']:.4e} * N^{fit['p']:.4f}")

    print("\n=== Pareto ===")
    pareto_res = run_pareto_sweep(cfg, str(data_dir), seed=0)
    print(f"  CSV: {pareto_res['csv_path']}")

    print("\n=== Validations ===")
    val_results = run_all_validations(quiet=False)
    n_pass = sum(1 for v in val_results.values() if v)
    n_total = len(val_results)
    print(f"  {n_pass}/{n_total} PASSED")

    print("\n=== Report ===")
    report_path = root / "benchmark_report.md"
    _write_report(report_path, summaries, complexity_res, pareto_res,
                  val_results, n_pass, n_total, cfg.N_mc)
    print(f"  Report: {report_path}")
    print("\nDONE.")


def _write_report(path, summaries, complexity_res, pareto_res,
                  val_results, n_pass, n_total, N_mc):
    lines = [
        "# SCA-BCD Benchmark Report",
        "",
        f"## 1. Baseline Comparison (MC, N={N_mc})",
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
        "Bar plots with error bars in plots directory.",
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
        lines.append(f"- {'PASS' if ok else 'FAIL'}: {name}")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
