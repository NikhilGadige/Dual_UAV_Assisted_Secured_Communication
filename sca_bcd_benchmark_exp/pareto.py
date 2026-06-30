from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from sca_bcd_benchmark_exp.baselines import run_baseline, BaselineMethod
from sca_bcd_benchmark_exp.configs import BenchmarkConfig


def run_pareto_sweep(
    cfg: BenchmarkConfig,
    output_dir: str,
    seed: int = 0,
    alpha_vals: list[float] | None = None,
) -> dict:
    if alpha_vals is None:
        alpha_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for alpha in alpha_vals:
        mod = BenchmarkConfig(**{**cfg.__dict__, "alpha": alpha, "seed": seed})
        r = run_baseline(BaselineMethod.SCA_BCD_FULL, mod, seed=seed)
        rows.append({
            "alpha": alpha,
            "objective": r.objective,
            "secrecy_rate": r.secrecy_rate,
            "sensing_utility": r.sensing_utility,
            "runtime_s": r.runtime_s,
            "n_bcd_iters": r.n_bcd_iters,
        })

    # Write CSV
    csv_path = out / "pareto_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "objective", "secrecy_rate",
                                           "sensing_utility", "runtime_s", "n_bcd_iters"])
        w.writeheader()
        w.writerows(rows)

    # Plot
    _plot_pareto(rows, str(out))

    return {
        "rows": rows,
        "csv_path": str(csv_path),
    }


def _plot_pareto(rows: list[dict], output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    alphas = [r["alpha"] for r in rows]
    secs = [r["secrecy_rate"] for r in rows]
    sens = [r["sensing_utility"] for r in rows]
    objs = [r["objective"] for r in rows]

    out = Path(output_dir)

    # 1. Secrecy vs alpha
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, secs, "o-", linewidth=1.5, color="coral")
    ax.set_xlabel("alpha (weight)")
    ax.set_ylabel("Secrecy rate")
    ax.set_title("Secrecy rate vs alpha")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(str(out / "secrecy_vs_alpha.png"), dpi=150)
    plt.close(fig)

    # 2. Sensing vs alpha
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, sens, "s-", linewidth=1.5, color="mediumpurple")
    ax.set_xlabel("alpha (weight)")
    ax.set_ylabel("Sensing utility")
    ax.set_title("Sensing utility vs alpha")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(str(out / "sensing_vs_alpha.png"), dpi=150)
    plt.close(fig)

    # 3. Pareto frontier: secrecy vs sensing (colored by alpha)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(secs, sens, c=alphas, cmap="viridis", s=60, edgecolor="k", zorder=5)
    cbar = fig.colorbar(sc, ax=ax, label="alpha")
    ax.plot(secs, sens, "--", color="gray", linewidth=0.8, alpha=0.5)
    for i, a in enumerate(alphas):
        ax.annotate(f"{a:.1f}", (secs[i], sens[i]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Secrecy rate")
    ax.set_ylabel("Sensing utility")
    ax.set_title("Pareto frontier: secrecy vs sensing")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(str(out / "pareto_frontier.png"), dpi=150)
    plt.close(fig)

    # 4. Objective vs alpha
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, objs, "d-", linewidth=1.5, color="steelblue")
    ax.set_xlabel("alpha (weight)")
    ax.set_ylabel("Weighted objective")
    ax.set_title("Weighted objective vs alpha")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(str(out / "objective_vs_alpha.png"), dpi=150)
    plt.close(fig)
