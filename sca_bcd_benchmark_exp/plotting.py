from __future__ import annotations

from pathlib import Path

import numpy as np

from sca_bcd_benchmark_exp.evaluation import MCSummary


def _import_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _short_label(m: str) -> str:
    mapping = {
        "random_feasible": "Random",
        "power_only": "Power only",
        "trajectory_only": "Trajectory only",
        "jammer_only": "Jammer only",
        "no_ris": "No RIS",
        "no_jammer": "No jammer",
        "no_secrecy": "No secrecy",
        "no_sensing": "No sensing",
        "sca_bcd_full": "SCA-BCD",
    }
    return mapping.get(m, m)


_ORDER = [
    "random_feasible",
    "power_only",
    "trajectory_only",
    "jammer_only",
    "no_ris",
    "no_jammer",
    "no_secrecy",
    "no_sensing",
    "sca_bcd_full",
]


def _ordered_items(summaries: dict[str, MCSummary]):
    items = []
    for k in _ORDER:
        if k in summaries:
            items.append((k, summaries[k]))
    for k, v in summaries.items():
        if k not in _ORDER:
            items.append((k, v))
    return items


def save_ablation_plots(
    summaries: dict[str, MCSummary],
    output_dir: str,
) -> dict[str, str]:
    plt = _import_plt()
    if plt is None:
        return {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}

    items = _ordered_items(summaries)
    labels = [_short_label(k) for k, _ in items]

    # ── 1. Objective comparison ──────────────────────────────────
    means = [v.objective_mean for _, v in items]
    stds = [v.objective_std for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Weighted objective")
    ax.set_title("Objective comparison across baselines")
    ax.grid(alpha=0.2, axis="y")
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.1,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    fig.tight_layout()
    p = str(out / "objective_comparison.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["objective_comparison"] = p

    # ── 2. Secrecy comparison ────────────────────────────────────
    means = [v.secrecy_mean for _, v in items]
    stds = [v.secrecy_std for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, means, yerr=stds, capsize=4, color="coral", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Secrecy rate")
    ax.set_title("Secrecy rate comparison across baselines")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    p = str(out / "secrecy_comparison.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["secrecy_comparison"] = p

    # ── 3. Sensing comparison ────────────────────────────────────
    means = [v.sensing_mean for _, v in items]
    stds = [v.sensing_std for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, means, yerr=stds, capsize=4, color="mediumpurple", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Sensing utility")
    ax.set_title("Sensing utility comparison across baselines")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    p = str(out / "sensing_comparison.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["sensing_comparison"] = p

    # ── 4. Runtime comparison ────────────────────────────────────
    means = [v.runtime_mean for _, v in items]
    stds = [v.runtime_std for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, means, yerr=stds, capsize=4, color="seagreen", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime comparison across baselines")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    p = str(out / "runtime_comparison.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["runtime_comparison"] = p

    return paths
