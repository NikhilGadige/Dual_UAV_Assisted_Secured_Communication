"""Cross-algorithm overlay plots: MAPPO vs MATD3PG vs MADDPG vs Random Walk."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

_COLORS = {
    "MAPPO": "#ff7f0e",
    "MATD3PG": "#d62728",
    "MADDPG": "#1f77b4",
    "Random Walk": "#7f7f7f",
}


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rolling(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < window:
        window = max(1, len(x) // 2) or 1
    roll = np.convolve(x, np.ones(window) / window, mode="valid")
    ep = np.arange(window, len(x) + 1)
    return ep, roll


def generate_comparison_plots(csv_paths: dict[str, str], output_dir: str, window: int = 20) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for label, path in csv_paths.items():
        rows = _read_csv(path)
        if not rows:
            continue
        data[label] = {
            "crb": np.array([float(r["avg_crb"]) for r in rows]),
            "pd": np.array([float(r["avg_pd"]) for r in rows]),
            "reward": np.array([float(r["avg_reward"]) for r in rows]),
        }
    if not data:
        return {}

    paths = {}

    for metric, ylabel, fname in [
        ("crb", "CRB trace (lower = better)", "comparison_crb.png"),
        ("pd", "Detection probability Pd (higher = better)", "comparison_pd.png"),
        ("reward", "Shared team reward", "comparison_reward.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, series in data.items():
            w = min(window, max(1, len(series[metric]) // 5))
            roll_ep, roll_v = _rolling(series[metric], w)
            color = _COLORS.get(label, None)
            ax.plot(roll_ep, roll_v, label=label, color=color, linewidth=2.0)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        if metric == "pd":
            ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"MAPPO vs MATD3PG vs MADDPG vs Random Walk — {metric.upper()} convergence")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        fig.tight_layout()
        p = str(out / fname)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths[fname.replace(".png", "")] = p

    return paths
