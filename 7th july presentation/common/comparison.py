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


def _rolling(x: np.ndarray, episodes: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < window:
        window = max(1, len(x) // 2) or 1
    roll = np.convolve(x, np.ones(window) / window, mode="valid")
    ep = episodes[window - 1:]
    return ep, roll


def _extract_series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    episodes = []
    values = []
    for r in rows:
        val = r.get(key, "")
        if val in ("", None):
            continue
        episodes.append(int(r["episode"]))
        values.append(float(val))
    return np.array(episodes), np.array(values, dtype=float)


def generate_comparison_plots(csv_paths: dict[str, str], output_dir: str, window: int = 50) -> dict:
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
        train_ep = np.array([int(r["episode"]) for r in rows])
        crb = np.array([float(r["avg_crb"]) for r in rows])
        data[label] = {
            "episodes": train_ep,
            "crb": np.clip(crb, None, 5.0),
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
            color = _COLORS.get(label, None)
            w = window
            if len(series[metric]) < w:
                w = max(1, len(series[metric]) // 2) or 1
            roll_ep, roll_v = _rolling(series[metric], series["episodes"], w)
            x = series["episodes"] if len(series[metric]) == 1 else roll_ep
            y = series[metric] if len(series[metric]) == 1 else roll_v
            ax.plot(x, y, label=f"{label} (Rolling-{w})", color=color, linewidth=2.0)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        
        if metric == "crb":
            ax.set_yscale('log')
            ax.set_ylim(bottom=0.01, top=10.5)
            ax.set_ylabel("CRB trace (log scale, lower = better)")
            ax.grid(True, which="both", alpha=0.15)
        else:
            ax.grid(True, alpha=0.25)

        if metric == "pd":
            ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"MAPPO vs MATD3PG vs MADDPG vs Random Walk — {metric.upper()} convergence")
        ax.legend(fontsize=9)
        fig.tight_layout()
        p = str(out / fname)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths[fname.replace(".png", "")] = p

    return paths
