"""Convergence plots for the sensing study: CRB + Pd, per algorithm."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


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


def generate_convergence_plots(csv_path: str, output_dir: str, title: str, color: str = "#1f77b4") -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(csv_path)
    if not rows:
        return {}

    episodes = np.array([int(r["episode"]) for r in rows])
    crb = np.array([float(r["avg_crb"]) for r in rows])
    pd = np.array([float(r["avg_pd"]) for r in rows])
    reward = np.array([float(r["avg_reward"]) for r in rows])

    # Clip CRB trace at 5.0 to ignore abnormally high values and keep the plots stable
    crb_clipped = np.clip(crb, None, 5.0)

    # Set default rolling window to 50 as requested
    window = 50
    if len(episodes) < window:
        window = max(1, len(episodes) // 2) or 1

    paths = {}

    # CRB convergence plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episodes, crb_clipped, color=color, alpha=0.2, linewidth=0.8, label="Episode CRB (raw)")
    roll_ep, roll_crb = _rolling(crb_clipped, episodes, window)
    ax.plot(roll_ep, roll_crb, color=color, linewidth=2.0, label=f"CRB Rolling-{window}")
    
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.01, top=10.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("CRB trace (log scale, lower = better)")
    ax.set_title(f"{title} — CRB convergence")
    ax.grid(True, which="both", alpha=0.15)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = str(out / "crb_convergence.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["crb_convergence"] = p

    # Pd convergence plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episodes, pd, color="#2ca02c", alpha=0.2, linewidth=0.8, label="Episode Pd (raw)")
    roll_ep_pd, roll_pd = _rolling(pd, episodes, window)
    ax.plot(roll_ep_pd, roll_pd, color="#2ca02c", linewidth=2.0, label=f"Pd Rolling-{window}")
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Detection probability Pd (higher = better)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{title} — Pd convergence")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = str(out / "pd_convergence.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["pd_convergence"] = p

    # Combined dual-axis plot: CRB (left, log scale) + Pd (right)
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax1.plot(roll_ep, roll_crb, color=color, linewidth=2.0, label=f"CRB Rolling-{window}")
    
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=0.01, top=10.5)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("CRB trace (log scale)", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, which="both", alpha=0.15)
    
    ax2 = ax1.twinx()
    ax2.plot(roll_ep_pd, roll_pd, color="#2ca02c", linewidth=2.0, label=f"Pd Rolling-{window}")
        
    ax2.set_ylabel("Detection probability Pd", color="#2ca02c")
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    
    # Combined legend for twin axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=8)
    
    ax1.set_title(f"{title} — CRB & Pd convergence")
    fig.tight_layout()
    p = str(out / "combined_convergence.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["combined_convergence"] = p

    # Reward convergence plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episodes, reward, color="#9467bd", alpha=0.2, linewidth=0.8, label="Episode Reward (raw)")
    roll_ep_r, roll_r = _rolling(reward, episodes, window)
    ax.plot(roll_ep_r, roll_r, color="#9467bd", linewidth=2.0, label=f"Reward Rolling-{window}")
        
    ax.set_xlabel("Episode")
    ax.set_ylabel("Shared team reward")
    ax.set_title(f"{title} — Reward convergence")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = str(out / "reward_convergence.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["reward_convergence"] = p

    return paths
