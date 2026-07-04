"""Convergence plots for the sensing study: CRB + Pd, per algorithm."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rolling(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < window:
        window = max(1, len(x) // 2) or 1
    roll = np.convolve(x, np.ones(window) / window, mode="valid")
    ep = np.arange(window, len(x) + 1)
    return ep, roll


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
    window = min(20, max(1, len(episodes) // 5))

    paths = {}

    # CRB convergence
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episodes, crb, color=color, alpha=0.3, linewidth=0.8, label="episode CRB")
    roll_ep, roll_crb = _rolling(crb, window)
    ax.plot(roll_ep, roll_crb, color=color, linewidth=2.0, label=f"rolling-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("CRB trace (lower = better)")
    ax.set_title(f"{title} — CRB convergence")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = str(out / "crb_convergence.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["crb_convergence"] = p

    # Pd convergence
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episodes, pd, color="#2ca02c", alpha=0.3, linewidth=0.8, label="episode Pd")
    roll_ep, roll_pd = _rolling(pd, window)
    ax.plot(roll_ep, roll_pd, color="#2ca02c", linewidth=2.0, label=f"rolling-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Detection probability Pd (higher = better)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{title} — Pd convergence")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = str(out / "pd_convergence.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["pd_convergence"] = p

    # Combined dual-axis: CRB (left) + Pd (right)
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax1.plot(roll_ep, roll_crb, color=color, linewidth=2.0, label="CRB (rolling)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("CRB trace", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax2 = ax1.twinx()
    ax2.plot(roll_ep, roll_pd, color="#2ca02c", linewidth=2.0, label="Pd (rolling)")
    ax2.set_ylabel("Detection probability Pd", color="#2ca02c")
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax1.set_title(f"{title} — CRB & Pd convergence")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    p = str(out / "combined_convergence.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["combined_convergence"] = p

    # Reward convergence
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episodes, reward, color="#9467bd", alpha=0.3, linewidth=0.8, label="episode reward")
    roll_ep, roll_r = _rolling(reward, window)
    ax.plot(roll_ep, roll_r, color="#9467bd", linewidth=2.0, label=f"rolling-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Shared team reward")
    ax.set_title(f"{title} — Reward convergence")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = str(out / "reward_convergence.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["reward_convergence"] = p

    return paths
