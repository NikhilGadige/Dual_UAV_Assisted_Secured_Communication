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
    eval_ep_crb, eval_crb = _extract_series(rows, "eval_avg_crb")
    eval_ep_pd, eval_pd = _extract_series(rows, "eval_avg_pd")
    eval_ep_reward, eval_reward = _extract_series(rows, "eval_avg_reward")
    use_eval = len(eval_crb) >= 2 and len(eval_pd) >= 2
    plot_ep_crb, plot_crb = (eval_ep_crb, eval_crb) if use_eval else (episodes, crb)
    plot_ep_pd, plot_pd = (eval_ep_pd, eval_pd) if use_eval else (episodes, pd)
    plot_ep_reward, plot_reward = (eval_ep_reward, eval_reward) if len(eval_reward) >= 2 else (episodes, reward)
    window = min(20, max(1, len(plot_ep_crb) // 5))

    paths = {}

    # CRB convergence
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episodes, crb, color=color, alpha=0.3, linewidth=0.8, label="episode CRB")
    roll_ep, roll_crb = _rolling(plot_crb, plot_ep_crb, window)
    ax.plot(plot_ep_crb if len(plot_crb) == 1 else roll_ep, plot_crb if len(plot_crb) == 1 else roll_crb,
            color=color, linewidth=2.0,
            label=("evaluation CRB" if use_eval else f"rolling-{window}"))
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
    roll_ep_pd, roll_pd = _rolling(plot_pd, plot_ep_pd, window)
    ax.plot(plot_ep_pd if len(plot_pd) == 1 else roll_ep_pd, plot_pd if len(plot_pd) == 1 else roll_pd,
            color="#2ca02c", linewidth=2.0,
            label=("evaluation Pd" if use_eval else f"rolling-{window}"))
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
    x_crb = plot_ep_crb if len(plot_crb) == 1 else roll_ep
    y_crb = plot_crb if len(plot_crb) == 1 else roll_crb
    x_pd = plot_ep_pd if len(plot_pd) == 1 else roll_ep_pd
    y_pd = plot_pd if len(plot_pd) == 1 else roll_pd
    ax1.plot(x_crb, y_crb, color=color, linewidth=2.0,
             label="CRB (evaluation)" if use_eval else "CRB (rolling)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("CRB trace", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax2 = ax1.twinx()
    ax2.plot(x_pd, y_pd, color="#2ca02c", linewidth=2.0,
             label="Pd (evaluation)" if use_eval else "Pd (rolling)")
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
    reward_window = min(20, max(1, len(plot_reward) // 5))
    roll_ep_r, roll_r = _rolling(plot_reward, plot_ep_reward, reward_window)
    ax.plot(plot_ep_reward if len(plot_reward) == 1 else roll_ep_r,
            plot_reward if len(plot_reward) == 1 else roll_r,
            color="#9467bd", linewidth=2.0,
            label=("evaluation reward" if len(eval_reward) >= 2 else f"rolling-{reward_window}"))
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
