from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "rician": "#2563eb",
    "rayleigh": "#dc2626",
}


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "lines.linewidth": 2.0,
        }
    )


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def read_log(csv_path: str | Path) -> list[dict[str, float | str | None]]:
    path = Path(csv_path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, float | str | None]] = []
    for row in rows:
        parsed.append(
            {
                "episode": float(row["episode"]),
                "fading_model": row["fading_model"],
                "avg_R_sec_mbps": float(row["avg_R_sec_mbps"]),
                "avg_R_legit_mbps": float(row["avg_R_legit_mbps"]),
                "avg_R_eve_mbps": float(row["avg_R_eve_mbps"]),
                "avg_shaped_reward": float(row["avg_shaped_reward"]),
                "rolling20_avg_R_sec_mbps": float(row["rolling20_avg_R_sec_mbps"]),
                "rolling100_avg_R_sec_mbps": float(row["rolling100_avg_R_sec_mbps"]),
                "eval_R_sec_mbps": _float_or_none(row.get("eval_R_sec_mbps")),
                "convergence_gap20_100_mbps": float(row["convergence_gap20_100_mbps"]),
                "critic_loss": _float_or_none(row.get("critic_loss")),
                "actor_loss": _float_or_none(row.get("actor_loss")),
                "exploration_noise": float(row["exploration_noise"]),
            }
        )
    return parsed


def _save_line(
    episodes: list[float],
    values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
    color: str,
) -> str:
    _style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(episodes, values, color=color)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path.resolve())


def _rolling_mean_std(values: list[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for idx in range(len(values)):
        chunk = np.asarray(values[max(0, idx - window + 1): idx + 1], dtype=float)
        means.append(float(chunk.mean()))
        stds.append(float(chunk.std()))
    return np.asarray(means), np.asarray(stds)


def generate_single_run_plots(csv_path: str, output_dir: str) -> list[str]:
    rows = read_log(csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fading = str(rows[0]["fading_model"])
    color = COLORS.get(fading, "#111827")
    label = f"TD3PG + {fading.title()}"
    episodes = [float(row["episode"]) for row in rows]
    secrecy = [float(row["avg_R_sec_mbps"]) for row in rows]
    rolling20 = [float(row["rolling20_avg_R_sec_mbps"]) for row in rows]
    rolling100 = [float(row["rolling100_avg_R_sec_mbps"]) for row in rows]
    shaped_reward = [float(row["avg_shaped_reward"]) for row in rows]
    gap = [float(row["convergence_gap20_100_mbps"]) for row in rows]
    noise = [float(row["exploration_noise"]) for row in rows]
    best = list(np.maximum.accumulate(np.asarray(secrecy, dtype=float)))
    saved = [
        _save_line(episodes, secrecy, label, "Average Secrecy Rate (Mbps)", out_dir / "secrecy_vs_episode.png", color),
        _save_line(episodes, rolling20, label, "Rolling 20-Episode Secrecy Rate (Mbps)", out_dir / "rolling20_vs_episode.png", color),
        _save_line(episodes, rolling100, label, "Rolling 100-Episode Secrecy Rate (Mbps)", out_dir / "rolling100_vs_episode.png", color),
        _save_line(episodes, shaped_reward, label, "Average Shaped Reward", out_dir / "shaped_reward_vs_episode.png", color),
        _save_line(episodes, gap, label, "|Rolling20 - Rolling100| (Mbps)", out_dir / "convergence_gap.png", color),
        _save_line(episodes, best, label, "Best-So-Far Secrecy Rate (Mbps)", out_dir / "best_so_far.png", color),
        _save_line(episodes, noise, label, "Exploration Noise Std", out_dir / "exploration_noise.png", color),
    ]

    eval_pairs = [(float(row["episode"]), row["eval_R_sec_mbps"]) for row in rows if row["eval_R_sec_mbps"] is not None]
    _style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(episodes, rolling100, color=color, label="Training rolling100")
    if eval_pairs:
        ax.plot(
            [ep for ep, _ in eval_pairs],
            [float(val) for _, val in eval_pairs],
            color="#111827",
            marker="o",
            linestyle="--",
            label="Evaluation secrecy",
        )
    ax.set_title(label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Secrecy Rate (Mbps)")
    ax.legend(frameon=True)
    ax.margins(x=0.01)
    fig.tight_layout()
    path = out_dir / "evaluation_vs_training.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path.resolve()))

    mean50, std50 = _rolling_mean_std(secrecy, 50)
    _style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(episodes, secrecy, color="#64748b", alpha=0.35, linewidth=1.0, label="Episode secrecy")
    ax.plot(episodes, mean50, color=color, label="Rolling mean (50)")
    ax.fill_between(episodes, mean50 - std50, mean50 + std50, color=color, alpha=0.20, label="Rolling std band (50)")
    ax.set_title(label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Secrecy Rate (Mbps)")
    ax.legend(frameon=True)
    ax.margins(x=0.01)
    fig.tight_layout()
    path = out_dir / "variance_band.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path.resolve()))
    return saved


def generate_channel_comparison(output_root: str = "td3pg_study/output") -> list[str]:
    root = Path(output_root)
    csvs = sorted(root.glob("td3pg_*_h*/td3pg_training_log.csv"))
    if len(csvs) < 2:
        return []
    by_fading = {read_log(csv_path)[0]["fading_model"]: (csv_path, read_log(csv_path)) for csv_path in csvs}
    if "rician" not in by_fading or "rayleigh" not in by_fading:
        return []
    plots_dir = root / "comparison_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    _style()
    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    for fading in ["rayleigh", "rician"]:
        _, rows = by_fading[fading]
        ax.plot(
            [float(row["episode"]) for row in rows],
            [float(row["rolling100_avg_R_sec_mbps"]) for row in rows],
            color=COLORS[fading],
            label=f"TD3PG + {fading.title()}",
        )
    ax.set_title("TD3PG Training Convergence: Rician vs Rayleigh")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling 100-Episode Secrecy Rate (Mbps)")
    ax.legend(frameon=True)
    ax.margins(x=0.01)
    fig.tight_layout()
    path = plots_dir / "td3pg_rician_vs_rayleigh_rolling100.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path.resolve()))

    _style()
    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    for fading in ["rayleigh", "rician"]:
        _, rows = by_fading[fading]
        eval_pairs = [(float(row["episode"]), row["eval_R_sec_mbps"]) for row in rows if row["eval_R_sec_mbps"] is not None]
        if eval_pairs:
            ax.plot(
                [ep for ep, _ in eval_pairs],
                [float(val) for _, val in eval_pairs],
                color=COLORS[fading],
                marker="o",
                linestyle="--",
                label=f"TD3PG + {fading.title()}",
            )
    ax.set_title("TD3PG Evaluation Convergence: Rician vs Rayleigh")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Evaluation Secrecy Rate (Mbps)")
    ax.legend(frameon=True)
    ax.margins(x=0.01)
    fig.tight_layout()
    path = plots_dir / "td3pg_rician_vs_rayleigh_evaluation.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path.resolve()))
    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TD3PG convergence plots.")
    parser.add_argument("--csv-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--output-root", type=str, default="td3pg_study/output")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.csv_path:
        if not args.output_dir:
            raise SystemExit("--output-dir is required with --csv-path")
        generated = generate_single_run_plots(args.csv_path, args.output_dir)
    else:
        generated = generate_channel_comparison(args.output_root)
    for path in generated:
        print(path)
