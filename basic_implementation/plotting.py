from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from convergence_study.plot_convergence import (
    _apply_plot_style,
    _color_for_run,
    _label_for_run,
    generate_single_run_plots,
    read_training_csv,
)


def _metric_or(row: dict[str, float | None], primary: str, fallback: str) -> float:
    value = row.get(primary)
    if value is None:
        value = row.get(fallback)
    return float(value if value is not None else 0.0)


def generate_four_way_plots(
    csv_map: dict[str, str],
    output_dir: str = "outputs/basic_outputs/plots_update",
) -> list[str]:
    plot_root = Path(output_dir)
    plot_root.mkdir(parents=True, exist_ok=True)
    ordered_keys = [
        "dqn_rayleigh_h32",
        "dqn_rician_h32",
        "ddpg_rayleigh_h32",
        "ddpg_rician_h32",
    ]
    rows_by_key = {key: read_training_csv(Path(csv_map[key])) for key in ordered_keys}
    saved_paths: list[str] = []

    _apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    for ax, run_key in zip(axes.flat, ordered_keys):
        rows = rows_by_key[run_key]
        episodes = [float(row["episode"]) for row in rows]
        secrecy = [_metric_or(row, "rolling100_avg_R_sec_mbps", "avg_R_sec_mbps") for row in rows]
        raw = [float(row["avg_R_sec_mbps"]) for row in rows]
        color = _color_for_run(run_key)
        ax.plot(episodes, raw, color="#888888", alpha=0.3, linewidth=1.0, label="Episode secrecy")
        ax.plot(episodes, secrecy, color=color, linewidth=2.2, label="Rolling100 secrecy")
        ax.set_title(_label_for_run(run_key))
        ax.set_xlabel("Episode")
        ax.set_ylabel("Secrecy Rate (Mbps)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True)
    fig.tight_layout()
    training_grid = plot_root / "four_way_training_convergence.png"
    fig.savefig(training_grid, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(training_grid.resolve()))

    _apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    for ax, run_key in zip(axes.flat, ordered_keys):
        rows = rows_by_key[run_key]
        eval_pairs = [
            (float(row["episode"]), row["eval_R_sec_mbps"])
            for row in rows
            if row["eval_R_sec_mbps"] is not None
        ]
        color = _color_for_run(run_key)
        if eval_pairs:
            ax.plot(
                [episode for episode, _ in eval_pairs],
                [float(value) for _, value in eval_pairs],
                color=color,
                marker="o",
                linestyle="--",
                linewidth=2.0,
                label="Evaluation secrecy",
            )
        ax.set_title(_label_for_run(run_key))
        ax.set_xlabel("Episode")
        ax.set_ylabel("Evaluation Secrecy Rate (Mbps)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True)
    fig.tight_layout()
    testing_grid = plot_root / "four_way_testing_convergence.png"
    fig.savefig(testing_grid, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(testing_grid.resolve()))

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_key in ordered_keys:
        rows = rows_by_key[run_key]
        episodes = [float(row["episode"]) for row in rows]
        secrecy = [_metric_or(row, "rolling100_avg_R_sec_mbps", "avg_R_sec_mbps") for row in rows]
        ax.plot(episodes, secrecy, color=_color_for_run(run_key), label=_label_for_run(run_key))
    ax.set_title("All Four Training Convergence Curves")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling100 Secrecy Rate (Mbps)")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    overlay_training = plot_root / "all_four_training_overlay.png"
    fig.savefig(overlay_training, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(overlay_training.resolve()))

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_key in ordered_keys:
        rows = rows_by_key[run_key]
        eval_pairs = [
            (float(row["episode"]), row["eval_R_sec_mbps"])
            for row in rows
            if row["eval_R_sec_mbps"] is not None
        ]
        if not eval_pairs:
            continue
        ax.plot(
            [episode for episode, _ in eval_pairs],
            [float(value) for _, value in eval_pairs],
            color=_color_for_run(run_key),
            marker="o",
            linestyle="--",
            label=_label_for_run(run_key),
        )
    ax.set_title("All Four Testing Convergence Curves")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Evaluation Secrecy Rate (Mbps)")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    overlay_testing = plot_root / "all_four_testing_overlay.png"
    fig.savefig(overlay_testing, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(overlay_testing.resolve()))

    return saved_paths


def write_summary_csv(summary_rows: list[dict], output_root: str = "outputs/basic_outputs") -> str:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "basic_update_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return str(summary_path.resolve())


def generate_per_run_plots(csv_path: str, plot_dir: str, run_key: str) -> list[str]:
    return generate_single_run_plots(csv_path, plot_dir, run_key)
