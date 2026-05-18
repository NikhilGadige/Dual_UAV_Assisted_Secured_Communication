import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN_RE = re.compile(r"^(dqn|ddpg)_(rician|rayleigh)_h(\d+)$")
BASE_COLORS = {
    "dqn_rician": "#1f77b4",
    "dqn_rayleigh": "#ff7f0e",
    "ddpg_rician": "#2ca02c",
    "ddpg_rayleigh": "#d62728",
}


def _label_for_run(run_key: str) -> str:
    match = RUN_RE.match(run_key)
    if not match:
        return run_key.replace("_", " ").title()
    algorithm, fading_model, hidden_dim = match.groups()
    return f"{algorithm.upper()} + {fading_model.title()} (h{hidden_dim})"


def _color_for_run(run_key: str) -> str | None:
    match = RUN_RE.match(run_key)
    if not match:
        return None
    algorithm, fading_model, hidden_dim = match.groups()
    base = BASE_COLORS.get(f"{algorithm}_{fading_model}")
    if base is None:
        return None
    if hidden_dim == "32":
        return base
    return _lighten_color(base, 0.35)


def _lighten_color(color: str, amount: float) -> str:
    rgb = np.array(plt.matplotlib.colors.to_rgb(color))
    blended = rgb + (1.0 - rgb) * amount
    return plt.matplotlib.colors.to_hex(blended)


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return float(text)


def read_training_csv(csv_path: Path) -> list[dict[str, float | None]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        rolling20 = row["rolling20_avg_R_sec_mbps"] if "rolling20_avg_R_sec_mbps" in row else row["rolling20"]
        rolling100 = row["rolling100_avg_R_sec_mbps"] if "rolling100_avg_R_sec_mbps" in row else row["rolling100"]
        convergence_gap = row["convergence_gap"] if "convergence_gap" in row else row["convergence_gap20_100_mbps"]
        parsed.append(
            {
                "episode": float(row["episode"]),
                "avg_R_sec_mbps": float(row["avg_R_sec_mbps"]),
                "rolling20_avg_R_sec_mbps": float(rolling20),
                "rolling100_avg_R_sec_mbps": float(rolling100),
                "avg_shaped_reward": float(row["avg_shaped_reward"]),
                "eval_R_sec_mbps": _parse_optional_float(row.get("eval_R_sec_mbps")),
                "convergence_gap": float(convergence_gap),
            }
        )
    return parsed


def _apply_plot_style() -> None:
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


def _rolling_mean_std(values: list[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for idx in range(len(values)):
        chunk = values[max(0, idx - window + 1): idx + 1]
        arr = np.asarray(chunk, dtype=float)
        means.append(float(arr.mean()))
        stds.append(float(arr.std()))
    return np.asarray(means, dtype=float), np.asarray(stds, dtype=float)


def _save_line_plot(
    x_values: list[float],
    y_values: list[float],
    title: str,
    y_label: str,
    output_path: Path,
    color: str | None = None,
) -> str:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(x_values, y_values, color=color)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path.resolve())


def generate_single_run_plots(csv_path: str, output_dir: str, run_key: str) -> list[str]:
    rows = read_training_csv(Path(csv_path))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    color = _color_for_run(run_key)
    label = _label_for_run(run_key)
    episodes = [row["episode"] for row in rows]
    avg_r_sec = [float(row["avg_R_sec_mbps"]) for row in rows]
    rolling20 = [float(row["rolling20_avg_R_sec_mbps"]) for row in rows]
    rolling100 = [float(row["rolling100_avg_R_sec_mbps"]) for row in rows]
    shaped_reward = [float(row["avg_shaped_reward"]) for row in rows]
    convergence_gap = [float(row["convergence_gap"]) for row in rows]
    eval_pairs = [(row["episode"], row["eval_R_sec_mbps"]) for row in rows if row["eval_R_sec_mbps"] is not None]
    best_so_far = list(np.maximum.accumulate(np.asarray(avg_r_sec, dtype=float)))
    rolling50_mean, rolling50_std = _rolling_mean_std(avg_r_sec, window=50)
    saved_paths: list[str] = []

    saved_paths.append(_save_line_plot(episodes, avg_r_sec, label, "Average Secrecy Rate (Mbps)", out_dir / "secrecy_vs_episode.png", color))
    saved_paths.append(_save_line_plot(episodes, rolling20, label, "Rolling 20-Episode Secrecy Rate (Mbps)", out_dir / "rolling20_vs_episode.png", color))
    saved_paths.append(_save_line_plot(episodes, rolling100, label, "Rolling 100-Episode Secrecy Rate (Mbps)", out_dir / "rolling100_vs_episode.png", color))
    saved_paths.append(_save_line_plot(episodes, shaped_reward, label, "Average Shaped Reward", out_dir / "shaped_reward_vs_episode.png", color))
    saved_paths.append(_save_line_plot(episodes, best_so_far, label, "Best-So-Far Secrecy Rate (Mbps)", out_dir / "best_so_far.png", color))
    saved_paths.append(_save_line_plot(episodes, convergence_gap, label, "|Rolling20 - Rolling100| (Mbps)", out_dir / "convergence_gap.png", color))

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(episodes, rolling100, label="Training rolling100", color=color)
    if eval_pairs:
        ax.plot(
            [ep for ep, _ in eval_pairs],
            [float(val) for _, val in eval_pairs],
            marker="o",
            linestyle="--",
            color="#111111",
            label="Evaluation secrecy",
        )
    ax.set_title(label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Secrecy Rate (Mbps)")
    ax.legend(frameon=True)
    ax.margins(x=0.01)
    fig.tight_layout()
    eval_plot_path = out_dir / "evaluation_vs_training.png"
    fig.savefig(eval_plot_path, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(eval_plot_path.resolve()))

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(episodes, rolling50_mean, color=color, label="Rolling mean (50)")
    ax.fill_between(
        episodes,
        rolling50_mean - rolling50_std,
        rolling50_mean + rolling50_std,
        color=color,
        alpha=0.2,
        label="Rolling std band (50)",
    )
    ax.plot(episodes, avg_r_sec, color="#666666", alpha=0.45, linewidth=1.2, label="Episode secrecy")
    ax.set_title(label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Secrecy Rate (Mbps)")
    ax.legend(frameon=True)
    ax.margins(x=0.01)
    fig.tight_layout()
    variance_plot_path = out_dir / "variance_band.png"
    fig.savefig(variance_plot_path, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(variance_plot_path.resolve()))

    return saved_paths


def discover_run_csvs(base_dir: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for child in base_dir.iterdir() if base_dir.exists() else []:
        if not child.is_dir() or child.name == "plots" or RUN_RE.match(child.name) is None:
            continue
        matches = sorted(child.glob("*.csv"))
        if matches:
            discovered[child.name] = matches[0]
    return discovered


def _plot_comparison(
    series_map: dict[str, list[dict[str, float | None]]],
    title: str,
    output_dir: Path,
    metric_key: str,
    filename: str,
    y_label: str,
    evaluations_only: bool = False,
) -> str:
    _apply_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for run_key, rows in series_map.items():
        label = _label_for_run(run_key)
        color = _color_for_run(run_key)
        if evaluations_only:
            eval_pairs = [(row["episode"], row[metric_key]) for row in rows if row[metric_key] is not None]
            if not eval_pairs:
                continue
            ax.plot(
                [ep for ep, _ in eval_pairs],
                [float(val) for _, val in eval_pairs],
                marker="o",
                linestyle="--",
                color=color,
                label=label,
            )
        else:
            ax.plot(
                [float(row["episode"]) for row in rows],
                [float(row[metric_key]) for row in rows],
                color=color,
                label=label,
            )
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=True)
    ax.margins(x=0.01)
    fig.tight_layout()
    plot_path = output_dir / filename
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return str(plot_path.resolve())


def generate_comparison_plots(base_dir: str = "outputs/convergence") -> dict[str, list[str]]:
    root = Path(base_dir)
    csv_map = discover_run_csvs(root)
    plots_root = root / "plots"
    results: dict[str, list[str]] = {}
    hidden_dim_layouts = {
        "dqn_rician_h32_vs_h64": ["dqn_rician_h32", "dqn_rician_h64"],
        "dqn_rayleigh_h32_vs_h64": ["dqn_rayleigh_h32", "dqn_rayleigh_h64"],
        "ddpg_rician_h32_vs_h64": ["ddpg_rician_h32", "ddpg_rician_h64"],
        "ddpg_rayleigh_h32_vs_h64": ["ddpg_rayleigh_h32", "ddpg_rayleigh_h64"],
    }
    for comparison_name, run_keys in hidden_dim_layouts.items():
        if not all(run_key in csv_map for run_key in run_keys):
            continue
        series_map = {run_key: read_training_csv(csv_map[run_key]) for run_key in run_keys}
        output_dir = plots_root / comparison_name
        title = comparison_name.replace("_", " ").upper()
        results[comparison_name] = [
            _plot_comparison(
                series_map,
                title,
                output_dir,
                metric_key="rolling100_avg_R_sec_mbps",
                filename="rolling100_comparison.png",
                y_label="Rolling 100-Episode Secrecy Rate (Mbps)",
            ),
            _plot_comparison(
                series_map,
                title,
                output_dir,
                metric_key="eval_R_sec_mbps",
                filename="evaluation_curve_comparison.png",
                y_label="Evaluation Secrecy Rate (Mbps)",
                evaluations_only=True,
            ),
        ]
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate convergence plots from training CSV outputs.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="outputs/convergence",
        help="Base output directory containing per-run convergence folders",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generated = generate_comparison_plots(args.base_dir)
    if not generated:
        print("No comparison plots generated. Expected run CSVs under outputs/convergence/<run_name>/." )
    else:
        print("Generated comparison plots:")
        for name, paths in generated.items():
            print(f"  {name}: {len(paths)} files")
