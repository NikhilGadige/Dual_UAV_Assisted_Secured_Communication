from __future__ import annotations

import csv
from pathlib import Path

from sca_bcd_exp.analysis.convergence_analysis import rolling_average


def _safe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def plot_convergence(training_log: str, plot_dir: str) -> dict[str, str]:
    plt = _safe_import_matplotlib()
    if plt is None:
        return {}

    with Path(training_log).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)
    iterations = [int(row["iteration"]) for row in rows]
    objective = [float(row["objective"]) for row in rows]
    secrecy = [float(row["average_secrecy_rate"]) for row in rows]
    outputs = {}

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iterations, objective, linewidth=2.0)
    ax.set_xlabel("BCD iteration")
    ax.set_ylabel("Objective")
    ax.set_title("SCA-BCD Objective Convergence")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    file_path = plot_path / "objective_convergence.png"
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    outputs["objective_convergence"] = str(file_path)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iterations, secrecy, linewidth=2.0)
    ax.set_xlabel("BCD iteration")
    ax.set_ylabel("Average secrecy rate")
    ax.set_title("Average Secrecy Rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    file_path = plot_path / "average_secrecy_rate.png"
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    outputs["average_secrecy_rate"] = str(file_path)
    return outputs


def plot_diagnostics(diagnostics_log: str, plot_dir: str) -> dict[str, str]:
    plt = _safe_import_matplotlib()
    if plt is None:
        return {}

    if not Path(diagnostics_log).exists():
        return {}

    with Path(diagnostics_log).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)
    iterations = [int(row["iteration"]) for row in rows]
    relay_norms = [float(row.get("relay_update_norm", 0)) for row in rows]
    jammer_norms = [float(row.get("jammer_update_norm", 0)) for row in rows]
    power_norms = [float(row.get("power_update_norm", 0)) for row in rows]
    rel_improvement = [float(row.get("relative_improvement", 0)) for row in rows]
    outputs = {}

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iterations, relay_norms, label="Relay", linewidth=1.8)
    ax.plot(iterations, jammer_norms, label="Jammer", linewidth=1.8)
    ax.plot(iterations, power_norms, label="Power", linewidth=1.8)
    ax.set_xlabel("BCD iteration")
    ax.set_ylabel("Update norm")
    ax.set_title("Variable Update Norms")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    file_path = plot_path / "update_norms.png"
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    outputs["update_norms"] = str(file_path)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(iterations, [max(r, 1e-12) for r in rel_improvement], linewidth=1.8)
    ax.set_xlabel("BCD iteration")
    ax.set_ylabel("Relative improvement")
    ax.set_title("Relative Objective Improvement (log scale)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    file_path = plot_path / "relative_improvement.png"
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    outputs["relative_improvement"] = str(file_path)

    return outputs


def plot_alpha_convergence(diagnostics_log: str, plot_dir: str) -> dict[str, str]:
    plt = _safe_import_matplotlib()
    if plt is None:
        return {}

    if not Path(diagnostics_log).exists():
        return {}

    with Path(diagnostics_log).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)
    iterations = [int(row["iteration"]) for row in rows]
    alpha_norms = [float(row.get("alpha_update_norm", 0)) for row in rows]
    alpha_step_sizes = [float(row.get("alpha_accepted_step_size", 0)) for row in rows]
    outputs = {}

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iterations, alpha_norms, linewidth=1.8)
    ax.set_xlabel("BCD iteration")
    ax.set_ylabel("Alpha update norm")
    ax.set_title("Alpha Variable Update Norms")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    file_path = plot_path / "alpha_convergence.png"
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    outputs["alpha_convergence"] = str(file_path)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iterations, alpha_step_sizes, marker=".", linewidth=1.8)
    ax.set_xlabel("BCD iteration")
    ax.set_ylabel("Alpha step size")
    ax.set_title("Alpha Accepted Step Sizes")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    file_path = plot_path / "alpha_step_sizes.png"
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    outputs["alpha_step_sizes"] = str(file_path)

    return outputs


def plot_mean_alpha_vs_iteration(training_log: str, plot_dir: str) -> dict[str, str]:
    plt = _safe_import_matplotlib()
    if plt is None:
        return {}

    if not Path(training_log).exists():
        return {}

    with Path(training_log).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)
    iterations = [int(row["iteration"]) for row in rows]
    if "mean_alpha" not in rows[0] or "min_alpha" not in rows[0]:
        return {}

    mean_alpha = [float(row["mean_alpha"]) for row in rows]
    min_alpha = [float(row["min_alpha"]) for row in rows]
    max_alpha = [float(row["max_alpha"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(iterations, min_alpha, max_alpha, alpha=0.2, label="Min–Max range")
    ax.plot(iterations, mean_alpha, linewidth=2.0, label="Mean alpha")
    ax.set_xlabel("BCD iteration")
    ax.set_ylabel("Alpha")
    ax.set_title("Mean Alpha vs. Iteration")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    file_path = plot_path / "mean_alpha_vs_iteration.png"
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    return {"mean_alpha_vs_iteration": str(file_path)}


def prepare_future_long_run_plots(training_log: str, diagnostics_log: str, plot_dir: str) -> list[str]:
    plt = _safe_import_matplotlib()
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    placeholders = [
        str(plot_path / "rolling100_objective.png"),
        str(plot_path / "rolling100_secrecy.png"),
        str(plot_path / "variance_band.png"),
        str(plot_path / "convergence_gap.png"),
    ]

    if plt is None or not Path(diagnostics_log).exists():
        return placeholders

    with Path(diagnostics_log).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return placeholders

    raw_objective = [float(row.get("raw_objective", 0)) for row in rows]
    secrecy = [float(row.get("average_secrecy_rate", 0)) for row in rows]

    if len(raw_objective) >= 100:
        rolling_obj = rolling_average(raw_objective, window=100)
        rolling_sec = rolling_average(secrecy, window=100)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rolling_obj, linewidth=2.0)
        ax.set_title("Rolling-100 Objective")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_path / "rolling100_objective.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rolling_sec, linewidth=2.0)
        ax.set_title("Rolling-100 Secrecy")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_path / "rolling100_secrecy.png", dpi=150)
        plt.close(fig)

    return placeholders