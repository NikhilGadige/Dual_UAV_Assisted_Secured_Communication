from __future__ import annotations

from pathlib import Path

import numpy as np


def _safe_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def save_all_plots(
    output_dir: str,
    objective_history: list[float],
    constraint_history: list[dict],
    secrecy_history: list[float],
    sensing_history: list[float],
) -> dict[str, str]:
    plt = _safe_import_matplotlib()
    if plt is None:
        return {}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}

    iters = list(range(len(objective_history)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, objective_history, linewidth=2.0)
    ax.set_xlabel("BCD iteration")
    ax.set_ylabel("Objective")
    ax.set_title("Objective History")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = out / "objective_history.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["objective_history"] = str(p)

    if constraint_history:
        viol_keys = list(constraint_history[0].keys())
        fig, ax = plt.subplots(figsize=(8, 4))
        for key in viol_keys:
            vals = [v.get(key, 0.0) for v in constraint_history]
            ax.plot(iters, vals, label=key, linewidth=1.5)
        ax.set_xlabel("BCD iteration")
        ax.set_ylabel("Violation")
        ax.set_title("Constraint Violations")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        p = out / "constraint_history.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths["constraint_history"] = str(p)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, secrecy_history, linewidth=2.0, color="green")
    ax.set_xlabel("BCD iteration")
    ax.set_ylabel("Secrecy Rate")
    ax.set_title("Secrecy History")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = out / "secrecy_history.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["secrecy_history"] = str(p)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, sensing_history, linewidth=2.0, color="purple")
    ax.set_xlabel("BCD iteration")
    ax.set_ylabel("Sensing Utility")
    ax.set_title("Sensing History")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = out / "sensing_history.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["sensing_history"] = str(p)

    return paths


def save_convergence_audit_plots(
    output_dir: str,
    results: dict[int, object],
) -> dict[str, str]:
    plt = _safe_import_matplotlib()
    if plt is None:
        return {}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}

    seeds = sorted(results.keys())
    max_len = max(len(results[s].objective_history) for s in seeds)

    # ── 1. objective_history.png ──────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for s in seeds:
        r = results[s]
        iters = list(range(len(r.objective_history)))
        ax.plot(iters, r.objective_history, linewidth=1.5, label=f"seed {s}")
    ax.set_xlabel("BCD iteration")
    ax.set_ylabel("Objective (weighted)")
    ax.set_title("Objective History Across Seeds")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = out / "objective_history.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["objective_history"] = str(p)

    # ── 2. update_norms.png ───────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for s in seeds:
        r = results[s]
        bcd_iters = list(range(1, len(r.objective_history)))
        axes[0].plot(bcd_iters, r.delta_w_norms, linewidth=1.2, label=f"seed {s}")
        axes[1].plot(bcd_iters, r.delta_q_norms, linewidth=1.2, label=f"seed {s}")
        axes[2].plot(bcd_iters, r.delta_v_norms, linewidth=1.2, label=f"seed {s}")
    axes[0].set_ylabel("||Δw||")
    axes[0].set_title("Beamforming Update Norm")
    axes[0].legend(fontsize=6)
    axes[0].grid(alpha=0.25)
    axes[1].set_ylabel("||Δq||")
    axes[1].set_title("Trajectory Update Norm")
    axes[1].legend(fontsize=6)
    axes[1].grid(alpha=0.25)
    axes[2].set_xlabel("BCD iteration")
    axes[2].set_ylabel("||Δv||")
    axes[2].set_title("Jammer Update Norm")
    axes[2].legend(fontsize=6)
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    p = out / "update_norms.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["update_norms"] = str(p)

    # ── 3. block_contributions.png ────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for s in seeds:
        r = results[s]
        bcd_iters = list(range(1, len(r.objective_history)))
        axes[0].plot(bcd_iters, r.block_contributions["power"], linewidth=1.2, label=f"seed {s}")
        axes[1].plot(bcd_iters, r.block_contributions["trajectory"], linewidth=1.2, label=f"seed {s}")
        axes[2].plot(bcd_iters, r.block_contributions["jammer"], linewidth=1.2, label=f"seed {s}")
    axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[0].set_ylabel("Δ obj")
    axes[0].set_title("Power Block Improvement")
    axes[0].legend(fontsize=6)
    axes[0].grid(alpha=0.25)
    axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[1].set_ylabel("Δ obj")
    axes[1].set_title("Trajectory Block Improvement")
    axes[1].legend(fontsize=6)
    axes[1].grid(alpha=0.25)
    axes[2].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[2].set_xlabel("BCD iteration")
    axes[2].set_ylabel("Δ obj")
    axes[2].set_title("Jammer Block Improvement")
    axes[2].legend(fontsize=6)
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    p = out / "block_contributions.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["block_contributions"] = str(p)

    # ── 4. relative_improvement.png ───────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for s in seeds:
        r = results[s]
        obj0 = r.objective_history[0]
        rel_imp = [
            (v - obj0) / max(abs(obj0), 1e-12)
            for v in r.objective_history
        ]
        iters = list(range(len(rel_imp)))
        ax.plot(iters, rel_imp, linewidth=1.5, label=f"seed {s}")
    ax.set_xlabel("BCD iteration")
    ax.set_ylabel("Relative improvement")
    ax.set_title("Relative Objective Improvement")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    p = out / "relative_improvement.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["relative_improvement"] = str(p)

    return paths
