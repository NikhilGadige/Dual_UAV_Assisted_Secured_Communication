"""Conditioning analysis: compare sensitivity ratios before and after
block-wise variable scaling.

Computes:
  - per-block gradient norms (unscaled and scaled coordinates)
  - per-block sensitivity (unscaled and scaled coordinates)
  - condition ratio before and after scaling
  - verification that condition_ratio_after < 50
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
from sca_bcd_exp.optimization.bcd_solver import BCDSolver
from sca_bcd_exp.optimization.secrecy_optimizer import SolutionState


def _safe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def run_conditioning_analysis() -> dict:
    """Run the full conditioning analysis.

    Returns a dict with condition metrics.
    """
    config = SCABCDConfig(jammer_mode="given", max_bcd_iters=50, seed=0)
    output_dir = config.output_root() / "conditioning"
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks_to_audit = ["power", "trajectory", "jammer"]
    block_labels = {
        "power": "Power (w_bs)",
        "trajectory": "Trajectory (q_uav)",
        "jammer": "Jammer (v_jammer)",
    }
    perturbation_pcts = [0.01, 0.05, 0.10]
    signs = [+1, -1]

    # ── 1. Feasible solution ─────────────────────────────────────────
    print("Obtaining feasible solution via BCD ...")
    env = SCABCDEnvironment(config)
    solver = BCDSolver(config)
    bcd_result = solver.solve(env)

    solution = bcd_result.solution
    x_nominal = env._unpack_decision_vars(solution.decision_vars)
    nom_eval = env.evaluate(solution)
    f_nominal = float(nom_eval["objective"])
    sec_nominal = float(nom_eval["secrecy"]["R_s_total"])
    scaler = env.scaler
    block_slices = env.block_slices()

    print(f"  Objective: {f_nominal:.6f},  Secrecy: {sec_nominal:.6f}")
    print(f"  Iterations: {bcd_result.n_iters}")

    # ── 2. Unscaled gradient norms ────────────────────────────────────
    grad_unscaled = {}
    for bn in blocks_to_audit:
        sl = block_slices[bn]
        xb = x_nominal[sl]
        g = env.finite_diff_gradient_for_block(xb, sl, solution)
        grad_unscaled[bn] = {
            "norm": float(np.linalg.norm(g)),
            "max_abs": float(np.max(np.abs(g))),
        }

    # ── 3. Scaled gradient norms ──────────────────────────────────────
    grad_scaled = {}
    for bn in blocks_to_audit:
        sl = block_slices[bn]
        xb_scaled = scaler.scale_block(x_nominal[sl].copy(), bn)
        g = env.finite_diff_gradient_for_block_scaled(xb_scaled, sl, solution)
        grad_scaled[bn] = {
            "norm": float(np.linalg.norm(g)),
            "max_abs": float(np.max(np.abs(g))),
        }

    # ── 4. Perturbation-based sensitivity (unscaled) ─────────────────
    records_unscaled = []
    records_scaled = []
    f_nominal = float(nom_eval["objective"])

    for bn in blocks_to_audit:
        sl = block_slices[bn]
        x_block_nom = x_nominal[sl].copy()

        for pct in perturbation_pcts:
            for sign in signs:
                factor = 1.0 + sign * pct
                x_pert = x_nominal.copy()
                x_pert[sl] = x_block_nom * factor

                dv = env._pack_decision_vars(x_pert, solution)
                sol = SolutionState(decision_vars=dv)
                eval_result = env.evaluate(sol)
                f_pert = float(eval_result["objective"])
                delta_f = f_pert - f_nominal

                # Unscaled delta-x norm
                dx_unscaled = float(np.linalg.norm(
                    x_pert[sl] - x_block_nom
                ))

                # Scaled delta-x norm
                xb_scaled_nom = scaler.scale_block(x_block_nom, bn)
                xb_scaled_pert = scaler.scale_block(x_pert[sl], bn)
                dx_scaled = float(np.linalg.norm(
                    xb_scaled_pert - xb_scaled_nom
                ))

                sens_u = delta_f / dx_unscaled if dx_unscaled > 1e-15 else 0.0
                sens_s = delta_f / dx_scaled if dx_scaled > 1e-15 else 0.0

                records_unscaled.append({
                    "block": bn,
                    "pert_pct": sign * pct * 100,
                    "sens": sens_u,
                    "abs_sens": abs(sens_u),
                })
                records_scaled.append({
                    "block": bn,
                    "pert_pct": sign * pct * 100,
                    "sens": sens_s,
                    "abs_sens": abs(sens_s),
                })

    df_u = pd.DataFrame(records_unscaled)
    df_s = pd.DataFrame(records_scaled)

    # ── 5. Compute condition ratios ──────────────────────────────────
    # Condition ratio = max(mean |sens|) / min(mean |sens|) across blocks
    mean_u = df_u.groupby("block")["abs_sens"].mean()
    mean_s = df_s.groupby("block")["abs_sens"].mean()

    ratio_before = float(mean_u.max() / max(mean_u.min(), 1e-30))
    ratio_after = float(mean_s.max() / max(mean_s.min(), 1e-30))

    gradient_ratio_before = max(
        v["norm"] for v in grad_unscaled.values()
    ) / max(max(v["norm"] for v in grad_unscaled.values()) * 1e-30, 1e-30)
    gradient_ratio_before = float(
        max(v["norm"] for v in grad_unscaled.values())
        / max(min(v["norm"] for v in grad_unscaled.values()), 1e-30)
    )
    gradient_ratio_after = float(
        max(v["norm"] for v in grad_scaled.values())
        / max(min(v["norm"] for v in grad_scaled.values()), 1e-30)
    )

    success = ratio_after < 50.0

    print(f"\nCondition ratio (before): {ratio_before:.2f}")
    print(f"Condition ratio (after):  {ratio_after:.2f}")
    print(f"Gradient ratio (before):  {gradient_ratio_before:.2f}")
    print(f"Gradient ratio (after):   {gradient_ratio_after:.2f}")
    print(f"Success (ratio_after < 50): {success}")

    # ── 6. Generate plots ────────────────────────────────────────────
    plt = _safe_import_matplotlib()
    if plt is not None:
        _plot_gradient_norms(
            grad_unscaled, grad_scaled, blocks_to_audit, block_labels, output_dir, plt,
        )
        _plot_sensitivity_comparison(
            mean_u, mean_s, blocks_to_audit, block_labels,
            ratio_before, ratio_after, output_dir, plt,
        )

    # ── 7. Generate report ───────────────────────────────────────────
    _write_report(
        output_dir,
        blocks_to_audit, block_labels,
        grad_unscaled, grad_scaled,
        df_u, df_s, mean_u, mean_s,
        ratio_before, ratio_after,
        gradient_ratio_before, gradient_ratio_after,
        success,
    )

    return {
        "ratio_before": ratio_before,
        "ratio_after": ratio_after,
        "gradient_ratio_before": gradient_ratio_before,
        "gradient_ratio_after": gradient_ratio_after,
        "success": success,
        "output_dir": str(output_dir),
    }


# ── Plots ─────────────────────────────────────────────────────────────

def _plot_gradient_norms(
    grad_u, grad_s, blocks, labels, output_dir, plt,
):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(blocks))
    w = 0.35

    norms_u = [grad_u[b]["norm"] for b in blocks]
    norms_s = [grad_s[b]["norm"] for b in blocks]

    bars1 = ax.bar(x - w/2, norms_u, w, label="Unscaled", color="#1f77b4", alpha=0.85)
    bars2 = ax.bar(x + w/2, norms_s, w, label="Scaled", color="#ff7f0e", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([labels[b] for b in blocks], fontsize=10)
    ax.set_ylabel("Gradient norm  ||g||", fontsize=12)
    ax.set_title("Per-Block Gradient Norms: Unscaled vs Scaled", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")

    # Annotate ratios
    ru = max(norms_u) / max(min(norms_u), 1e-30)
    rs = max(norms_s) / max(min(norms_s), 1e-30)
    ax.text(0.02, 0.95, f"Ratio (unscaled): {ru:.0f}", transform=ax.transAxes,
            fontsize=10, verticalalignment="top")
    ax.text(0.02, 0.88, f"Ratio (scaled):   {rs:.0f}", transform=ax.transAxes,
            fontsize=10, verticalalignment="top")

    fig.tight_layout()
    p = output_dir / "gradient_norms.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


def _plot_sensitivity_comparison(
    mean_u, mean_s, blocks, labels,
    ratio_before, ratio_after, output_dir, plt,
):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(blocks))
    w = 0.35

    vals_u = [mean_u[b] for b in blocks]
    vals_s = [mean_s[b] for b in blocks]

    ax.bar(x - w/2, vals_u, w, label=f"Unscaled (ratio={ratio_before:.0f})",
           color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, vals_s, w, label=f"Scaled (ratio={ratio_after:.0f})",
           color="#ff7f0e", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([labels[b] for b in blocks], fontsize=10)
    ax.set_ylabel("Mean |sensitivity|", fontsize=12)
    ax.set_title("Sensitivity per Block: Unscaled vs Scaled", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")

    fig.tight_layout()
    p = output_dir / "scaled_vs_unscaled_sensitivities.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ── Report ────────────────────────────────────────────────────────────

def _write_report(
    output_dir, blocks, labels,
    grad_u, grad_s,
    df_u, df_s,
    mean_u, mean_s,
    ratio_before, ratio_after,
    grad_ratio_before, grad_ratio_after,
    success,
):
    lines = []
    def L(s=""):
        lines.append(s)

    L("# Conditioning Report")
    L()
    L("## Summary")
    L()
    L(f"| Metric | Before (Unscaled) | After (Scaled) |")
    L(f"|--------|:-----------------:|:--------------:|")
    L(f"| Condition ratio (sensitivity) | {ratio_before:.2f} | {ratio_after:.2f} |")
    L(f"| Condition ratio (gradient)    | {grad_ratio_before:.2f} | {grad_ratio_after:.2f} |")
    L(f"| Success (ratio_after < 50)    | | {'YES' if success else 'NO'} |")
    L()
    L("---")
    L()
    L("## Per-Block Mean Sensitivity")
    L()
    L("| Block | |f/x| (unscaled) | |f/x_scaled| (scaled) |")
    L("|-------|:---------------------------:|:----------------------------:|")
    for bn in blocks:
        L(f"| {labels.get(bn, bn)} | {mean_u[bn]:.6e} | {mean_s[bn]:.6e} |")
    L()
    L("### Detailed (unscaled)")
    L()
    L("| Block | Pert (%) | df/dx |")
    L("|-------|:--------:|:-----:|")
    for _, row in df_u.iterrows():
        L(f"| {row['block']:12s} | {row['pert_pct']:+6.1f} | {row['sens']:+10.3e} |")
    L()
    L("### Detailed (scaled)")
    L()
    L("| Block | Pert (%) | df/dx_scaled |")
    L("|-------|:--------:|:------------:|")
    for _, row in df_s.iterrows():
        L(f"| {row['block']:12s} | {row['pert_pct']:+6.1f} | {row['sens']:+10.3e} |")
    L()
    L("---")
    L()
    L("## Gradient Norms")
    L()
    L("| Block | ||g|| (unscaled) | ||g|| (scaled) |")
    L("|-------|:------------------------:|:-----------------------:|")
    for bn in blocks:
        L(f"| {labels.get(bn, bn)} | {grad_u[bn]['norm']:.6e} | {grad_s[bn]['norm']:.6e} |")
    L()
    L("---")
    L()
    L("## Scaling Details")
    L()
    L("```")
    L("Power:      w_scaled = w / sqrt(P_bs_max)")
    L("Trajectory: q_scaled = (q - q_center) / q_scale")
    L("Jammer:     v_scaled = v / sqrt(P_j_max)")
    L()
    L("Adaptive FD step: eps_i = max(1e-6, 1e-3 * |x_scaled_i|)")
    L("```")
    L()

    (output_dir / "conditioning_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"  Saved {output_dir / 'conditioning_report.md'}")


if __name__ == "__main__":
    run_conditioning_analysis()
