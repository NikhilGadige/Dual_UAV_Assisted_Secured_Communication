"""Coupling audit for SCA-BCD optimization blocks.

Perturbs each block independently from a feasible solution,
measures sensitivity of the objective and secrecy rate,
and determines whether the problem is genuinely coupled.
"""

from __future__ import annotations

import sys
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


def run_coupling_audit() -> pd.DataFrame:
    """Run the full coupling audit.

    Returns the detailed perturbation DataFrame.
    """
    # -- Configuration ------------------------------------------------
    # NOTE: The default config has jammer_mode="mixed", which means the
    # jammer beamforming variables (v_jammer) are IGNORED during
    # evaluation -- heuristic beams are designed instead.  For a
    # meaningful coupling audit we set jammer_mode="given" so the
    # optimizer's decision variables actually affect the objective.
    # The report notes this difference.
    config = SCABCDConfig(
        jammer_mode="given",
        max_bcd_iters=50,
        seed=0,
    )

    output_dir = config.output_root() / "coupling_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks_to_audit = ["power", "trajectory", "jammer"]
    block_labels = {
        "power": "Power (w_bs)",
        "trajectory": "Trajectory (q_uav)",
        "jammer": "Jammer (v_jammer)",
    }
    perturbation_pcts = [0.01, 0.05, 0.10]
    signs = [+1, -1]
    INACTIVE_THRESHOLD = 1e-6

    # ===================================================================
    # STEP 1 -- Obtain a feasible solution via BCD
    # ===================================================================
    print("=" * 64)
    print("  COUPLING AUDIT -- SCA-BCD")
    print("=" * 64)
    print(f"  channel_model = {config.channel_model}")
    print(f"  jammer_mode   = {config.jammer_mode}")
    print(f"  N_time={config.N_time}  N_j={config.N_j}  N_ris={config.N_ris}")
    print(f"  P_bs_max={config.P_bs_max}  P_j_max={config.P_j_max}")
    print()

    print("[1/5] Running SCA-BCD to obtain a feasible solution ...")
    env = SCABCDEnvironment(config)
    solver = BCDSolver(config)
    bcd_result = solver.solve(env)

    solution = bcd_result.solution
    x_flat = env._unpack_decision_vars(solution.decision_vars)

    nom_eval = env.evaluate(solution)
    f_nominal = float(nom_eval["objective"])
    sec_nominal = float(nom_eval["secrecy"]["R_s_total"])
    sinr_eve_nominal = nom_eval["secrecy"]["SINR_eve"]
    sinr_eve_max_nom = float(np.max(sinr_eve_nominal))

    print(f"    Converged: {bcd_result.converged}  ({bcd_result.n_iters} iterations)")
    print(f"    Objective : {f_nominal:.6f}")
    obj_val = float(f_nominal)
    sec_val = float(sec_nominal)
    print(f"    Secrecy   : {sec_nominal:.6f}  (max Eve SINR = {sinr_eve_max_nom:.6e})")

    # -- Block info ---------------------------------------------------
    block_slices = env.block_slices()
    print("\n  Block dimensions:")
    for bn in blocks_to_audit:
        sl = block_slices[bn]
        n = sl.stop - sl.start
        print(f"    {bn:12s}  {n:3d} entries   slice [{sl.start}:{sl.stop})")

    # ===================================================================
    # STEP 2 -- Perturb each block and evaluate
    # ===================================================================
    print("\n[2/5] Perturbing each block (+/-1%, +/-5%, +/-10%) ...")

    records = []

    for block_name in blocks_to_audit:
        sl = block_slices[block_name]
        x_block_nom = x_flat[sl].copy()
        block_norm = float(np.linalg.norm(x_block_nom))

        for pct in perturbation_pcts:
            for sign in signs:
                factor = 1.0 + sign * pct

                x_pert = x_flat.copy()
                x_pert[sl] = x_block_nom * factor

                dv = env._pack_decision_vars(x_pert, solution)
                sol = SolutionState(decision_vars=dv)
                eval_result = env.evaluate(sol)

                f_pert = float(eval_result["objective"])
                sec_pert = float(eval_result["secrecy"]["R_s_total"])
                sinr_eve_pert = eval_result["secrecy"]["SINR_eve"]
                sinr_eve_max_pert = float(np.max(sinr_eve_pert))

                dx = float(np.linalg.norm(x_pert[sl] - x_block_nom))
                df = f_pert - f_nominal
                dsec = sec_pert - sec_nominal
                dsinr_eve = sinr_eve_max_pert - sinr_eve_max_nom

                sens_obj = df / dx if dx > 1e-15 else 0.0
                sens_sec = dsec / dx if dx > 1e-15 else 0.0

                records.append(
                    {
                        "block": block_name,
                        "perturbation_pct": sign * pct * 100,
                        "factor": factor,
                        "dx_norm": dx,
                        "block_norm": block_norm,
                        "df": df,
                        "dsecrecy": dsec,
                        "dsinr_eve_max": dsinr_eve,
                        "sensitivity_obj": sens_obj,
                        "sensitivity_secrecy": sens_sec,
                        "f_perturbed": f_pert,
                        "sec_perturbed": sec_pert,
                        "sinr_eve_max_perturbed": sinr_eve_max_pert,
                    }
                )

    df_results = pd.DataFrame(records)

    # ===================================================================
    # STEP 3 -- Compute average sensitivities
    # ===================================================================
    print("\n[3/5] Computing average sensitivities ...")

    summary_rows = []
    for block_name in blocks_to_audit:
        bd = df_results[df_results["block"] == block_name]
        obj_sens_abs = bd["sensitivity_obj"].abs().values
        sec_sens_abs = bd["sensitivity_secrecy"].abs().values
        obj_sens_raw = bd["sensitivity_obj"].values
        sec_sens_raw = bd["sensitivity_secrecy"].values

        summary_rows.append(
            {
                "block": block_name,
                "obj_sens_mean": float(np.mean(obj_sens_abs)),
                "obj_sens_std": float(np.std(obj_sens_abs)),
                "obj_sens_max": float(np.max(obj_sens_abs)),
                "obj_sens_raw_mean": float(np.mean(obj_sens_raw)),
                "obj_sens_raw_std": float(np.std(obj_sens_raw)),
                "sec_sens_mean": float(np.mean(sec_sens_abs)),
                "sec_sens_std": float(np.std(sec_sens_abs)),
                "sec_sens_max": float(np.max(sec_sens_abs)),
                "sec_sens_raw_mean": float(np.mean(sec_sens_raw)),
                "sec_sens_raw_std": float(np.std(sec_sens_raw)),
            }
        )

        print(
            f"    {block_name:12s}  "
            f"|df/dx| = {np.mean(obj_sens_abs):.6e} +- {np.std(obj_sens_abs):.1e}  "
            f"|dRs/dx| = {np.mean(sec_sens_abs):.6e} +- {np.std(sec_sens_abs):.1e}"
        )

    df_summary = pd.DataFrame(summary_rows)

    # ===================================================================
    # STEP 4 -- Specific verifications
    # ===================================================================
    print("\n[4/5] Specific verifications ...")

    # 4a -- Jammer beamforming -> eavesdropper SINR
    jammer_data = df_results[df_results["block"] == "jammer"]
    jammer_dsinr_abs = jammer_data["dsinr_eve_max"].abs()
    jammer_affects_sinr = bool(np.any(jammer_dsinr_abs > 1e-12))
    print(
        f"    4a. Jammer -> Eve SINR:  max|DeltaSINR_eve| = {jammer_dsinr_abs.max():.6e}"
        f"  {'[OK] AFFECTS' if jammer_affects_sinr else '[X] NO EFFECT'}"
    )

    # 4b -- Jammer power -> secrecy rate
    jammer_dsec_abs = jammer_data["dsecrecy"].abs()
    jammer_affects_sec = bool(np.any(jammer_dsec_abs > 1e-12))
    print(
        f"    4b. Jammer -> Secrecy:   max|DeltaR_s|     = {jammer_dsec_abs.max():.6e}"
        f"  {'[OK] AFFECTS' if jammer_affects_sec else '[X] NO EFFECT'}"
    )

    # 4c -- Finite-difference gradients for all blocks
    fd_results = {}
    for block_name in blocks_to_audit:
        sl = block_slices[block_name]
        xb = x_flat[sl]
        g = env.finite_diff_gradient_for_block(xb, sl, solution)
        g_norm = float(np.linalg.norm(g))
        g_max_abs = float(np.max(np.abs(g)))
        g_nonzero = int(np.sum(np.abs(g) > 1e-12))
        g_total = len(g)
        fd_results[block_name] = {
            "norm": g_norm,
            "max_abs": g_max_abs,
            "nonzero": g_nonzero,
            "total": g_total,
        }
        status = "[OK] nonzero" if g_nonzero > 0 else "[X] ALL ZERO"
        print(
            f"    4c. FD gradient ({block_name:10s}):  "
            f"||g||={g_norm:.6e}  max|g_i|={g_max_abs:.6e}  "
            f"nonzero={g_nonzero}/{g_total}  {status}"
        )

    fd_jammer_nonzero = fd_results["jammer"]["nonzero"] > 0

    # ===================================================================
    # STEP 5 -- Determine coupling status
    # ===================================================================
    print("\n[5/5] Coupling assessment ...")

    all_obj_sens = df_summary["obj_sens_mean"].values
    all_sec_sens = df_summary["sec_sens_mean"].values
    max_obj_sens = float(np.max(all_obj_sens)) if len(all_obj_sens) > 0 else 0.0
    max_sec_sens = float(np.max(all_sec_sens)) if len(all_sec_sens) > 0 else 0.0

    inactive_blocks = []
    for i, bn in enumerate(blocks_to_audit):
        obj_sens_ok = (
            all_obj_sens[i] > INACTIVE_THRESHOLD * max(max_obj_sens, 1e-15)
        )
        sec_sens_ok = (
            all_sec_sens[i] > INACTIVE_THRESHOLD * max(max_sec_sens, 1e-15)
        )
        is_inactive = not (obj_sens_ok or sec_sens_ok)
        if is_inactive:
            inactive_blocks.append(bn)
        print(f"    {bn:12s}  {'INACTIVE' if is_inactive else 'ACTIVE'}  "
              f"(obj_sens={all_obj_sens[i]:.6e}, sec_sens={all_sec_sens[i]:.6e})")

    # Conditioning
    active_sens = [s for s in all_obj_sens if s > 0]
    cond_ratio = max(active_sens) / max(min(active_sens), 1e-30) if len(active_sens) >= 2 else float("inf")
    WELL_CONDITIONED_THRESHOLD = 100.0
    well_conditioned = cond_ratio < WELL_CONDITIONED_THRESHOLD
    print(f"    Condition ratio (max/min active):  {cond_ratio:.2f}  "
          f"{'[OK] well-conditioned' if well_conditioned else '[X] ill-conditioned'}")

    # Final verdict
    if len(inactive_blocks) > 0:
        conclusion = "C"
        conclusion_text = (
            "C. Degenerate optimisation problem with inactive variables"
        )
    elif not well_conditioned:
        conclusion = "B"
        conclusion_text = "B. Partially coupled optimisation problem"
    else:
        conclusion = "A"
        conclusion_text = "A. Fully coupled optimisation problem"

    print(f"\n  Verdict: {conclusion_text}")

    # ===================================================================
    # GENERATE OUTPUTS
    # ===================================================================

    # -- Plots --------------------------------------------------------
    print("\nGenerating plots ...")
    _generate_plots(
        df_results,
        blocks_to_audit,
        block_labels,
        output_dir,
    )

    # -- CSV ----------------------------------------------------------
    cm_df = df_summary.rename(columns={
        "block": "block",
        "obj_sens_mean": "objective_sensitivity_mean",
        "obj_sens_std": "objective_sensitivity_std",
        "obj_sens_max": "objective_sensitivity_max",
        "sec_sens_mean": "secrecy_sensitivity_mean",
        "sec_sens_std": "secrecy_sensitivity_std",
        "sec_sens_max": "secrecy_sensitivity_max",
    })
    cm_path = output_dir / "coupling_matrix.csv"
    cm_df[["block", "objective_sensitivity_mean", "objective_sensitivity_std",
           "secrecy_sensitivity_mean", "secrecy_sensitivity_std"]].to_csv(
        cm_path, index=False, float_format="%.6e"
    )
    print(f"  Saved {cm_path}")

    # -- Report -------------------------------------------------------
    report_text = _generate_report(
        config=config,
        bcd_result=bcd_result,
        f_nominal=f_nominal,
        sec_nominal=sec_nominal,
        sinr_eve_max_nom=sinr_eve_max_nom,
        df_results=df_results,
        df_summary=df_summary,
        fd_results=fd_results,
        blocks_to_audit=blocks_to_audit,
        block_labels=block_labels,
        jammer_affects_sinr=jammer_affects_sinr,
        jammer_affects_sec=jammer_affects_sec,
        fd_jammer_nonzero=fd_jammer_nonzero,
        inactive_blocks=inactive_blocks,
        cond_ratio=cond_ratio,
        well_conditioned=well_conditioned,
        conclusion=conclusion,
        conclusion_text=conclusion_text,
    )
    report_path = output_dir / "coupling_audit.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"  Saved {report_path}")

    print("\n[OK] Coupling audit complete.")
    return df_results


# =======================================================================
# PLOTTING
# =======================================================================

def _generate_plots(
    df_results: pd.DataFrame,
    blocks_to_audit: list[str],
    block_labels: dict[str, str],
    output_dir: Path,
) -> None:
    plt = _safe_import_matplotlib()
    if plt is None:
        print("  (matplotlib not available -- skipping plots)")
        return

    colours = {"power": "#1f77b4", "trajectory": "#ff7f0e", "jammer": "#2ca02c"}

    # -- Objective sensitivity ----------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for bn in blocks_to_audit:
        bd = df_results[df_results["block"] == bn]
        pcts = bd["perturbation_pct"].values
        sens = bd["sensitivity_obj"].abs().values
        ax.plot(pcts, sens, "o-", linewidth=2, markersize=8,
                color=colours[bn], label=block_labels[bn])

    ax.set_xlabel("Perturbation (%)", fontsize=12)
    ax.set_ylabel("|Deltaf / Deltax|  (objective sensitivity)", fontsize=12)
    ax.set_title("Objective Sensitivity per Block", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    p = output_dir / "objective_sensitivity.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")

    # -- Secrecy sensitivity ------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for bn in blocks_to_audit:
        bd = df_results[df_results["block"] == bn]
        pcts = bd["perturbation_pct"].values
        sens = bd["sensitivity_secrecy"].abs().values
        ax.plot(pcts, sens, "o-", linewidth=2, markersize=8,
                color=colours[bn], label=block_labels[bn])

    ax.set_xlabel("Perturbation (%)", fontsize=12)
    ax.set_ylabel("|DeltaR_s / Deltax|  (secrecy sensitivity)", fontsize=12)
    ax.set_title("Secrecy Sensitivity per Block", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    p = output_dir / "secrecy_sensitivity.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# =======================================================================
# REPORT
# =======================================================================

def _generate_report(
    config: SCABCDConfig,
    bcd_result,
    f_nominal: float,
    sec_nominal: float,
    sinr_eve_max_nom: float,
    df_results: pd.DataFrame,
    df_summary: pd.DataFrame,
    fd_results: dict,
    blocks_to_audit: list[str],
    block_labels: dict[str, str],
    jammer_affects_sinr: bool,
    jammer_affects_sec: bool,
    fd_jammer_nonzero: bool,
    inactive_blocks: list[str],
    cond_ratio: float,
    well_conditioned: bool,
    conclusion: str,
    conclusion_text: str,
) -> str:
    lines = []
    def L(s=""):
        lines.append(s)

    L("# Coupling Audit -- SCA-BCD")
    L()
    L("## Overview")
    L()
    L(f"- **Objective**:  weighted ISAC objective  f = alpha.R_s/R_s_ref + (1-alpha).U_sense/U_sense_ref  (alpha = {config.alpha})")
    L("- **Secrecy**:  total secrecy rate R_s_total (sum over time slots)")
    L(f"- **Perturbations**:  each block scaled by (1 +/- p) for p in {{1%, 5%, 10%}}")
    L("- **Sensitivity**:  Deltaf / Deltax_norm  and  DeltaR_s / Deltax_norm")
    L(f"- **Config**:  channel_model = {config.channel_model},  jammer_mode = {config.jammer_mode}")
    L()
    L("---")
    L()
    L("## Nominal (Feasible) Solution")
    L()
    L(f"| Metric | Value |")
    L(f"|--------|-------|")
    L(f"| Converged | {bcd_result.converged} ({bcd_result.n_iters} iterations) |")
    L(f"| Objective f | {f_nominal:.6f} |")
    L(f"| Secrecy R_s_total | {sec_nominal:.6f} |")
    L(f"| Max Eve SINR | {sinr_eve_max_nom:.6e} |")
    L()
    L("---")
    L()
    L("## Per-Block Sensitivity")
    L()
    L("### Summary (mean +/- std of |sensitivity|)")
    L()
    L("| Block | |df/dx| (obj) | |dR_s/dx| (sec) | Status |")
    L("|-------|:----:|:----:|:----:|")
    for _, row in df_summary.iterrows():
        bn = row["block"]
        is_inactive = bn in inactive_blocks
        status = "[!] INACTIVE" if is_inactive else "ACTIVE"
        L(f"| {block_labels.get(bn, bn)} | {row['obj_sens_mean']:.6e} +/- {row['obj_sens_std']:.1e} | {row['sec_sens_mean']:.6e} +/- {row['sec_sens_std']:.1e} | {status} |")
    L()
    L("### Raw sensitivity (signed)")
    L()
    L("| Block | Deltaf/Deltax (mean +/- std) | DeltaR_s/Deltax (mean +/- std) |")
    L("|-------|:----:|:----:|")
    for _, row in df_summary.iterrows():
        bn = row["block"]
        L(f"| {block_labels.get(bn, bn)} | {row['obj_sens_raw_mean']:.6e} +/- {row['obj_sens_raw_std']:.1e} | {row['sec_sens_raw_mean']:.6e} +/- {row['sec_sens_raw_std']:.1e} |")
    L()
    L("### Detailed perturbation results")
    L()
    L("| Block | Pert (%) | Deltax_norm | Deltaf | DeltaR_s | df/dx | dR_s/dx |")
    L("|-------|:--------:|:-------:|:---:|:----:|:-----:|:-------:|")
    for _, row in df_results.iterrows():
        L(f"| {row['block']:12s} | {row['perturbation_pct']:+6.1f} | {row['dx_norm']:.6e} | {row['df']:+9.6f} | {row['dsecrecy']:+9.6f} | {row['sensitivity_obj']:+10.3e} | {row['sensitivity_secrecy']:+10.3e} |")
    L()
    L("---")
    L()
    L("## Specific Verification")
    L()
    L("### 1. Jammer beamforming variables change eavesdropper SINR")
    L()
    max_dsinr = float(df_results[df_results["block"] == "jammer"]["dsinr_eve_max"].abs().max())
    L(f"- When jammer variables are perturbed, max |DeltaSINR_eve| = {max_dsinr:.6e}")
    L(f"- **Result**: {'[OK] Jammer variables DO affect Eve SINR' if jammer_affects_sinr else '[X] Jammer variables do NOT affect Eve SINR'}")
    L()
    L("### 2. Jammer power changes secrecy rate")
    L()
    max_dsec = float(df_results[df_results["block"] == "jammer"]["dsecrecy"].abs().max())
    L(f"- When jammer variables are perturbed, max |DeltaR_s| = {max_dsec:.6e}")
    L(f"- **Result**: {'[OK] Jammer power DOES affect secrecy rate' if jammer_affects_sec else '[X] Jammer power does NOT affect secrecy rate'}")
    L()
    L("### 3. Finite-difference gradients for jammer variables are nonzero")
    L()
    fdj = fd_results["jammer"]
    L(f"- ||g_jammer|| = {fdj['norm']:.6e}")
    L(f"- max|g_i| = {fdj['max_abs']:.6e}")
    L(f"- Non-zero entries: {fdj['nonzero']} / {fdj['total']}")
    L(f"- **Result**: {'[OK] FD gradients are nonzero' if fd_jammer_nonzero else '[X] FD gradients are ALL ZERO'}")
    L()

    # Also show FD gradients for all blocks
    L("### FD gradients across all blocks")
    L()
    L("| Block | ||g|| | max|g_i| | nonzero/total |")
    L("|-------|:-----:|:--------:|:-------------:|")
    for bn in blocks_to_audit:
        fd = fd_results[bn]
        L(f"| {block_labels.get(bn, bn)} | {fd['norm']:.6e} | {fd['max_abs']:.6e} | {fd['nonzero']}/{fd['total']} |")
    L()
    L("---")
    L()
    L("## Conditioning Assessment")
    L()
    L(f"- **Condition ratio** (max |df/dx| / min |df/dx| among active blocks): **{cond_ratio:.2f}**")
    L(f"- Threshold: well-conditioned if ratio < {100}")
    L(f"- **Verdict**: {'Well-conditioned' if well_conditioned else 'Ill-conditioned -- sensitivity varies significantly across blocks'}")
    L()
    L("---")
    L()
    L("## Coupling Matrix")
    L()
    L("| Block | Objective Sensitivity (mean +/- std) | Secrecy Sensitivity (mean +/- std) |")
    L("|-------|:-------------------------------:|:-------------------------------:|")
    for _, row in df_summary.iterrows():
        bn = row["block"]
        L(f"| {block_labels.get(bn, bn)} | {row['obj_sens_mean']:.6e} +/- {row['obj_sens_std']:.1e} | {row['sec_sens_mean']:.6e} +/- {row['sec_sens_std']:.1e} |")
    L()
    L("---")
    L()
    L("## Notes")
    L()
    L(f"1. The default configuration uses `jammer_mode = \"mixed\"`, which causes the jammer beamforming")
    L("   variables (v_jammer) to be **ignored** during evaluation -- heuristic beams are designed instead.")
    L(f"   For this audit, `jammer_mode` was set to `\"given\"` so that the optimizer's jammer decisions")
    L("   actually affect the objective.  Under the default configuration, the jammer block would appear")
    L("   **completely inactive** (zero sensitivity, zero gradients, no effect on secrecy or SINR).")
    L()
    L("2. The RIS phase variables (phi_rad) are not included as an optimisation block in the current BCD loop.")
    L("   They remain at their initial value (zeros) throughout and are not optimised.  A full coupling")
    L("   analysis would include an RIS block.")
    L()
    L("---")
    L()
    L("## Conclusion")
    L()
    L(f"**{conclusion_text}**")
    L()

    return "\n".join(lines)


if __name__ == "__main__":
    run_coupling_audit()
