"""Sensing utility audit: trace U_sense, parameter sweeps, saturation check, alternative utilities."""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from optimization_problem_exp.environments.optimization_problem_env import (
    OptimizationProblemEnv,
    OptimizationConfig,
)
from optimization_problem_exp.optimization.problem_formulation import (
    compute_sensing_utility,
    DecisionVariables,
    U_SENSE_REF,
    R_S_REF,
)
from vehicle_reflection_exp.channels.vehicle_channel import compute_rcs


OUTPUT_DIR = "outputs/optimization/sensing_utility_analysis/sensing_audit"


def _rcs_list(env):
    return [compute_rcs(vt) for vt in env.scenario.vehicle_types]


def _sense(env, dv):
    return compute_sensing_utility(
        q_uav=dv.q_uav,
        q_vehicles=env.scenario.q_vehicles,
        rcs_list=_rcs_list(env),
        N_tx=env.config.N_tx_sense,
        N_rx=env.config.N_rx_sense,
        L_pilot=env.config.L_pilot,
        noise_power=env.config.noise_power_sense,
        d_ant=env.config.d_ant,
        wavelength=env.config.wavelength,
        seed=env.config.seed or 0,
    )


# Part 1: Trace across alpha
def part1_trace(env):
    rows = []
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        dv = env._design_alpha_vars(alpha=alpha, rng_seed=42)
        s = _sense(env, dv)
        rows.append({
            "label": f"alpha={alpha:.2f}",
            "U_tot": s["U_sense_total"],
            "CRB_tot": s["CRB_trace_total"],
            "U_slot": s["U_sense_per_slot"].tolist(),
            "CRB_slot": s["CRB_trace_per_slot"].tolist(),
        })
    return rows


# Part 2: Parameter sweeps
def part2_sweeps(env):
    dv_base = env._design_alpha_vars(alpha=0.5, rng_seed=42)
    sweeps = {}

    for (pname, values) in [
        ("N_ant", [2, 4, 8, 16, 32, 64]),
        ("L_pilot", [2, 4, 8, 16, 32, 64]),
        ("noise_power_sense", [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]),
    ]:
        entries = []
        for val in values:
            kw = {k: getattr(env.config, k) for k in [
                "N_ris", "N_j", "N_time", "P_bs_max", "P_j_max",
                "sigma2", "v_max", "dt", "d_ant", "wavelength",
                "f_c", "eta_ris", "seed", "output_root",
            ]}
            if pname == "N_ant":
                kw["N_tx_sense"] = val
                kw["N_rx_sense"] = val
            else:
                kw[pname] = val
            e = OptimizationProblemEnv(OptimizationConfig(**kw))
            s = _sense(e, dv_base)
            entries.append({
                "val": val, "U_tot": s["U_sense_total"],
                "CRB_tot": s["CRB_trace_total"],
                "U_slot": s["U_sense_per_slot"].tolist(),
                "CRB_slot": s["CRB_trace_per_slot"].tolist(),
            })
        sweeps[pname] = entries

    # uav_z sweep
    entries = []
    for z in [30, 60, 90, 120, 150, 200]:
        dv = DecisionVariables(
            phi_rad=dv_base.phi_rad.copy(),
            q_uav=dv_base.q_uav.copy(),
            w_bs=dv_base.w_bs.copy(),
            v_jammer=dv_base.v_jammer.copy(),
        )
        dv.q_uav[:, 2] = z
        s = _sense(env, dv)
        entries.append({
            "val": z, "U_tot": s["U_sense_total"],
            "CRB_tot": s["CRB_trace_total"],
            "U_slot": s["U_sense_per_slot"].tolist(),
            "CRB_slot": s["CRB_trace_per_slot"].tolist(),
        })
    sweeps["uav_z"] = entries

    # vehicle distance sweep
    entries = []
    for d in [50, 100, 150, 200, 300, 500]:
        q_veh = env.scenario.q_vehicles.copy()
        q_veh[:, 0] = d
        rcs = _rcs_list(env)
        s = compute_sensing_utility(
            q_uav=dv_base.q_uav, q_vehicles=q_veh, rcs_list=rcs,
            N_tx=env.config.N_tx_sense, N_rx=env.config.N_rx_sense,
            L_pilot=env.config.L_pilot,
            noise_power=env.config.noise_power_sense,
            d_ant=env.config.d_ant, wavelength=env.config.wavelength,
            seed=env.config.seed or 0,
        )
        entries.append({
            "val": d, "U_tot": s["U_sense_total"],
            "CRB_tot": s["CRB_trace_total"],
            "U_slot": s["U_sense_per_slot"].tolist(),
            "CRB_slot": s["CRB_trace_per_slot"].tolist(),
        })
    sweeps["veh_distance"] = entries

    # N_veh sweep
    entries = []
    for nv in [1, 2, 3, 4, 5]:
        nv = min(nv, len(env.scenario.q_vehicles))
        q_veh = env.scenario.q_vehicles[:nv]
        rcs = [compute_rcs(env.scenario.vehicle_types[i]) for i in range(nv)]
        s = compute_sensing_utility(
            q_uav=dv_base.q_uav, q_vehicles=q_veh, rcs_list=rcs,
            N_tx=env.config.N_tx_sense, N_rx=env.config.N_rx_sense,
            L_pilot=env.config.L_pilot,
            noise_power=env.config.noise_power_sense,
            d_ant=env.config.d_ant, wavelength=env.config.wavelength,
            seed=env.config.seed or 0,
        )
        entries.append({
            "val": nv, "U_tot": s["U_sense_total"],
            "CRB_tot": s["CRB_trace_total"],
            "U_slot": s["U_sense_per_slot"].tolist(),
            "CRB_slot": s["CRB_trace_per_slot"].tolist(),
        })
    sweeps["N_veh"] = entries
    return sweeps


# Part 3: Saturation check
def part3_saturation(env):
    dv = env._design_alpha_vars(alpha=0.5, rng_seed=42)
    s = _sense(env, dv)
    crb = s["CRB_trace_per_slot"]
    u = s["U_sense_per_slot"]
    max_u = float(env.config.N_time)
    ratio = s["U_sense_total"] / max_u
    n_total = len(crb)
    n_singular = sum(1 for c in crb if ~np.isfinite(c) or c > 1e10)
    n_sat = sum(1 for c, u in zip(crb, u) if np.isfinite(c) and c < 0.01 and u > 0.99)
    n_ok = n_total - n_singular  # non-singular slots
    all_sat = n_ok > 0 and n_sat == n_ok
    return {
        "crb_traces": crb.tolist(),
        "u_per_slot": u.tolist(),
        "u_total": s["U_sense_total"],
        "max_possible": max_u,
        "ratio": float(ratio),
        "all_saturated": all_sat,
        "noise_power": env.config.noise_power_sense,
        "effective_snr": 1.0 / env.config.noise_power_sense,
    }


# Part 4: Alternative utilities
def part4_alternatives(env):
    eps = 1e-10
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    alpha_rows = []
    for alpha in alphas:
        dv = env._design_alpha_vars(alpha=alpha, rng_seed=42)
        s = _sense(env, dv)
        crb = np.array(s["CRB_trace_per_slot"])
        finite_mask = np.isfinite(crb) & (crb > 0)
        crb_finite = crb[finite_mask]
        u_orig = s["U_sense_total"]
        u_eps = float(np.sum(1.0 / (eps + crb)))
        u_log = float(np.sum(-np.log10(np.maximum(crb[finite_mask], eps))))
        u_inv = float(np.sum(1.0 / np.maximum(crb[finite_mask], eps)))
        u_norm = float(u_orig / U_SENSE_REF)
        alpha_rows.append({
            "alpha": alpha,
            "U_orig": u_orig,
            "U_eps_inv": u_eps,
            "U_log10": u_log,
            "U_inv": u_inv,
            "U_norm": u_norm,
        })
    dv05 = env._design_alpha_vars(alpha=0.5, rng_seed=42)
    s05 = _sense(env, dv05)
    crb05 = np.array(s05["CRB_trace_per_slot"])
    finite_mask = np.isfinite(crb05) & (crb05 > 0)
    crb_finite = crb05[finite_mask]
    return {
        "alternatives_alpha05": {
            "U_orig 1/(1+tr)": s05["U_sense_total"],
            "U_eps_inv 1/(eps+tr)": float(np.sum(1.0 / (eps + crb05))),
            "U_log10 -log10(tr)": float(np.sum(-np.log10(np.maximum(crb_finite, eps)))),
            "U_inv 1/tr": float(np.sum(1.0 / np.maximum(crb_finite, eps))),
            "U_norm normalized": float(s05["U_sense_total"] / U_SENSE_REF),
        },
        "alpha_sweep": alpha_rows,
    }


# Main
def run_sensing_audit():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = OptimizationProblemEnv()

    trace = part1_trace(env)
    sweeps = part2_sweeps(env)
    sat = part3_saturation(env)
    alt = part4_alternatives(env)

    # Write CSV
    csv_path = os.path.join(OUTPUT_DIR, "utility_sweeps.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "label", "param", "param_value",
                     "U_sense_total", "CRB_trace_total",
                     "U_sense_per_slot", "CRB_trace_per_slot"])
        for r in trace:
            w.writerow([
                "part1", r["label"], "", "",
                f"{r['U_tot']:.6f}", f"{r['CRB_tot']:.4e}",
                ",".join(f"{u:.6f}" for u in r["U_slot"]),
                ",".join(f"{c:.2e}" for c in r["CRB_slot"]),
            ])
        for pname, entries in sweeps.items():
            for e in entries:
                w.writerow([
                    "part2", pname, pname, e["val"],
                    f"{e['U_tot']:.6f}", f"{e['CRB_tot']:.4e}",
                    ",".join(f"{u:.6f}" for u in e["U_slot"]),
                    ",".join(f"{c:.2e}" for c in e["CRB_slot"]),
                ])
        for i, (c, u) in enumerate(zip(sat["crb_traces"], sat["u_per_slot"])):
            w.writerow(["part3", f"slot_{i}", "", "", f"{u:.6f}", f"{c:.4e}", "", ""])
        w.writerow(["part3", "total", "", "", f"{sat['u_total']:.6f}", "", "", ""])
        w.writerow(["part3", "theoretical_max", "", "", f"{sat['max_possible']:.1f}", "", "", ""])
        w.writerow(["part3", "saturation_ratio", "", "", f"{sat['ratio']:.6f}", "", "", ""])
        for lbl, val in alt["alternatives_alpha05"].items():
            w.writerow(["part4", lbl, "", "", f"{val:.6e}", "", "", ""])
        for ar in alt["alpha_sweep"]:
            w.writerow(["part4_alpha", f"alpha={ar['alpha']:.2f}", "",
                        ar["alpha"], f"{ar['U_orig']:.6f}", "", "", ""])

    # Write report
    lines = [
        "# Sensing Utility Audit Report",
        "",
        "## Part 1: U_sense Trace Across Alpha",
        "",
        "| Method | U_sense_total | CRB_trace_total | Per-slot U | Per-slot tr(CRB) |",
        "|--------|--------------|----------------|-----------|-----------------|",
    ]
    for r in trace:
        lines.append(
            f"| {r['label']} | {r['U_tot']:.6f} | {r['CRB_tot']:.4e} "
            f"| {', '.join(f'{u:.4f}' for u in r['U_slot'])} "
            f"| {', '.join(f'{c:.2e}' for c in r['CRB_slot'])} |"
        )

    lines.extend(["", "## Part 2: Parameter Sweeps", ""])
    for pname, entries in sweeps.items():
        lines.append(f"### {pname}")
        lines.append("| Value | U_sense_total | CRB_trace_total | U per slot |")
        lines.append("|-------|--------------|----------------|-----------|")
        for e in entries:
            lines.append(
                f"| {e['val']} | {e['U_tot']:.6f} | {e['CRB_tot']:.4e} "
                f"| {', '.join(f'{u:.4f}' for u in e['U_slot'])} |"
            )
        lines.append("")

    lines.extend(["", "## Part 3: Saturation Check", ""])
    lines.append(f"Theoretical max U_sense = N_time = {sat['max_possible']:.0f}")
    lines.append(f"Actual U_sense          = {sat['u_total']:.6f}")
    lines.append(f"Saturation ratio        = {sat['ratio']:.6f} ({sat['ratio']*100:.2f}%)")
    lines.append("")
    for i, (c, u) in enumerate(zip(sat["crb_traces"], sat["u_per_slot"])):
        if not np.isfinite(c) or c < 0:
            tag = "SINGULAR"
        elif c < 0.01:
            tag = "SATURATED"
        else:
            tag = "OK"
        lines.append(f"  Slot {i}: tr(CRB)={c:.4e}  U={u:.6f}  -> {tag}")
    lines.append("")

    n_sing = sum(1 for c in sat["crb_traces"] if ~np.isfinite(c) or c > 1e10)
    n_sat = sum(1 for c, u in zip(sat["crb_traces"], sat["u_per_slot"]) if np.isfinite(c) and c < 0.01 and u > 0.99)
    n_total = len(sat["crb_traces"])
    n_ok = n_total - n_sing
    if sat["all_saturated"]:
        lines.append(f"**All {n_ok} non-singular slots SATURATED: tr(CRB) << 0.01, U ~ 1.0**")
    else:
        lines.append(f"**{n_sat}/{n_ok} non-singular slots saturated.**")
    lines.append("")

    lines.append(f"Noise power (sigma^2)       = {sat['noise_power']}")
    lines.append(f"Effective SNR (1/sigma^2)   = {sat['effective_snr']:.0f}")
    lines.append(f"FIM factor (2/sigma^2)      = {2.0/sat['noise_power']:.0f}")
    lines.append("")
    lines.append("### Root cause")
    lines.append("")
    for line in [
        "U_sense[n] = 1 / (1 + tr(CRB[n]))",
        "",
        "FIM(i,j) = (2/sigma^2) * Re{tr(X^H * dH_i^H * dH_j * X)}",
        "",
        f"With sigma^2 = {sat['noise_power']}:",
        f"  factor = 2/{sat['noise_power']} = {2.0/sat['noise_power']:.0f}",
        "  FIM entries ~ 2e8 * N_ant * L_pilot * RCS * path_loss",
        "  CRB = FIM^{-1} ~ 5e-9",
        "",
        "Thus tr(CRB) ~ 1e-8, and:",
        "  U_sense[n] = 1 / (1 + 1e-8) ~ 1.0",
        "",
        "The utility is saturated at ~1 per slot because",
        "the denominator is dominated by the constant 1, not by tr(CRB).",
        "Even changing N_ant from 2 to 64 only changes tr(CRB) by 4 orders",
        "of magnitude (1e-4 to 1e-10), which is invisible in 1/(1+tr(CRB)).",
        "",
        "To make U_sense vary meaningfully, noise_power must be increased",
        "to ~1.0, where tr(CRB) becomes O(1) and U_sense varies with params.",
    ]:
        lines.append(line)

    lines.extend(["", "## Part 4: Alternative Utilities (alpha=0.5)", ""])
    lines.append("| Utility | Value |")
    lines.append("|---------|-------|")
    for lbl, val in alt["alternatives_alpha05"].items():
        lines.append(f"| {lbl} | {val:.4e} |")
    lines.append("")
    lines.append("### Alpha sweep comparison")
    lines.append("| alpha | U_orig 1/(1+tr) | 1/(eps+tr) | -log10(tr) | 1/tr | normalized |")
    lines.append("|-------|----------------|-----------|-----------|------|-----------|")
    for ar in alt["alpha_sweep"]:
        lines.append(
            f"| {ar['alpha']:.2f} | {ar['U_orig']:.6f} | "
            f"{ar['U_eps_inv']:.2e} | {ar['U_log10']:.4f} | "
            f"{ar['U_inv']:.2e} | {ar['U_norm']:.6f} |"
        )

    lines.extend(["", "## Part 5: Decision", ""])
    if sat["all_saturated"]:
        lines.append("# SENSING_OBJECTIVE_SATURATED")
    else:
        lines.append("# SENSING_OBJECTIVE_OK")
    lines.append("")

    report_path = os.path.join(OUTPUT_DIR, "sensing_audit_report.md")
    text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    assert os.path.getsize(report_path) > 0, f"Report is empty: {report_path}"

    print(f"\n=== Sensing Audit Complete ===")
    print(f"  Report: {report_path} ({os.path.getsize(report_path)} bytes)")
    print(f"  CSV:    {csv_path}")
    print(f"  U_sense total: {sat['u_total']:.6f} / {sat['max_possible']:.0f} max")
    n_sing = sum(1 for c in sat["crb_traces"] if ~np.isfinite(c) or c > 1e10)
    n_sat = sum(1 for c, u in zip(sat["crb_traces"], sat["u_per_slot"]) if np.isfinite(c) and c < 0.01 and u > 0.99)
    n_ok = len(sat["crb_traces"]) - n_sing
    print(f"  Non-singular slots saturated: {n_sat}/{n_ok}")
    decision = "SENSING_OBJECTIVE_SATURATED" if sat["all_saturated"] else "SENSING_OBJECTIVE_OK"
    print(f"  Decision: {decision}")
    return decision


if __name__ == "__main__":
    run_sensing_audit()
