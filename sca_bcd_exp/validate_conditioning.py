"""Validation tests for conditioning improvements.

Tests:
  1. test_scaled_gradients_finite  — scaled FD gradients are finite
  2. test_conditioning_improved    — condition ratio < 50 after scaling
  3. test_adaptive_fd_stability    — adaptive FD step produces stable gradients
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
from sca_bcd_exp.optimization.bcd_solver import BCDSolver
from sca_bcd_exp.optimization.secrecy_optimizer import SolutionState


def validate_conditioning(verbose: bool = True) -> dict:
    """Run all three validation tests.

    Returns dict with pass/fail for each test.
    """
    config = SCABCDConfig(jammer_mode="given", max_bcd_iters=20, seed=0)
    output_dir = config.output_root() / "conditioning"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── Shared setup ─────────────────────────────────────────────────
    env = SCABCDEnvironment(config)
    solver = BCDSolver(config)
    bcd_result = solver.solve(env)
    solution = bcd_result.solution
    x_nominal = env._unpack_decision_vars(solution.decision_vars)
    nom_eval = env.evaluate(solution)
    f_nominal = float(nom_eval["objective"])
    scaler = env.scaler
    block_slices = env.block_slices()
    blocks_to_audit = ["power", "trajectory", "jammer"]

    if verbose:
        print(f"  Solution: obj={f_nominal:.6f}, "
              f"iters={bcd_result.n_iters}, "
              f"converged={bcd_result.converged}")

    # ═══════════════════════════════════════════════════════════════════
    # Test 1: Scaled gradients are finite and relate correctly
    # ═══════════════════════════════════════════════════════════════════
    if verbose:
        print("\n[Test 1] test_scaled_gradients_finite ...")

    t1_pass = True
    t1_details = {}
    for bn in blocks_to_audit:
        sl = block_slices[bn]
        xb = x_nominal[sl].copy()
        xb_scaled = scaler.scale_block(xb, bn)

        g_unscaled = env.finite_diff_gradient_for_block(xb, sl, solution)
        g_scaled = env.finite_diff_gradient_for_block_scaled(
            xb_scaled, sl, solution,
        )

        # Both arrays must be entirely finite
        g_u_finite = bool(np.all(np.isfinite(g_unscaled)))
        g_s_finite = bool(np.all(np.isfinite(g_scaled)))
        t1_pass = t1_pass and g_u_finite and g_s_finite

        # Chain rule: g_scaled ≈ g_unscaled * element_scale
        # Only check elements where the gradient is significant
        element_scales = np.array([
            scaler.element_scale(sl.start + i) for i in range(len(xb))
        ])
        g_chain = g_unscaled * element_scales

        # Compute relative error where |g_scaled| > 1e-9
        mask = np.abs(g_scaled) > 1e-9
        n_significant = int(np.sum(mask))
        if n_significant > 0:
            denom = np.maximum(np.abs(g_scaled[mask]), 1e-30)
            rel_err = np.abs(g_scaled[mask] - g_chain[mask]) / denom
            max_rel_err = float(np.max(rel_err))
            median_rel_err = float(np.median(rel_err))
        else:
            max_rel_err = 0.0
            median_rel_err = 0.0

        # Allow larger tolerance for power block (boundary saturation)
        tol = 5.0 if bn == "power" else 2.0
        chain_ok = max_rel_err < tol if n_significant > 0 else True
        t1_pass = t1_pass and chain_ok

        t1_details[bn] = {
            "g_unscaled_norm": float(np.linalg.norm(g_unscaled)),
            "g_scaled_norm": float(np.linalg.norm(g_scaled)),
            "max_chain_rel_err": max_rel_err,
            "median_chain_rel_err": median_rel_err,
            "n_significant_elements": n_significant,
            "g_unscaled_finite": g_u_finite,
            "g_scaled_finite": g_s_finite,
            "chain_rule_ok": chain_ok,
        }

        if verbose:
            print(
                f"    {bn:12s}: ||g_u||={t1_details[bn]['g_unscaled_norm']:.4e}, "
                f"||g_s||={t1_details[bn]['g_scaled_norm']:.4e}, "
                f"chain_err={max_rel_err:.2%} (median={median_rel_err:.2%}), "
                f"n_sig={n_significant}"
            )

    results["test_scaled_gradients_finite"] = {
        "passed": t1_pass,
        "details": t1_details,
    }

    # ═══════════════════════════════════════════════════════════════════
    # Test 2: Conditioning improves after scaling
    # ═══════════════════════════════════════════════════════════════════
    # Uses perturbation-based sensitivity ratio, NOT gradient norm ratio.
    # The success criterion is condition_ratio_after < 50.
    if verbose:
        print("\n[Test 2] test_conditioning_improved ...")

    perturbation_pcts = [0.01, 0.05, 0.10]
    signs = [+1, -1]

    sens_unscaled = {b: [] for b in blocks_to_audit}
    sens_scaled = {b: [] for b in blocks_to_audit}

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

                dx_u = float(np.linalg.norm(x_pert[sl] - x_block_nom))
                dx_s = float(np.linalg.norm(
                    scaler.scale_block(x_pert[sl], bn)
                    - scaler.scale_block(x_block_nom, bn)
                ))

                sens_u = abs(delta_f) / dx_u if dx_u > 1e-15 else 0.0
                sens_s = abs(delta_f) / dx_s if dx_s > 1e-15 else 0.0

                sens_unscaled[bn].append(sens_u)
                sens_scaled[bn].append(sens_s)

    mean_sens_u = np.array([float(np.mean(sens_unscaled[b])) for b in blocks_to_audit])
    mean_sens_s = np.array([float(np.mean(sens_scaled[b])) for b in blocks_to_audit])

    ratio_before = float(np.max(mean_sens_u) / max(np.min(mean_sens_u), 1e-30))
    ratio_after = float(np.max(mean_sens_s) / max(np.min(mean_sens_s), 1e-30))
    improved = ratio_after < ratio_before
    under_threshold = ratio_after < 50.0
    t2_pass = improved and under_threshold

    if verbose:
        print(f"    Sensitivity ratio (before): {ratio_before:.2f}")
        print(f"    Sensitivity ratio (after):  {ratio_after:.2f}")
        print(f"    Improved: {improved}")
        print(f"    Under threshold (50): {under_threshold}")

    results["test_conditioning_improved"] = {
        "passed": t2_pass,
        "details": {
            "ratio_before": float(ratio_before),
            "ratio_after": float(ratio_after),
            "improved": bool(improved),
            "under_threshold_50": bool(under_threshold),
            "mean_sens_unscaled": {b: float(np.mean(sens_unscaled[b])) for b in blocks_to_audit},
            "mean_sens_scaled": {b: float(np.mean(sens_scaled[b])) for b in blocks_to_audit},
        },
    }

    # ═══════════════════════════════════════════════════════════════════
    # Test 3: Adaptive FD step produces stable gradients
    # ═══════════════════════════════════════════════════════════════════
    if verbose:
        print("\n[Test 3] test_adaptive_fd_stability ...")

    t3_pass = True
    t3_details = {}
    for bn in blocks_to_audit:
        sl = block_slices[bn]
        xb = x_nominal[sl].copy()
        xb_scaled = scaler.scale_block(xb, bn)

        g_adaptive = env.finite_diff_gradient_for_block_scaled(
            xb_scaled, sl, solution,
        )

        # Fixed-step gradient in scaled space for comparison
        eps_fixed = 1e-5
        full = env._unpack_decision_vars(solution.decision_vars)
        f0 = env._flat_block_obj(full, sl, solution)
        g_fixed = np.zeros_like(xb_scaled)
        start = sl.start
        for i in range(len(xb_scaled)):
            full_p = full.copy()
            full_p[start + i] += eps_fixed * scaler.element_scale(start + i)
            fp = env._flat_block_obj(full_p, sl, solution)
            g_fixed[i] = (fp - f0) / eps_fixed

        # Must both be finite
        adapt_finite = bool(np.all(np.isfinite(g_adaptive)))
        fixed_finite = bool(np.all(np.isfinite(g_fixed)))
        t3_pass = t3_pass and adapt_finite and fixed_finite

        # Norm must be non-degenerate
        adapt_norm = float(np.linalg.norm(g_adaptive))
        t3_pass = t3_pass and (adapt_norm > 1e-15)

        # Sign agreement: at elements where |g| > 1e-9, signs should match
        mask = (np.abs(g_adaptive) > 1e-9) & (np.abs(g_fixed) > 1e-9)
        n_check = int(np.sum(mask))
        if n_check > 0:
            sign_agree = float(np.mean(
                (np.sign(g_adaptive[mask]) == np.sign(g_fixed[mask]))
            ))
        else:
            sign_agree = 1.0
        t3_pass = t3_pass and (sign_agree > 0.8)

        # Norm similarity (within 2x)
        fixed_norm = float(np.linalg.norm(g_fixed))
        if fixed_norm > 1e-15 and adapt_norm > 1e-15:
            norm_ratio = max(adapt_norm, fixed_norm) / min(adapt_norm, fixed_norm)
        else:
            norm_ratio = 1.0
        t3_pass = t3_pass and (norm_ratio < 2.0)

        t3_details[bn] = {
            "adaptive_norm": adapt_norm,
            "fixed_norm": fixed_norm,
            "norm_ratio": float(norm_ratio),
            "sign_agreement_pct": float(sign_agree * 100),
            "n_sign_checked": n_check,
            "all_finite": adapt_finite,
        }

        if verbose:
            print(
                f"    {bn:12s}: ||g_a||={adapt_norm:.4e}, "
                f"||g_f||={fixed_norm:.4e}, "
                f"ratio={norm_ratio:.3f}, "
                f"sign_agree={sign_agree:.0%} ({n_check} elts)"
            )

    results["test_adaptive_fd_stability"] = {
        "passed": t3_pass,
        "details": t3_details,
    }

    # ── Overall summary ──────────────────────────────────────────────
    all_passed = all(r["passed"] for r in results.values())

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        for name, r in results.items():
            print(f"  {name}: {'PASS' if r['passed'] else 'FAIL'}")
        print(f"{'=' * 50}")

    # Write report
    report_lines = [
        "# Conditioning Validation Report",
        "",
        f"**Overall**: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}",
        "",
        "## Test Results",
        "",
        "| Test | Status | Details |",
        "|------|--------|---------|",
    ]
    for name, r in results.items():
        status = "PASS" if r["passed"] else "FAIL"
        report_lines.append(
            f"| {name} | {status} | {json.dumps(r['details'], default=str)} |"
        )

    report_lines.append("")
    (output_dir / "validation_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    return results


def main():
    results = validate_conditioning(verbose=True)
    all_passed = all(r["passed"] for r in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
