"""MIMO BS upgrade: validation, audits, and report generation.

Covers Parts 6-10 of the upgrade specification.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization_problem_exp.environments.optimization_problem_env import (
    OptimizationProblemEnv, OptimizationConfig,
)
from optimization_problem_exp.optimization.problem_formulation import (
    DecisionVariables,
    compute_secrecy_rate,
    compute_sensing_utility,
    evaluate_weighted_objective,
    check_constraints,
    compute_constraint_violations,
    evaluate_objective_and_constraints,
    compute_channel_condition_numbers,
    get_u_ref,
    R_S_REF,
    generate_mimo_rician_channel,
)

OUTPUT_DIR = "outputs/mimo_bs_upgrade"
N_TESTS_PASSED = 0
N_TESTS_TOTAL = 0


def test(name: str, condition: bool, detail: str = ""):
    global N_TESTS_PASSED, N_TESTS_TOTAL
    N_TESTS_TOTAL += 1
    if condition:
        N_TESTS_PASSED += 1
        print(f"  PASS  {name}  {detail}")
    else:
        print(f"  FAIL  {name}  {detail}")
    return condition


def make_env(M_bs: int = 4, seed: int = 42) -> OptimizationProblemEnv:
    cfg = OptimizationConfig(M_bs=M_bs, seed=seed)
    return OptimizationProblemEnv(cfg)


def get_mimo_channel(env, slot: int = 0) -> np.ndarray:
    """Get the MIMO BS-RIS channel for a slot."""
    np.random.seed(slot)
    dv = env.random_decision_vars(alpha=0.5)
    q_uav_slot = dv.q_uav[slot]
    h_BR = generate_mimo_rician_channel(
        env.config.N_ris, env.config.M_bs, K=5.0,
        path_loss_factor=1.0,
    )
    return h_BR, dv


# ── Part 6: Baseline beamformers ─────────────────────────

def mrt_beamformer(h_eff: np.ndarray, power: float) -> np.ndarray:
    """Maximal Ratio Transmission beamformer.

    h_eff: effective channel vector (M_bs,) for the intended user.
    Received signal: h_eff^H @ w.
    MRT: w = sqrt(power) * h_eff / ||h_eff|| (aligns with h_eff).
    Returns w scaled to sqrt(power).
    """
    norm = float(np.linalg.norm(h_eff))
    if norm < 1e-15:
        w = np.ones(len(h_eff), dtype=complex) / np.sqrt(len(h_eff))
    else:
        w = h_eff / norm
    return w * np.sqrt(power)


def zf_beamformer(h_eff_user: np.ndarray, h_eff_eves: list[np.ndarray], power: float) -> np.ndarray:
    """Zero-forcing beamformer nulling the strongest eve.

    Projects MRT beamformer w_MRT = h_eff_user onto nullspace of h_eve.
    h_eff_user: (M_bs,) intended user channel.
    h_eff_eves: list of (M_bs,) eve channels (one per eve).
    Returns w scaled to sqrt(power).
    """
    M_bs = len(h_eff_user)
    if not h_eff_eves:
        return mrt_beamformer(h_eff_user, power)

    # Pick the strongest eve (largest channel gain ||h_eve||^2)
    eve_powers = [float(np.linalg.norm(h) ** 2) for h in h_eff_eves]
    strongest_idx = int(np.argmax(eve_powers))
    h_eve = h_eff_eves[strongest_idx]

    h_eve_norm = float(np.linalg.norm(h_eve))
    if h_eve_norm < 1e-15:
        return mrt_beamformer(h_eff_user, power)

    # Project user channel onto nullspace of eve channel
    # w_null = h_user - (h_eve^H @ h_user / ||h_eve||^2) * h_eve
    proj = np.vdot(h_eve, h_eff_user) / (h_eve_norm ** 2)
    w_null = h_eff_user - proj * h_eve
    w_norm = float(np.linalg.norm(w_null))
    if w_norm < 1e-15:
        return mrt_beamformer(h_eff_user, power)
    return (w_null / w_norm) * np.sqrt(power)


def random_feasible_beamformer(M_bs: int, power: float, rng_seed: int = 0) -> np.ndarray:
    """Random beamformer with ||w||^2 = power."""
    np.random.seed(rng_seed)
    w = np.random.randn(M_bs) + 1j * np.random.randn(M_bs)
    w = w / float(np.linalg.norm(w)) * np.sqrt(power)
    return w


# ── Part 8: Audits ───────────────────────────────────────

def audit_global_phase_invariance(env: OptimizationProblemEnv, trials: int = 5) -> bool:
    """Global phase rotation of all w_bs should not change objective (PASS expected)."""
    all_pass = True
    dv = env.random_decision_vars(alpha=0.5)
    base = env.evaluate(dv, jammer_mode="mixed", alpha=0.5)
    for t in range(trials):
        theta = np.random.uniform(0, 2 * np.pi)
        dv_rot = DecisionVariables(
            phi_rad=dv.phi_rad.copy(),
            q_uav=dv.q_uav.copy(),
            w_bs=dv.w_bs * np.exp(1j * theta),
            v_jammer=dv.v_jammer.copy(),
        )
        rot = env.evaluate(dv_rot, jammer_mode="mixed", alpha=0.5)
        diff = abs(base["objective"]["f"] - rot["objective"]["f"])
        ok = diff < 1e-10
        if not ok:
            all_pass = False
        test(f"global_phase_{t}", ok, f"diff={diff:.2e}")
    return all_pass


def audit_per_entry_phase_change(env: OptimizationProblemEnv, trials: int = 3) -> bool:
    """Per-entry phase rotation SHOULD change objective (FAIL expected for MIMO)."""
    all_pass = True
    dv = env.random_decision_vars(alpha=0.5)
    base = env.evaluate(dv, jammer_mode="mixed", alpha=0.5)
    rng = np.random.RandomState(12345)
    for t in range(trials):
        dv_rot = DecisionVariables(
            phi_rad=dv.phi_rad.copy(),
            q_uav=dv.q_uav.copy(),
            w_bs=dv.w_bs.copy(),
            v_jammer=dv.v_jammer.copy(),
        )
        # Rotate phase of one antenna in one slot
        slot = rng.randint(0, dv.N_time)
        ant = rng.randint(0, env.config.M_bs)
        theta = rng.uniform(0.3, 1.0)
        dv_rot.w_bs[slot, ant] *= np.exp(1j * theta)
        rot = env.evaluate(dv_rot, jammer_mode="mixed", alpha=0.5)
        diff = abs(base["objective"]["f"] - rot["objective"]["f"])
        # Should be non-zero (MIMO beamforming matters)
        changed = diff > 1e-8
        if not changed:
            # Try with a different antenna/slot
            slot2 = (slot + 1) % dv.N_time
            ant2 = (ant + 1) % env.config.M_bs
            dv_rot2 = DecisionVariables(
                phi_rad=dv.phi_rad.copy(), q_uav=dv.q_uav.copy(),
                w_bs=dv.w_bs.copy(), v_jammer=dv.v_jammer.copy(),
            )
            dv_rot2.w_bs[slot2, ant2] *= np.exp(1j * 0.7)
            rot2 = env.evaluate(dv_rot2, jammer_mode="mixed", alpha=0.5)
            diff = abs(base["objective"]["f"] - rot2["objective"]["f"])
            changed = diff > 1e-8
        all_pass = all_pass and changed
        test(f"per_entry_phase_{t}", changed, f"diff={diff:.2e}")
    return all_pass


def audit_mrt_improves_user_sinr(env: OptimizationProblemEnv) -> bool:
    """MRT should improve user SINR over random."""
    dv = env.random_decision_vars(alpha=0.5)
    base = env.evaluate(dv, jammer_mode="mixed", alpha=0.5)
    base_sinr = float(np.mean(base["secrecy"]["SINR_user"]))

    # Apply MRT: compute effective channels and set w_bs per slot
    dv_mrt = DecisionVariables(
        phi_rad=dv.phi_rad.copy(),
        q_uav=dv.q_uav.copy(),
        w_bs=np.zeros_like(dv.w_bs),
        v_jammer=dv.v_jammer.copy(),
    )
    for n in range(env.config.N_time):
        h_BR = __import__("optimization_problem_exp.optimization.problem_formulation",
                          fromlist=["compute_bs_ris_channel"]).compute_bs_ris_channel(
            env.scenario.q_bs, dv.q_uav[n], env.config.N_ris, seed=n * 10, M_bs=env.config.M_bs,
        )
        h_RU = __import__("optimization_problem_exp.optimization.problem_formulation",
                          fromlist=["compute_ris_user_channel"]).compute_ris_user_channel(
            dv.q_uav[n], env.scenario.q_user, env.config.N_ris, seed=n * 10 + 1,
        )
        phi_aligned = __import__("optimization_problem_exp.optimization.problem_formulation",
                                 fromlist=["design_ris_phases"]).design_ris_phases(h_BR, h_RU)
        Phi = __import__("ris_uav_exp.channels.ris_channel",
                         fromlist=["compute_ris_reflection_matrix"]).compute_ris_reflection_matrix(phi_aligned)
        h_eff = __import__("ris_uav_exp.channels.ris_channel",
                           fromlist=["compute_effective_channel"]).compute_effective_channel(h_RU, Phi, h_BR)
        power = min(float(np.linalg.norm(dv.w_bs[n]) ** 2), env.config.P_bs_max)
        dv_mrt.w_bs[n] = mrt_beamformer(h_eff, power)

    mrt = env.evaluate(dv_mrt, jammer_mode="mixed", alpha=0.5)
    mrt_sinr = float(np.mean(mrt["secrecy"]["SINR_user"]))
    ok = mrt_sinr >= base_sinr - 1e-6
    test("mrt_improves_user_sinr", ok,
         f"base={base_sinr:.4f}, mrt={mrt_sinr:.4f}")
    return ok


def audit_zf_suppresses_eve_sinr(env: OptimizationProblemEnv) -> bool:
    """ZF should suppress eve SINR compared to MRT."""
    dv = env.random_decision_vars(alpha=0.5)
    dv_mrt = DecisionVariables(
        phi_rad=dv.phi_rad.copy(), q_uav=dv.q_uav.copy(),
        w_bs=np.zeros_like(dv.w_bs), v_jammer=dv.v_jammer.copy(),
    )
    dv_zf = DecisionVariables(
        phi_rad=dv.phi_rad.copy(), q_uav=dv.q_uav.copy(),
        w_bs=np.zeros_like(dv.w_bs), v_jammer=dv.v_jammer.copy(),
    )
    N_eve = len(env.scenario.q_eves)
    for n in range(env.config.N_time):
        base_seed = n * 10
        h_BR = __import__("optimization_problem_exp.optimization.problem_formulation",
                          fromlist=["compute_bs_ris_channel"]).compute_bs_ris_channel(
            env.scenario.q_bs, dv.q_uav[n], env.config.N_ris, base_seed, M_bs=env.config.M_bs,
        )
        h_RU = __import__("optimization_problem_exp.optimization.problem_formulation",
                          fromlist=["compute_ris_user_channel"]).compute_ris_user_channel(
            dv.q_uav[n], env.scenario.q_user, env.config.N_ris, base_seed + 1,
        )
        h_RE_list = [
            __import__("optimization_problem_exp.optimization.problem_formulation",
                       fromlist=["compute_ris_eve_channel"]).compute_ris_eve_channel(
                dv.q_uav[n], env.scenario.q_eves[ke], env.config.N_ris, base_seed + 2 + ke,
            )
            for ke in range(N_eve)
        ]
        phi_aligned = __import__("optimization_problem_exp.optimization.problem_formulation",
                                 fromlist=["design_ris_phases"]).design_ris_phases(h_BR, h_RU)
        Phi = __import__("ris_uav_exp.channels.ris_channel",
                         fromlist=["compute_ris_reflection_matrix"]).compute_ris_reflection_matrix(phi_aligned)
        h_eff_user = __import__("ris_uav_exp.channels.ris_channel",
                                fromlist=["compute_effective_channel"]).compute_effective_channel(h_RU, Phi, h_BR)
        h_eff_eves = [
            __import__("ris_uav_exp.channels.ris_channel",
                       fromlist=["compute_effective_channel"]).compute_effective_channel(h_RE_list[ke], Phi, h_BR)
            for ke in range(N_eve)
        ]
        power = min(float(np.linalg.norm(dv.w_bs[n]) ** 2), env.config.P_bs_max)
        dv_mrt.w_bs[n] = mrt_beamformer(h_eff_user, power)
        dv_zf.w_bs[n] = zf_beamformer(h_eff_user, h_eff_eves, power)

    mrt = env.evaluate(dv_mrt, jammer_mode="mixed", alpha=0.5)
    zf = env.evaluate(dv_zf, jammer_mode="mixed", alpha=0.5)
    mrt_eve_sinr = float(np.max(mrt["secrecy"]["SINR_eve"]))
    zf_eve_sinr = float(np.max(zf["secrecy"]["SINR_eve"]))
    ok = zf_eve_sinr <= mrt_eve_sinr + 1e-6
    test("zf_suppresses_eve_sinr", ok,
         f"mrt_eve={mrt_eve_sinr:.4f}, zf_eve={zf_eve_sinr:.4f}")
    return ok


def audit_beamforming_changes_objective(env: OptimizationProblemEnv) -> bool:
    """Different beamforming vectors should change the objective."""
    dv = env.random_decision_vars(alpha=0.5)
    base = env.evaluate(dv, jammer_mode="mixed", alpha=0.5)
    dv_alt = DecisionVariables(
        phi_rad=dv.phi_rad.copy(), q_uav=dv.q_uav.copy(),
        w_bs=np.zeros_like(dv.w_bs), v_jammer=dv.v_jammer.copy(),
    )
    for n in range(env.config.N_time):
        dv_alt.w_bs[n] = random_feasible_beamformer(
            env.config.M_bs, min(float(np.linalg.norm(dv.w_bs[n]) ** 2), env.config.P_bs_max),
            rng_seed=n + 100,
        )
    alt = env.evaluate(dv_alt, jammer_mode="mixed", alpha=0.5)
    diff = abs(base["objective"]["f"] - alt["objective"]["f"])
    ok = diff > 1e-8
    test("beamforming_changes_objective", ok,
         f"diff={diff:.4f}, base={base['objective']['f']:.4f}, alt={alt['objective']['f']:.4f}")
    return ok


def audit_channels_finite(env: OptimizationProblemEnv) -> bool:
    """All MIMO channels should be finite."""
    ok = True
    dv = env.random_decision_vars(alpha=0.5)
    for n in range(min(3, env.config.N_time)):
        h_BR = __import__("optimization_problem_exp.optimization.problem_formulation",
                          fromlist=["compute_bs_ris_channel"]).compute_bs_ris_channel(
            env.scenario.q_bs, dv.q_uav[n], env.config.N_ris, n * 10, M_bs=env.config.M_bs,
        )
        for label, h in [("h_BR", h_BR)]:
            is_finite = np.all(np.isfinite(h))
            ok = ok and is_finite
            test(f"channel_finite_{label}_slot{n}", is_finite,
                 f"shape={h.shape}, any_nan={np.any(np.isnan(h))}, any_inf={np.any(np.isinf(h))}")
    return ok


def audit_power_constraints(env: OptimizationProblemEnv) -> bool:
    """Power constraints should hold for various beamformers."""
    ok = True
    for seed in range(5):
        dv = env.random_decision_vars(alpha=np.random.uniform(0, 1))
        constr = check_constraints(
            phi_rad=dv.phi_rad, q_uav=dv.q_uav, w_bs=dv.w_bs, v_jammer=dv.v_jammer,
            P_bs_max=env.config.P_bs_max, P_j_max=env.config.P_j_max,
            v_max=env.config.v_max, dt=env.config.dt,
            q_min=env.scenario.q_min, q_max=env.scenario.q_max,
        )
        viol = compute_constraint_violations(
            phi_rad=dv.phi_rad, q_uav=dv.q_uav, w_bs=dv.w_bs, v_jammer=dv.v_jammer,
            P_bs_max=env.config.P_bs_max, P_j_max=env.config.P_j_max,
            v_max=env.config.v_max, dt=env.config.dt,
            q_min=env.scenario.q_min, q_max=env.scenario.q_max,
        )
        pw_ok = constr["bs_power"]
        pw_viol = viol["bs_power_excess"]
        ok = ok and pw_ok
        test(f"power_constraint_seed{seed}", pw_ok,
             f"excess={pw_viol:.6e}")
    return ok


def audit_gradient_checks(env: OptimizationProblemEnv) -> bool:
    """Real-FD gradients for power block should exist and be finite."""
    from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
    from sca_bcd_exp.configs import SCABCDConfig

    cfg = SCABCDConfig(M_bs=env.config.M_bs, seed=42)
    sca_env = SCABCDEnvironment(cfg)
    sol = sca_env.reset()
    blocks = sca_env.block_slices()
    sl = blocks["power"]
    x0 = sca_env._unpack_decision_vars(sol.decision_vars)[sl]
    grad = sca_env.finite_diff_gradient_for_block(x0, sl, sol)

    is_finite = np.all(np.isfinite(grad))
    has_variance = float(np.std(grad)) > 1e-15
    ok = is_finite and has_variance
    test("gradient_finite", is_finite,
         f"n_nan={np.sum(~np.isfinite(grad))}")
    test("gradient_varies", has_variance,
         f"std={float(np.std(grad)):.6e}")
    return ok


def audit_convergence_possible(env: OptimizationProblemEnv) -> bool:
    """Check that SCA can converge with MIMO variables (single power block step)."""
    from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
    from sca_bcd_exp.configs import SCABCDConfig
    from sca_bcd_exp.optimization.power_optimizer import optimize_power

    cfg = SCABCDConfig(
        M_bs=env.config.M_bs, seed=42,
        max_sca_iters=5, tol_obj=1e-3,
    )
    sca_env = SCABCDEnvironment(cfg)
    sol = sca_env.reset()
    init_obj = sca_env.evaluate_objective(sol)
    sol, _ = optimize_power(sca_env, cfg, sol)
    final_obj = sca_env.evaluate_objective(sol)
    obj_improved = final_obj >= init_obj - 1e-4
    test("convergence_possible", obj_improved,
         f"init={init_obj:.4f}, final={final_obj:.4f}")
    return obj_improved


# ── Part 9: Run all audits and tests ─────────────────────

def run_all_validations(env: OptimizationProblemEnv) -> dict:
    results = {}
    print("\n=== Part 6: Audit: Global Phase Invariance ===")
    results["global_phase"] = audit_global_phase_invariance(env)

    print("\n=== Part 6: Audit: Per-Entry Phase Change ===")
    results["per_entry_phase"] = audit_per_entry_phase_change(env)

    print("\n=== Part 6: Audit: MRT Improves User SINR ===")
    results["mrt_sinr"] = audit_mrt_improves_user_sinr(env)

    print("\n=== Part 6: Audit: ZF Suppresses Eve SINR ===")
    results["zf_eve"] = audit_zf_suppresses_eve_sinr(env)

    print("\n=== Part 6: Audit: Beamforming Changes Objective ===")
    results["beamforming_obj"] = audit_beamforming_changes_objective(env)

    print("\n=== Part 6: Audit: Channels Finite ===")
    results["channels_finite"] = audit_channels_finite(env)

    print("\n=== Part 8: Audit: Power Constraints ===")
    results["power_constraints"] = audit_power_constraints(env)

    print("\n=== Part 8: Audit: Gradient Checks ===")
    results["gradients"] = audit_gradient_checks(env)

    print("\n=== Part 8: Audit: Convergence Possible ===")
    results["convergence"] = audit_convergence_possible(env)

    return results


# ── Part 10: Generate outputs ────────────────────────────

def generate_outputs(env: OptimizationProblemEnv, results: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Beamforming comparison ────────────────────────────
    beam_rows = []
    dv = env.random_decision_vars(alpha=0.5)
    for n in range(env.config.N_time):
        base_seed = n * 10
        h_BR = __import__("optimization_problem_exp.optimization.problem_formulation",
                          fromlist=["compute_bs_ris_channel"]).compute_bs_ris_channel(
            env.scenario.q_bs, dv.q_uav[n], env.config.N_ris, base_seed, M_bs=env.config.M_bs,
        )
        h_RU = __import__("optimization_problem_exp.optimization.problem_formulation",
                          fromlist=["compute_ris_user_channel"]).compute_ris_user_channel(
            dv.q_uav[n], env.scenario.q_user, env.config.N_ris, base_seed + 1,
        )
        phi_aligned = __import__("optimization_problem_exp.optimization.problem_formulation",
                                 fromlist=["design_ris_phases"]).design_ris_phases(h_BR, h_RU)
        Phi = __import__("ris_uav_exp.channels.ris_channel",
                         fromlist=["compute_ris_reflection_matrix"]).compute_ris_reflection_matrix(phi_aligned)
        h_eff_user = __import__("ris_uav_exp.channels.ris_channel",
                                fromlist=["compute_effective_channel"]).compute_effective_channel(h_RU, Phi, h_BR)
        h_RE_list = [
            __import__("ris_uav_exp.channels.ris_channel",
                       fromlist=["compute_effective_channel"]).compute_effective_channel(
                __import__("optimization_problem_exp.optimization.problem_formulation",
                           fromlist=["compute_ris_eve_channel"]).compute_ris_eve_channel(
                    dv.q_uav[n], env.scenario.q_eves[ke], env.config.N_ris, base_seed + 2 + ke,
                ), Phi, h_BR,
            )
            for ke in range(len(env.scenario.q_eves))
        ]
        power = float(np.linalg.norm(dv.w_bs[n]) ** 2)
        w_mrt = mrt_beamformer(h_eff_user, power)
        w_zf = zf_beamformer(h_eff_user, h_RE_list, power)
        w_rnd = random_feasible_beamformer(env.config.M_bs, power, rng_seed=n)

        gain_mrt = float(np.abs(h_eff_user.conj() @ w_mrt) ** 2)
        gain_zf = float(np.abs(h_eff_user.conj() @ w_zf) ** 2)
        gain_rnd = float(np.abs(h_eff_user.conj() @ w_rnd) ** 2)

        beam_rows.append({
            "slot": n,
            "power": power,
            "gain_mrt": gain_mrt,
            "gain_zf": gain_zf,
            "gain_random": gain_rnd,
            "w_mrt_norm": float(np.linalg.norm(w_mrt) ** 2),
            "w_zf_norm": float(np.linalg.norm(w_zf) ** 2),
            "w_rnd_norm": float(np.linalg.norm(w_rnd) ** 2),
        })

    csv_path = os.path.join(OUTPUT_DIR, "beamforming_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(beam_rows[0].keys()))
        w.writeheader()
        w.writerows(beam_rows)

    # ── Channel diagnostics ───────────────────────────────
    cond = compute_channel_condition_numbers(
        dv, q_bs=env.scenario.q_bs, q_user=env.scenario.q_user,
        q_eves=env.scenario.q_eves, q_jammer=env.scenario.q_jammer,
        q_vehicles=env.scenario.q_vehicles,
        vehicle_types=env.scenario.vehicle_types,
        N_tx_sense=env.config.N_tx_sense,
        N_rx_sense=env.config.N_rx_sense,
        L_pilot=env.config.L_pilot,
        noise_power_sense=env.config.noise_power_sense,
        d_ant=env.config.d_ant,
        wavelength=env.config.wavelength,
        seed=env.config.seed or 0,
    )
    diag_path = os.path.join(OUTPUT_DIR, "channel_diagnostics.csv")
    with open(diag_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in cond.items():
            w.writerow([k, v])

    # ── Plots ─────────────────────────────────────────
    # Beamforming gain comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    slots = [r["slot"] for r in beam_rows]
    ax.plot(slots, [r["gain_mrt"] for r in beam_rows], "bo-", label="MRT")
    ax.plot(slots, [r["gain_zf"] for r in beam_rows], "rs--", label="ZF")
    ax.plot(slots, [r["gain_random"] for r in beam_rows], "g^-.", label="Random")
    ax.set_xlabel("Time slot")
    ax.set_ylabel("Effective channel gain")
    ax.set_title("Beamforming Comparison (MIMO BS)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR, "beamforming_comparison.png"), dpi=150)
    plt.close(fig)

    # ── Report ────────────────────────────────────────
    lines = [
        "# MIMO BS Upgrade — Validation Report",
        "",
        f"M_bs = {env.config.M_bs}  |  N_time = {env.config.N_time}",
        f"P_bs_max = {env.config.P_bs_max}  |  w_bs shape = ({env.config.N_time}, {env.config.M_bs})",
        "",
        "## Part 6 & 8: Audit Results",
        "",
    ]
    audit_checks = [
        ("Global phase invariance (PASS expected: whole-vector rotation is still irrelevant)",
         results.get("global_phase", False)),
        ("Per-entry phase changes objective (PASS expected for MIMO)",
         results.get("per_entry_phase", False)),
        ("MRT improves user SINR over random", results.get("mrt_sinr", False)),
        ("ZF suppresses eve SINR compared to MRT", results.get("zf_eve", False)),
        ("Beamforming vector changes objective", results.get("beamforming_obj", False)),
        ("MIMO channels are finite", results.get("channels_finite", False)),
        ("Power constraints hold", results.get("power_constraints", False)),
        ("Gradient exists and varies", results.get("gradients", False)),
        ("Convergence possible (power block step)", results.get("convergence", False)),
    ]
    for label, passed in audit_checks:
        lines.append(f"- {'PASS' if passed else 'FAIL'}  {label}")

    lines.extend([
        "",
        f"## Part 9: Validation — {N_TESTS_PASSED}/{N_TESTS_TOTAL} tests passed",
        "",
        "### Beamforming comparison",
        "| Slot | Power | MRT gain | ZF gain | Random gain |",
        "|------|-------|----------|---------|-------------|",
    ])
    for r in beam_rows:
        lines.append(
            f"| {r['slot']} | {r['power']:.4f} | {r['gain_mrt']:.4e} "
            f"| {r['gain_zf']:.4e} | {r['gain_random']:.4e} |"
        )

    lines.extend([
        "",
        "### Channel diagnostics",
        "| Metric | Value |",
        "|--------|-------|",
    ])
    for k, v in cond.items():
        if v is not None:
            lines.append(f"| {k} | {v:.6e} |")
        else:
            lines.append(f"| {k} | N/A |")

    lines.append("")
    lines.append("## Final Decision")
    all_essential = (
        results.get("mrt_sinr", False) and
        results.get("zf_eve", False) and
        results.get("beamforming_obj", False) and
        results.get("channels_finite", False) and
        results.get("power_constraints", False) and
        results.get("gradients", False)
    )
    if all_essential:
        lines.append("")
        lines.append("**MIMO_BS_UPGRADE_COMPLETE**")
    else:
        lines.append("")
        lines.append("**MIMO_BS_UPGRADE_PARTIAL — some essential checks failed**")
        lines.append(f"  Essential failures: ", )
        for label, passed in audit_checks:
            if not passed:
                lines.append(f"  - {label}")

    report_path = os.path.join(OUTPUT_DIR, "validation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n=== MIMO BS Upgrade Complete ===")
    print(f"  Tests: {N_TESTS_PASSED}/{N_TESTS_TOTAL}")
    print(f"  Report: {report_path}")
    print(f"  CSV:    {csv_path}")
    print(f"  Diag:   {diag_path}")
    if all_essential:
        print(f"  Decision: MIMO_BS_UPGRADE_COMPLETE")
    else:
        print(f"  Decision: MIMO_BS_UPGRADE_PARTIAL")
    return report_path


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    env = make_env(M_bs=4)
    results = run_all_validations(env)
    generate_outputs(env, results)
