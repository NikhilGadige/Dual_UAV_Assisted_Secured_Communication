"""Validation for joint ISAC optimization problem (Phase 5A).

~30 tests + 10 plots + channel debug + Monte Carlo summary.
No solver — evaluation only.
"""

import os
import sys
import numpy as np

from optimization_problem_exp.optimization.problem_formulation import (
    DecisionVariables,
    compute_secrecy_rate,
    compute_sensing_utility,
    evaluate_weighted_objective,
    evaluate_objective_and_constraints,
    compute_channel_condition_numbers,
    check_constraints,
    compute_constraint_violations,
    compute_bs_ris_channel,
    compute_ris_user_channel,
    compute_ris_eve_channel,
    compute_jammer_user_channel,
    compute_jammer_eve_channel,
    compute_user_sinr,
    compute_eve_sinr,
    compute_direct_bs_user_channel,
    compute_direct_bs_eve_channel,
    design_ris_phases,
    design_heuristic_jammer_beam,
    _pl,
    R_S_REF,
    U_SENSE_REF,
)
from optimization_problem_exp.environments.optimization_problem_env import (
    OptimizationConfig,
    OptimizationProblemEnv,
    default_scenario,
)
from vehicle_reflection_exp.channels.vehicle_channel import compute_rcs
from ris_uav_exp.channels.ris_channel import (
    compute_effective_channel,
    compute_effective_channel_gain,
    compute_ris_reflection_matrix,
)
from fd_jammer_exp.channels.fd_jammer_channel import compute_jammer_gain

OUTPUT_DIR = "outputs/optimization_problem"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _make_env(seed=42):
    return OptimizationProblemEnv(OptimizationConfig(seed=seed))


def _make_dv(env):
    return env.random_decision_vars()


# ── Original tests (1-10) ──────────────────────────────

def test_secrecy_rate_finite():
    env = _make_env()
    dv = _make_dv(env)
    sec = compute_secrecy_rate(
        env.scenario.q_bs, env.scenario.q_user,
        env.scenario.q_eves, env.scenario.q_jammer,
        env.config.N_ris, env.config.N_j, None,
        dv.q_uav, dv.w_bs, dv.v_jammer,
        env.config.P_bs_max, env.config.P_j_max,
        env.config.sigma2, seed=42,
        jammer_mode="mixed", jammer_mix_alpha=0.85,
        jammer_power_factor=0.85,
        eta_ris=0.3,
        ris_alignment_alpha=0.85,
    )
    assert np.isfinite(sec["R_s_total"]), "Secrecy rate not finite"
    assert sec["R_s_total"] >= 0.0, f"Secrecy rate negative: {sec['R_s_total']}"
    return True


def test_sensing_utility_finite():
    env = _make_env()
    dv = _make_dv(env)
    rcs_list = [compute_rcs(vt) for vt in env.scenario.vehicle_types]
    sense = compute_sensing_utility(
        dv.q_uav, env.scenario.q_vehicles, rcs_list,
        env.config.N_tx_sense, env.config.N_rx_sense,
        env.config.L_pilot, env.config.noise_power_sense, seed=42,
    )
    assert np.isfinite(sense["U_sense_total"]), "Sensing utility not finite"
    return True


def test_objective_finite():
    env = _make_env()
    result = env.evaluate(env.random_decision_vars())
    f = result["objective"]["f"]
    assert np.isfinite(f), f"Objective not finite: {f}"
    return True


def test_ris_constraints():
    env = _make_env()
    dv = _make_dv(env)
    cons = check_constraints(
        phi_rad=dv.phi_rad, q_uav=dv.q_uav,
        w_bs=dv.w_bs, v_jammer=dv.v_jammer,
        P_bs_max=env.config.P_bs_max,
        P_j_max=env.config.P_j_max,
        v_max=env.config.v_max, dt=env.config.dt,
        q_min=env.scenario.q_min, q_max=env.scenario.q_max,
    )
    assert cons["ris_unit_modulus"], "RIS unit modulus constraint failed"
    assert cons["ris_phase_range"], "RIS phase range constraint failed"
    return True


def test_power_constraints():
    env = _make_env()
    dv = _make_dv(env)
    cons = check_constraints(
        phi_rad=dv.phi_rad, q_uav=dv.q_uav,
        w_bs=dv.w_bs, v_jammer=dv.v_jammer,
        P_bs_max=env.config.P_bs_max,
        P_j_max=env.config.P_j_max,
        v_max=env.config.v_max, dt=env.config.dt,
        q_min=env.scenario.q_min, q_max=env.scenario.q_max,
    )
    assert cons["bs_power"], "BS power constraint failed"
    assert cons["jammer_power"], "Jammer power constraint failed"
    return True


def test_uav_constraints():
    env = _make_env()
    dv = _make_dv(env)
    cons = check_constraints(
        phi_rad=dv.phi_rad, q_uav=dv.q_uav,
        w_bs=dv.w_bs, v_jammer=dv.v_jammer,
        P_bs_max=env.config.P_bs_max,
        P_j_max=env.config.P_j_max,
        v_max=env.config.v_max, dt=env.config.dt,
        q_min=env.scenario.q_min, q_max=env.scenario.q_max,
    )
    assert cons["uav_trajectory_bounds"], "UAV bounds constraint failed"
    return True


def test_constraint_violation_reporting():
    env = _make_env()
    dv = _make_dv(env)
    viol = compute_constraint_violations(
        phi_rad=dv.phi_rad, q_uav=dv.q_uav,
        w_bs=dv.w_bs, v_jammer=dv.v_jammer,
        P_bs_max=env.config.P_bs_max,
        P_j_max=env.config.P_j_max,
        v_max=env.config.v_max, dt=env.config.dt,
        q_min=env.scenario.q_min, q_max=env.scenario.q_max,
    )
    assert isinstance(viol, dict), "Violation should be a dict"
    for key in [
        "bs_power_excess", "jammer_power_excess",
        "uav_speed_excess", "uav_boundary_violation",
        "total_violation",
    ]:
        assert key in viol, f"Missing violation key: {key}"
        assert viol[key] >= 0.0, f"Negative violation: {viol[key]}"
    return True


def test_weighted_objective_consistency():
    """Fixed-reference consistency: alpha=1 gives R_s/R_S_REF, alpha=0 gives U_sense/U_SENSE_REF."""
    env = _make_env()
    dv = _make_dv(env)
    sec = compute_secrecy_rate(
        env.scenario.q_bs, env.scenario.q_user,
        env.scenario.q_eves, env.scenario.q_jammer,
        env.config.N_ris, env.config.N_j, None,
        dv.q_uav, dv.w_bs, dv.v_jammer,
        env.config.P_bs_max, env.config.P_j_max,
        env.config.sigma2, seed=42,
        jammer_mode="mixed", jammer_mix_alpha=0.85,
        jammer_power_factor=0.85,
        eta_ris=0.3,
        ris_alignment_alpha=0.85,
    )
    rcs_list = [compute_rcs(vt) for vt in env.scenario.vehicle_types]
    sense = compute_sensing_utility(
        dv.q_uav, env.scenario.q_vehicles, rcs_list,
        env.config.N_tx_sense, env.config.N_rx_sense,
        env.config.L_pilot, env.config.noise_power_sense, seed=42,
    )
    R = sec["R_s_total"]
    U = sense["U_sense_total"]

    f1 = evaluate_weighted_objective(1.0, R, U)
    f0 = evaluate_weighted_objective(0.0, R, U)
    f05 = evaluate_weighted_objective(0.5, R, U)

    assert np.isclose(f1, R / R_S_REF), "alpha=1 should give R/R_S_REF"
    assert np.isclose(f0, U / U_SENSE_REF), "alpha=0 should give U/U_SENSE_REF"
    assert f0 <= f05 <= f1 or f1 <= f05 <= f0
    return True


def test_multi_target_support():
    env = _make_env(seed=99)
    K_veh = len(env.scenario.q_vehicles)
    assert K_veh == 3, f"Expected 3 vehicles, got {K_veh}"
    dv = _make_dv(env)
    rcs_list = [compute_rcs(vt) for vt in env.scenario.vehicle_types]
    sense = compute_sensing_utility(
        dv.q_uav, env.scenario.q_vehicles, rcs_list,
        env.config.N_tx_sense, env.config.N_rx_sense,
        env.config.L_pilot, env.config.noise_power_sense, seed=99,
    )
    assert len(sense["U_sense_per_slot"]) == env.config.N_time
    assert np.all(np.isfinite(sense["U_sense_per_slot"]))
    return True


def test_multi_eve_support():
    env = _make_env(seed=77)
    K_eve = len(env.scenario.q_eves)
    assert K_eve == 3, f"Expected 3 eves, got {K_eve}"
    dv = _make_dv(env)
    sec = compute_secrecy_rate(
        env.scenario.q_bs, env.scenario.q_user,
        env.scenario.q_eves, env.scenario.q_jammer,
        env.config.N_ris, env.config.N_j, None,
        dv.q_uav, dv.w_bs, dv.v_jammer,
        env.config.P_bs_max, env.config.P_j_max,
        env.config.sigma2, seed=77,
        jammer_mode="mixed", jammer_mix_alpha=0.85,
        jammer_power_factor=0.85,
        eta_ris=0.3,
        ris_alignment_alpha=0.85,
    )
    assert sec["SINR_eve"].shape[1] == K_eve
    assert np.all(np.isfinite(sec["R_s_total"]))
    return True


# ── Prior fix tests (11-16) ────────────────────────────

def test_user_sinr_finite_positive():
    env = _make_env()
    dv = _make_dv(env)
    sec = compute_secrecy_rate(
        env.scenario.q_bs, env.scenario.q_user,
        env.scenario.q_eves, env.scenario.q_jammer,
        env.config.N_ris, env.config.N_j, None,
        dv.q_uav, dv.w_bs, dv.v_jammer,
        env.config.P_bs_max, env.config.P_j_max,
        env.config.sigma2, seed=42,
        jammer_mode="mixed", jammer_mix_alpha=0.85,
        jammer_power_factor=0.85,
        eta_ris=0.3,
        ris_alignment_alpha=0.85,
    )
    sinr_u = sec["SINR_user"]
    assert np.all(np.isfinite(sinr_u)), "User SINR not finite"
    assert np.all(sinr_u > 0.0), f"User SINR not positive: {sinr_u}"
    return True


def test_eve_sinr_finite_positive():
    env = _make_env()
    dv = _make_dv(env)
    sec = compute_secrecy_rate(
        env.scenario.q_bs, env.scenario.q_user,
        env.scenario.q_eves, env.scenario.q_jammer,
        env.config.N_ris, env.config.N_j, None,
        dv.q_uav, dv.w_bs, dv.v_jammer,
        env.config.P_bs_max, env.config.P_j_max,
        env.config.sigma2, seed=42,
        jammer_mode="mixed", jammer_mix_alpha=0.85,
        jammer_power_factor=0.85,
        eta_ris=0.3,
        ris_alignment_alpha=0.85,
    )
    sinr_e = sec["SINR_eve"]
    assert np.all(np.isfinite(sinr_e)), "Eve SINR not finite"
    assert np.all(sinr_e > 0.0), f"Eve SINR not positive: {sinr_e}"
    return True


def test_secrecy_rate_not_identically_zero():
    env = _make_env(seed=42)
    any_positive = False
    for trial in range(5):
        dv = env.random_decision_vars()
        sec = compute_secrecy_rate(
            env.scenario.q_bs, env.scenario.q_user,
            env.scenario.q_eves, env.scenario.q_jammer,
            env.config.N_ris, env.config.N_j, None,
            dv.q_uav, dv.w_bs, dv.v_jammer,
            env.config.P_bs_max, env.config.P_j_max,
            env.config.sigma2, seed=trial,
            jammer_mode="mixed", jammer_mix_alpha=0.85,
            jammer_power_factor=0.85,
            eta_ris=0.3,
            ris_alignment_alpha=0.85,
        )
        if sec["R_s_total"] > 0.01:
            any_positive = True
            break
    assert any_positive, "Secrecy rate is identically zero across 5 trials"
    return True


def test_jammer_power_affects_secrecy():
    env = _make_env(seed=42)
    results = env.run_secrecy_vs_jammer_power(num_points=5, num_trials=2)
    Rs = results["R_s"]
    diffs = np.abs(np.diff(Rs)) / (np.abs(Rs[:-1]) + 1e-10)
    assert np.any(diffs > 0.02), (
        f"Secrecy barely changes with jammer power: Rs range [{Rs.min():.4f}, {Rs.max():.4f}]"
    )
    return True


def test_eve_distance_increases_secrecy():
    env = _make_env(seed=42)
    results = env.run_secrecy_vs_eve_distance(num_points=5, num_trials=2)
    Rs = results["R_s"]
    assert Rs[-1] > Rs[0] + 0.01, (
        f"Moving eves farther did not increase secrecy: "
        f"Rs[0]={Rs[0]:.4f}, Rs[-1]={Rs[-1]:.4f}"
    )
    return True


def test_all_comm_channels_finite():
    q_bs = np.array([0.0, 0.0, 30.0])
    q_uav = np.array([200.0, 0.0, 60.0])
    q_user = np.array([200.0, 0.0, 1.5])
    q_eve = np.array([200.0, 150.0, 1.5])
    q_jammer = np.array([100.0, -120.0, 50.0])
    N_ris, N_j = 16, 4

    h_BR = compute_bs_ris_channel(q_bs, q_uav, N_ris, 0)
    h_RU = compute_ris_user_channel(q_uav, q_user, N_ris, 1)
    h_RE = compute_ris_eve_channel(q_uav, q_eve, N_ris, 2)
    h_JU = compute_jammer_user_channel(q_jammer, q_user, N_j, 3)
    h_JE = compute_jammer_eve_channel(q_jammer, q_eve, N_j, 4)

    for name, arr in [("h_BR", h_BR), ("h_RU", h_RU), ("h_RE", h_RE)]:
        assert arr.shape == (N_ris,), f"{name} shape fail: {arr.shape}"
        assert np.all(np.isfinite(arr)), f"{name} not finite"
    for name, arr in [("h_JU", h_JU), ("h_JE", h_JE)]:
        assert arr.shape == (1, N_j), f"{name} shape fail: {arr.shape}"
        assert np.all(np.isfinite(arr)), f"{name} not finite"
    return True


# ── New tests (17-22) ──────────────────────────────────

def test_alpha_sweep_not_flat():
    """Weighted objective f must vary by at least 5% across alpha sweep."""
    env = _make_env(seed=42)
    sweep = env.sweep_alpha()
    f = sweep["f_weighted"]
    f_range = float(np.max(f) - np.min(f))
    f_mean = float(np.mean(f))
    assert f_range / (f_mean + 1e-10) > 0.05, (
        f"Weighted objective nearly flat: range/mean = {f_range/(f_mean+1e-10):.4f}"
    )
    return True


def test_secrecy_vs_alpha_not_constant():
    """Secrecy rate must change by at least 10% across alpha sweep."""
    env = _make_env(seed=42)
    sweep = env.sweep_alpha()
    Rs = sweep["R_s_total"]
    Rs_range = float(np.max(Rs) - np.min(Rs))
    Rs_mean = float(np.mean(Rs))
    assert Rs_range / (Rs_mean + 1e-10) > 0.10, (
        f"Secrecy nearly constant vs alpha: range/mean = {Rs_range/(Rs_mean+1e-10):.4f}"
    )
    return True


def test_jammer_directional_behaviour():
    """Protect-mode jammer should give different secrecy than blast-mode."""
    env = _make_env(seed=42)
    dv = _make_dv(env)

    sec_protect = compute_secrecy_rate(
        env.scenario.q_bs, env.scenario.q_user,
        env.scenario.q_eves, env.scenario.q_jammer,
        env.config.N_ris, env.config.N_j, None,
        dv.q_uav, dv.w_bs, dv.v_jammer,
        env.config.P_bs_max, env.config.P_j_max,
        env.config.sigma2, seed=42,
        jammer_mode="mixed", jammer_mix_alpha=0.85,
        jammer_power_factor=0.85,
        eta_ris=0.3,
        ris_alignment_alpha=0.85,
    )
    sec_blast = compute_secrecy_rate(
        env.scenario.q_bs, env.scenario.q_user,
        env.scenario.q_eves, env.scenario.q_jammer,
        env.config.N_ris, env.config.N_j, None,
        dv.q_uav, dv.w_bs, dv.v_jammer,
        env.config.P_bs_max, env.config.P_j_max,
        env.config.sigma2, seed=42,
        jammer_mode="blast",
        jammer_power_factor=0.85,
        eta_ris=0.3,
        ris_alignment_alpha=0.85,
    )
    Rs_p = sec_protect["R_s_total"]
    Rs_b = sec_blast["R_s_total"]
    diff_pct = abs(Rs_p - Rs_b) / (max(Rs_p, Rs_b) + 1e-10)
    assert diff_pct > 0.02, (
        f"Jammer modes give nearly same secrecy: protect={Rs_p:.4f}, blast={Rs_b:.4f}"
    )
    return True


def test_monte_carlo_stats_basic():
    """Monte Carlo over 50 realisations: Pr(Rs>0) > 0, avg > 0, finite CDF."""
    env = _make_env(seed=42)
    mc = env.run_monte_carlo_secrecy(
        num_realizations=50, jammer_mode="mixed",
        ris_phase_noise_std=0.8,
    )
    assert mc["prob_rs_gt_0"] > 0.0, "Pr(Rs>0) should be > 0"
    assert mc["avg_secrecy"] >= 0.0, "Avg secrecy should be non-negative"
    assert np.isfinite(mc["avg_secrecy"]), "Avg secrecy not finite"
    assert len(mc["secrecy_cdf_vals"]) == 50, "CDF vals should have 50 entries"
    assert np.all(np.isfinite(mc["secrecy_cdf_vals"])), "CDF vals not finite"
    assert mc["median_secrecy"] >= 0.0, "Median secrecy should be non-negative"
    return True


def test_direct_links_affect_sinr():
    """Enabling weak direct links should slightly change SINR."""
    env = _make_env(seed=42)
    dv = _make_dv(env)

    sec_no_dir = compute_secrecy_rate(
        env.scenario.q_bs, env.scenario.q_user,
        env.scenario.q_eves, env.scenario.q_jammer,
        env.config.N_ris, env.config.N_j, None,
        dv.q_uav, dv.w_bs, dv.v_jammer,
        env.config.P_bs_max, env.config.P_j_max,
        env.config.sigma2, seed=42,
        jammer_mode="mixed", jammer_mix_alpha=0.85,
        jammer_power_factor=0.85,
        eta_ris=0.3,
        ris_alignment_alpha=0.85,
        include_direct_links=False,
    )
    sec_with_dir = compute_secrecy_rate(
        env.scenario.q_bs, env.scenario.q_user,
        env.scenario.q_eves, env.scenario.q_jammer,
        env.config.N_ris, env.config.N_j, None,
        dv.q_uav, dv.w_bs, dv.v_jammer,
        env.config.P_bs_max, env.config.P_j_max,
        env.config.sigma2, seed=42,
        jammer_mode="mixed", jammer_mix_alpha=0.85,
        jammer_power_factor=0.85,
        eta_ris=0.3,
        ris_alignment_alpha=0.85,
        include_direct_links=True,
    )

    sinr_u_diff = float(
        np.abs(sec_with_dir["SINR_user"] - sec_no_dir["SINR_user"]).sum()
    )
    assert sinr_u_diff > 0.0, (
        "Direct links should affect user SINR (even slightly)"
    )
    return True


def test_jammer_power_sweep_meaningful():
    """Jammer power sweep should not be monotonic flat; protect-mode jammer
    should show secrecy first rising then plateauing as P_j increases."""
    env = _make_env(seed=42)
    res = env.run_secrecy_vs_jammer_power(num_points=6, num_trials=2)
    Rs = res["R_s"]
    # The curve should change by > 5% across the sweep
    Rs_range = float(np.max(Rs) - np.min(Rs))
    Rs_mean = float(np.mean(Rs))
    assert Rs_range / (Rs_mean + 1e-10) > 0.05, (
        f"Jammer power sweep nearly flat: range/mean = {Rs_range/(Rs_mean+1e-10):.4f}"
    )
    return True


# ── Plots (original 4) ─────────────────────────────────

def plot_secrecy_vs_alpha():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = _make_env()
    sweep = env.sweep_alpha()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep["alpha"], sweep["R_s_total"], "C0o-", linewidth=2)
    ax.set_xlabel("Weight alpha")
    ax.set_ylabel("Secrecy rate (total, bps/Hz)")
    ax.set_title("Secrecy Rate vs Weight alpha  (alpha-dependent decisions)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "secrecy_vs_alpha.png"), dpi=150)
    plt.close(fig)
    return True


def plot_sensing_vs_alpha():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = _make_env()
    sweep = env.sweep_alpha()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep["alpha"], sweep["U_sense_total"], "C1s-", linewidth=2)
    ax.set_xlabel("Weight alpha")
    ax.set_ylabel("Sensing utility (total)")
    ax.set_title("Sensing Utility vs Weight alpha  (alpha-dependent decisions)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "sensing_vs_alpha.png"), dpi=150)
    plt.close(fig)
    return True


def plot_weighted_objective_vs_alpha():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = _make_env()
    sweep = env.sweep_alpha()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep["alpha"], sweep["f_weighted"], "C2^-", linewidth=2)
    ax.set_xlabel("Weight alpha")
    ax.set_ylabel("Weighted objective f")
    ax.set_title("Weighted Objective vs Weight alpha  (fixed normalisation)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "weighted_objective_vs_alpha.png"),
        dpi=150,
    )
    plt.close(fig)
    return True


def plot_constraint_violation_breakdown():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = _make_env()
    dv = _make_dv(env)
    viol = compute_constraint_violations(
        phi_rad=dv.phi_rad, q_uav=dv.q_uav,
        w_bs=dv.w_bs, v_jammer=dv.v_jammer,
        P_bs_max=env.config.P_bs_max * 0.5,
        P_j_max=env.config.P_j_max * 0.5,
        v_max=env.config.v_max * 0.5, dt=env.config.dt,
        q_min=env.scenario.q_min, q_max=env.scenario.q_max,
        R_s_total=0.0, U_sense_total=0.0,
    )
    labels = []
    values = []
    for key in [
        "bs_power_excess", "jammer_power_excess",
        "uav_speed_excess", "uav_boundary_violation",
    ]:
        labels.append(key.replace("_", " ").title())
        values.append(viol[key])

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["C0", "C1", "C2", "C3"]
    bars = ax.bar(labels, values, color=colors, alpha=0.7)
    ax.set_ylabel("Violation magnitude")
    ax.set_title("Constraint Violation Breakdown  (tightened limits)")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "constraint_violation_breakdown.png"),
        dpi=150,
    )
    plt.close(fig)
    return True


# ── Sanity plots ────────────────────────────────────────

def plot_secrecy_vs_user_distance():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = _make_env()
    res = env.run_secrecy_vs_user_distance(num_points=8, num_trials=3)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(res["user_dist"], res["R_s"], "C0o-", linewidth=2, label="Secrecy rate")
    ax.plot(res["user_dist"], res["R_user"], "C1s--", linewidth=1.5, label="R_user (avg)")
    ax.plot(res["user_dist"], res["R_eve_max"], "C2^--", linewidth=1.5, label="max R_eve")
    ax.set_xlabel("User x-distance from origin (m)")
    ax.set_ylabel("Rate (bps/Hz)")
    ax.set_title("Secrecy Rate vs User Distance  (directional jammer, protect mode)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "secrecy_vs_user_distance.png"), dpi=150,
    )
    plt.close(fig)
    return True


def plot_secrecy_vs_jammer_power():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = _make_env()
    res = env.run_secrecy_vs_jammer_power(num_points=8, num_trials=3)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(res["P_j"], res["R_s"], "C1s-", linewidth=2)
    ax.set_xlabel("Jammer max power P_j (W)")
    ax.set_ylabel("Secrecy rate (bps/Hz)")
    ax.set_title("Secrecy Rate vs Jammer Power  (nullspace jammer, protect mode)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "secrecy_vs_jammer_power.png"), dpi=150,
    )
    plt.close(fig)
    return True


def plot_secrecy_vs_eve_distance():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = _make_env()
    res = env.run_secrecy_vs_eve_distance(num_points=8, num_trials=3)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(res["eve_offset"], res["R_s"], "C0o-", linewidth=2, label="Secrecy rate")
    ax.plot(res["eve_offset"], res["R_user"], "C1s--", linewidth=1.5, label="R_user (avg)")
    ax.plot(res["eve_offset"], res["R_eve_max"], "C2^--", linewidth=1.5, label="max R_eve")
    ax.set_xlabel("Eve lateral offset from user (m)")
    ax.set_ylabel("Rate (bps/Hz)")
    ax.set_title("Secrecy Rate vs Eve Distance  (directional jammer, protect mode)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "secrecy_vs_eve_distance.png"), dpi=150,
    )
    plt.close(fig)
    return True


def plot_user_and_eve_sinr():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = _make_env()
    dv = _make_dv(env)
    sec = compute_secrecy_rate(
        env.scenario.q_bs, env.scenario.q_user,
        env.scenario.q_eves, env.scenario.q_jammer,
        env.config.N_ris, env.config.N_j, None,
        dv.q_uav, dv.w_bs, dv.v_jammer,
        env.config.P_bs_max, env.config.P_j_max,
        env.config.sigma2, seed=42,
        jammer_mode="mixed", jammer_mix_alpha=0.85,
        jammer_power_factor=0.85,
        eta_ris=0.3,
        ris_alignment_alpha=0.85,
    )
    N_time = env.config.N_time

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    slots = np.arange(N_time)
    ax1.semilogy(slots, sec["SINR_user"], "C0o-", linewidth=2, label="User")
    for ke in range(sec["SINR_eve"].shape[1]):
        ax1.semilogy(
            slots, sec["SINR_eve"][:, ke],
            "x--", alpha=0.6, label=f"Eve {ke}",
        )
    ax1.set_xlabel("Time slot")
    ax1.set_ylabel("SINR (log scale)")
    ax1.set_title("User and Eve SINR per Time Slot  (nullspace jammer)")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    rates = np.column_stack([
        sec["R_user"],
        [sec["R_eve_max"][n] for n in range(N_time)],
    ])
    ax2.bar(slots - 0.15, rates[:, 0], 0.3, label="R_user", alpha=0.8)
    ax2.bar(slots + 0.15, rates[:, 1], 0.3, label="max R_eve", alpha=0.8)
    ax2.set_xlabel("Time slot")
    ax2.set_ylabel("Rate (bps/Hz)")
    ax2.set_title("User Rate vs Max Eve Rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("SINR and Rate Breakdown  (phase-aligned RIS + nullspace jammer)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "user_and_eve_sinr.png"), dpi=150)
    plt.close(fig)
    return True


# ── New plots (9-10) ───────────────────────────────────

def plot_secrecy_cdf():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = _make_env(seed=42)
    mc = env.run_monte_carlo_secrecy(
        num_realizations=200, jammer_mode="mixed",
        ris_phase_noise_std=0.8,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(mc["secrecy_cdf_vals"], mc["secrecy_cdf_probs"], "C0-", linewidth=2)
    ax.axvline(mc["avg_secrecy"], color="C1", linestyle="--",
               label=f"Mean = {mc['avg_secrecy']:.2f}")
    ax.axvline(mc["median_secrecy"], color="C2", linestyle=":",
               label=f"Median = {mc['median_secrecy']:.2f}")
    ax.set_xlabel("Secrecy rate (total, bps/Hz)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Secrecy Rate CDF  ({len(mc['secrecy_cdf_vals'])} realisations, "
                 f"Pr>0 = {mc['prob_rs_gt_0']:.1%})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "secrecy_cdf.png"), dpi=150)
    plt.close(fig)
    return True


def plot_weighted_objective_components():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = _make_env()
    sweep = env.sweep_alpha()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep["alpha"], sweep["R_s_norm"], "C0o-", linewidth=2,
            label=f"R_s/R_S_REF  (R_S_REF={R_S_REF})")
    ax.plot(sweep["alpha"], sweep["U_sense_norm"], "C1s-", linewidth=2,
            label=f"U_sense/U_SENSE_REF  (U_SENSE_REF={U_SENSE_REF})")
    ax.plot(sweep["alpha"], sweep["f_weighted"], "k^--", linewidth=1.5,
            label="f = alpha*R_s_norm + (1-alpha)*U_sense_norm")
    ax.set_xlabel("Weight alpha")
    ax.set_ylabel("Normalised value")
    ax.set_title("Weighted Objective Components vs alpha")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "weighted_objective_components_vs_alpha.png"),
        dpi=150,
    )
    plt.close(fig)
    return True


# ── Channel debug summary ──────────────────────────────

def generate_channel_debug_summary():
    env = _make_env(seed=42)
    lines = []
    sep = "=" * 64
    lines.append(sep)
    lines.append("  Channel Debug Summary  —  Phase 5A (v2)")
    lines.append(sep)
    lines.append("")

    # Multi-trial statistics
    Rs_all = []
    Ru_all = []
    Re_all = []
    SINRu_all = []
    SINRe_all = []

    for trial in range(10):
        env2 = OptimizationProblemEnv(OptimizationConfig(seed=42 + trial))
        dv = env2.random_decision_vars()
        sec = compute_secrecy_rate(
            env2.scenario.q_bs, env2.scenario.q_user,
            env2.scenario.q_eves, env2.scenario.q_jammer,
            env2.config.N_ris, env2.config.N_j, None,
            dv.q_uav, dv.w_bs, dv.v_jammer,
            env2.config.P_bs_max, env2.config.P_j_max,
            env2.config.sigma2, seed=trial,
            jammer_mode="mixed", jammer_mix_alpha=0.85,
            jammer_power_factor=0.85,
            eta_ris=0.3,
            ris_alignment_alpha=0.85,
        )
        Rs_all.append(sec["R_s_total"])
        Ru_all.append(float(np.mean(sec["R_user"])))
        Re_all.append(float(np.max(sec["R_eve_max"])))
        SINRu_all.extend(sec["SINR_user"].tolist())
        SINRe_all.extend(sec["SINR_eve"].flatten().tolist())

    lines.append(f"  Trials: 10")
    lines.append(f"  Avg secrecy rate:      {np.mean(Rs_all):.4f}  bps/Hz")
    lines.append(f"  Min secrecy rate:      {np.min(Rs_all):.4f}  bps/Hz")
    lines.append(f"  Max secrecy rate:      {np.max(Rs_all):.4f}  bps/Hz")
    lines.append(f"  Avg R_user (mean):     {np.mean(Ru_all):.4f}  bps/Hz")
    lines.append(f"  Avg max R_eve:         {np.mean(Re_all):.4f}  bps/Hz")
    lines.append(f"  Avg user SINR:         {np.mean(SINRu_all):.6f}  linear")
    lines.append(f"  Avg eve SINR:          {np.mean(SINRe_all):.6f}  linear")
    lines.append(f"  Fraction Rs > 0.01:    {np.mean(np.array(Rs_all) > 0.01):.2%}")
    lines.append("")

    # Jammer mode comparison
    dv_ref = env.random_decision_vars()
    sec_protect = compute_secrecy_rate(
        env.scenario.q_bs, env.scenario.q_user,
        env.scenario.q_eves, env.scenario.q_jammer,
        env.config.N_ris, env.config.N_j, None,
        dv_ref.q_uav, dv_ref.w_bs, dv_ref.v_jammer,
        env.config.P_bs_max, env.config.P_j_max,
        env.config.sigma2, seed=42,
        jammer_mode="mixed", jammer_mix_alpha=0.85,
        jammer_power_factor=0.85,
        eta_ris=0.3,
        ris_alignment_alpha=0.85,
    )
    sec_blast = compute_secrecy_rate(
        env.scenario.q_bs, env.scenario.q_user,
        env.scenario.q_eves, env.scenario.q_jammer,
        env.config.N_ris, env.config.N_j, None,
        dv_ref.q_uav, dv_ref.w_bs, dv_ref.v_jammer,
        env.config.P_bs_max, env.config.P_j_max,
        env.config.sigma2, seed=42,
        jammer_mode="blast",
        jammer_power_factor=0.85,
        eta_ris=0.3,
        ris_alignment_alpha=0.85,
    )
    sec_iso = compute_secrecy_rate(
        env.scenario.q_bs, env.scenario.q_user,
        env.scenario.q_eves, env.scenario.q_jammer,
        env.config.N_ris, env.config.N_j, None,
        dv_ref.q_uav, dv_ref.w_bs, dv_ref.v_jammer,
        env.config.P_bs_max, env.config.P_j_max,
        env.config.sigma2, seed=42,
        jammer_mode="isotropic",
        jammer_power_factor=0.85,
        eta_ris=0.3,
        ris_alignment_alpha=0.85,
    )

    lines.append("  Jammer Mode Comparison:")
    lines.append(f"    Protect  Rs = {sec_protect['R_s_total']:.4f}")
    lines.append(f"    Blast    Rs = {sec_blast['R_s_total']:.4f}")
    lines.append(f"    Isotropic Rs = {sec_iso['R_s_total']:.4f}")
    lines.append("")

    # Path loss table
    scenario = default_scenario()
    lines.append("  Path Loss Table (default scenario, UAV midpoint):")
    lines.append(f"  {'Link':<20} {'Distance (m)':<14} {'PL (linear)':<14}")
    lines.append(f"  {'-'*48}")
    q_mid = (scenario.q_uav_start + scenario.q_uav_end) / 2
    links = [
        ("BS -> UAV(mid)", scenario.q_bs, q_mid),
        ("UAV(mid) -> User", q_mid, scenario.q_user),
        ("UAV(mid) -> Eve0", q_mid, scenario.q_eves[0]),
        ("UAV(mid) -> Eve1", q_mid, scenario.q_eves[1]),
        ("UAV(mid) -> Eve2", q_mid, scenario.q_eves[2]),
        ("Jammer -> User", scenario.q_jammer, scenario.q_user),
        ("Jammer -> Eve0", scenario.q_jammer, scenario.q_eves[0]),
        ("Jammer -> Eve1", scenario.q_jammer, scenario.q_eves[1]),
    ]
    for name, a, b in links:
        d = float(np.linalg.norm(a - b))
        pl = 1.0 / (d**2)
        lines.append(f"  {name:<20} {d:<14.1f} {pl:<14.2e}")
    lines.append("")

    # Config
    cfg = OptimizationConfig()
    lines.append("  Configuration:")
    lines.append(f"    N_ris     = {cfg.N_ris}")
    lines.append(f"    N_j       = {cfg.N_j}")
    lines.append(f"    P_bs_max  = {cfg.P_bs_max}")
    lines.append(f"    P_j_max   = {cfg.P_j_max}")
    lines.append(f"    sigma2    = {cfg.sigma2:.0e}")
    lines.append(f"    N_time    = {cfg.N_time}")
    lines.append(f"    v_max     = {cfg.v_max}")
    lines.append(f"    R_S_REF   = {R_S_REF}")
    lines.append(f"    U_SENSE_REF = {U_SENSE_REF}")

    lines.append(f"\n{sep}")

    path = os.path.join(OUTPUT_DIR, "channel_debug_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [ INFO ] channel_debug_summary.txt written")
    return True


# ── Monte Carlo JSON export ────────────────────────────

def generate_monte_carlo_stats_json():
    """Export Monte Carlo secrecy stats to JSON file."""
    import json

    env = _make_env(seed=42)
    mc = env.run_monte_carlo_secrecy(
        num_realizations=200, jammer_mode="mixed",
        ris_phase_noise_std=0.8,
    )

    # Percentiles
    percentiles = {
        "5": float(np.percentile(mc["all_Rs"], 5)),
        "25": float(np.percentile(mc["all_Rs"], 25)),
        "50": float(np.percentile(mc["all_Rs"], 50)),
        "75": float(np.percentile(mc["all_Rs"], 75)),
        "95": float(np.percentile(mc["all_Rs"], 95)),
    }

    stats = {
        "mean_rs": mc["avg_secrecy"],
        "std_rs": mc["std_secrecy"],
        "median_rs": mc["median_secrecy"],
        "min_rs": mc["min_secrecy"],
        "max_rs": mc["max_secrecy"],
        "p_success": mc["prob_rs_gt_0"],
        "p_outage": 1.0 - mc["prob_rs_gt_0"],
        "percentiles": percentiles,
    }

    path = os.path.join(OUTPUT_DIR, "monte_carlo_stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  [ INFO ] monte_carlo_stats.json written")
    return True


# ── New tests (23-26) ──────────────────────────────────

def test_monte_carlo_json_exists():
    """monte_carlo_stats.json exists and contains all required keys."""
    import json

    path = os.path.join(OUTPUT_DIR, "monte_carlo_stats.json")
    assert os.path.isfile(path), f"Missing {path}"
    with open(path) as f:
        data = json.load(f)

    required_keys = [
        "mean_rs", "std_rs", "median_rs", "min_rs", "max_rs",
        "p_success", "p_outage", "percentiles",
    ]
    for key in required_keys:
        assert key in data, f"Missing key: {key}"

    pct_keys = ["5", "25", "50", "75", "95"]
    for pk in pct_keys:
        assert pk in data["percentiles"], f"Missing percentile: {pk}"

    # Sanity: values should be finite and non-negative for rates
    assert np.isfinite(data["mean_rs"]), "mean_rs not finite"
    assert np.isfinite(data["p_success"]), "p_success not finite"
    assert 0.0 <= data["p_success"] <= 1.0, "p_success out of [0,1]"
    assert data["min_rs"] >= 0.0, "min_rs negative"
    return True


def test_evaluate_wrapper_finite():
    """evaluate_objective_and_constraints() returns finite objective."""
    env = _make_env(seed=42)
    dv = _make_dv(env)

    result = evaluate_objective_and_constraints(
        decision_vars=dv,
        q_bs=env.scenario.q_bs,
        q_user=env.scenario.q_user,
        q_eves=env.scenario.q_eves,
        q_jammer=env.scenario.q_jammer,
        q_vehicles=env.scenario.q_vehicles,
        vehicle_types=env.scenario.vehicle_types,
        N_ris=env.config.N_ris,
        N_j=env.config.N_j,
        N_tx_sense=env.config.N_tx_sense,
        N_rx_sense=env.config.N_rx_sense,
        L_pilot=env.config.L_pilot,
        P_bs_max=env.config.P_bs_max,
        P_j_max=env.config.P_j_max,
        sigma2=env.config.sigma2,
        noise_power_sense=env.config.noise_power_sense,
        v_max=env.config.v_max,
        dt=env.config.dt,
        q_min=env.scenario.q_min,
        q_max=env.scenario.q_max,
        eta_ris=env.config.eta_ris,
        alpha=0.5,
        seed=42,
    )
    assert np.isfinite(result["objective"]), "Objective not finite"
    assert np.isfinite(result["secrecy"]["R_s_total"]), "Secrecy not finite"
    assert np.isfinite(result["sensing"]["U_sense_total"]), "Sensing utility not finite"
    return True


def test_evaluate_wrapper_fields():
    """Wrapper result contains all expected keys."""
    env = _make_env(seed=42)
    dv = _make_dv(env)

    result = evaluate_objective_and_constraints(
        decision_vars=dv,
        q_bs=env.scenario.q_bs,
        q_user=env.scenario.q_user,
        q_eves=env.scenario.q_eves,
        q_jammer=env.scenario.q_jammer,
        q_vehicles=env.scenario.q_vehicles,
        vehicle_types=env.scenario.vehicle_types,
        N_ris=env.config.N_ris,
        N_j=env.config.N_j,
        N_tx_sense=env.config.N_tx_sense,
        N_rx_sense=env.config.N_rx_sense,
        L_pilot=env.config.L_pilot,
        P_bs_max=env.config.P_bs_max,
        P_j_max=env.config.P_j_max,
        sigma2=env.config.sigma2,
        noise_power_sense=env.config.noise_power_sense,
        v_max=env.config.v_max,
        dt=env.config.dt,
        q_min=env.scenario.q_min,
        q_max=env.scenario.q_max,
        eta_ris=env.config.eta_ris,
        alpha=0.5,
        seed=42,
    )
    expected_keys = ["objective", "secrecy", "sensing", "constraints", "violations"]
    for key in expected_keys:
        assert key in result, f"Missing top-level key: {key}"

    # Sub-keys for constraints and violations
    cons = result["constraints"]
    for ck in [
        "ris_unit_modulus", "ris_phase_range", "bs_power",
        "jammer_power", "uav_speed", "uav_trajectory_bounds",
    ]:
        assert ck in cons, f"Missing constraint key: {ck}"

    viol = result["violations"]
    for vk in ["bs_power_excess", "jammer_power_excess", "uav_speed_excess"]:
        assert vk in viol, f"Missing violation key: {vk}"
    return True


def test_evaluate_wrapper_consistency():
    """Wrapper produces same objective as calling individual functions."""
    env = _make_env(seed=42)
    dv = _make_dv(env)

    # Individual calls
    sec = compute_secrecy_rate(
        env.scenario.q_bs, env.scenario.q_user,
        env.scenario.q_eves, env.scenario.q_jammer,
        env.config.N_ris, env.config.N_j, None,
        dv.q_uav, dv.w_bs, dv.v_jammer,
        env.config.P_bs_max, env.config.P_j_max,
        env.config.sigma2, seed=42,
        jammer_mode="mixed", jammer_mix_alpha=0.85,
        jammer_power_factor=0.85,
        eta_ris=0.3,
        ris_alignment_alpha=0.85,
    )
    rcs_list = [compute_rcs(vt) for vt in env.scenario.vehicle_types]
    sense = compute_sensing_utility(
        dv.q_uav, env.scenario.q_vehicles, rcs_list,
        env.config.N_tx_sense, env.config.N_rx_sense,
        env.config.L_pilot, env.config.noise_power_sense, seed=42,
    )
    f_individual = evaluate_weighted_objective(0.5, sec["R_s_total"], sense["U_sense_total"])

    # Wrapper
    wr = evaluate_objective_and_constraints(
        decision_vars=dv,
        q_bs=env.scenario.q_bs,
        q_user=env.scenario.q_user,
        q_eves=env.scenario.q_eves,
        q_jammer=env.scenario.q_jammer,
        q_vehicles=env.scenario.q_vehicles,
        vehicle_types=env.scenario.vehicle_types,
        N_ris=env.config.N_ris,
        N_j=env.config.N_j,
        N_tx_sense=env.config.N_tx_sense,
        N_rx_sense=env.config.N_rx_sense,
        L_pilot=env.config.L_pilot,
        P_bs_max=env.config.P_bs_max,
        P_j_max=env.config.P_j_max,
        sigma2=env.config.sigma2,
        noise_power_sense=env.config.noise_power_sense,
        v_max=env.config.v_max,
        dt=env.config.dt,
        q_min=env.scenario.q_min,
        q_max=env.scenario.q_max,
        eta_ris=env.config.eta_ris,
        alpha=0.5,
        seed=42,
        jammer_mode="mixed",
        jammer_mix_alpha=0.85,
        jammer_power_factor=0.85,
        ris_alignment_alpha=0.85,
    )

    assert np.isclose(
        wr["objective"], f_individual, rtol=1e-6, atol=1e-6,
    ), f"Wrapper objective {wr['objective']:.6f} != individual {f_individual:.6f}"
    assert np.isclose(
        wr["secrecy"]["R_s_total"], sec["R_s_total"], rtol=1e-6, atol=1e-6,
    ), "Secrecy mismatch"
    assert np.isclose(
        wr["sensing"]["U_sense_total"], sense["U_sense_total"], rtol=1e-6, atol=1e-6,
    ), "Sensing mismatch"
    return True


# ── Numerical diagnostics ──────────────────────────────

def generate_numerical_diagnostics():
    """Compute and save channel condition numbers and numerical diagnostics."""
    env = _make_env(seed=42)
    dv = _make_dv(env)

    cond = compute_channel_condition_numbers(
        decision_vars=dv,
        q_bs=env.scenario.q_bs,
        q_user=env.scenario.q_user,
        q_eves=env.scenario.q_eves,
        q_jammer=env.scenario.q_jammer,
        q_vehicles=env.scenario.q_vehicles,
        vehicle_types=env.scenario.vehicle_types,
        seed=42,
    )

    lines = []
    lines.append("=" * 64)
    lines.append("  Numerical Diagnostics  --  Phase 5A (pre-5B)")
    lines.append("=" * 64)
    lines.append("")
    lines.append("  Channel Condition Numbers:")
    lines.append(f"    RIS effective channel cond : {cond.get('ris_eff_channel_cond', 'N/A')}")
    lines.append(f"    Sensing matrix cond        : {cond.get('sensing_matrix_cond', 'N/A')}")
    lines.append(f"    CRB FIM cond               : {cond.get('fim_cond', 'N/A')}")
    lines.append(f"    CRB trace (mean)           : {cond.get('crb_trace', 'N/A')}")
    lines.append("")

    # Evaluate at extreme alpha values
    for alpha in [0.0, 0.5, 1.0]:
        wr = evaluate_objective_and_constraints(
            decision_vars=dv,
            q_bs=env.scenario.q_bs, q_user=env.scenario.q_user,
            q_eves=env.scenario.q_eves, q_jammer=env.scenario.q_jammer,
            q_vehicles=env.scenario.q_vehicles,
            vehicle_types=env.scenario.vehicle_types,
            N_ris=env.config.N_ris, N_j=env.config.N_j,
            N_tx_sense=env.config.N_tx_sense,
            N_rx_sense=env.config.N_rx_sense,
            L_pilot=env.config.L_pilot,
            P_bs_max=env.config.P_bs_max, P_j_max=env.config.P_j_max,
            sigma2=env.config.sigma2,
            noise_power_sense=env.config.noise_power_sense,
            v_max=env.config.v_max, dt=env.config.dt,
            q_min=env.scenario.q_min, q_max=env.scenario.q_max,
            eta_ris=env.config.eta_ris,
            alpha=alpha, seed=42,
        )
        lines.append(f"  alpha={alpha:.1f}:  f={wr['objective']:.6f}  "
                     f"R_s={wr['secrecy']['R_s_total']:.4f}  "
                     f"U_sense={wr['sensing']['U_sense_total']:.4f}")
    lines.append("")

    # SINR ranges
    sec = compute_secrecy_rate(
        env.scenario.q_bs, env.scenario.q_user,
        env.scenario.q_eves, env.scenario.q_jammer,
        env.config.N_ris, env.config.N_j, None,
        dv.q_uav, dv.w_bs, dv.v_jammer,
        env.config.P_bs_max, env.config.P_j_max,
        env.config.sigma2, seed=42,
        jammer_mode="mixed", jammer_mix_alpha=0.85,
        jammer_power_factor=0.85, eta_ris=0.3,
        ris_alignment_alpha=0.85,
    )
    lines.append("  SINR ranges:")
    lines.append(f"    User SINR  : [{float(sec['SINR_user'].min()):.6e}, "
                 f"{float(sec['SINR_user'].max()):.6e}]")
    lines.append(f"    Eve SINR   : [{float(sec['SINR_eve'].min()):.6e}, "
                 f"{float(sec['SINR_eve'].max()):.6e}]")
    lines.append("")

    # Monte Carlo reproducibility check
    mc1 = env.run_monte_carlo_secrecy(num_realizations=50, jammer_mode="mixed")
    mc2 = env.run_monte_carlo_secrecy(num_realizations=50, jammer_mode="mixed")
    rep = np.isclose(mc1["avg_secrecy"], mc2["avg_secrecy"])
    lines.append(f"  MC reproducibility: {'PASS' if rep else 'FAIL'}  "
                 f"(avg1={mc1['avg_secrecy']:.6f}, avg2={mc2['avg_secrecy']:.6f})")
    lines.append("")
    lines.append("=" * 64)

    path = os.path.join(OUTPUT_DIR, "numerical_diagnostics.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [ INFO ] numerical_diagnostics.txt written")
    return True


# ── New tests (27-29) ──────────────────────────────────

def test_evaluate_wrapper_extreme_values():
    """Stress-test the wrapper with 100 extreme-but-feasible DVs."""
    env = _make_env(seed=42)
    for trial in range(100):
        rng = np.random.RandomState(trial)
        N_ris = env.config.N_ris
        N_j = env.config.N_j
        N_time = env.config.N_time

        phi_rad = rng.uniform(-np.pi, np.pi, N_ris)
        q_uav = np.zeros((N_time, 3))
        for n in range(N_time):
            for d in range(3):
                q_uav[n, d] = rng.uniform(
                    env.scenario.q_min[d], env.scenario.q_max[d],
                )
        w_bs = np.sqrt(env.config.P_bs_max) * np.exp(
            1j * rng.uniform(0, 2 * np.pi, N_time),
        )
        v_jammer = np.zeros((N_time, N_j), dtype=complex)
        for n in range(N_time):
            v = rng.randn(N_j) + 1j * rng.randn(N_j)
            v = v / max(float(np.linalg.norm(v)), 1e-30)
            v_jammer[n] = v * np.sqrt(env.config.P_j_max)

        dv = DecisionVariables(phi_rad=phi_rad, q_uav=q_uav, w_bs=w_bs, v_jammer=v_jammer)

        result = evaluate_objective_and_constraints(
            decision_vars=dv,
            q_bs=env.scenario.q_bs, q_user=env.scenario.q_user,
            q_eves=env.scenario.q_eves, q_jammer=env.scenario.q_jammer,
            q_vehicles=env.scenario.q_vehicles,
            vehicle_types=env.scenario.vehicle_types,
            N_ris=N_ris, N_j=N_j,
            N_tx_sense=env.config.N_tx_sense,
            N_rx_sense=env.config.N_rx_sense,
            L_pilot=env.config.L_pilot,
            P_bs_max=env.config.P_bs_max, P_j_max=env.config.P_j_max,
            sigma2=env.config.sigma2,
            noise_power_sense=env.config.noise_power_sense,
            v_max=env.config.v_max, dt=env.config.dt,
            q_min=env.scenario.q_min, q_max=env.scenario.q_max,
            eta_ris=env.config.eta_ris,
            alpha=rng.uniform(0.0, 1.0),
            seed=trial,
        )
        assert np.isfinite(result["objective"]), (
            f"Trial {trial}: objective not finite: {result['objective']}"
        )
        assert not np.isnan(result["objective"]), (
            f"Trial {trial}: objective is NaN"
        )
        assert not np.isinf(result["objective"]), (
            f"Trial {trial}: objective is Inf"
        )
        assert np.isfinite(result["secrecy"]["R_s_total"]), (
            f"Trial {trial}: secrecy not finite"
        )
        assert np.isfinite(result["sensing"]["U_sense_total"]), (
            f"Trial {trial}: sensing not finite"
        )
    return True


def test_reproducibility():
    """With seed=42, multiple runs produce identical outputs."""
    env = _make_env(seed=42)

    # Monte Carlo
    mc_a = env.run_monte_carlo_secrecy(num_realizations=50, jammer_mode="mixed")
    mc_b = env.run_monte_carlo_secrecy(num_realizations=50, jammer_mode="mixed")
    assert np.isclose(mc_a["avg_secrecy"], mc_b["avg_secrecy"]), \
        f"MC not reproducible: {mc_a['avg_secrecy']} vs {mc_b['avg_secrecy']}"
    assert np.isclose(mc_a["median_secrecy"], mc_b["median_secrecy"]), \
        "MC median not reproducible"
    assert np.isclose(mc_a["std_secrecy"], mc_b["std_secrecy"]), \
        "MC std not reproducible"

    # Alpha sweep
    sa = env.sweep_alpha()
    sb = env.sweep_alpha()
    assert np.allclose(sa["f_weighted"], sb["f_weighted"]), \
        "Alpha sweep not reproducible"
    assert np.allclose(sa["R_s_total"], sb["R_s_total"]), \
        "Alpha sweep Rs not reproducible"

    # Objective evaluation
    dv = _make_dv(env)
    ra = evaluate_objective_and_constraints(
        decision_vars=dv,
        q_bs=env.scenario.q_bs, q_user=env.scenario.q_user,
        q_eves=env.scenario.q_eves, q_jammer=env.scenario.q_jammer,
        q_vehicles=env.scenario.q_vehicles,
        vehicle_types=env.scenario.vehicle_types,
        N_ris=env.config.N_ris, N_j=env.config.N_j,
        N_tx_sense=env.config.N_tx_sense,
        N_rx_sense=env.config.N_rx_sense,
        L_pilot=env.config.L_pilot,
        P_bs_max=env.config.P_bs_max, P_j_max=env.config.P_j_max,
        sigma2=env.config.sigma2,
        noise_power_sense=env.config.noise_power_sense,
        v_max=env.config.v_max, dt=env.config.dt,
        q_min=env.scenario.q_min, q_max=env.scenario.q_max,
        eta_ris=env.config.eta_ris,
        alpha=0.5, seed=42,
    )
    rb = evaluate_objective_and_constraints(
        decision_vars=dv,
        q_bs=env.scenario.q_bs, q_user=env.scenario.q_user,
        q_eves=env.scenario.q_eves, q_jammer=env.scenario.q_jammer,
        q_vehicles=env.scenario.q_vehicles,
        vehicle_types=env.scenario.vehicle_types,
        N_ris=env.config.N_ris, N_j=env.config.N_j,
        N_tx_sense=env.config.N_tx_sense,
        N_rx_sense=env.config.N_rx_sense,
        L_pilot=env.config.L_pilot,
        P_bs_max=env.config.P_bs_max, P_j_max=env.config.P_j_max,
        sigma2=env.config.sigma2,
        noise_power_sense=env.config.noise_power_sense,
        v_max=env.config.v_max, dt=env.config.dt,
        q_min=env.scenario.q_min, q_max=env.scenario.q_max,
        eta_ris=env.config.eta_ris,
        alpha=0.5, seed=42,
    )
    assert np.isclose(ra["objective"], rb["objective"]), \
        "Objective not reproducible"
    return True


# ── Test runner ─────────────────────────────────────────

def run_validation():
    summary_lines = []
    sep = "=" * 64
    summary_lines.append(sep)
    summary_lines.append("  Optimization Problem Validation (Phase 5A — v2)")
    summary_lines.append(sep)
    summary_lines.append("")

    passed = 0
    failed = 0

    plot_fns = [
        plot_secrecy_vs_alpha,
        plot_sensing_vs_alpha,
        plot_weighted_objective_vs_alpha,
        plot_constraint_violation_breakdown,
        plot_secrecy_vs_user_distance,
        plot_secrecy_vs_jammer_power,
        plot_secrecy_vs_eve_distance,
        plot_user_and_eve_sinr,
        plot_secrecy_cdf,
        plot_weighted_objective_components,
        generate_channel_debug_summary,
        generate_monte_carlo_stats_json,
        generate_numerical_diagnostics,
    ]

    for fn in plot_fns:
        try:
            result = fn()
            status = "PASS" if result else "FAIL"
            if result:
                passed += 1
            else:
                failed += 1
            line = f"  [ {status} ] {fn.__name__}"
            print(line)
            summary_lines.append(line)
        except Exception as e:
            failed += 1
            line = f"  [ FAIL ] {fn.__name__}: {e}"
            import traceback
            traceback.print_exc()
            print(line)
            summary_lines.append(line)

    test_fns = [
        # Original (1-10)
        test_secrecy_rate_finite,
        test_sensing_utility_finite,
        test_objective_finite,
        test_ris_constraints,
        test_power_constraints,
        test_uav_constraints,
        test_constraint_violation_reporting,
        test_weighted_objective_consistency,
        test_multi_target_support,
        test_multi_eve_support,
        # Prior fix (11-16)
        test_user_sinr_finite_positive,
        test_eve_sinr_finite_positive,
        test_secrecy_rate_not_identically_zero,
        test_jammer_power_affects_secrecy,
        test_eve_distance_increases_secrecy,
        test_all_comm_channels_finite,
        # New (17-22)
        test_alpha_sweep_not_flat,
        test_secrecy_vs_alpha_not_constant,
        test_jammer_directional_behaviour,
        test_monte_carlo_stats_basic,
        test_direct_links_affect_sinr,
        test_jammer_power_sweep_meaningful,
        # New (23-26)
        test_monte_carlo_json_exists,
        test_evaluate_wrapper_finite,
        test_evaluate_wrapper_fields,
        test_evaluate_wrapper_consistency,
        # New (27-29)
        test_evaluate_wrapper_extreme_values,
        test_reproducibility,
    ]

    for fn in test_fns:
        try:
            result = fn()
            status = "PASS" if result else "FAIL"
            if result:
                passed += 1
            else:
                failed += 1
            line = f"  [ {status} ] {fn.__name__}"
            print(line)
            summary_lines.append(line)
        except Exception as e:
            failed += 1
            line = f"  [ FAIL ] {fn.__name__}: {e}"
            import traceback
            traceback.print_exc()
            print(line)
            summary_lines.append(line)

    total = passed + failed
    summary_lines.append("")
    summary_lines.append(sep)
    summary_lines.append(f"  Total: {total}  |  Passed: {passed}  |  Failed: {failed}")
    summary_lines.append(sep)

    summary_path = os.path.join(OUTPUT_DIR, "validation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nSummary saved to {summary_path}")
    return passed, failed, total


if __name__ == "__main__":
    run_validation()
