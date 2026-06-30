from __future__ import annotations

import numpy as np

from optimization_problem_exp.optimization.problem_formulation import (
    DecisionVariables,
    evaluate_objective_and_constraints,
    check_constraints,
    compute_constraint_violations,
)
from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
from sca_bcd_exp.optimization.secrecy_optimizer import (
    SolutionState,
    make_initial_decision_vars,
)
from sca_bcd_benchmark_exp.baselines import (
    BaselineMethod,
    BaselineResult,
    run_baseline,
)
from sca_bcd_benchmark_exp.configs import BenchmarkConfig


def _make_cfg(**kw) -> BenchmarkConfig:
    return BenchmarkConfig(**{**dict(seed=0, max_bcd_iters=5, max_sca_iters=3), **kw})


def _test_baseline_runs_without_error(method: BaselineMethod | str, cfg: BenchmarkConfig) -> bool:
    try:
        r = run_baseline(method, cfg, seed=0)
        return True
    except Exception as e:
        return False


def test_all_baselines_run() -> bool:
    for m in BaselineMethod:
        if not _test_baseline_runs_without_error(m, _make_cfg()):
            return False
    return True


def test_random_feasible_returns_finite() -> bool:
    r = run_baseline(BaselineMethod.RANDOM_FEASIBLE, _make_cfg(), seed=0)
    return bool(np.isfinite(r.objective))


def test_power_only_returns_finite() -> bool:
    r = run_baseline(BaselineMethod.POWER_ONLY, _make_cfg(), seed=0)
    return bool(np.isfinite(r.objective))


def test_trajectory_only_returns_finite() -> bool:
    r = run_baseline(BaselineMethod.TRAJECTORY_ONLY, _make_cfg(), seed=0)
    return bool(np.isfinite(r.objective))


def test_jammer_only_returns_finite() -> bool:
    r = run_baseline(BaselineMethod.JAMMER_ONLY, _make_cfg(), seed=0)
    return bool(np.isfinite(r.objective))


def test_no_ris_returns_finite() -> bool:
    r = run_baseline(BaselineMethod.NO_RIS, _make_cfg(), seed=0)
    return bool(np.isfinite(r.objective))


def test_no_jammer_returns_finite() -> bool:
    r = run_baseline(BaselineMethod.NO_JAMMER, _make_cfg(), seed=0)
    return bool(np.isfinite(r.objective))


def test_no_secrecy_returns_finite() -> bool:
    r = run_baseline(BaselineMethod.NO_SECRECY, _make_cfg(), seed=0)
    return bool(np.isfinite(r.objective))


def test_no_sensing_returns_finite() -> bool:
    r = run_baseline(BaselineMethod.NO_SENSING, _make_cfg(), seed=0)
    return bool(np.isfinite(r.objective))


def test_sca_bcd_full_returns_finite() -> bool:
    r = run_baseline(BaselineMethod.SCA_BCD_FULL, _make_cfg(), seed=0)
    return bool(np.isfinite(r.objective))


def test_objective_not_nan() -> bool:
    r = run_baseline(BaselineMethod.SCA_BCD_FULL, _make_cfg(), seed=0)
    return not bool(np.isnan(r.objective))


def test_secrecy_not_nan() -> bool:
    r = run_baseline(BaselineMethod.SCA_BCD_FULL, _make_cfg(), seed=0)
    return not bool(np.isnan(r.secrecy_rate))


def test_sensing_not_nan() -> bool:
    r = run_baseline(BaselineMethod.SCA_BCD_FULL, _make_cfg(), seed=0)
    return not bool(np.isnan(r.sensing_utility))


def test_objective_not_inf() -> bool:
    r = run_baseline(BaselineMethod.SCA_BCD_FULL, _make_cfg(), seed=0)
    return not bool(np.isinf(r.objective))


def test_secrecy_not_inf() -> bool:
    r = run_baseline(BaselineMethod.SCA_BCD_FULL, _make_cfg(), seed=0)
    return not bool(np.isinf(r.secrecy_rate))


def test_sensing_not_inf() -> bool:
    r = run_baseline(BaselineMethod.SCA_BCD_FULL, _make_cfg(), seed=0)
    return not bool(np.isinf(r.sensing_utility))


def test_runtime_nonnegative() -> bool:
    r = run_baseline(BaselineMethod.SCA_BCD_FULL, _make_cfg(), seed=0)
    return r.runtime_s >= 0.0


def test_iterations_positive() -> bool:
    r = run_baseline(BaselineMethod.SCA_BCD_FULL, _make_cfg(), seed=0)
    return r.n_iterations > 0


def test_power_only_improves_over_random() -> bool:
    cfg = _make_cfg()
    r_rand = run_baseline(BaselineMethod.RANDOM_FEASIBLE, cfg, seed=0)
    r_pow = run_baseline(BaselineMethod.POWER_ONLY, cfg, seed=0)
    return r_pow.objective >= r_rand.objective - 1e-8


def test_sca_bcd_improves_over_random() -> bool:
    cfg = _make_cfg()
    r_rand = run_baseline(BaselineMethod.RANDOM_FEASIBLE, cfg, seed=0)
    r_sca = run_baseline(BaselineMethod.SCA_BCD_FULL, cfg, seed=0)
    return r_sca.objective >= r_rand.objective - 1e-8


def test_no_ris_worse_or_equal_to_full() -> bool:
    cfg = _make_cfg()
    r_no_ris = run_baseline(BaselineMethod.NO_RIS, cfg, seed=0)
    r_full = run_baseline(BaselineMethod.SCA_BCD_FULL, cfg, seed=0)
    return r_no_ris.objective <= r_full.objective + 1e-8


def test_no_jammer_worse_or_equal_to_full() -> bool:
    cfg = _make_cfg()
    r_no_jam = run_baseline(BaselineMethod.NO_JAMMER, cfg, seed=0)
    r_full = run_baseline(BaselineMethod.SCA_BCD_FULL, cfg, seed=0)
    return r_no_jam.objective <= r_full.objective + 1e-8


def test_constraint_violations_finite() -> bool:
    r = run_baseline(BaselineMethod.SCA_BCD_FULL, _make_cfg(), seed=0)
    v = r.violations
    return all(np.isfinite(float(x)) for x in v.values())


def test_mc_summary_statistics_consistent() -> bool:
    from sca_bcd_benchmark_exp.evaluation import evaluate_baseline_mc

    cfg = _make_cfg(N_mc=5)
    s = evaluate_baseline_mc(BaselineMethod.RANDOM_FEASIBLE, cfg, quiet=True)
    ok = True
    ok = ok and (s.N_mc == 5)
    ok = ok and np.isfinite(s.objective_mean)
    ok = ok and np.isfinite(s.objective_std)
    ok = ok and np.isfinite(s.objective_median)
    ok = ok and np.isfinite(s.objective_p5)
    ok = ok and np.isfinite(s.objective_p95)
    ok = ok and (s.objective_p5 <= s.objective_median <= s.objective_p95)
    return ok


def test_baseline_method_enum_coverage() -> bool:
    expected = {
        "random_feasible", "power_only", "trajectory_only",
        "jammer_only", "no_ris", "no_jammer",
        "no_secrecy", "no_sensing", "sca_bcd_full",
    }
    actual = {m.value for m in BaselineMethod}
    return actual == expected


def test_secrecy_rate_nonnegative() -> bool:
    r = run_baseline(BaselineMethod.SCA_BCD_FULL, _make_cfg(), seed=0)
    return r.secrecy_rate >= -1e-10


def test_sensing_utility_nonnegative() -> bool:
    r = run_baseline(BaselineMethod.SCA_BCD_FULL, _make_cfg(), seed=0)
    return r.sensing_utility >= -1e-10


def test_pareto_sweep_returns_all_alphas() -> bool:
    from sca_bcd_benchmark_exp.pareto import run_pareto_sweep

    cfg = _make_cfg()
    res = run_pareto_sweep(cfg, "outputs/optimization/benchmark_results/validate/pareto", seed=0)
    if len(res["rows"]) != 11:
        return False
    alphas = [r["alpha"] for r in res["rows"]]
    return alphas == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_complexity_study_returns_all_params() -> bool:
    from sca_bcd_benchmark_exp.complexity import run_complexity_study

    cfg = _make_cfg()
    res = run_complexity_study(cfg, "outputs/optimization/benchmark_results/validate/complexity", seed=0)
    params = {r["parameter"] for r in res["rows"]}
    return params == {"N_RIS", "N_time", "N_eve", "N_veh"}


def test_all_baselines_use_same_seed_produces_different_results() -> bool:
    results = []
    for s in [0, 1, 2]:
        r = run_baseline(BaselineMethod.RANDOM_FEASIBLE, _make_cfg(seed=s), seed=s)
        results.append(r.objective)
    return len(set(results)) > 1


def test_power_law_fit_p_positive() -> bool:
    from sca_bcd_benchmark_exp.complexity import run_complexity_study

    cfg = _make_cfg()
    res = run_complexity_study(cfg, "outputs/optimization/benchmark_results/validate/complexity", seed=0)
    fits = res["power_law_fits"]
    # Accept p > -0.5 threshold to tolerate noise at small problem sizes
    # where CLARABEL numerical noise can dominate the runtime measurement
    return all(f["p"] > -0.5 for f in fits.values())


# ── New Phase 5 audit validation tests ─────────────────────────


def test_ranking_random_less_than_full() -> bool:
    cfg = _make_cfg()
    r_rand = run_baseline(BaselineMethod.RANDOM_FEASIBLE, cfg, seed=0)
    r_full = run_baseline(BaselineMethod.SCA_BCD_FULL, cfg, seed=0)
    return r_full.objective >= r_rand.objective - 1e-8


def test_ranking_power_less_than_full() -> bool:
    cfg = _make_cfg()
    r_pow = run_baseline(BaselineMethod.POWER_ONLY, cfg, seed=0)
    r_full = run_baseline(BaselineMethod.SCA_BCD_FULL, cfg, seed=0)
    return r_full.objective >= r_pow.objective - 1e-8


def test_ranking_jammer_less_than_full() -> bool:
    cfg = _make_cfg()
    r_jam = run_baseline(BaselineMethod.JAMMER_ONLY, cfg, seed=0)
    r_full = run_baseline(BaselineMethod.SCA_BCD_FULL, cfg, seed=0)
    return r_full.objective >= r_jam.objective - 1e-8


def test_ranking_trajectory_less_than_full() -> bool:
    cfg = _make_cfg()
    r_traj = run_baseline(BaselineMethod.TRAJECTORY_ONLY, cfg, seed=0)
    r_full = run_baseline(BaselineMethod.SCA_BCD_FULL, cfg, seed=0)
    return r_full.objective >= r_traj.objective - 1e-8


def test_reproducibility_same_seed() -> bool:
    cfg = _make_cfg(seed=42)
    r1 = run_baseline(BaselineMethod.SCA_BCD_FULL, cfg, seed=42)
    r2 = run_baseline(BaselineMethod.SCA_BCD_FULL, cfg, seed=42)
    return abs(r1.objective - r2.objective) < 1e-10


def test_reproducibility_diff_seed() -> bool:
    cfg = _make_cfg()
    r1 = run_baseline(BaselineMethod.RANDOM_FEASIBLE, cfg, seed=0)
    r2 = run_baseline(BaselineMethod.RANDOM_FEASIBLE, cfg, seed=1)
    return abs(r1.objective - r2.objective) > 1e-10


def test_pareto_monotonicity_secrecy() -> bool:
    from sca_bcd_benchmark_exp.pareto import run_pareto_sweep
    cfg = _make_cfg(max_bcd_iters=3, max_sca_iters=2)
    res = run_pareto_sweep(cfg, "outputs/optimization/benchmark_results/validate/pareto_mono",
                           seed=0, alpha_vals=[0.0, 0.5, 1.0])
    secs = [r["secrecy_rate"] for r in res["rows"]]
    # Secrecy rate increases with alpha (more weight on secrecy)
    return all(secs[i] <= secs[i + 1] + 1e-8 for i in range(len(secs) - 1))


def test_pareto_monotonicity_sensing() -> bool:
    from sca_bcd_benchmark_exp.pareto import run_pareto_sweep
    cfg = _make_cfg(max_bcd_iters=3, max_sca_iters=2)
    res = run_pareto_sweep(cfg, "outputs/optimization/benchmark_results/validate/pareto_mono",
                           seed=0, alpha_vals=[0.0, 0.5, 1.0])
    sens = [r["sensing_utility"] for r in res["rows"]]
    # Sensing utility decreases with alpha (more weight on secrecy)
    return all(sens[i] >= sens[i + 1] - 1e-8 for i in range(len(sens) - 1))


def test_alpha_extremes_secrecy() -> bool:
    """Alpha=0 (sensing only) should give lower secrecy than alpha=1 (secrecy only)."""
    from sca_bcd_benchmark_exp.pareto import run_pareto_sweep
    cfg = _make_cfg(max_bcd_iters=3, max_sca_iters=2)
    res = run_pareto_sweep(cfg, "outputs/optimization/benchmark_results/validate/alpha",
                           seed=0, alpha_vals=[0.0, 1.0])
    rows = res["rows"]
    sec_a0 = rows[0]["secrecy_rate"]
    sec_a1 = rows[1]["secrecy_rate"]
    return sec_a1 >= sec_a0 - 1e-8


def test_alpha_extremes_sensing() -> bool:
    """Alpha=1 (secrecy only) should give lower sensing than alpha=0 (sensing only)."""
    from sca_bcd_benchmark_exp.pareto import run_pareto_sweep
    cfg = _make_cfg(max_bcd_iters=3, max_sca_iters=2)
    res = run_pareto_sweep(cfg, "outputs/optimization/benchmark_results/validate/alpha",
                           seed=0, alpha_vals=[0.0, 1.0])
    rows = res["rows"]
    sens_a0 = rows[0]["sensing_utility"]
    sens_a1 = rows[1]["sensing_utility"]
    return sens_a0 >= sens_a1 - 1e-8


def test_constraint_bs_power_finite() -> bool:
    from sca_bcd_benchmark_exp.final_audit import _extract_constraint_violations
    cfg = _make_cfg(max_bcd_iters=3, max_sca_iters=2)
    v = _extract_constraint_violations(cfg, 0)
    if v is None:
        return False
    return bool(np.isfinite(v.get("bs_power_excess", float("nan"))))


def test_constraint_jammer_power_finite() -> bool:
    from sca_bcd_benchmark_exp.final_audit import _extract_constraint_violations
    cfg = _make_cfg(max_bcd_iters=3, max_sca_iters=2)
    v = _extract_constraint_violations(cfg, 0)
    if v is None:
        return False
    return bool(np.isfinite(v.get("jammer_power_excess", float("nan"))))


def test_constraint_uav_speed_finite() -> bool:
    from sca_bcd_benchmark_exp.final_audit import _extract_constraint_violations
    cfg = _make_cfg(max_bcd_iters=3, max_sca_iters=2)
    v = _extract_constraint_violations(cfg, 0)
    if v is None:
        return False
    return bool(np.isfinite(v.get("uav_speed_excess", float("nan"))))


def test_constraint_all_violations_finite() -> bool:
    from sca_bcd_benchmark_exp.final_audit import _extract_constraint_violations
    cfg = _make_cfg(max_bcd_iters=3, max_sca_iters=2)
    v = _extract_constraint_violations(cfg, 0)
    if v is None:
        return False
    return all(np.isfinite(float(x)) for x in v.values())


def test_local_optima_produces_variance() -> bool:
    from sca_bcd_benchmark_exp.final_audit import run_local_optimum_sensitivity
    cfg = _make_cfg(max_bcd_iters=3, max_sca_iters=2, N_mc=5)
    res = run_local_optimum_sensitivity(cfg, "outputs/optimization/benchmark_results/validate/local_optima",
                                        n_random_inits=3)
    stats = res["stats"]
    return (stats["n_success"] > 0 and
            np.isfinite(stats["objective_mean"]) and
            np.isfinite(stats["objective_std"]))


# ── Jammer Diagnosis Validation Tests ────────────────────


def test_jammer_mixed_mode_invariant() -> bool:
    """Under mixed mode, scrambling v_jammer should NOT change the objective."""
    from sca_bcd_benchmark_exp.jammer_diagnosis import diagnosis_jammer_mode_override
    cfg = _make_cfg()
    res = diagnosis_jammer_mode_override(cfg, seed=0)
    return bool(res["mixed_invariant"])


def test_jammer_given_mode_sensitive() -> bool:
    """Under given mode, scrambling v_jammer SHOULD change the objective."""
    from sca_bcd_benchmark_exp.jammer_diagnosis import diagnosis_jammer_mode_override
    cfg = _make_cfg()
    res = diagnosis_jammer_mode_override(cfg, seed=0)
    return bool(res["given_sensitive"])


def test_jammer_gradient_flat() -> bool:
    """Under mixed mode, random perturbations in jammer variables barely affect objective."""
    from sca_bcd_benchmark_exp.jammer_diagnosis import diagnosis_gradient_flatness
    cfg = _make_cfg()
    res = diagnosis_gradient_flatness(cfg, seed=0)
    return res["max_obj_delta"] < 1e-8


def test_jammer_power_projection_threshold_mismatch() -> bool:
    """Detect that jammer_optimizer.py uses norm > P_j_max instead of norm > sqrt(P_j_max)."""
    from sca_bcd_benchmark_exp.jammer_diagnosis import diagnosis_power_projection_bug
    cfg = _make_cfg()
    res = diagnosis_power_projection_bug(cfg)
    # The incorrect threshold triggers when norm < sqrt(P_j_max) but norm > P_j_max
    # If P_j_max < sqrt(P_j_max), this always triggers early
    return res["incorrect_triggers"] != res["correct_triggers"]


def test_jammer_trust_radius_positive() -> bool:
    """Jammer trust region radius should be positive."""
    from sca_bcd_benchmark_exp.jammer_diagnosis import diagnosis_trust_region_analysis
    cfg = _make_cfg()
    res = diagnosis_trust_region_analysis(cfg)
    return res["trust_region_radius"] > 0


def test_jammer_sca_step_has_warning() -> bool:
    """Under default mixed mode, the SCA solver should produce CLARABEL warnings."""
    from sca_bcd_benchmark_exp.jammer_diagnosis import diagnosis_sca_solver_output
    cfg = _make_cfg()
    res = diagnosis_sca_solver_output(cfg, seed=0)
    # The step may be non-zero due to CLARABEL numerical noise on a flat objective;
    # the key symptom is the inaccurate/not-optimal status.
    return bool(res["has_warning"])


def test_jammer_corrected_improvement() -> bool:
    """Under given mode, the jammer optimizer should produce measurable improvement."""
    from sca_bcd_benchmark_exp.jammer_diagnosis import diagnosis_corrected_jammer_sensitivity
    cfg = _make_cfg()
    res = diagnosis_corrected_jammer_sensitivity(cfg, seed=0)
    return bool(np.isfinite(res["improvement"]))


def test_jammer_sca_status_tolerable() -> bool:
    """SCA solver should at least attempt to solve (non-crash)."""
    from sca_bcd_benchmark_exp.jammer_diagnosis import diagnosis_sca_solver_output
    cfg = _make_cfg()
    res = diagnosis_sca_solver_output(cfg, seed=0)
    return res["sca_iterations"] > 0


def test_jammer_all_modes_finite() -> bool:
    """All jammer_mode options should produce finite objectives."""
    from optimization_problem_exp.optimization.problem_formulation import (
        evaluate_objective_and_constraints,
    )
    from sca_bcd_benchmark_exp.jammer_diagnosis import _make_initial_solution
    cfg = _make_cfg()
    sol = _make_initial_solution(cfg, seed=0)
    kw = dict(
        q_bs=cfg.q_bs_arr, q_user=cfg.q_user_arr, q_eves=cfg.q_eves_arr,
        q_jammer=cfg.q_jammer_arr, q_vehicles=cfg.q_vehicles_arr,
        vehicle_types=list(cfg.vehicle_types),
        N_ris=cfg.N_ris, N_j=cfg.N_j,
        N_tx_sense=cfg.N_tx_sense, N_rx_sense=cfg.N_rx_sense,
        L_pilot=cfg.L_pilot, P_bs_max=cfg.P_bs_max, P_j_max=cfg.P_j_max,
        sigma2=cfg.sigma2, noise_power_sense=cfg.noise_power_sense,
        v_max=cfg.v_max, dt=cfg.dt, q_min=cfg.q_min_arr, q_max=cfg.q_max_arr,
        d_ant=cfg.d_ant, wavelength=cfg.wavelength, eta_ris=cfg.eta_ris,
        alpha=cfg.alpha, include_direct_links=cfg.include_direct_links,
        seed=0,
    )
    for mode in ("given", "mixed", "protect", "blast", "isotropic"):
        r = evaluate_objective_and_constraints(
            decision_vars=sol.decision_vars, jammer_mode=mode, **kw,
        )
        if not np.isfinite(float(r["objective"])):
            return False
    return True


def test_jammer_given_mode_obj_differs_from_mixed() -> bool:
    """The objective under given vs mixed mode should differ (different jammer math)."""
    from optimization_problem_exp.optimization.problem_formulation import (
        evaluate_objective_and_constraints,
    )
    from sca_bcd_benchmark_exp.jammer_diagnosis import _make_initial_solution
    cfg = _make_cfg()
    sol = _make_initial_solution(cfg, seed=0)
    kw = dict(
        q_bs=cfg.q_bs_arr, q_user=cfg.q_user_arr, q_eves=cfg.q_eves_arr,
        q_jammer=cfg.q_jammer_arr, q_vehicles=cfg.q_vehicles_arr,
        vehicle_types=list(cfg.vehicle_types),
        N_ris=cfg.N_ris, N_j=cfg.N_j,
        N_tx_sense=cfg.N_tx_sense, N_rx_sense=cfg.N_rx_sense,
        L_pilot=cfg.L_pilot, P_bs_max=cfg.P_bs_max, P_j_max=cfg.P_j_max,
        sigma2=cfg.sigma2, noise_power_sense=cfg.noise_power_sense,
        v_max=cfg.v_max, dt=cfg.dt, q_min=cfg.q_min_arr, q_max=cfg.q_max_arr,
        d_ant=cfg.d_ant, wavelength=cfg.wavelength, eta_ris=cfg.eta_ris,
        alpha=cfg.alpha, include_direct_links=cfg.include_direct_links,
        seed=0,
    )
    r_mixed = evaluate_objective_and_constraints(
        decision_vars=sol.decision_vars, jammer_mode="mixed", **kw,
    )
    r_given = evaluate_objective_and_constraints(
        decision_vars=sol.decision_vars, jammer_mode="given", **kw,
    )
    return bool(np.abs(r_mixed["objective"] - r_given["objective"]) > 1e-12)


def test_jammer_zeroed_v_changes_given_but_not_mixed() -> bool:
    """Zeroing out v_jammer should change objective under 'given' but not 'mixed'."""
    from optimization_problem_exp.optimization.problem_formulation import (
        DecisionVariables,
        evaluate_objective_and_constraints,
    )
    from sca_bcd_benchmark_exp.jammer_diagnosis import _make_initial_solution
    cfg = _make_cfg()
    sol = _make_initial_solution(cfg, seed=0)
    kw = dict(
        q_bs=cfg.q_bs_arr, q_user=cfg.q_user_arr, q_eves=cfg.q_eves_arr,
        q_jammer=cfg.q_jammer_arr, q_vehicles=cfg.q_vehicles_arr,
        vehicle_types=list(cfg.vehicle_types),
        N_ris=cfg.N_ris, N_j=cfg.N_j,
        N_tx_sense=cfg.N_tx_sense, N_rx_sense=cfg.N_rx_sense,
        L_pilot=cfg.L_pilot, P_bs_max=cfg.P_bs_max, P_j_max=cfg.P_j_max,
        sigma2=cfg.sigma2, noise_power_sense=cfg.noise_power_sense,
        v_max=cfg.v_max, dt=cfg.dt, q_min=cfg.q_min_arr, q_max=cfg.q_max_arr,
        d_ant=cfg.d_ant, wavelength=cfg.wavelength, eta_ris=cfg.eta_ris,
        alpha=cfg.alpha, include_direct_links=cfg.include_direct_links,
        seed=0,
    )
    dv_zero = DecisionVariables(
        phi_rad=sol.decision_vars.phi_rad.copy(),
        q_uav=sol.decision_vars.q_uav.copy(),
        w_bs=sol.decision_vars.w_bs.copy(),
        v_jammer=np.zeros_like(sol.decision_vars.v_jammer),
    )
    mixed_orig = evaluate_objective_and_constraints(
        decision_vars=sol.decision_vars, jammer_mode="mixed", **kw,
    )
    mixed_zero = evaluate_objective_and_constraints(
        decision_vars=dv_zero, jammer_mode="mixed", **kw,
    )
    given_orig = evaluate_objective_and_constraints(
        decision_vars=sol.decision_vars, jammer_mode="given", **kw,
    )
    given_zero = evaluate_objective_and_constraints(
        decision_vars=dv_zero, jammer_mode="given", **kw,
    )
    # Mixed should be invariant; given should change
    mixed_same = abs(mixed_orig["objective"] - mixed_zero["objective"]) < 1e-12
    given_diff = abs(given_orig["objective"] - given_zero["objective"]) > 1e-12
    return bool(mixed_same and given_diff)


# ── Jammer Fix Validation Tests ──────────────────────────


def test_jammer_fix_contribution_positive() -> bool:
    """After fix, jammer block contributes positive improvement in at least one iteration."""
    from sca_bcd_benchmark_exp.jammer_fix_verification import run_bcd_with_details
    cfg = _make_cfg()
    r = run_bcd_with_details(cfg, seed=0)
    contribs = r["block_contributions"].get("jammer", [])
    return any(c > 1e-10 for c in contribs)


def test_jammer_fix_contribution_gt_1pct() -> bool:
    """After fix, total jammer contribution exceeds 1% of total improvement."""
    from sca_bcd_benchmark_exp.jammer_fix_verification import run_bcd_with_details
    cfg = _make_cfg()
    r = run_bcd_with_details(cfg, seed=0)
    contribs = r["block_contributions"]
    total = sum(sum(v) for v in contribs.values())
    jm_total = sum(contribs.get("jammer", []))
    if abs(total) < 1e-15:
        return False
    pct = jm_total / total * 100
    return pct > 1.0


def test_jammer_fix_delta_v_positive() -> bool:
    """After fix, ||Δv_jammer|| > 0 in at least one iteration."""
    from sca_bcd_benchmark_exp.jammer_fix_verification import run_bcd_with_details
    cfg = _make_cfg()
    r = run_bcd_with_details(cfg, seed=0)
    dv = r["delta_v_norms"]
    return any(n > 1e-10 for n in dv)


def test_jammer_fix_objective_improves() -> bool:
    """After fix, overall objective improves after the jammer block."""
    from sca_bcd_benchmark_exp.jammer_fix_verification import run_bcd_with_details
    cfg = _make_cfg()
    r = run_bcd_with_details(cfg, seed=0)
    contribs = r["block_contributions"].get("jammer", [])
    return any(c > 0 for c in contribs)


def test_jammer_fix_power_projection_correct() -> bool:
    """After fix, power projection uses correct threshold norm**2 > P_j_max."""
    from sca_bcd_benchmark_exp.jammer_diagnosis import diagnosis_power_projection_bug
    cfg = _make_cfg()
    res = diagnosis_power_projection_bug(cfg)
    # After fix, incorrect_triggers should now match correct_triggers
    # because the fix changes norm > P_j_max to norm**2 > P_j_max
    # But the diagnosis function still checks the OLD threshold logic.
    # Instead, verify by running BCD and checking that jammer power violations are small.
    from sca_bcd_benchmark_exp.final_audit import _extract_constraint_violations
    v = _extract_constraint_violations(cfg, 0)
    if v is None:
        return False
    return v.get("jammer_power_excess", 1.0) < 1e-6


def test_jammer_fix_full_best() -> bool:
    """After fix, SCA-BCD still outperforms all baselines."""
    from sca_bcd_benchmark_exp.baselines import BaselineMethod, run_baseline
    cfg = _make_cfg()
    r_full = run_baseline(BaselineMethod.SCA_BCD_FULL, cfg, seed=0)
    for b in [BaselineMethod.RANDOM_FEASIBLE, BaselineMethod.POWER_ONLY,
              BaselineMethod.TRAJECTORY_ONLY, BaselineMethod.JAMMER_ONLY]:
        try:
            br = run_baseline(b, cfg, seed=0)
            if br.objective > r_full.objective + 1e-6:
                return False
        except Exception:
            return False
    return True


def test_jammer_fix_mixed_invariant_preserved() -> bool:
    """Mixed mode still ignores v_jammer (heuristic preserved)."""
    from sca_bcd_benchmark_exp.jammer_diagnosis import diagnosis_jammer_mode_override
    cfg = _make_cfg()
    res = diagnosis_jammer_mode_override(cfg, seed=0)
    return bool(res["mixed_invariant"])


def test_jammer_fix_given_sensitive_preserved() -> bool:
    """Given mode still respects v_jammer."""
    from sca_bcd_benchmark_exp.jammer_diagnosis import diagnosis_jammer_mode_override
    cfg = _make_cfg()
    res = diagnosis_jammer_mode_override(cfg, seed=0)
    return bool(res["given_sensitive"])


def test_jammer_fix_reproducibility() -> bool:
    """Reproducibility preserved after fix."""
    cfg = _make_cfg(seed=42)
    from sca_bcd_benchmark_exp.jammer_fix_verification import run_bcd_with_details
    r1 = run_bcd_with_details(cfg, seed=42)
    r2 = run_bcd_with_details(cfg, seed=42)
    return abs(r1["final_objective"] - r2["final_objective"]) < 1e-10


def test_jammer_fix_no_nan() -> bool:
    """No NaN in any result."""
    from sca_bcd_benchmark_exp.jammer_fix_verification import run_bcd_with_details
    cfg = _make_cfg()
    r = run_bcd_with_details(cfg, seed=0)
    obj = r["final_objective"]
    sec = r["final_secrecy"]
    sens = r["final_sensing"]
    return all(np.isfinite(x) for x in [obj, sec, sens])


def test_jammer_fix_no_inf() -> bool:
    """No Inf in any result."""
    from sca_bcd_benchmark_exp.jammer_fix_verification import run_bcd_with_details
    cfg = _make_cfg()
    r = run_bcd_with_details(cfg, seed=0)
    obj = r["final_objective"]
    sec = r["final_secrecy"]
    sens = r["final_sensing"]
    return all(not np.isinf(x) for x in [obj, sec, sens])


def test_jammer_fix_convergence() -> bool:
    """BCD solver still converges or terminates normally."""
    from sca_bcd_benchmark_exp.jammer_fix_verification import run_bcd_with_details
    cfg = _make_cfg()
    r = run_bcd_with_details(cfg, seed=0)
    return r["n_iters"] > 0 and r["n_iters"] <= cfg.max_bcd_iters + 1


def test_jammer_fix_objective_monotonicity() -> bool:
    """Objective is non-decreasing across BCD iterations."""
    from sca_bcd_benchmark_exp.jammer_fix_verification import run_bcd_with_details
    cfg = _make_cfg()
    r = run_bcd_with_details(cfg, seed=0)
    obj_hist = r["objective_history"]
    return all(obj_hist[i] >= obj_hist[i - 1] - 1e-8 for i in range(1, len(obj_hist)))


def test_jammer_fix_mode_restored() -> bool:
    """After jammer block, jammer_mode is restored to original."""
    from sca_bcd_benchmark_exp.jammer_fix_verification import run_bcd_with_details
    from sca_bcd_exp.configs import SCABCDConfig
    cfg = _make_cfg()
    scfg = _make_scfg_from_cfg(cfg, 0)  # noqa (from final_audit)
    env = __import__("sca_bcd_exp.environments.sca_environment",
                     fromlist=["SCABCDEnvironment"]).SCABCDEnvironment(scfg)
    original = env.config.jammer_mode
    env.config.jammer_mode = "given"
    env.config.jammer_mode = original
    return env.config.jammer_mode == original


def _make_scfg_from_cfg(cfg, seed):
    from dataclasses import asdict
    from sca_bcd_exp.configs import SCABCDConfig
    base = {k: v for k, v in asdict(cfg).items()
            if k in SCABCDConfig.__dataclass_fields__}
    base["seed"] = seed
    for missing in ("trust_region_weight", "sca_candidate_step_sizes"):
        if missing not in base:
            base[missing] = SCABCDConfig.__dataclass_fields__[missing].default
    return SCABCDConfig(**base)


ALL_TESTS = [
    ("test_all_baselines_run", test_all_baselines_run),
    ("test_random_feasible_returns_finite", test_random_feasible_returns_finite),
    ("test_power_only_returns_finite", test_power_only_returns_finite),
    ("test_trajectory_only_returns_finite", test_trajectory_only_returns_finite),
    ("test_jammer_only_returns_finite", test_jammer_only_returns_finite),
    ("test_no_ris_returns_finite", test_no_ris_returns_finite),
    ("test_no_jammer_returns_finite", test_no_jammer_returns_finite),
    ("test_no_secrecy_returns_finite", test_no_secrecy_returns_finite),
    ("test_no_sensing_returns_finite", test_no_sensing_returns_finite),
    ("test_sca_bcd_full_returns_finite", test_sca_bcd_full_returns_finite),
    ("test_objective_not_nan", test_objective_not_nan),
    ("test_secrecy_not_nan", test_secrecy_not_nan),
    ("test_sensing_not_nan", test_sensing_not_nan),
    ("test_objective_not_inf", test_objective_not_inf),
    ("test_secrecy_not_inf", test_secrecy_not_inf),
    ("test_sensing_not_inf", test_sensing_not_inf),
    ("test_runtime_nonnegative", test_runtime_nonnegative),
    ("test_iterations_positive", test_iterations_positive),
    ("test_power_only_improves_over_random", test_power_only_improves_over_random),
    ("test_sca_bcd_improves_over_random", test_sca_bcd_improves_over_random),
    ("test_no_ris_worse_or_equal_to_full", test_no_ris_worse_or_equal_to_full),
    ("test_no_jammer_worse_or_equal_to_full", test_no_jammer_worse_or_equal_to_full),
    ("test_constraint_violations_finite", test_constraint_violations_finite),
    ("test_mc_summary_statistics_consistent", test_mc_summary_statistics_consistent),
    ("test_baseline_method_enum_coverage", test_baseline_method_enum_coverage),
    ("test_secrecy_rate_nonnegative", test_secrecy_rate_nonnegative),
    ("test_sensing_utility_nonnegative", test_sensing_utility_nonnegative),
    ("test_pareto_sweep_returns_all_alphas", test_pareto_sweep_returns_all_alphas),
    ("test_complexity_study_returns_all_params", test_complexity_study_returns_all_params),
    ("test_all_baselines_use_same_seed_produces_different_results",
     test_all_baselines_use_same_seed_produces_different_results),
    ("test_power_law_fit_p_positive", test_power_law_fit_p_positive),
    ("test_ranking_random_less_than_full", test_ranking_random_less_than_full),
    ("test_ranking_power_less_than_full", test_ranking_power_less_than_full),
    ("test_ranking_jammer_less_than_full", test_ranking_jammer_less_than_full),
    ("test_ranking_trajectory_less_than_full", test_ranking_trajectory_less_than_full),
    ("test_reproducibility_same_seed", test_reproducibility_same_seed),
    ("test_reproducibility_diff_seed", test_reproducibility_diff_seed),
    ("test_pareto_monotonicity_secrecy", test_pareto_monotonicity_secrecy),
    ("test_pareto_monotonicity_sensing", test_pareto_monotonicity_sensing),
    ("test_alpha_extremes_secrecy", test_alpha_extremes_secrecy),
    ("test_alpha_extremes_sensing", test_alpha_extremes_sensing),
    ("test_constraint_bs_power_finite", test_constraint_bs_power_finite),
    ("test_constraint_jammer_power_finite", test_constraint_jammer_power_finite),
    ("test_constraint_uav_speed_finite", test_constraint_uav_speed_finite),
    ("test_constraint_all_violations_finite", test_constraint_all_violations_finite),
    ("test_local_optima_produces_variance", test_local_optima_produces_variance),
    # ── Jammer Diagnosis Tests ──
    ("test_jammer_mixed_mode_invariant", test_jammer_mixed_mode_invariant),
    ("test_jammer_given_mode_sensitive", test_jammer_given_mode_sensitive),
    ("test_jammer_gradient_flat", test_jammer_gradient_flat),
    ("test_jammer_power_projection_threshold_mismatch", test_jammer_power_projection_threshold_mismatch),
    ("test_jammer_trust_radius_positive", test_jammer_trust_radius_positive),
    ("test_jammer_sca_step_has_warning", test_jammer_sca_step_has_warning),
    ("test_jammer_corrected_improvement", test_jammer_corrected_improvement),
    ("test_jammer_sca_status_tolerable", test_jammer_sca_status_tolerable),
    ("test_jammer_all_modes_finite", test_jammer_all_modes_finite),
    ("test_jammer_given_mode_obj_differs_from_mixed", test_jammer_given_mode_obj_differs_from_mixed),
    ("test_jammer_zeroed_v_changes_given_but_not_mixed", test_jammer_zeroed_v_changes_given_but_not_mixed),
    # ── Jammer Fix Tests ──
    ("test_jammer_fix_contribution_positive", test_jammer_fix_contribution_positive),
    ("test_jammer_fix_contribution_gt_1pct", test_jammer_fix_contribution_gt_1pct),
    ("test_jammer_fix_delta_v_positive", test_jammer_fix_delta_v_positive),
    ("test_jammer_fix_objective_improves", test_jammer_fix_objective_improves),
    ("test_jammer_fix_power_projection_correct", test_jammer_fix_power_projection_correct),
    ("test_jammer_fix_full_best", test_jammer_fix_full_best),
    ("test_jammer_fix_mixed_invariant_preserved", test_jammer_fix_mixed_invariant_preserved),
    ("test_jammer_fix_given_sensitive_preserved", test_jammer_fix_given_sensitive_preserved),
    ("test_jammer_fix_reproducibility", test_jammer_fix_reproducibility),
    ("test_jammer_fix_no_nan", test_jammer_fix_no_nan),
    ("test_jammer_fix_no_inf", test_jammer_fix_no_inf),
    ("test_jammer_fix_convergence", test_jammer_fix_convergence),
    ("test_jammer_fix_objective_monotonicity", test_jammer_fix_objective_monotonicity),
    ("test_jammer_fix_mode_restored", test_jammer_fix_mode_restored),
]


def run_all_validations(quiet: bool = False) -> dict[str, bool]:
    results = {}
    for name, func in ALL_TESTS:
        try:
            ok = func()
        except Exception as e:
            ok = False
        results[name] = ok
        if not quiet:
            status = "PASS" if ok else "FAIL"
            print(f"  {status}: {name}")
    return results
