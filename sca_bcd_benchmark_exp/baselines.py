from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from optimization_problem_exp.optimization.problem_formulation import (
    DecisionVariables,
    evaluate_objective_and_constraints,
    compute_ris_reflection_matrix,
)
from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
from sca_bcd_exp.optimization.bcd_solver import BCDSolver, BCDResult
from sca_bcd_exp.optimization.jammer_optimizer import optimize_jammer
from sca_bcd_exp.optimization.power_optimizer import optimize_power
from sca_bcd_exp.optimization.secrecy_optimizer import (
    SolutionState,
    clone_solution,
    make_initial_decision_vars,
)
from sca_bcd_exp.optimization.trajectory_optimizer import optimize_trajectory
from sca_bcd_benchmark_exp.configs import BenchmarkConfig


class BaselineMethod(str, Enum):
    RANDOM_FEASIBLE = "random_feasible"
    POWER_ONLY = "power_only"
    TRAJECTORY_ONLY = "trajectory_only"
    JAMMER_ONLY = "jammer_only"
    NO_RIS = "no_ris"
    NO_JAMMER = "no_jammer"
    NO_SECRECY = "no_secrecy"
    NO_SENSING = "no_sensing"
    SCA_BCD_FULL = "sca_bcd_full"


@dataclass
class BaselineResult:
    method: str
    objective: float
    secrecy_rate: float
    sensing_utility: float
    runtime_s: float
    n_iterations: int
    violation_total: float
    violations: dict = field(default_factory=dict)
    converged: bool = False
    n_bcd_iters: int = 0
    n_sca_iters: int = 0


def _make_env_and_solution(
    cfg: BenchmarkConfig,
    seed: int,
) -> tuple[SCABCDEnvironment, SolutionState]:
    from sca_bcd_exp.configs import SCABCDConfig

    scfg = SCABCDConfig(
        channel_model=cfg.channel_model,
        seed=seed,
        N_time=cfg.N_time,
        N_ris=cfg.N_ris,
        N_j=cfg.N_j,
        N_tx_sense=cfg.N_tx_sense,
        N_rx_sense=cfg.N_rx_sense,
        L_pilot=cfg.L_pilot,
        P_bs_max=cfg.P_bs_max,
        P_j_max=cfg.P_j_max,
        sigma2=cfg.sigma2,
        noise_power_sense=cfg.noise_power_sense,
        v_max=cfg.v_max,
        dt=cfg.dt,
        d_ant=cfg.d_ant,
        wavelength=cfg.wavelength,
        eta_ris=cfg.eta_ris,
        alpha=cfg.alpha,
        jammer_mode=cfg.jammer_mode,
        include_direct_links=cfg.include_direct_links,
        max_bcd_iters=cfg.max_bcd_iters,
        max_sca_iters=cfg.max_sca_iters,
        tol_obj=cfg.tol_obj,
        tol_var=cfg.tol_var,
        reg_eps=cfg.reg_eps,
        rho_penalty=cfg.rho_penalty,
        fd_h=cfg.fd_h,
        sensing_utility_mode=cfg.sensing_utility_mode,
        q_bs=cfg.q_bs,
        q_user=cfg.q_user,
        q_jammer=cfg.q_jammer,
        q_eves=cfg.q_eves,
        q_vehicles=cfg.q_vehicles,
        vehicle_types=cfg.vehicle_types,
        q_min=cfg.q_min,
        q_max=cfg.q_max,
    )
    env = SCABCDEnvironment(scfg)
    solution = env.reset()
    return env, solution, scfg


def _evaluate(env: SCABCDEnvironment, solution: SolutionState) -> dict:
    return env.evaluate(solution)


def _extract_metrics(env, solution, runtime, n_iter=0, violations_extra=None) -> BaselineResult:
    r = _evaluate(env, solution)
    viol = dict(r["violations"])
    if violations_extra:
        viol.update(violations_extra)
    return BaselineResult(
        method="",
        objective=float(r["objective"]),
        secrecy_rate=float(r["secrecy"]["R_s_total"]),
        sensing_utility=float(r["sensing"]["U_sense_total"]),
        runtime_s=runtime,
        n_iterations=n_iter,
        violation_total=float(sum(viol.values())),
        violations=viol,
    )


def _make_scfg(cfg: BenchmarkConfig, seed: int, **overrides):
    from sca_bcd_exp.configs import SCABCDConfig

    base = dict(
        channel_model=cfg.channel_model,
        seed=seed,
        N_time=cfg.N_time,
        N_ris=cfg.N_ris,
        N_j=cfg.N_j,
        N_tx_sense=cfg.N_tx_sense,
        N_rx_sense=cfg.N_rx_sense,
        L_pilot=cfg.L_pilot,
        P_bs_max=cfg.P_bs_max,
        P_j_max=cfg.P_j_max,
        sigma2=cfg.sigma2,
        noise_power_sense=cfg.noise_power_sense,
        v_max=cfg.v_max,
        dt=cfg.dt,
        d_ant=cfg.d_ant,
        wavelength=cfg.wavelength,
        eta_ris=cfg.eta_ris,
        alpha=cfg.alpha,
        jammer_mode=cfg.jammer_mode,
        include_direct_links=cfg.include_direct_links,
        sensing_utility_mode=cfg.sensing_utility_mode,
        max_bcd_iters=cfg.max_bcd_iters,
        max_sca_iters=cfg.max_sca_iters,
        tol_obj=cfg.tol_obj,
        tol_var=cfg.tol_var,
        reg_eps=cfg.reg_eps,
        rho_penalty=cfg.rho_penalty,
        fd_h=cfg.fd_h,
        q_bs=cfg.q_bs,
        q_user=cfg.q_user,
        q_jammer=cfg.q_jammer,
        q_eves=cfg.q_eves,
        q_vehicles=cfg.q_vehicles,
        vehicle_types=cfg.vehicle_types,
        q_min=cfg.q_min,
        q_max=cfg.q_max,
    )
    base.update(overrides)
    return SCABCDConfig(**base)


# ── Baseline 1: Random feasible ────────────────────────────────────


def _baseline_random_feasible(env, solution, scfg) -> BaselineResult:
    t0 = time.perf_counter()
    r = _evaluate(env, solution)
    t = time.perf_counter() - t0
    res = _extract_metrics(env, solution, t, n_iter=1)
    res.method = BaselineMethod.RANDOM_FEASIBLE.value
    return res


# ── Baseline 2: Power-only ─────────────────────────────────────────


def _baseline_power_only(env, solution, scfg) -> BaselineResult:
    t0 = time.perf_counter()
    sol, _ = optimize_power(env, scfg, solution)
    t = time.perf_counter() - t0
    r = _evaluate(env, sol)
    res = _extract_metrics(env, sol, t, n_iter=scfg.max_sca_iters)
    res.method = BaselineMethod.POWER_ONLY.value
    return res


# ── Baseline 3: Trajectory-only ────────────────────────────────────


def _baseline_trajectory_only(env, solution, scfg) -> BaselineResult:
    t0 = time.perf_counter()
    sol, _ = optimize_trajectory(env, scfg, solution)
    t = time.perf_counter() - t0
    res = _extract_metrics(env, sol, t, n_iter=scfg.max_sca_iters)
    res.method = BaselineMethod.TRAJECTORY_ONLY.value
    return res


# ── Baseline 4: Jammer-only ────────────────────────────────────────


def _baseline_jammer_only(env, solution, scfg) -> BaselineResult:
    t0 = time.perf_counter()
    sol, _ = optimize_jammer(env, scfg, solution)
    t = time.perf_counter() - t0
    res = _extract_metrics(env, sol, t, n_iter=scfg.max_sca_iters)
    res.method = BaselineMethod.JAMMER_ONLY.value
    return res


# ── Baseline 5: No RIS ─────────────────────────────────────────────


def _baseline_no_ris(env, solution, scfg) -> BaselineResult:
    t0 = time.perf_counter()
    dv = clone_solution(solution).decision_vars
    dv.phi_rad = np.zeros(scfg.N_ris, dtype=float)
    sol = SolutionState(decision_vars=dv)
    t = time.perf_counter() - t0
    res = _extract_metrics(env, sol, t, n_iter=0)
    res.method = BaselineMethod.NO_RIS.value
    return res


# ── Baseline 6: No jammer ──────────────────────────────────────────


def _baseline_no_jammer(env, solution, scfg) -> BaselineResult:
    t0 = time.perf_counter()
    dv = clone_solution(solution).decision_vars
    dv.v_jammer = np.zeros_like(dv.v_jammer)
    sol = SolutionState(decision_vars=dv)
    t = time.perf_counter() - t0
    res = _extract_metrics(env, sol, t, n_iter=0)
    res.method = BaselineMethod.NO_JAMMER.value
    return res


# ── Baseline 7: No secrecy (alpha=1) ───────────────────────────────


def _baseline_no_secrecy(env, solution, scfg) -> BaselineResult:
    t0 = time.perf_counter()
    from sca_bcd_exp.configs import SCABCDConfig

    mod_cfg = SCABCDConfig(**{**scfg.__dict__, "alpha": 0.0})
    from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment

    mod_env = SCABCDEnvironment(mod_cfg)
    mod_solution = mod_env.reset()
    solver = BCDSolver(mod_cfg)
    bcd_res = solver.solve(mod_env)
    t = time.perf_counter() - t0
    r = mod_env.evaluate(bcd_res.solution)
    res = _extract_metrics(mod_env, bcd_res.solution, t,
                            n_iter=bcd_res.n_iters)
    res.method = BaselineMethod.NO_SECRECY.value
    res.n_bcd_iters = bcd_res.n_iters
    res.converged = bcd_res.converged
    return res


# ── Baseline 8: No sensing (alpha=0) ───────────────────────────────


def _baseline_no_sensing(env, solution, scfg) -> BaselineResult:
    t0 = time.perf_counter()
    from sca_bcd_exp.configs import SCABCDConfig

    mod_cfg = SCABCDConfig(**{**scfg.__dict__, "alpha": 1.0})
    from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment

    mod_env = SCABCDEnvironment(mod_cfg)
    mod_solution = mod_env.reset()
    solver = BCDSolver(mod_cfg)
    bcd_res = solver.solve(mod_env)
    t = time.perf_counter() - t0
    r = mod_env.evaluate(bcd_res.solution)
    res = _extract_metrics(mod_env, bcd_res.solution, t,
                            n_iter=bcd_res.n_iters)
    res.method = BaselineMethod.NO_SENSING.value
    res.n_bcd_iters = bcd_res.n_iters
    res.converged = bcd_res.converged
    return res


# ── Baseline 9: SCA-BCD full ───────────────────────────────────────


def _baseline_sca_bcd_full(env, solution, scfg) -> BaselineResult:
    t0 = time.perf_counter()
    solver = BCDSolver(scfg)
    bcd_res = solver.solve(env)
    t = time.perf_counter() - t0
    r = env.evaluate(bcd_res.solution)
    res = _extract_metrics(env, bcd_res.solution, t,
                            n_iter=bcd_res.n_iters)
    res.method = BaselineMethod.SCA_BCD_FULL.value
    res.n_bcd_iters = bcd_res.n_iters
    res.converged = bcd_res.converged
    return res


# ── Dispatcher ─────────────────────────────────────────────────────


BASELINE_FUNCS = {
    BaselineMethod.RANDOM_FEASIBLE: _baseline_random_feasible,
    BaselineMethod.POWER_ONLY: _baseline_power_only,
    BaselineMethod.TRAJECTORY_ONLY: _baseline_trajectory_only,
    BaselineMethod.JAMMER_ONLY: _baseline_jammer_only,
    BaselineMethod.NO_RIS: _baseline_no_ris,
    BaselineMethod.NO_JAMMER: _baseline_no_jammer,
    BaselineMethod.NO_SECRECY: _baseline_no_secrecy,
    BaselineMethod.NO_SENSING: _baseline_no_sensing,
    BaselineMethod.SCA_BCD_FULL: _baseline_sca_bcd_full,
}


def run_baseline(
    method: BaselineMethod | str,
    cfg: BenchmarkConfig,
    seed: int = 0,
) -> BaselineResult:
    if isinstance(method, str):
        method = BaselineMethod(method)

    env, solution, scfg = _make_env_and_solution(cfg, seed)
    func = BASELINE_FUNCS[method]
    return func(env, solution, scfg)
