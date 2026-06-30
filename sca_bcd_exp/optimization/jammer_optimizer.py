from __future__ import annotations

import numpy as np

from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
from sca_bcd_exp.optimization.sca_solver import SCAResult, solve_sca_block
from sca_bcd_exp.optimization.secrecy_optimizer import SolutionState, clone_solution


def optimize_jammer(
    env: SCABCDEnvironment,
    config: SCABCDConfig,
    solution: SolutionState,
) -> tuple[SolutionState, SCAResult]:
    blocks = env.block_slices()
    sl = blocks["jammer"]
    n_time = config.N_time
    n_j = config.N_j

    x0 = env._unpack_decision_vars(solution.decision_vars)[sl]

    sqrt_Pj = np.sqrt(config.P_j_max / n_j)
    lb = np.full(2 * n_time * n_j, -sqrt_Pj)
    ub = np.full(2 * n_time * n_j, sqrt_Pj)

    def block_objective(x: np.ndarray) -> float:
        full = env._unpack_decision_vars(solution.decision_vars).copy()
        full[sl] = x
        dv = env._pack_decision_vars(full, solution)
        sol = SolutionState(decision_vars=dv)
        return env.evaluate_objective(sol)

    def block_gradient(x: np.ndarray) -> np.ndarray:
        return env.finite_diff_gradient_for_block(
            x, sl, solution,
        )

    result = solve_sca_block(
        x0=x0,
        objective_fn=block_objective,
        gradient_fn=block_gradient,
        var_lb=lb,
        var_ub=ub,
        config=config,
        trust_region_radius=0.5 * sqrt_Pj * n_time * n_j,
    )

    v_flat = result.x
    n_v = n_time * n_j
    v_re = v_flat[:n_v].reshape(n_time, n_j)
    v_im = v_flat[n_v:].reshape(n_time, n_j)
    v_jammer = v_re + 1j * v_im
    for n in range(n_time):
        power = float(np.linalg.norm(v_jammer[n]) ** 2)
        if power > config.P_j_max:
            v_jammer[n] *= np.sqrt(config.P_j_max) / np.sqrt(power)

    full = env._unpack_decision_vars(solution.decision_vars).copy()
    full[sl] = result.x
    dv = env._pack_decision_vars(full, solution)
    dv.v_jammer = v_jammer
    updated = clone_solution(solution)
    updated.decision_vars = dv
    return updated, result
