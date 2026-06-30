from __future__ import annotations

import numpy as np

from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
from sca_bcd_exp.optimization.sca_solver import SCAResult, solve_sca_block
from sca_bcd_exp.optimization.secrecy_optimizer import SolutionState, clone_solution


def optimize_power(
    env: SCABCDEnvironment,
    config: SCABCDConfig,
    solution: SolutionState,
) -> tuple[SolutionState, SCAResult]:
    blocks = env.block_slices()
    sl = blocks["power"]
    n_time = config.N_time
    m_bs = config.M_bs

    x0 = env._unpack_decision_vars(solution.decision_vars)[sl]

    sqrt_Pmax = np.sqrt(config.P_bs_max)
    lb = np.full(2 * n_time * m_bs, -sqrt_Pmax)
    ub = np.full(2 * n_time * m_bs, sqrt_Pmax)

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
        trust_region_radius=2.0 * sqrt_Pmax,
    )

    full = env._unpack_decision_vars(solution.decision_vars).copy()
    full[sl] = result.x
    dv = env._pack_decision_vars(full, solution)
    updated = clone_solution(solution)
    updated.decision_vars = dv
    return updated, result
