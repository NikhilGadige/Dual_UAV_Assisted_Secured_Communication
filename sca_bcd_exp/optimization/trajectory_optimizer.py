from __future__ import annotations

import numpy as np

from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
from sca_bcd_exp.optimization.sca_solver import SCAResult, solve_sca_block
from sca_bcd_exp.optimization.secrecy_optimizer import SolutionState, clone_solution


def optimize_trajectory(
    env: SCABCDEnvironment,
    config: SCABCDConfig,
    solution: SolutionState,
) -> tuple[SolutionState, SCAResult]:
    blocks = env.block_slices()
    sl = blocks["trajectory"]
    n_time = config.N_time

    x0 = env._unpack_decision_vars(solution.decision_vars)[sl]

    q_min = config.q_min_arr
    q_max = config.q_max_arr
    lb = np.tile(q_min, n_time)
    ub = np.tile(q_max, n_time)

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

    traj_range = float(np.max(q_max - q_min))
    trust_radius = 0.2 * traj_range

    result = solve_sca_block(
        x0=x0,
        objective_fn=block_objective,
        gradient_fn=block_gradient,
        var_lb=lb,
        var_ub=ub,
        config=config,
        trust_region_radius=trust_radius,
    )

    x_final = result.x.reshape(n_time, 3)
    x_final = np.clip(x_final, q_min, q_max)
    for n in range(1, n_time):
        prev = x_final[n - 1]
        curr = x_final[n]
        dist = np.linalg.norm(curr - prev)
        max_step = config.v_max * config.dt
        if dist > max_step:
            x_final[n] = prev + (curr - prev) * (max_step / dist)

    full = env._unpack_decision_vars(solution.decision_vars).copy()
    full[sl] = x_final.ravel()
    dv = env._pack_decision_vars(full, solution)
    updated = clone_solution(solution)
    updated.decision_vars = dv
    return updated, result
