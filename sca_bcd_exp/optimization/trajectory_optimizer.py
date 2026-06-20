from __future__ import annotations

from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np

from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.optimization.sca_solver import SCAResult, solve_sca
from sca_bcd_exp.optimization.secrecy_optimizer import SolutionState, clone_solution

if TYPE_CHECKING:
    from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment


def _trajectory_constraints(
    var: cp.Variable,
    start: np.ndarray,
    end: np.ndarray,
    fixed_other: np.ndarray,
    current_self: np.ndarray,
    config: SCABCDConfig,
    trust_region_radius: float,
) -> list:
    horizon = config.horizon
    path = cp.reshape(var, (horizon, 2), order="C")
    constraints = [
        path[0, :] == start,
        path[horizon - 1, :] == end,
        path >= -config.half_area,
        path <= config.half_area,
        cp.norm(var - current_self.reshape(-1), 2) <= trust_region_radius,
    ]
    max_step = config.max_speed * config.slot_duration
    for m in range(horizon):
        constraints.append(cp.norm(path[m, :], 2) <= config.max_flight_radius)
        diff0 = current_self[m] - fixed_other[m]
        lhs = np.dot(diff0, diff0) + 2.0 * diff0 @ (path[m, :] - current_self[m])
        constraints.append(lhs >= config.collision_distance ** 2)
    for m in range(horizon - 1):
        constraints.append(cp.norm(path[m + 1, :] - path[m, :], 2) <= max_step)
    return constraints


def optimize_relay_trajectory(
    env: SCABCDEnvironment,
    config: SCABCDConfig,
    solution: SolutionState,
) -> tuple[SolutionState, SCAResult]:
    x0 = solution.relay_trajectory.reshape(-1)

    def objective_fn(x: np.ndarray) -> float:
        trial = clone_solution(solution)
        trial.relay_trajectory = x.reshape(config.horizon, 2)
        return env.evaluate_solution(trial)["objective"]

    def gradient_fn(x: np.ndarray) -> np.ndarray:
        trial = clone_solution(solution)
        trial.relay_trajectory = x.reshape(config.horizon, 2)
        return env.relay_gradient(trial)

    def constraint_builder(var: cp.Variable, current_x: np.ndarray) -> list:
        return _trajectory_constraints(
            var,
            env.relay_start,
            env.relay_end,
            solution.jammer_trajectory,
            current_x.reshape(config.horizon, 2),
            config,
            config.trajectory_trust_region_radius,
        )

    def projector(x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x.reshape(config.horizon, 2), -config.half_area, config.half_area)
        for idx in range(config.horizon):
            norm = float(np.linalg.norm(clipped[idx]))
            if norm > config.max_flight_radius > 0.0:
                clipped[idx] *= config.max_flight_radius / norm
        clipped[0] = env.relay_start
        clipped[-1] = env.relay_end
        return clipped.reshape(-1)

    result = solve_sca(
        initial_x=x0,
        objective_fn=objective_fn,
        gradient_fn=gradient_fn,
        constraint_builder=constraint_builder,
        max_iters=config.max_sca_iters,
        tolerance=config.sca_tolerance,
        trust_region_weight=config.trust_region_weight,
        candidate_step_sizes=config.candidate_step_sizes,
        projector=projector,
    )
    updated = clone_solution(solution)
    updated.relay_trajectory = result.x.reshape(config.horizon, 2)
    return updated, result


def optimize_jammer_trajectory(
    env: SCABCDEnvironment,
    config: SCABCDConfig,
    solution: SolutionState,
) -> tuple[SolutionState, SCAResult]:
    x0 = solution.jammer_trajectory.reshape(-1)

    def objective_fn(x: np.ndarray) -> float:
        trial = clone_solution(solution)
        trial.jammer_trajectory = x.reshape(config.horizon, 2)
        return env.evaluate_solution(trial)["objective"]

    def gradient_fn(x: np.ndarray) -> np.ndarray:
        trial = clone_solution(solution)
        trial.jammer_trajectory = x.reshape(config.horizon, 2)
        return env.jammer_gradient(trial)

    def constraint_builder(var: cp.Variable, current_x: np.ndarray) -> list:
        return _trajectory_constraints(
            var,
            env.jammer_start,
            env.jammer_end,
            solution.relay_trajectory,
            current_x.reshape(config.horizon, 2),
            config,
            config.trajectory_trust_region_radius,
        )

    def projector(x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x.reshape(config.horizon, 2), -config.half_area, config.half_area)
        for idx in range(config.horizon):
            norm = float(np.linalg.norm(clipped[idx]))
            if norm > config.max_flight_radius > 0.0:
                clipped[idx] *= config.max_flight_radius / norm
        clipped[0] = env.jammer_start
        clipped[-1] = env.jammer_end
        return clipped.reshape(-1)

    result = solve_sca(
        initial_x=x0,
        objective_fn=objective_fn,
        gradient_fn=gradient_fn,
        constraint_builder=constraint_builder,
        max_iters=config.max_sca_iters,
        tolerance=config.sca_tolerance,
        trust_region_weight=config.trust_region_weight,
        candidate_step_sizes=config.candidate_step_sizes,
        projector=projector,
    )
    updated = clone_solution(solution)
    updated.jammer_trajectory = result.x.reshape(config.horizon, 2)
    return updated, result
