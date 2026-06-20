from __future__ import annotations

from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np

from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.optimization.sca_solver import SCAResult, solve_sca
from sca_bcd_exp.optimization.secrecy_optimizer import SolutionState, clone_solution

if TYPE_CHECKING:
    from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment


def _project_average_bounded(values: np.ndarray, lower: float, upper: float, avg_budget: float) -> np.ndarray:
    projected = np.clip(np.asarray(values, dtype=float), lower, upper)
    mean_val = float(projected.mean())
    if mean_val > avg_budget and mean_val > 0.0:
        projected *= avg_budget / mean_val
        projected = np.clip(projected, lower, upper)
    return projected


def optimize_powers(
    env: SCABCDEnvironment,
    config: SCABCDConfig,
    solution: SolutionState,
) -> tuple[SolutionState, SCAResult]:
    horizon = config.horizon
    x0 = np.concatenate([solution.source_power, solution.relay_power, solution.jammer_power])

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return x[:horizon], x[horizon: 2 * horizon], x[2 * horizon:]

    def projector(x: np.ndarray) -> np.ndarray:
        src, rel, jam = unpack(x)
        src = _project_average_bounded(src, config.user_power_min, config.user_power_max, config.avg_user_power_budget)
        rel = _project_average_bounded(rel, config.relay_power_min, config.relay_power_max, config.avg_relay_power_budget)
        jam = _project_average_bounded(jam, config.jammer_power_min, config.jammer_power_max, config.avg_jammer_power_budget)
        return np.concatenate([src, rel, jam])

    def objective_fn(x: np.ndarray) -> float:
        src, rel, jam = unpack(projector(x))
        trial = clone_solution(solution)
        trial.source_power = src
        trial.relay_power = rel
        trial.jammer_power = jam
        return env.evaluate_solution(trial)["objective"]

    def gradient_fn(x: np.ndarray) -> np.ndarray:
        src, rel, jam = unpack(projector(x))
        trial = clone_solution(solution)
        trial.source_power = src
        trial.relay_power = rel
        trial.jammer_power = jam
        return env.power_gradient(trial)

    def constraint_builder(var: cp.Variable, current_x: np.ndarray) -> list:
        src = var[:horizon]
        rel = var[horizon: 2 * horizon]
        jam = var[2 * horizon:]
        return [
            src >= config.user_power_min,
            src <= config.user_power_max,
            rel >= config.relay_power_min,
            rel <= config.relay_power_max,
            jam >= config.jammer_power_min,
            jam <= config.jammer_power_max,
            cp.sum(src) / horizon <= config.avg_user_power_budget,
            cp.sum(rel) / horizon <= config.avg_relay_power_budget,
            cp.sum(jam) / horizon <= config.avg_jammer_power_budget,
            cp.norm(var - current_x, 2) <= config.power_trust_region_radius,
        ]

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

    src, rel, jam = unpack(result.x)
    updated = clone_solution(solution)
    updated.source_power = src
    updated.relay_power = rel
    updated.jammer_power = jam
    return updated, result
