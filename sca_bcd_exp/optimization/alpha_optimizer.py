from __future__ import annotations

from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np

from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.optimization.sca_solver import SCAResult, solve_sca
from sca_bcd_exp.optimization.secrecy_optimizer import SolutionState, clone_solution

if TYPE_CHECKING:
    from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment

def optimize_alpha(
    env: SCABCDEnvironment,
    config: SCABCDConfig,
    solution: SolutionState,
) -> tuple[SolutionState, SCAResult]:
    x0 = solution.alpha_trajectory.copy()

    def objective_fn(x: np.ndarray) -> float:
        trial = clone_solution(solution)
        trial.alpha_trajectory = np.asarray(x, dtype=float)
        return env.evaluate_solution(trial)["objective"]

    def gradient_fn(x: np.ndarray) -> np.ndarray:
        trial = clone_solution(solution)
        trial.alpha_trajectory = np.asarray(x, dtype=float)
        return env.alpha_gradient(trial)

    def constraint_builder(var: cp.Variable, current_x: np.ndarray) -> list:
        return [
            var >= config.alpha_min,
            var <= config.alpha_max,
            cp.norm(var - current_x, 2) <= config.alpha_trust_region_radius,
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
    )

    updated = clone_solution(solution)
    updated.alpha_trajectory = np.asarray(result.x, dtype=float)
    return updated, result