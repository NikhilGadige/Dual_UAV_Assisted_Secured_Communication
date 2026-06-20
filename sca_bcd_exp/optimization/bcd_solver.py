from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.optimization.alpha_optimizer import optimize_alpha
from sca_bcd_exp.optimization.power_optimizer import optimize_powers
from sca_bcd_exp.optimization.secrecy_optimizer import SolutionState, clone_solution
from sca_bcd_exp.optimization.trajectory_optimizer import optimize_jammer_trajectory, optimize_relay_trajectory

if TYPE_CHECKING:
    from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment


@dataclass
class BCDResult:
    solution: SolutionState
    raw_objective_history: list[float]
    display_objective_history: list[float]
    sca_objective_history: dict[str, list[list[float]]]
    iteration_metrics: list[dict]
    diagnostics: list[dict]


class BCDSolver:
    def __init__(self, config: SCABCDConfig):
        self.config = config

    def solve(self, env: SCABCDEnvironment) -> BCDResult:
        solution = env.reset()
        metrics = env.evaluate_solution(solution)
        raw_history = [float(metrics["raw_objective"])]
        display_history = [float(metrics["raw_objective"])]
        iteration_metrics = [self._pack_iteration_metrics(0, metrics, solution)]
        diagnostics = [self._pack_diagnostics(0, metrics, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
        sca_history = {"power": [], "relay": [], "jammer": [], "alpha": []}
        patience_counter = 0

        for iteration in range(1, self.config.max_bcd_iters + 1):
            previous_solution = clone_solution(solution)
            previous_raw = raw_history[-1]

            solution, power_result = optimize_powers(env, self.config, solution)
            solution, relay_result = optimize_relay_trajectory(env, self.config, solution)
            solution, jammer_result = optimize_jammer_trajectory(env, self.config, solution)
            solution, alpha_result = optimize_alpha(env, self.config, solution)

            sca_history["power"].append(power_result.objective_history)
            sca_history["relay"].append(relay_result.objective_history)
            sca_history["jammer"].append(jammer_result.objective_history)
            sca_history["alpha"].append(alpha_result.objective_history)

            metrics = env.evaluate_solution(solution)
            raw_objective = float(metrics["raw_objective"])
            display_objective = max(display_history[-1], raw_objective)
            raw_history.append(raw_objective)
            display_history.append(display_objective)
            iteration_metrics.append(self._pack_iteration_metrics(iteration, metrics, solution))

            abs_improvement = raw_objective - previous_raw
            rel_gap = abs(abs_improvement) / max(abs(previous_raw), 1e-12)
            power_norm = self._power_norm(solution, previous_solution)
            relay_norm = float(np.linalg.norm(solution.relay_trajectory - previous_solution.relay_trajectory))
            jammer_norm = float(np.linalg.norm(solution.jammer_trajectory - previous_solution.jammer_trajectory))
            alpha_norm = float(np.linalg.norm(solution.alpha_trajectory - previous_solution.alpha_trajectory))
            diagnostics.append(
                self._pack_diagnostics(
                    iteration,
                    metrics,
                    display_objective,
                    abs_improvement,
                    rel_gap,
                    relay_norm,
                    jammer_norm,
                    power_norm,
                    alpha_norm,
                    power_result.accepted_step_sizes[-1] if power_result.accepted_step_sizes else 0.0,
                    relay_result.accepted_step_sizes[-1] if relay_result.accepted_step_sizes else 0.0,
                    jammer_result.accepted_step_sizes[-1] if jammer_result.accepted_step_sizes else 0.0,
                    alpha_result.accepted_step_sizes[-1] if alpha_result.accepted_step_sizes else 0.0,
                )
            )

            if abs_improvement <= self.config.bcd_abs_tolerance or rel_gap <= self.config.bcd_rel_tolerance:
                patience_counter += 1
            else:
                patience_counter = 0

            if iteration >= self.config.min_bcd_iters and patience_counter >= self.config.bcd_patience:
                break

        return BCDResult(
            solution=solution,
            raw_objective_history=raw_history,
            display_objective_history=display_history,
            sca_objective_history=sca_history,
            iteration_metrics=iteration_metrics,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _power_norm(solution: SolutionState, previous_solution: SolutionState) -> float:
        return float(
            np.linalg.norm(
                np.concatenate(
                    [
                        solution.source_power - previous_solution.source_power,
                        solution.relay_power - previous_solution.relay_power,
                        solution.jammer_power - previous_solution.jammer_power,
                    ]
                )
            )
        )

    @staticmethod
    def _pack_iteration_metrics(iteration: int, metrics: dict, solution: SolutionState | None = None) -> dict:
        d = {
            "iteration": iteration,
            "objective": float(metrics["raw_objective"]),
            "average_secrecy_rate": float(metrics["average_secrecy_rate"]),
            "average_num_eves": float(metrics["average_num_eves"]),
            "average_nearest_eve_distance": float(metrics["average_nearest_eve_distance"]),
            "average_max_eve_capacity": float(metrics["average_max_eve_capacity"]),
        }
        if solution is not None:
            alpha = solution.alpha_trajectory
            d["mean_alpha"] = float(np.mean(alpha))
            d["min_alpha"] = float(np.min(alpha))
            d["max_alpha"] = float(np.max(alpha))
        return d

    @staticmethod
    def _pack_diagnostics(
        iteration: int,
        metrics: dict,
        display_objective: float,
        abs_improvement: float,
        rel_gap: float,
        relay_norm: float,
        jammer_norm: float,
        power_norm: float,
        alpha_norm: float,
        power_step_size: float,
        relay_step_size: float,
        jammer_step_size: float,
        alpha_step_size: float,
    ) -> dict:
        return {
            "iteration": iteration,
            "raw_objective": float(metrics["raw_objective"]),
            "display_objective": float(display_objective),
            "average_secrecy_rate": float(metrics["average_secrecy_rate"]),
            "absolute_improvement": float(abs_improvement),
            "relative_improvement": float(rel_gap),
            "relay_update_norm": float(relay_norm),
            "jammer_update_norm": float(jammer_norm),
            "power_update_norm": float(power_norm),
            "alpha_update_norm": float(alpha_norm),
            "power_accepted_step_size": float(power_step_size),
            "relay_accepted_step_size": float(relay_step_size),
            "jammer_accepted_step_size": float(jammer_step_size),
            "alpha_accepted_step_size": float(alpha_step_size),
        }
