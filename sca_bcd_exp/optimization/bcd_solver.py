from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
from sca_bcd_exp.optimization.jammer_optimizer import optimize_jammer
from sca_bcd_exp.optimization.power_optimizer import optimize_power
from sca_bcd_exp.optimization.secrecy_optimizer import SolutionState, clone_solution
from sca_bcd_exp.optimization.trajectory_optimizer import optimize_trajectory


@dataclass
class BCDResult:
    solution: SolutionState
    objective_history: list[float]
    secrecy_history: list[float]
    sensing_history: list[float]
    violation_history: list[dict]
    block_results: dict[str, list]
    n_iters: int
    converged: bool
    delta_w_norms: list[float] = None
    delta_q_norms: list[float] = None
    delta_v_norms: list[float] = None
    block_contributions: dict[str, list[float]] = None
    obj_changes: list[float] = None
    var_changes: list[float] = None
    convergence_reason: str = ""


class BCDSolver:
    def __init__(self, config: SCABCDConfig):
        self.config = config

    def solve(self, env: SCABCDEnvironment) -> BCDResult:
        solution = env.reset()
        result = env.evaluate(solution)

        obj = float(result["objective"])
        sec = float(result["secrecy"]["R_s_total"])
        sens = float(result["sensing"]["U_sense_total"])
        viol = dict(result["violations"])

        obj_history = [obj]
        secrecy_history = [sec]
        sensing_history = [sens]
        violation_history = [viol]
        block_results = {"power": [], "trajectory": [], "jammer": []}
        delta_w_norms = []
        delta_q_norms = []
        delta_v_norms = []
        block_contributions = {"power": [], "trajectory": [], "jammer": []}
        obj_changes = []
        var_changes = []
        converged = False
        convergence_reason = "max_iterations"

        for iteration in range(self.config.max_bcd_iters):
            prev_obj = obj_history[-1]
            prev_x = env._unpack_decision_vars(solution.decision_vars).copy()
            blocks = env.block_slices()
            prev_w = prev_x[blocks["power"]].copy()
            prev_q = prev_x[blocks["trajectory"]].copy()
            prev_v = prev_x[blocks["jammer"]].copy()

            # --- Block 1: Power ---
            sol_pw, res_pw = optimize_power(env, self.config, solution)
            block_results["power"].append(res_pw)
            solution = sol_pw

            pw_x = env._unpack_decision_vars(solution.decision_vars)
            dw = float(np.linalg.norm(pw_x[blocks["power"]] - prev_w))
            pw_result = env.evaluate(solution)
            pw_obj = float(pw_result["objective"])
            pw_impr = pw_obj - prev_obj

            # --- Block 2: Trajectory ---
            sol_tj, res_tj = optimize_trajectory(env, self.config, solution)
            block_results["trajectory"].append(res_tj)
            solution = sol_tj

            tj_x = env._unpack_decision_vars(solution.decision_vars)
            dq = float(np.linalg.norm(tj_x[blocks["trajectory"]] - prev_q))
            tj_result = env.evaluate(solution)
            tj_obj = float(tj_result["objective"])
            tj_impr = tj_obj - pw_obj

            # --- Block 3: Jammer ---
            original_mode = env.config.jammer_mode
            env.config.jammer_mode = "given"

            tj_obj_given = float(env.evaluate(solution)["objective"])

            sol_jm, res_jm = optimize_jammer(env, self.config, solution)
            block_results["jammer"].append(res_jm)
            solution = sol_jm

            jm_x = env._unpack_decision_vars(solution.decision_vars)
            dv = float(np.linalg.norm(jm_x[blocks["jammer"]] - prev_v))

            result_given = env.evaluate(solution)
            obj_given = float(result_given["objective"])
            jm_impr = obj_given - tj_obj_given

            env.config.jammer_mode = original_mode

            result = env.evaluate(solution)
            obj = float(result["objective"])
            sec = float(result["secrecy"]["R_s_total"])
            sens = float(result["sensing"]["U_sense_total"])
            viol = dict(result["violations"])

            obj_history.append(obj)
            secrecy_history.append(sec)
            sensing_history.append(sens)
            violation_history.append(viol)
            delta_w_norms.append(dw)
            delta_q_norms.append(dq)
            delta_v_norms.append(dv)
            block_contributions["power"].append(pw_impr)
            block_contributions["trajectory"].append(tj_impr)
            block_contributions["jammer"].append(jm_impr)

            obj_change = abs(obj - prev_obj)
            x_current = env._unpack_decision_vars(solution.decision_vars)
            var_change = float(np.linalg.norm(x_current - prev_x))
            obj_changes.append(obj_change)
            var_changes.append(var_change)

            if obj_change < self.config.tol_obj and var_change < self.config.tol_var:
                converged = True
                convergence_reason = "tol_obj_and_tol_var"
                break
            elif obj_change < 1e-12:
                converged = True
                convergence_reason = "zero_obj_change"
                break

        return BCDResult(
            solution=solution,
            objective_history=obj_history,
            secrecy_history=secrecy_history,
            sensing_history=sensing_history,
            violation_history=violation_history,
            block_results=block_results,
            n_iters=len(obj_history),
            converged=converged,
            delta_w_norms=delta_w_norms,
            delta_q_norms=delta_q_norms,
            delta_v_norms=delta_v_norms,
            block_contributions=block_contributions,
            obj_changes=obj_changes,
            var_changes=var_changes,
            convergence_reason=convergence_reason,
        )
