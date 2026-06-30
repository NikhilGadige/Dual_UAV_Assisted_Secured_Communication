from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from sca_bcd_exp.configs import SCABCDConfig


Array = np.ndarray


@dataclass
class SCAResult:
    x: Array
    objective_history: list[float]
    obj_pen_history: list[float]
    violation_history: list[float]
    n_iters: int
    status: str


def solve_sca_block(
    x0: Array,
    objective_fn,
    gradient_fn,
    var_lb: Array,
    var_ub: Array,
    config: SCABCDConfig,
    trust_region_radius: float | None = None,
) -> SCAResult:
    xk = np.asarray(x0, dtype=float).copy()
    dim = xk.size
    reg = config.reg_eps
    rho = config.rho_penalty

    A_reg = np.eye(dim) * reg

    obj_history = []
    pen_history = []
    viol_history = []

    base_val = objective_fn(xk)
    obj_history.append(base_val)
    pen_history.append(0.0)
    viol_history.append(0.0)

    status = "max_iters"

    for iteration in range(config.max_sca_iters):
        grad = np.asarray(gradient_fn(xk), dtype=float)

        H = A_reg.copy()

        x = cp.Variable(dim)
        xi_lb = cp.Variable(dim, nonneg=True)
        xi_ub = cp.Variable(dim, nonneg=True)

        surrogate = grad @ (x - xk) - 0.5 * cp.quad_form(x - xk, H)
        penalty = rho * (cp.sum(xi_lb) + cp.sum(xi_ub))
        objective = cp.Maximize(surrogate - penalty)

        constraints = [
            var_lb - xi_lb <= x,
            x <= var_ub + xi_ub,
        ]
        if trust_region_radius is not None and trust_region_radius > 0:
            constraints.append(cp.norm(x - xk, 2) <= trust_region_radius)

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False, warm_start=True)
        except Exception:
            try:
                problem.solve(solver=cp.SCS, verbose=False, warm_start=True)
            except Exception:
                problem.solve(solver=cp.ECOS, verbose=False, warm_start=True)

        if x.value is None:
            status = f"solve_failed_{problem.status}"
            break

        candidate = np.asarray(x.value, dtype=float)

        accepted = False
        for step in config.sca_candidate_step_sizes:
            trial = xk + step * (candidate - xk)
            trial = np.clip(trial, var_lb, var_ub)
            f_trial = objective_fn(trial)
            f_current = obj_history[-1]
            if f_trial >= f_current - 1e-9:
                xk = trial.copy()
                obj_history.append(f_trial)
                pen_history.append(0.0)
                viol_history.append(0.0)
                accepted = True
                break

        if not accepted:
            status = "line_search_failed"
            break

        step_norm = float(np.linalg.norm(xk - np.asarray(x0 if iteration == 0 else obj_history[-2], dtype=float)))
        obj_change = abs(obj_history[-1] - obj_history[-2]) if len(obj_history) >= 2 else 1.0

        if obj_change < config.tol_obj and step_norm < config.tol_var:
            status = "converged"
            break

    return SCAResult(
        x=xk,
        objective_history=obj_history,
        obj_pen_history=pen_history,
        violation_history=viol_history,
        n_iters=iteration + 1,
        status=status,
    )
