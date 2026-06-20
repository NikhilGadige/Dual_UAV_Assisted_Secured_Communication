from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cvxpy as cp
import numpy as np


Array = np.ndarray


@dataclass
class SCAResult:
    x: Array
    objective_history: list[float]
    accepted_steps: int
    accepted_step_sizes: list[float]
    status: str


def solve_sca(
    initial_x: Array,
    objective_fn: Callable[[Array], float],
    gradient_fn: Callable[[Array], Array],
    constraint_builder: Callable[[cp.Variable, Array], list],
    max_iters: int,
    tolerance: float,
    trust_region_weight: float,
    candidate_step_sizes: tuple[float, ...],
    projector: Callable[[Array], Array] | None = None,
    solver: str = "SCS",
) -> SCAResult:
    xk = np.asarray(initial_x, dtype=float).copy()
    if projector is not None:
        xk = projector(xk)

    history = [float(objective_fn(xk))]
    accepted_steps = 0
    accepted_step_sizes: list[float] = []
    status = "converged"

    for _ in range(max_iters):
        grad = np.asarray(gradient_fn(xk), dtype=float).reshape(-1)
        x = cp.Variable(xk.size)
        surrogate = grad @ (x - xk) - 0.5 * trust_region_weight * cp.sum_squares(x - xk)
        problem = cp.Problem(cp.Maximize(surrogate), constraint_builder(x, xk))
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except Exception:
            problem.solve(solver="SCS", warm_start=True, verbose=False)

        if x.value is None:
            status = str(problem.status)
            break

        candidate = np.asarray(x.value, dtype=float).reshape(xk.shape)
        if projector is not None:
            candidate = projector(candidate)

        base_value = history[-1]
        accepted = None
        accepted_step_size = None
        for step_size in candidate_step_sizes:
            trial = xk + step_size * (candidate - xk)
            if projector is not None:
                trial = projector(trial)
            trial_value = float(objective_fn(trial))
            if trial_value >= base_value - 1e-9:
                accepted = trial
                accepted_step_size = float(step_size)
                history.append(trial_value)
                accepted_steps += 1
                accepted_step_sizes.append(float(step_size))
                break

        if accepted is None:
            history.append(base_value)
            accepted_step_sizes.append(0.0)
            status = str(problem.status)
            break

        step_norm = float(np.linalg.norm(accepted - xk))
        improvement = float(abs(history[-1] - base_value))
        xk = accepted
        if step_norm <= tolerance and improvement <= tolerance:
            break
        if accepted_step_size is not None and accepted_step_size <= min(candidate_step_sizes):
            status = "small_step"

    return SCAResult(
        x=xk,
        objective_history=history,
        accepted_steps=accepted_steps,
        accepted_step_sizes=accepted_step_sizes,
        status=status,
    )
