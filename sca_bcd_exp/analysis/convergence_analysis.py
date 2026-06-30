from __future__ import annotations

import numpy as np


def has_converged(
    obj_history: list[float],
    var_norms: list[float],
    tol_obj: float = 1e-4,
    tol_var: float = 1e-4,
    patience: int = 3,
) -> bool:
    if len(obj_history) < patience + 1:
        return False
    obj_stable = all(
        abs(obj_history[-i] - obj_history[-i - 1]) < tol_obj
        for i in range(1, patience + 1)
    )
    var_stable = all(
        vn < tol_var for vn in var_norms[-patience:]
    ) if len(var_norms) >= patience else False
    return obj_stable or var_stable


def is_objective_finite(obj_history: list[float]) -> bool:
    return all(np.isfinite(v) for v in obj_history)


def violation_decreasing(violation_history: list[dict]) -> bool:
    if len(violation_history) < 2:
        return True
    totals = [sum(v.values()) for v in violation_history]
    return totals[-1] <= totals[0] + 1e-10


def objective_non_decreasing(obj_history: list[float], tol: float = 1e-8) -> bool:
    return all(
        obj_history[i + 1] + tol >= obj_history[i]
        for i in range(len(obj_history) - 1)
    )


def check_no_nan_inf(values: list[float]) -> bool:
    return all(np.isfinite(v) for v in values)
