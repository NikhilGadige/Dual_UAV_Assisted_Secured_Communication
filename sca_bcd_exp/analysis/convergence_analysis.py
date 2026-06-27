from __future__ import annotations

import numpy as np


def is_monotone_non_decreasing(values: list[float], tolerance: float = 1e-8) -> bool:
    return all(values[idx + 1] + tolerance >= values[idx] for idx in range(len(values) - 1))


def has_sca_converged(history: list[float], tolerance: float = 1e-4) -> bool:
    if len(history) < 2:
        return True
    return abs(history[-1] - history[-2]) <= tolerance or history[-1] >= history[0] - tolerance


def rolling_average(values: list[float] | np.ndarray, window: int = 100) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size < window:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="valid")


def future_convergence_placeholders() -> list[str]:
    return [
        "rolling100_objective.png",
        "rolling100_secrecy.png",
        "variance_band.png",
        "convergence_gap.png",
    ]
