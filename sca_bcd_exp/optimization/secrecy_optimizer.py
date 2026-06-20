from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SolutionState:
    relay_trajectory: np.ndarray
    jammer_trajectory: np.ndarray
    source_power: np.ndarray
    relay_power: np.ndarray
    jammer_power: np.ndarray
    alpha_trajectory: np.ndarray


def clone_solution(solution: SolutionState) -> SolutionState:
    return SolutionState(
        relay_trajectory=solution.relay_trajectory.copy(),
        jammer_trajectory=solution.jammer_trajectory.copy(),
        source_power=solution.source_power.copy(),
        relay_power=solution.relay_power.copy(),
        jammer_power=solution.jammer_power.copy(),
        alpha_trajectory=solution.alpha_trajectory.copy(),
    )
