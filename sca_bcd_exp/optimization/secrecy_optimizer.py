from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimization_problem_exp.optimization.problem_formulation import (
    DecisionVariables,
)


@dataclass
class SolutionState:
    decision_vars: DecisionVariables

    def __post_init__(self):
        pass


def clone_solution(solution: SolutionState) -> SolutionState:
    return SolutionState(
        decision_vars=DecisionVariables(
            phi_rad=solution.decision_vars.phi_rad.copy(),
            q_uav=solution.decision_vars.q_uav.copy(),
            w_bs=solution.decision_vars.w_bs.copy(),
            v_jammer=solution.decision_vars.v_jammer.copy(),
        )
    )


def make_initial_decision_vars(
    N_time: int, N_ris: int, N_j: int,
    M_bs: int = 4,
    P_bs_max: float = 10.0, P_j_max: float = 0.05,
    q_min: np.ndarray | None = None, q_max: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> DecisionVariables:
    if rng is None:
        rng = np.random.default_rng(0)
    if q_min is None:
        q_min = np.zeros(3)
    if q_max is None:
        q_max = np.array([400.0, 150.0, 120.0])

    phi_rad = np.zeros(N_ris, dtype=float)
    t = np.linspace(0.0, 1.0, N_time)
    q_uav = np.zeros((N_time, 3), dtype=float)
    for d in range(3):
        q_uav[:, d] = q_min[d] + t * (q_max[d] - q_min[d])
    power_per_entry = np.sqrt(P_bs_max / (N_time * M_bs))
    w_bs = np.full((N_time, M_bs), power_per_entry, dtype=complex)
    v_jammer = np.zeros((N_time, N_j), dtype=complex)
    for n in range(N_time):
        v = rng.standard_normal(N_j) + 1j * rng.standard_normal(N_j)
        v = v * np.sqrt(P_j_max / N_j) / np.linalg.norm(v)
        v_jammer[n] = v

    return DecisionVariables(
        phi_rad=phi_rad,
        q_uav=q_uav,
        w_bs=w_bs,
        v_jammer=v_jammer,
    )
