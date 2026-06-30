"""Reward normalizer and reward modes for Phase 6C5 reward repair."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from optimization_problem_exp.optimization.problem_formulation import R_S_REF


class RewardNormalizer:
    """Welford's online algorithm for running mean/std."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return math.sqrt(self.M2 / (self.count - 1))

    def zscore(self, x: float) -> float:
        s = self.std
        if s < 1e-8:
            return 0.0
        return (x - self.mean) / s


def compute_reward_original(
    R_total: float, U_total: float, u_ref: float,
    cons_viol: float, secrecy_outage: float,
    alpha: float, lambda_constraint: float, lambda_outage: float,
) -> dict:
    R_norm = R_total / R_S_REF
    U_norm = U_total / u_ref if u_ref > 0 else U_total
    f = alpha * R_norm + (1.0 - alpha) * U_norm
    reward = f - lambda_constraint * cons_viol - lambda_outage * secrecy_outage
    return {"reward": reward, "f": f, "R_norm": R_norm, "U_norm": U_norm}


def compute_reward_normalized(
    R_total: float, U_total: float,
    normalizer_R: RewardNormalizer, normalizer_U: RewardNormalizer,
    cons_viol: float, secrecy_outage: float,
    alpha: float, lambda_constraint: float, lambda_outage: float,
) -> dict:
    z_R = normalizer_R.zscore(R_total)
    z_U = normalizer_U.zscore(U_total)
    f = alpha * z_R + (1.0 - alpha) * z_U
    reward = f - lambda_constraint * cons_viol - lambda_outage * secrecy_outage
    return {"reward": reward, "f": f, "R_norm": z_R, "U_norm": z_U}


def compute_reward_full(
    R_total: float, U_total: float,
    normalizer_R: RewardNormalizer, normalizer_U: RewardNormalizer,
    cons_viol: float, secrecy_outage: float,
    alpha: float, lambda_constraint: float, lambda_outage: float,
    lambda_secret: float, R_target: float,
    actions: dict[str, np.ndarray] | None,
    lambda_action: float,
) -> dict:
    z_R = normalizer_R.zscore(R_total)
    z_U = normalizer_U.zscore(U_total)
    f = alpha * z_R + (1.0 - alpha) * z_U

    penalty_secret = lambda_secret * max(0.0, R_target - R_total)

    penalty_action = 0.0
    if actions is not None and lambda_action > 0:
        for name, act in actions.items():
            penalty_action += lambda_action * float(np.sum(act ** 2))
        penalty_action /= max(len(actions), 1)

    reward = f - lambda_constraint * cons_viol - lambda_outage * secrecy_outage - penalty_secret - penalty_action
    return {
        "reward": reward,
        "f": f,
        "R_norm": z_R,
        "U_norm": z_U,
        "penalty_secret": penalty_secret,
        "penalty_action": penalty_action,
    }
