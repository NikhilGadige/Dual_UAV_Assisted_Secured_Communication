"""Sensing metrics for the multi-agent sensing study (CRB + Pd).

Reuses already-validated repo modules instead of re-deriving the physics:
- crb_sensing_exp (Fisher Information Matrix / Cramer-Rao Bound for target
  angle estimation from a monostatic ULA)
- detection_sensing_exp (energy-detector Neyman-Pearson test statistic)
- vehicle_reflection_exp (RCS-per-vehicle-type, scalar Rician reflection
  channel — this already implements the K-factor Rician model from the
  proposed ISAC model's channel-model section)
- core.channel.path_loss (the repo's standard d^-alpha path-loss model)

Each sensing agent (a UAV at some 2D position) independently illuminates
the same K vehicle targets with a monostatic pilot and is scored on:
  - CRB trace (lower is better) for target-angle estimation
  - Pd, detection probability of the energy detector at a fixed
    Neyman-Pearson threshold (higher is better)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma as gamma_dist

from core.channel import path_loss
from vehicle_reflection_exp.channels.vehicle_channel import _generate_scalar_rician
from crb_sensing_exp.channels.crb_channel import (
    ula_steering_vector,
    ula_steering_derivative,
    target_response_matrix,
    target_response_derivative,
    composite_sensing_channel,
    compute_channel_derivatives,
    compute_fim,
    compute_crb,
)
from detection_sensing_exp.channels.detection_channel import (
    generate_h1,
    energy_detector_statistic,
)


def energy_detector_threshold(N_r: int, L: int, noise_power: float, pfa: float = 0.05) -> float:
    """Analytic Neyman-Pearson threshold for the energy detector under H0.

    Under H0, T = ||Y||_F^2 sums N_r*L iid Exponential(noise_power)
    variables (each |noise entry|^2 with real/imag ~ N(0, noise_power/2)),
    i.e. T ~ Gamma(shape=N_r*L, scale=noise_power) exactly. This gives a
    fixed Pfa without having to Monte-Carlo-calibrate a threshold on every
    RL step.
    """
    return float(gamma_dist.ppf(1.0 - pfa, a=N_r * L, scale=noise_power))


def agent_sensing_metrics(
    agent_xy: np.ndarray,
    vehicle_xy: np.ndarray,
    rcs_lin_list: list[float],
    N_tx: int,
    N_rx: int,
    L_pilot: int,
    noise_power: float,
    d_ant: float,
    wavelength: float,
    rician_k_db: float,
    pfa: float,
    num_mc: int,
    rng: np.random.RandomState,
) -> dict:
    """CRB trace + Monte-Carlo Pd for one sensing agent against all
    vehicle targets, from that agent's current 2D position.

    Monostatic geometry: the agent's own position is both the pilot
    transmitter and the echo receiver (matches the convention already
    used in optimization_problem_exp.compute_sensing_utility).
    """
    K = len(vehicle_xy)
    alphas: list[complex] = []
    A_list: list[np.ndarray] = []
    dA_list: list[np.ndarray] = []
    thetas: list[float] = []

    for k in range(K):
        dx = float(vehicle_xy[k][0] - agent_xy[0])
        dy = float(vehicle_xy[k][1] - agent_xy[1])
        theta_deg = float(np.degrees(np.arctan2(dy, dx)))
        thetas.append(theta_deg)

        # Floor the range at 5 m: a sensor can't be physically coincident
        # with a target, and letting d -> 0 blows up path_loss (~d^-2),
        # which pushes the FIM into a huge-dynamic-range, near-singular
        # regime and produces spurious huge CRB traces rather than
        # meaningful sensing improvement.
        d = max(float(np.hypot(dx, dy)), 5.0)
        pl = path_loss(d, alpha=2.0, beta0=1.0)
        rcs_lin = rcs_lin_list[k]
        alphas.append(np.sqrt(rcs_lin) * _generate_scalar_rician(K=rician_k_db, path_loss_factor=pl))

        a = ula_steering_vector(N_tx, theta_deg, d_ant, wavelength)
        da = ula_steering_derivative(N_tx, theta_deg, d_ant, wavelength)
        A_list.append(target_response_matrix(a))
        dA_list.append(target_response_derivative(a, da))

    H_sense = composite_sensing_channel(alphas, A_list)
    dH = compute_channel_derivatives(alphas, dA_list)

    X = (rng.randn(N_tx, L_pilot) + 1j * rng.randn(N_tx, L_pilot)).astype(complex)
    for col in range(L_pilot):
        nrm = float(np.linalg.norm(X[:, col]))
        if nrm > 0.0:
            X[:, col] /= nrm

    FIM = compute_fim(dH, X, noise_power)
    crb = compute_crb(FIM)
    crb_trace = float(np.trace(crb["crb_matrix"]).real)
    if not np.isfinite(crb_trace) or crb_trace < 0:
        crb_trace = 1e10

    threshold = energy_detector_threshold(N_rx, L_pilot, noise_power, pfa)
    hits = 0
    for _ in range(num_mc):
        Y1 = generate_h1(H_sense, X, noise_power)
        if energy_detector_statistic(Y1) > threshold:
            hits += 1
    pd = hits / max(num_mc, 1)

    return {"crb_trace": crb_trace, "pd": pd, "thetas": thetas}


def crb_to_utility(crb_trace: float, eps: float = 1e-6) -> float:
    """log-utility, same convention as optimization_problem_exp (higher = better)."""
    return float(-np.log10(max(crb_trace, eps)))
