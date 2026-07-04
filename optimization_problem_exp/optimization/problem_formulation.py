"""Joint ISAC optimization problem formulation.

Decision variables, secrecy rate, sensing utility, objectives, constraints.
No solver — evaluation only.
"""

import os

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable

from ris_uav_exp.channels.ris_channel import (
    generate_ris_rician_channel,
    compute_ris_reflection_matrix,
    compute_effective_channel,
    compute_effective_channel_gain,
)
from fd_jammer_exp.channels.fd_jammer_channel import (
    generate_miso_rician_channel,
    compute_jammer_gain,
    isotropic_beamforming,
    mrt_beamforming,
    nullspace_beamforming,
)
from core.channel import path_loss
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
from vehicle_reflection_exp.channels.vehicle_channel import compute_rcs

# ── Fixed normalisation constants ────────────────────────
R_S_REF: float = 35.0
U_SENSE_REF: float = 3.0
DIRECT_LINK_ATTEN: float = 0.001

# ── Sensing utility modes ───────────────────────────────
SENSING_UTILITY_MODES = ["original", "log", "inverse", "normalized", "exponential"]
EPS_CRB: float = 1e-12
BETA_EXP: float = 1e6

# Will be populated by compute_normalization_constants()
_U_REF_CACHE: dict[str, float] = {}

# ── Decision variables ──────────────────────────────────

@dataclass
class DecisionVariables:
    phi_rad: np.ndarray
    q_uav: np.ndarray
    w_bs: np.ndarray
    v_jammer: np.ndarray

    def __post_init__(self):
        if self.phi_rad.ndim == 1:
            self.phi_rad = self.phi_rad.reshape(-1)
        if self.q_uav.ndim == 1:
            self.q_uav = self.q_uav.reshape(1, -1)

    @property
    def N_time(self) -> int:
        return self.q_uav.shape[0]

    @property
    def N_ris(self) -> int:
        return len(self.phi_rad)

    @property
    def N_j(self) -> int:
        return self.v_jammer.shape[-2] if self.v_jammer.ndim >= 2 else 1


# ── MIMO BS channel helper ──────────────────────────────

def generate_mimo_rician_channel(
    N_rx: int,
    N_tx: int,
    K: float = 5.0,
    path_loss_factor: float = 1.0,
) -> np.ndarray:
    """Generate Rician MIMO channel matrix (N_rx, N_tx).

    Each entry is a complex Gaussian with K-factor LOS component.
    """
    K_lin = 10.0 ** (K / 10.0) if K < 100 else K
    scale_los = np.sqrt(K_lin / (K_lin + 1.0))
    scale_nlos = np.sqrt(1.0 / (K_lin + 1.0))

    los = np.exp(1j * np.random.uniform(0.0, 2.0 * np.pi, size=(N_rx, N_tx)))
    nlos = (
        np.random.normal(0.0, 1.0 / np.sqrt(2.0), size=(N_rx, N_tx))
        + 1j * np.random.normal(0.0, 1.0 / np.sqrt(2.0), size=(N_rx, N_tx))
    )
    H = np.sqrt(path_loss_factor) * (scale_los * los + scale_nlos * nlos)
    return H


def compute_mimo_bs_user_channel(
    q_bs: np.ndarray, q_user: np.ndarray, M_bs: int, seed: int = 0,
) -> np.ndarray:
    """Direct BS -> User MISO channel vector (M_bs,)."""
    np.random.seed(seed)
    return generate_mimo_rician_channel(
        1, M_bs, K=0.0,
        path_loss_factor=_pl(q_bs, q_user) * DIRECT_LINK_ATTEN,
    ).ravel()


def compute_mimo_bs_eve_channel(
    q_bs: np.ndarray, q_eve: np.ndarray, M_bs: int, seed: int = 0,
) -> np.ndarray:
    """Direct BS -> Eve MISO channel vector (M_bs,)."""
    np.random.seed(seed)
    return generate_mimo_rician_channel(
        1, M_bs, K=0.0,
        path_loss_factor=_pl(q_bs, q_eve) * DIRECT_LINK_ATTEN,
    ).ravel()


# ── Path-loss helper ────────────────────────────────────

def _pl(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    return path_loss(float(np.linalg.norm(pos_a - pos_b)), alpha=2.0, beta0=1.0)


# ── Weak direct BS-User / BS-Eve links ──────────────────

def _compute_direct_link(q_tx: np.ndarray, q_rx: np.ndarray, seed: int) -> complex:
    """Very weak direct path channel gain (scalar)."""
    np.random.seed(seed)
    K = 0.0
    h_scalar = generate_ris_rician_channel(1, K=K, path_loss_factor=1.0)
    pl = _pl(q_tx, q_rx)
    return h_scalar[0] * np.sqrt(pl * DIRECT_LINK_ATTEN)


def compute_direct_bs_user_channel(
    q_bs: np.ndarray, q_user: np.ndarray, seed: int = 0,
) -> complex:
    """Direct BS -> User scalar channel gain."""
    return _compute_direct_link(q_bs, q_user, seed)


def compute_direct_bs_eve_channel(
    q_bs: np.ndarray, q_eve: np.ndarray, seed: int = 0,
) -> complex:
    """Direct BS -> Eve scalar channel gain."""
    return _compute_direct_link(q_bs, q_eve, seed)


# ── Heuristic directional jammer ─────────────────────────

def design_heuristic_jammer_beam(
    h_JU: np.ndarray,
    h_JE_list: list[np.ndarray],
    N_j: int,
    P_j: float,
    mode: str = "protect",
    mix_alpha: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Design jammer beamforming vector w (N_j, 1) heuristically.

    mode="protect": null toward user, MRT toward eves.
    mode="blast":   MRT toward user (maximises interference at user).
    mode="isotropic": equal-gain random phases.
    mode="mixed":   convex combination of protect and blast (mix_alpha blends).
    """
    np.random.seed(seed)

    if mode == "isotropic" or N_j < 1:
        return isotropic_beamforming(N_j) * np.sqrt(P_j)

    if mode == "protect":
        h_norm_sq = float(np.sum(np.abs(h_JU) ** 2))
        if h_norm_sq < 1e-15:
            return design_heuristic_jammer_beam(
                h_JU, h_JE_list, N_j, P_j, "isotropic", seed=seed + 1,
            )
        P_null = np.eye(N_j, dtype=complex) - (
            h_JU.conj().T @ h_JU
        ) / h_norm_sq
        h_JE_sum = sum(h_JE_list)
        v_dir = P_null @ h_JE_sum.conj().T

    elif mode == "blast":
        v_dir = h_JU.conj().T

    elif mode == "mixed":
        v_protect = design_heuristic_jammer_beam(
            h_JU, h_JE_list, N_j, 1.0, "protect", seed=seed,
        )
        v_blast = design_heuristic_jammer_beam(
            h_JU, h_JE_list, N_j, 1.0, "blast", seed=seed + 1,
        )
        v_dir = mix_alpha * v_protect + (1.0 - mix_alpha) * v_blast

    else:
        return design_heuristic_jammer_beam(
            h_JU, h_JE_list, N_j, P_j, "isotropic", seed=seed,
        )

    norm = float(np.linalg.norm(v_dir))
    if norm < 1e-15:
        return design_heuristic_jammer_beam(
            h_JU, h_JE_list, N_j, P_j, "isotropic", seed=seed + 2,
        )
    return v_dir / norm * np.sqrt(P_j)


# ── Dedicated channel functions ─────────────────────────

def compute_bs_ris_channel(
    q_bs: np.ndarray,
    q_uav: np.ndarray,
    N_ris: int,
    seed: int = 0,
    M_bs: int = 1,
) -> np.ndarray:
    """BS -> RIS channel.

    M_bs = 1 (default): returns vector (N_ris,).
    M_bs > 1:           returns matrix (N_ris, M_bs).
    """
    np.random.seed(seed)
    if M_bs <= 1:
        return generate_ris_rician_channel(
            N_ris, K=5.0,
            path_loss_factor=_pl(q_bs, q_uav),
        )
    return generate_mimo_rician_channel(
        N_ris, M_bs, K=5.0,
        path_loss_factor=_pl(q_bs, q_uav),
    )


def compute_ris_user_channel(
    q_uav: np.ndarray,
    q_user: np.ndarray,
    N_ris: int,
    seed: int = 0,
) -> np.ndarray:
    """RIS -> User channel vector (N_ris,).

    h_RU = sqrt(PL) * Rician channel vector.
    """
    np.random.seed(seed)
    return generate_ris_rician_channel(
        N_ris, K=5.0,
        path_loss_factor=_pl(q_uav, q_user),
    )


def compute_ris_eve_channel(
    q_uav: np.ndarray,
    q_eve: np.ndarray,
    N_ris: int,
    seed: int = 0,
) -> np.ndarray:
    """RIS -> Eve channel vector (N_ris,).

    h_RE = sqrt(PL) * Rician channel vector.
    """
    np.random.seed(seed)
    return generate_ris_rician_channel(
        N_ris, K=5.0,
        path_loss_factor=_pl(q_uav, q_eve),
    )


def compute_jammer_user_channel(
    q_jammer: np.ndarray,
    q_user: np.ndarray,
    N_j: int,
    seed: int = 0,
) -> np.ndarray:
    """Jammer -> User MISO channel (1 x N_j).

    h_JU = sqrt(PL) * Rician channel.
    """
    np.random.seed(seed)
    return generate_miso_rician_channel(
        N_j, K=5.0,
        path_loss_factor=_pl(q_jammer, q_user),
    )


def compute_jammer_eve_channel(
    q_jammer: np.ndarray,
    q_eve: np.ndarray,
    N_j: int,
    seed: int = 0,
) -> np.ndarray:
    """Jammer -> Eve MISO channel (1 x N_j).

    h_JE = sqrt(PL) * Rician channel.
    """
    np.random.seed(seed)
    return generate_miso_rician_channel(
        N_j, K=5.0,
        path_loss_factor=_pl(q_jammer, q_eve),
    )


# ── SINR computations ───────────────────────────────────

def compute_user_sinr(
    P_bs: float,
    gain_user: float,
    P_j: float,
    jammer_gain_user: float,
    sigma2: float,
) -> float:
    """User SINR.

    SINR_u = P_bs * G_u / (P_j * J_u + sigma2)
    """
    den = P_j * jammer_gain_user + sigma2
    return P_bs * gain_user / den if den > 0 else 0.0


def compute_eve_sinr(
    P_bs: float,
    gain_eve: float,
    P_j: float,
    jammer_gain_eve: float,
    sigma2: float,
) -> float:
    """Eve SINR.

    SINR_e = P_bs * G_e / (P_j * J_e + sigma2)
    """
    den = P_j * jammer_gain_eve + sigma2
    return P_bs * gain_eve / den if den > 0 else 0.0


# ── RIS phase design (heuristic, not optimization) ──────

def design_ris_phases(
    h_BR: np.ndarray,
    h_RU: np.ndarray,
) -> np.ndarray:
    """Set RIS phases for constructive beamforming at the user.

    phi_i = angle(h_RU[i]) - angle(h_BR_combined[i])

    For MIMO BS (h_BR matrix), combine the BS-antenna paths per RIS element.
    This aligns the cascaded BS-RIS-User channel so all elements
    add coherently, maximising |h_eff_user|.
    """
    if h_BR.ndim > 1:
        h_BR_combined = np.sum(h_BR, axis=1)
    else:
        h_BR_combined = h_BR
    return np.angle(h_RU) - np.angle(h_BR_combined)


# ── Phase interpolation helper ──────────────────────────

def _circular_interp(phi_a: np.ndarray, phi_b: np.ndarray, t: float) -> np.ndarray:
    """Smooth circular interpolation between two phase vectors.

    At t=1: phi = phi_a (aligned for user).
    At t=0: phi = phi_b (random).
    Intermediate values give a smooth blend.
    """
    z = t * np.exp(1j * phi_a) + (1.0 - t) * np.exp(1j * phi_b)
    z = z / (np.abs(z) + 1e-30)
    return np.angle(z)


# ── Secrecy rate ────────────────────────────────────────

def compute_secrecy_rate(
    q_bs: np.ndarray,
    q_user: np.ndarray,
    q_eves: np.ndarray,
    q_jammer: np.ndarray,
    N_ris: int,
    N_j: int,
    Phi: np.ndarray | None,
    q_uav: np.ndarray,
    w_bs: np.ndarray,
    v_jammer: np.ndarray,
    P_bs_max: float,
    P_j_max: float,
    sigma2: float,
    seed: int = 0,
    jammer_mode: str = "given",
    jammer_mix_alpha: float = 0.5,
    include_direct_links: bool = False,
    eta_ris: float = 1.0,
    ris_alignment_alpha: float = 1.0,
    ris_phase_noise_std: float = 0.0,
    jammer_power_factor: float = 1.0,
    M_bs: int = 1,
    phi_override: np.ndarray | None = None,
):
    """Compute secrecy rate per time slot and total.

    Supports both single-antenna BS (M_bs=1, default) and
    multi-antenna BS (M_bs > 1) with beamforming vectors w_bs shape
    (N_time,) or (N_time, M_bs) respectively.

    Uses dedicated channel functions plus smoothed RIS and jammer.

    jammer_mode:
        "given"    — use v_jammer as provided
        "protect"  — null toward user, MRT toward eves
        "blast"    — MRT toward user (degrades comm)
        "mixed"    — convex blend of protect and blast
    include_direct_links — add weak direct BS-user / BS-eve paths.
    eta_ris: RIS power efficiency (reflection loss), 0 < eta_ris <= 1.
    ris_alignment_alpha: 1=fully aligned for user, 0=random phases, smooth blend.
    ris_phase_noise_std: std of Gaussian phase noise added to RIS (rad).
    jammer_power_factor: scales the jammer transmit power (0..1).
    M_bs: number of BS antennas (1 = scalar w_bs, >1 = vector per slot).
    phi_override: explicit RIS phase-shift decision variable, shape
        (N_time, N_ris) in radians. When given, it takes precedence over
        Phi / ris_alignment_alpha for every slot — this is the hook used
        to let an external optimizer (e.g. an RL agent) actually control
        phi_n instead of relying on the closed-form alignment heuristic.

    Returns dict with R_s_per_slot, R_s_total, SINR_user, SINR_eve,
    and average channel gain info.
    """
    N_time = q_uav.shape[0]
    N_eve = len(q_eves)

    R_s_slots = np.zeros(N_time)
    SINR_user_slots = np.zeros(N_time)
    SINR_eve_slots = np.zeros((N_time, N_eve))
    R_user_slots = np.zeros(N_time)
    R_eve_max_slots = np.zeros(N_time)
    gain_user_list = []
    gain_eve_list = []

    for n in range(N_time):
        base_seed = seed + n * 10

        h_BR = compute_bs_ris_channel(q_bs, q_uav[n], N_ris, base_seed, M_bs=M_bs)
        h_RU = compute_ris_user_channel(q_uav[n], q_user, N_ris, base_seed + 1)

        h_RE_list = []
        for ke in range(N_eve):
            h_RE = compute_ris_eve_channel(
                q_uav[n], q_eves[ke], N_ris, base_seed + 2 + ke,
            )
            h_RE_list.append(h_RE)

        h_JU = compute_jammer_user_channel(
            q_jammer, q_user, N_j, base_seed + 10,
        )
        h_JE_list = []
        for ke in range(N_eve):
            h_JE = compute_jammer_eve_channel(
                q_jammer, q_eves[ke], N_j, base_seed + 11 + ke,
            )
            h_JE_list.append(h_JE)

        # Direct links (optional, very weak)
        if include_direct_links and M_bs > 1:
            h_direct_u = compute_mimo_bs_user_channel(
                q_bs, q_user, M_bs, base_seed + 20,
            )
            h_direct_e_list = [
                compute_mimo_bs_eve_channel(
                    q_bs, q_eves[ke], M_bs, base_seed + 21 + ke,
                )
                for ke in range(N_eve)
            ]
        elif include_direct_links:
            h_direct_u = compute_direct_bs_user_channel(
                q_bs, q_user, base_seed + 20,
            )
            h_direct_e_list = [
                compute_direct_bs_eve_channel(
                    q_bs, q_eves[ke], base_seed + 21 + ke,
                ) for ke in range(N_eve)
            ]
        else:
            if M_bs > 1:
                h_direct_u = np.zeros(M_bs, dtype=complex)
                h_direct_e_list = [np.zeros(M_bs, dtype=complex) for _ in range(N_eve)]
            else:
                h_direct_u = 0j
                h_direct_e_list = [0j] * N_eve

        # RIS phase design with smooth alpha interpolation
        if phi_override is not None:
            phi = phi_override[n]
            if ris_phase_noise_std > 0.0:
                phi = phi + np.random.normal(0.0, ris_phase_noise_std, N_ris)
            Phi_local = compute_ris_reflection_matrix(phi)
        elif Phi is not None:
            Phi_local = Phi
        elif ris_alignment_alpha >= 0.999:
            phi_aligned = design_ris_phases(h_BR, h_RU)
            phi = phi_aligned.copy()
            if ris_phase_noise_std > 0.0:
                phi += np.random.normal(0.0, ris_phase_noise_std, N_ris)
            Phi_local = compute_ris_reflection_matrix(phi)
        elif ris_alignment_alpha <= 0.001:
            phi = np.random.uniform(-np.pi, np.pi, N_ris)
            if ris_phase_noise_std > 0.0:
                phi += np.random.normal(0.0, ris_phase_noise_std, N_ris)
            Phi_local = compute_ris_reflection_matrix(phi)
        else:
            phi_aligned = design_ris_phases(h_BR, h_RU)
            phi_rand = np.random.uniform(-np.pi, np.pi, N_ris)
            phi = _circular_interp(phi_aligned, phi_rand, ris_alignment_alpha)
            if ris_phase_noise_std > 0.0:
                phi += np.random.normal(0.0, ris_phase_noise_std, N_ris)
            Phi_local = compute_ris_reflection_matrix(phi)

        # ── MIMO / SISO-SINR branching ──────────────────────────────
        if M_bs > 1:
            # Effective channel vectors: (M_bs,) each antenna → user/eve
            h_eff_user_vec = (
                compute_effective_channel(h_RU, Phi_local, h_BR)
                + h_direct_u
            )
            gain_user = eta_ris * float(np.abs(h_eff_user_vec.conj() @ w_bs[n]) ** 2)
            gain_user_list.append(gain_user)
        else:
            h_eff_user = (
                compute_effective_channel(h_RU, Phi_local, h_BR) + h_direct_u
            )
            gain_user = eta_ris * compute_effective_channel_gain(h_eff_user)
            gain_user_list.append(gain_user)

        # BS power
        if M_bs > 1:
            P_bs = min(float(np.linalg.norm(w_bs[n]) ** 2), P_bs_max)
        else:
            P_bs = min(float(np.abs(w_bs[n]) ** 2), P_bs_max)

        # Jammer beam: use provided v_jammer or design heuristically
        if jammer_mode == "given":
            v = v_jammer[n].reshape(N_j, 1)
        else:
            v = design_heuristic_jammer_beam(
                h_JU, h_JE_list, N_j, P_j_max,
                mode=jammer_mode, mix_alpha=jammer_mix_alpha,
                seed=base_seed + 30,
            )
            v = v * np.sqrt(max(jammer_power_factor, 0.0))
        P_j = min(float(np.linalg.norm(v) ** 2), P_j_max)

        jammer_gain_user = compute_jammer_gain(h_JU, v)

        if M_bs > 1:
            SINR_user = gain_user / max(P_j * jammer_gain_user + sigma2, 1e-30)
        else:
            SINR_user = compute_user_sinr(
                P_bs, gain_user, P_j, jammer_gain_user, sigma2,
            )

        SINR_eves_n = np.zeros(N_eve)
        for ke in range(N_eve):
            if M_bs > 1:
                h_eff_eve_vec = (
                    compute_effective_channel(h_RE_list[ke], Phi_local, h_BR)
                    + h_direct_e_list[ke]
                )
                gain_eve = eta_ris * float(np.abs(h_eff_eve_vec.conj() @ w_bs[n]) ** 2)
            else:
                h_eff_eve = (
                    compute_effective_channel(h_RE_list[ke], Phi_local, h_BR)
                    + h_direct_e_list[ke]
                )
                gain_eve = eta_ris * compute_effective_channel_gain(h_eff_eve)
            gain_eve_list.append(gain_eve)

            jammer_gain_eve = compute_jammer_gain(h_JE_list[ke], v)
            if M_bs > 1:
                SINR_eve = gain_eve / max(P_j * jammer_gain_eve + sigma2, 1e-30)
            else:
                SINR_eve = compute_eve_sinr(
                    P_bs, gain_eve, P_j, jammer_gain_eve, sigma2,
                )
            SINR_eves_n[ke] = SINR_eve

        R_user = np.log2(np.maximum(1.0 + SINR_user, 1e-12))
        R_eve_max = float(np.max(np.log2(np.maximum(1.0 + SINR_eves_n, 1e-12))))
        R_s = max(R_user - R_eve_max, 0.0)

        R_s_slots[n] = R_s
        SINR_user_slots[n] = SINR_user
        SINR_eve_slots[n, :] = SINR_eves_n
        R_user_slots[n] = R_user
        R_eve_max_slots[n] = R_eve_max

    return {
        "R_s_per_slot": R_s_slots,
        "R_s_total": float(np.sum(R_s_slots)),
        "SINR_user": SINR_user_slots,
        "SINR_eve": SINR_eve_slots,
        "R_user": R_user_slots,
        "R_eve_max": R_eve_max_slots,
        "gain_user_avg": float(np.mean(gain_user_list)) if gain_user_list else 0.0,
        "gain_eve_avg": float(np.mean(gain_eve_list)) if gain_eve_list else 0.0,
    }


# ── Sensing utility functions (multi-mode) ──────────────

def compute_utility_original(crb_trace: float) -> float:
    return 1.0 / (1.0 + crb_trace)


def compute_utility_log(crb_trace: float, eps: float = EPS_CRB) -> float:
    return float(-np.log10(max(crb_trace, eps)))


def compute_utility_inverse(crb_trace: float, eps: float = EPS_CRB) -> float:
    return 1.0 / max(crb_trace, eps)


def compute_utility_normalized(
    crb_trace: float, tr_max: float, tr_min: float, eps: float = EPS_CRB,
) -> float:
    denom = tr_max - tr_min + eps
    return (tr_max - crb_trace) / denom


def compute_utility_exponential(crb_trace: float, beta: float = BETA_EXP) -> float:
    return float(np.exp(-beta * crb_trace))


def apply_sensing_utility(
    crb_trace_per_slot: np.ndarray,
    mode: str = "log",
    tr_max: float | None = None,
    tr_min: float | None = None,
    beta: float = BETA_EXP,
) -> dict:
    """Apply a sensing utility mode to per-slot CRB traces.

    Returns dict with all mode utilities plus the selected one.
    """
    N = len(crb_trace_per_slot)
    u_orig = np.array([compute_utility_original(c) for c in crb_trace_per_slot])
    u_log = np.array([compute_utility_log(c) for c in crb_trace_per_slot])
    u_inv = np.array([compute_utility_inverse(c) for c in crb_trace_per_slot])

    # Compute tr_max/tr_min from the data for normalized mode
    finite = crb_trace_per_slot[np.isfinite(crb_trace_per_slot) & (crb_trace_per_slot > 0)]
    tm = float(np.max(finite)) if len(finite) > 0 and tr_max is None else (tr_max or 1.0)
    tn = float(np.min(finite)) if len(finite) > 0 and tr_min is None else (tr_min or 0.0)
    u_norm = np.array([compute_utility_normalized(c, tm, tn) for c in crb_trace_per_slot])
    u_exp = np.array([compute_utility_exponential(c, beta) for c in crb_trace_per_slot])

    mode_map = {
        "original": u_orig,
        "log": u_log,
        "inverse": u_inv,
        "normalized": u_norm,
        "exponential": u_exp,
    }
    selected = mode_map.get(mode, u_log)

    return {
        "U_sense_per_slot": selected.tolist() if isinstance(selected, np.ndarray) else selected,
        "U_sense_total": float(np.sum(selected)),
        "U_original_total": float(np.sum(u_orig)),
        "U_log_total": float(np.sum(u_log)),
        "U_inverse_total": float(np.sum(u_inv)),
        "U_normalized_total": float(np.sum(u_norm)),
        "U_exponential_total": float(np.sum(u_exp)),
        "U_original_per_slot": u_orig.tolist(),
        "U_log_per_slot": u_log.tolist(),
        "U_inverse_per_slot": u_inv.tolist(),
        "U_normalized_per_slot": u_norm.tolist(),
        "U_exponential_per_slot": u_exp.tolist(),
        "mode": mode,
    }


def compute_sensing_utility(
    q_uav: np.ndarray,
    q_vehicles: np.ndarray,
    rcs_list: list[float],
    N_tx: int,
    N_rx: int,
    L_pilot: int,
    noise_power: float,
    d_ant: float = 0.5,
    wavelength: float = 1.0,
    seed: int = 0,
    mode: str = "log",
):
    """Compute sensing utility using selected mode.

    Always returns all mode utilities plus the selected one.
    Modes: original, log, inverse, normalized, exponential.
    Default: log.
    """
    N_time = q_uav.shape[0]
    N_veh = len(q_vehicles)
    CRB_trace_per_slot = np.zeros(N_time)

    for n in range(N_time):
        rng = np.random.RandomState(seed + n)
        thetas = []
        alphas = []
        for kv in range(N_veh):
            dx = q_vehicles[kv][0] - q_uav[n][0]
            dy = q_vehicles[kv][1] - q_uav[n][1]
            theta_deg = float(np.degrees(np.arctan2(dy, dx)))
            thetas.append(theta_deg)

            rcs_lin = 10.0 ** (rcs_list[kv] / 10.0)
            pl = _pl(q_uav[n], q_vehicles[kv])
            alpha_val = complex(
                np.sqrt(pl * rcs_lin) * rng.randn() / np.sqrt(2),
                np.sqrt(pl * rcs_lin) * rng.randn() / np.sqrt(2),
            )
            alphas.append(alpha_val)

        A_list = []
        dA_list = []
        for th in thetas:
            a = ula_steering_vector(N_tx, th, d_ant, wavelength)
            da = ula_steering_derivative(N_tx, th, d_ant, wavelength)
            a_rx = ula_steering_vector(N_rx, th, d_ant, wavelength)
            da_rx = ula_steering_derivative(N_rx, th, d_ant, wavelength)
            A_list.append(target_response_matrix(a_rx))
            dA_list.append(target_response_derivative(a_rx, da_rx))

        H_sense = composite_sensing_channel(alphas, A_list)
        dH = compute_channel_derivatives(alphas, dA_list)

        X_pilot = rng.randn(N_tx, L_pilot) + 1j * rng.randn(N_tx, L_pilot)
        for col in range(L_pilot):
            nrm = float(np.linalg.norm(X_pilot[:, col]))
            if nrm > 0.0:
                X_pilot[:, col] /= nrm

        FIM = compute_fim(dH, X_pilot, noise_power)
        crb_result = compute_crb(FIM)

        CRB_trace = float(np.trace(crb_result["crb_matrix"]).real)
        if not np.isfinite(CRB_trace) or CRB_trace < 0:
            CRB_trace = 1e10

        CRB_trace_per_slot[n] = CRB_trace

    result = apply_sensing_utility(CRB_trace_per_slot, mode=mode)
    result["CRB_trace_per_slot"] = CRB_trace_per_slot
    result["CRB_trace_total"] = float(np.sum(CRB_trace_per_slot))
    return result


# ── Weighted objective ──────────────────────────────────

def evaluate_weighted_objective(
    alpha: float,
    R_s_total: float,
    U_sense_total: float,
    R_s_ref: float | None = None,
    U_sense_ref: float | None = None,
):
    """f = alpha * R_s/R_s_ref + (1-alpha) * U_sense/U_sense_ref

    Uses fixed constants R_S_REF / U_SENSE_REF unless overridden.
    """
    r_ref = R_S_REF if R_s_ref is None else R_s_ref
    u_ref = U_SENSE_REF if U_sense_ref is None else U_sense_ref
    R_s_norm = R_s_total / r_ref if r_ref > 0 else R_s_total
    U_sense_norm = U_sense_total / u_ref if u_ref > 0 else U_sense_total
    f = alpha * R_s_norm + (1.0 - alpha) * U_sense_norm
    return f


def compute_normalization_constants(
    env: Any,
    n_mc: int = 100,
    modes: list[str] | None = None,
    seed_offset: int = 0,
) -> dict[str, float]:
    """Compute U_ref for each sensing utility mode via Monte Carlo.

    Returns dict mapping mode -> mean U across n_mc random realizations.
    """
    if modes is None:
        modes = SENSING_UTILITY_MODES
    from optimization_problem_exp.environments.optimization_problem_env import (
        OptimizationProblemEnv,
    )

    u_refs: dict[str, list[float]] = {m: [] for m in modes}
    for mc in range(n_mc):
        dv = env._design_alpha_vars(alpha=0.5, rng_seed=seed_offset + mc)
        for mode in modes:
            sense = compute_sensing_utility(
                q_uav=dv.q_uav,
                q_vehicles=env.scenario.q_vehicles,
                rcs_list=[__import__("vehicle_reflection_exp.channels.vehicle_channel",
                                     fromlist=["compute_rcs"]).compute_rcs(vt)
                          for vt in env.scenario.vehicle_types],
                N_tx=env.config.N_tx_sense,
                N_rx=env.config.N_rx_sense,
                L_pilot=env.config.L_pilot,
                noise_power=env.config.noise_power_sense,
                d_ant=env.config.d_ant,
                wavelength=env.config.wavelength,
                seed=(env.config.seed or 0) + mc,
                mode=mode,
            )
            u_refs[mode].append(sense["U_sense_total"])

    result = {}
    for mode in modes:
        arr = np.array(u_refs[mode])
        result[mode] = float(np.mean(arr))
        result[f"{mode}_std"] = float(np.std(arr))
        result[f"{mode}_min"] = float(np.min(arr))
        result[f"{mode}_max"] = float(np.max(arr))

    global _U_REF_CACHE
    _U_REF_CACHE.clear()
    _U_REF_CACHE.update({m: result[m] for m in modes})
    return result


def get_u_ref(mode: str) -> float:
    """Get cached U_ref for a mode, falling back to U_SENSE_REF."""
    if mode in _U_REF_CACHE:
        return _U_REF_CACHE[mode]
    return U_SENSE_REF


def load_normalization_constants(path: str | None = None) -> dict:
    """Load normalization constants from JSON into global cache."""
    if path is None:
        path = os.path.join(
            "outputs", "sca_bcd_benchmark", "sensing_utility_fix",
            "normalization_constants.json",
        )
    if not os.path.exists(path):
        print(f"Warning: normalization constants not found at {path}, using defaults")
        return {}
    import json
    with open(path) as f:
        data = json.load(f)
    global _U_REF_CACHE
    _U_REF_CACHE.clear()
    for mode in SENSING_UTILITY_MODES:
        if mode in data:
            _U_REF_CACHE[mode] = data[mode]
    return _U_REF_CACHE


# ── Constraints ─────────────────────────────────────────

def check_constraints(
    phi_rad: np.ndarray,
    q_uav: np.ndarray,
    w_bs: np.ndarray,
    v_jammer: np.ndarray,
    P_bs_max: float,
    P_j_max: float,
    v_max: float,
    dt: float,
    q_min: np.ndarray,
    q_max: np.ndarray,
    R_s_min: float = 0.0,
    U_sense_min: float = 0.0,
):
    """Check all constraints. Returns dict of boolean pass/fail."""
    N_time = q_uav.shape[0]
    N_ris = len(phi_rad)

    constraints = {}

    # 1. RIS unit modulus
    unit_mod = np.allclose(np.abs(np.exp(1j * phi_rad)), 1.0, atol=1e-10)
    constraints["ris_unit_modulus"] = bool(unit_mod)
    constraints["ris_phase_range"] = bool(
        np.all(phi_rad >= -2*np.pi) and np.all(phi_rad <= 2*np.pi)
    )

    # 2. BS power (works for scalar and vector w_bs)
    bs_power = np.array([
        float(np.linalg.norm(w_bs[n]) ** 2) for n in range(N_time)
    ])
    constraints["bs_power"] = bool(np.all(bs_power <= P_bs_max + 1e-10))
    constraints["bs_power_min"] = bool(np.all(bs_power >= 0.0))

    # 3. Jammer power
    j_power = np.array([
        float(np.linalg.norm(v_jammer[n]) ** 2) for n in range(N_time)
    ])
    constraints["jammer_power"] = bool(np.all(j_power <= P_j_max + 1e-10))

    # 4. UAV speed
    speeds = np.zeros(N_time - 1)
    for n in range(1, N_time):
        dist = float(np.linalg.norm(q_uav[n] - q_uav[n - 1]))
        speeds[n - 1] = dist / dt
    constraints["uav_speed"] = bool(np.all(speeds <= v_max + 1e-6))
    constraints["uav_speed_nonneg"] = bool(np.all(speeds >= 0.0))

    # 5. UAV trajectory boundaries
    in_bounds = True
    for d in range(3):
        in_bounds = in_bounds and bool(
            np.all(q_uav[:, d] >= q_min[d] - 1e-10)
        )
        in_bounds = in_bounds and bool(
            np.all(q_uav[:, d] <= q_max[d] + 1e-10)
        )
    constraints["uav_trajectory_bounds"] = in_bounds

    # 6. RIS dimension
    constraints["ris_dimension"] = N_ris > 0

    return constraints


def compute_constraint_violations(
    phi_rad: np.ndarray,
    q_uav: np.ndarray,
    w_bs: np.ndarray,
    v_jammer: np.ndarray,
    P_bs_max: float,
    P_j_max: float,
    v_max: float,
    dt: float,
    q_min: np.ndarray,
    q_max: np.ndarray,
    R_s_total: float = 0.0,
    U_sense_total: float = 0.0,
    R_s_min: float = 0.0,
    U_sense_min: float = 0.0,
):
    """Compute numerical violation magnitudes.

    Returns dict of violation values (0 = no violation, >0 = violation).
    """
    N_time = q_uav.shape[0]
    violations = {}

    bs_power = np.array([
        float(np.linalg.norm(w_bs[n]) ** 2) for n in range(N_time)
    ])
    violations["bs_power_excess"] = float(
        np.max(np.maximum(bs_power - P_bs_max, 0.0))
    )
    violations["bs_power_negative"] = float(
        np.abs(np.min(np.minimum(bs_power, 0.0)))
    )

    j_power = np.array([
        float(np.linalg.norm(v_jammer[n]) ** 2) for n in range(N_time)
    ])
    violations["jammer_power_excess"] = float(
        np.max(np.maximum(j_power - P_j_max, 0.0))
    )

    max_speed = 0.0
    for n in range(1, N_time):
        dist = float(np.linalg.norm(q_uav[n] - q_uav[n - 1]))
        speed = dist / dt
        max_speed = max(max_speed, speed)
    violations["uav_speed_excess"] = float(max(max_speed - v_max, 0.0))

    bound_viol = 0.0
    for d in range(3):
        below = float(np.max(np.maximum(q_min[d] - q_uav[:, d], 0.0)))
        above = float(np.max(np.maximum(q_uav[:, d] - q_max[d], 0.0)))
        bound_viol = max(bound_viol, below, above)
    violations["uav_boundary_violation"] = bound_viol

    violations["secrecy_rate_shortfall"] = float(
        max(R_s_min - R_s_total, 0.0)
    )
    violations["sensing_utility_shortfall"] = float(
        max(U_sense_min - U_sense_total, 0.0)
    )

    violations["total_violation"] = sum(violations.values())
    return violations


# ── Unified evaluation wrapper ─────────────────────────

def evaluate_objective_and_constraints(
    decision_vars: DecisionVariables,
    q_bs: np.ndarray,
    q_user: np.ndarray,
    q_eves: np.ndarray,
    q_jammer: np.ndarray,
    q_vehicles: np.ndarray,
    vehicle_types: list,
    N_ris: int,
    N_j: int,
    N_tx_sense: int,
    N_rx_sense: int,
    L_pilot: int,
    P_bs_max: float,
    P_j_max: float,
    sigma2: float,
    noise_power_sense: float,
    v_max: float,
    dt: float,
    q_min: np.ndarray,
    q_max: np.ndarray,
    d_ant: float = 0.5,
    wavelength: float = 1.0,
    eta_ris: float = 0.3,
    alpha: float = 0.5,
    jammer_mode: str = "mixed",
    jammer_mix_alpha: float | None = None,
    jammer_power_factor: float | None = None,
    ris_alignment_alpha: float | None = None,
    include_direct_links: bool = False,
    seed: int = 0,
    sensing_utility_mode: str = "log",
    sensing_u_ref: float | None = None,
    M_bs: int = 1,
):
    """Evaluate the joint ISAC objective and all constraints in one call.

    Internally calls the existing compute_secrecy_rate(),
    compute_sensing_utility(), evaluate_weighted_objective(),
    check_constraints(), and compute_constraint_violations().

    Parameters
    ----------
    decision_vars : DecisionVariables
        Current decision variables (phi_rad, q_uav, w_bs, v_jammer).
    q_bs, q_user, q_eves, q_jammer, q_vehicles : np.ndarray
        Scenario geometry.
    vehicle_types : list
        Vehicle type strings for RCS lookup.
    N_ris, N_j : int
        Number of RIS elements, jammer antennas.
    N_tx_sense, N_rx_sense, L_pilot : int
        Sensing array dimensions and pilot length.
    P_bs_max, P_j_max : float
        Max BS and jammer power.
    sigma2, noise_power_sense : float
        Noise powers for communication and sensing.
    v_max, dt : float
        UAV speed limit and time step.
    q_min, q_max : np.ndarray (3,)
        UAV position bounds.
    d_ant, wavelength : float
        Antenna spacing and wavelength for ULA.
    eta_ris : float
        RIS reflection efficiency.
    alpha : float
        Trade-off weight in [0, 1].
    jammer_mode : str
        Jammer beamforming mode (given/protect/blast/mixed/isotropic).
    jammer_mix_alpha : float | None
        Mix ratio for mixed-mode jammer (defaults to alpha).
    jammer_power_factor : float | None
        Jammer power scaling (defaults to max(0.01, alpha)).
    ris_alignment_alpha : float | None
        RIS alignment (defaults to alpha).
    include_direct_links : bool
        Include weak direct BS-user/eve paths.
    seed : int
        Random seed.
    sensing_utility_mode : str
        Sensing utility mode (original/log/inverse/normalized/exponential).
    sensing_u_ref : float | None
        Normalization constant for sensing utility (None = auto-compute).
    M_bs : int
        Number of BS antennas (1 = single-antenna, >1 = MIMO).

    Returns
    -------
    dict with keys:
        "objective" — weighted objective value f
        "secrecy"   — full secrecy result dict from compute_secrecy_rate
        "sensing"   — full sensing result dict from compute_sensing_utility
        "constraints" — dict of bool constraint checks
        "violations"  — dict of numerical violation magnitudes
    """
    from vehicle_reflection_exp.channels.vehicle_channel import compute_rcs

    rcs_list = [compute_rcs(vt) for vt in vehicle_types]

    _jammer_mix = alpha if jammer_mix_alpha is None else jammer_mix_alpha
    _jp_factor = max(0.01, alpha) if jammer_power_factor is None else jammer_power_factor
    _ris_align = alpha if ris_alignment_alpha is None else ris_alignment_alpha

    sec_result = compute_secrecy_rate(
        q_bs=q_bs, q_user=q_user, q_eves=q_eves, q_jammer=q_jammer,
        N_ris=N_ris, N_j=N_j, Phi=None,
        q_uav=decision_vars.q_uav,
        w_bs=decision_vars.w_bs,
        v_jammer=decision_vars.v_jammer,
        P_bs_max=P_bs_max, P_j_max=P_j_max, sigma2=sigma2,
        seed=seed,
        jammer_mode=jammer_mode,
        jammer_mix_alpha=_jammer_mix,
        jammer_power_factor=_jp_factor,
        include_direct_links=include_direct_links,
        eta_ris=eta_ris,
        ris_alignment_alpha=_ris_align,
        M_bs=M_bs,
    )
    sense_result = compute_sensing_utility(
        q_uav=decision_vars.q_uav,
        q_vehicles=q_vehicles,
        rcs_list=rcs_list,
        N_tx=N_tx_sense,
        N_rx=N_rx_sense,
        L_pilot=L_pilot,
        noise_power=noise_power_sense,
        d_ant=d_ant,
        wavelength=wavelength,
        seed=seed,
        mode=sensing_utility_mode,
    )

    u_ref = sensing_u_ref if sensing_u_ref is not None else get_u_ref(sensing_utility_mode)
    f = evaluate_weighted_objective(
        alpha,
        sec_result["R_s_total"],
        sense_result["U_sense_total"],
        U_sense_ref=u_ref,
    )

    constraints = check_constraints(
        phi_rad=decision_vars.phi_rad,
        q_uav=decision_vars.q_uav,
        w_bs=decision_vars.w_bs,
        v_jammer=decision_vars.v_jammer,
        P_bs_max=P_bs_max,
        P_j_max=P_j_max,
        v_max=v_max,
        dt=dt,
        q_min=q_min,
        q_max=q_max,
    )

    violations = compute_constraint_violations(
        phi_rad=decision_vars.phi_rad,
        q_uav=decision_vars.q_uav,
        w_bs=decision_vars.w_bs,
        v_jammer=decision_vars.v_jammer,
        P_bs_max=P_bs_max,
        P_j_max=P_j_max,
        v_max=v_max,
        dt=dt,
        q_min=q_min,
        q_max=q_max,
        R_s_total=sec_result["R_s_total"],
        U_sense_total=sense_result["U_sense_total"],
    )

    return {
        "objective": f,
        "secrecy": sec_result,
        "sensing": sense_result,
        "constraints": constraints,
        "violations": violations,
    }


# ── Channel condition numbers (diagnostic) ──────────────

def compute_channel_condition_numbers(
    decision_vars: DecisionVariables,
    q_bs: np.ndarray | None = None,
    q_user: np.ndarray | None = None,
    q_eves: np.ndarray | None = None,
    q_jammer: np.ndarray | None = None,
    q_vehicles: np.ndarray | None = None,
    vehicle_types: list | None = None,
    N_tx_sense: int = 16,
    N_rx_sense: int = 16,
    L_pilot: int = 32,
    noise_power_sense: float = 1e-8,
    d_ant: float = 0.5,
    wavelength: float = 1.0,
    seed: int = 0,
):
    """Compute condition numbers for RIS, sensing, and FIM matrices.

    Returns dict with:
        ris_eff_channel_cond  — condition number of the cascaded RIS effective channel matrix
        sensing_matrix_cond   — condition number of the composite sensing channel H_sense
        fim_cond              — condition number of the CRB FIM
        crb_trace             — trace of the CRB matrix
    """
    result = {
        "ris_eff_channel_cond": None,
        "sensing_matrix_cond": None,
        "fim_cond": None,
        "crb_trace": None,
    }

    q_uav_dv = decision_vars.q_uav
    # RIS effective channel condition
    if q_bs is not None and q_user is not None:
        try:
            N_ris = decision_vars.N_ris
            N_time = decision_vars.N_time
            conds = []
            for n in range(N_time):
                h_BR = compute_bs_ris_channel(q_bs, decision_vars.q_uav[n], N_ris, seed + n)
                h_RU = compute_ris_user_channel(
                    decision_vars.q_uav[n], q_user, N_ris, seed + n + 1,
                )
                phi = decision_vars.phi_rad
                Phi = compute_ris_reflection_matrix(phi)
                H_eff = compute_effective_channel(h_RU, Phi, h_BR)
                H_eff_arr = np.atleast_2d(H_eff)
                if np.all(np.abs(H_eff_arr) < 1e-30):
                    conds.append(1.0)
                else:
                    conds.append(float(np.linalg.cond(H_eff_arr)))
            result["ris_eff_channel_cond"] = float(np.mean(conds)) if conds else None
        except Exception:
            pass

    # Sensing matrix condition
    if q_vehicles is not None and vehicle_types is not None:
        from vehicle_reflection_exp.channels.vehicle_channel import compute_rcs
        try:
            N_time = decision_vars.N_time
            conds = []
            for n in range(N_time):
                rng = np.random.RandomState(seed + n)
                thetas = []
                alphas = []
                for kv in range(len(q_vehicles)):
                    dx = q_vehicles[kv][0] - decision_vars.q_uav[n][0]
                    dy = q_vehicles[kv][1] - decision_vars.q_uav[n][1]
                    thetas.append(float(np.degrees(np.arctan2(dy, dx))))
                    rcs_lin = 10.0 ** (compute_rcs(vehicle_types[kv]) / 10.0)
                    pl_val = _pl(decision_vars.q_uav[n], q_vehicles[kv])
                    alpha = complex(
                        np.sqrt(pl_val * rcs_lin) * rng.randn() / np.sqrt(2),
                        np.sqrt(pl_val * rcs_lin) * rng.randn() / np.sqrt(2),
                    )
                    alphas.append(alpha)

                A_list = []
                for th in thetas:
                    a_rx = ula_steering_vector(N_rx_sense, th, d_ant, wavelength)
                    A_list.append(target_response_matrix(a_rx))

                H_sense = composite_sensing_channel(alphas, A_list)
                if H_sense.size > 0:
                    conds.append(float(np.linalg.cond(H_sense)))
            result["sensing_matrix_cond"] = float(np.mean(conds)) if conds else None
        except Exception:
            pass

    # FIM condition number
    if q_vehicles is not None and vehicle_types is not None:
        from vehicle_reflection_exp.channels.vehicle_channel import compute_rcs
        try:
            N_time = decision_vars.N_time
            fim_conds = []
            crb_traces = []
            for n in range(N_time):
                rng = np.random.RandomState(seed + n)
                thetas = []
                alphas = []
                for kv in range(len(q_vehicles)):
                    dx = q_vehicles[kv][0] - decision_vars.q_uav[n][0]
                    dy = q_vehicles[kv][1] - decision_vars.q_uav[n][1]
                    thetas.append(float(np.degrees(np.arctan2(dy, dx))))
                    rcs_lin = 10.0 ** (compute_rcs(vehicle_types[kv]) / 10.0)
                    pl_val = _pl(decision_vars.q_uav[n], q_vehicles[kv])
                    alpha = complex(
                        np.sqrt(pl_val * rcs_lin) * rng.randn() / np.sqrt(2),
                        np.sqrt(pl_val * rcs_lin) * rng.randn() / np.sqrt(2),
                    )
                    alphas.append(alpha)

                A_list = []
                dA_list = []
                for th in thetas:
                    a_rx = ula_steering_vector(N_rx_sense, th, d_ant, wavelength)
                    da_rx = ula_steering_derivative(N_rx_sense, th, d_ant, wavelength)
                    A_list.append(target_response_matrix(a_rx))
                    dA_list.append(target_response_derivative(a_rx, da_rx))

                H_sense = composite_sensing_channel(alphas, A_list)
                dH = compute_channel_derivatives(alphas, dA_list)

                X_pilot = rng.randn(N_tx_sense, L_pilot) + 1j * rng.randn(N_tx_sense, L_pilot)
                for col in range(L_pilot):
                    nrm = float(np.linalg.norm(X_pilot[:, col]))
                    if nrm > 0.0:
                        X_pilot[:, col] /= nrm

                FIM = compute_fim(dH, X_pilot, noise_power_sense)
                eig = np.linalg.eigvalsh(FIM)
                eig = np.maximum(eig, 1e-30)
                fim_conds.append(float(eig[-1] / eig[0]))

                crb_result = compute_crb(FIM)
                crb_traces.append(float(np.trace(crb_result["crb_matrix"]).real))

            result["fim_cond"] = float(np.mean(fim_conds)) if fim_conds else None
            result["crb_trace"] = float(np.mean(crb_traces)) if crb_traces else None
        except Exception:
            pass

    return result


# ── Monte Carlo secrecy statistics ──────────────────────

def compute_monte_carlo_secrecy_stats(
    q_bs: np.ndarray,
    q_user: np.ndarray,
    q_eves: np.ndarray,
    q_jammer: np.ndarray,
    q_uav: np.ndarray,
    w_bs: np.ndarray,
    v_jammer: np.ndarray,
    N_ris: int,
    N_j: int,
    P_bs_max: float,
    P_j_max: float,
    sigma2: float,
    num_realizations: int = 500,
    jammer_mode: str = "protect",
    include_direct_links: bool = False,
    eta_ris: float = 1.0,
    ris_phase_noise_std: float = 0.0,
    jammer_power_factor: float = 1.0,
    ris_alignment_alpha: float = 1.0,
    M_bs: int = 1,
) -> dict:
    """Run many secrecy realisations and compute statistics.

    Returns dict with:
        prob_rs_gt_0    Pr(Rs_total > 0.01)
        avg_secrecy     mean(Rs_total)
        median_secrecy  median(Rs_total)
        std_secrecy     std(Rs_total)
        secrecy_cdf_vals    sorted Rs values for CDF plotting
        secrecy_cdf_probs   np.linspace(0,1) for CDF
        avg_user_sinr, avg_eve_sinr, ...
    """
    Rs_all = np.zeros(num_realizations)
    Ru_all = np.zeros(num_realizations)
    Re_all = np.zeros(num_realizations)
    SINRu_all = []
    SINRe_all = []

    for r in range(num_realizations):
        sec = compute_secrecy_rate(
            q_bs, q_user, q_eves, q_jammer,
            N_ris, N_j, None,
            q_uav, w_bs, v_jammer,
            P_bs_max, P_j_max, sigma2,
            seed=r,
            jammer_mode=jammer_mode,
            include_direct_links=include_direct_links,
            eta_ris=eta_ris,
            ris_phase_noise_std=ris_phase_noise_std,
            jammer_power_factor=jammer_power_factor,
            ris_alignment_alpha=ris_alignment_alpha,
            M_bs=M_bs,
        )
        Rs_all[r] = sec["R_s_total"]
        Ru_all[r] = float(np.mean(sec["R_user"]))
        Re_all[r] = float(np.max(sec["R_eve_max"]))
        SINRu_all.extend(sec["SINR_user"].tolist())
        SINRe_all.extend(sec["SINR_eve"].flatten().tolist())

    prob_gt_0 = float(np.mean(Rs_all > 0.01))
    sorted_rs = np.sort(Rs_all)
    cdf_probs = np.linspace(0.0, 1.0, num_realizations)

    return {
        "prob_rs_gt_0": prob_gt_0,
        "avg_secrecy": float(np.mean(Rs_all)),
        "median_secrecy": float(np.median(Rs_all)),
        "std_secrecy": float(np.std(Rs_all)),
        "min_secrecy": float(np.min(Rs_all)),
        "max_secrecy": float(np.max(Rs_all)),
        "secrecy_cdf_vals": sorted_rs,
        "secrecy_cdf_probs": cdf_probs,
        "avg_user_sinr": float(np.mean(SINRu_all)),
        "avg_eve_sinr": float(np.mean(SINRe_all)),
        "avg_user_rate": float(np.mean(Ru_all)),
        "avg_eve_rate": float(np.mean(Re_all)),
        "all_Rs": Rs_all,
    }
