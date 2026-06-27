import numpy as np
from core.channel import path_loss


def generate_ris_rician_channel(
    N: int,
    K: float = 5.0,
    path_loss_factor: float = 1.0,
    los_phases: np.ndarray | None = None,
) -> np.ndarray:
    K_linear = 10.0 ** (K / 10.0) if K < 100 else K
    scale_los = np.sqrt(K_linear / (K_linear + 1.0))
    scale_nlos = np.sqrt(1.0 / (K_linear + 1.0))

    if los_phases is None:
        los = np.exp(1j * np.random.uniform(0.0, 2.0 * np.pi, size=N))
    else:
        los = np.exp(1j * los_phases)

    nlos = (
        np.random.normal(0.0, 1.0 / np.sqrt(2.0), size=N)
        + 1j * np.random.normal(0.0, 1.0 / np.sqrt(2.0), size=N)
    )

    h = np.sqrt(path_loss_factor) * (scale_los * los + scale_nlos * nlos)
    return h


def compute_ris_reflection_matrix(phases: np.ndarray) -> np.ndarray:
    return np.diag(np.exp(1j * phases))


def compute_effective_channel(
    h_rx: np.ndarray,
    Phi: np.ndarray,
    h_tx: np.ndarray,
) -> complex:
    return h_rx.conj().T @ Phi @ h_tx


def compute_effective_channel_gain(h_eff: complex) -> float:
    return float(np.abs(h_eff) ** 2)
