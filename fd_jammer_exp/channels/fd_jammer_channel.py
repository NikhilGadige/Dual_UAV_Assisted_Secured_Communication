import numpy as np


def generate_miso_rician_channel(
    N_j: int,
    K: float = 5.0,
    path_loss_factor: float = 1.0,
    los_phases: np.ndarray | None = None,
) -> np.ndarray:
    """Generate a MISO Rician channel row vector (1 x N_j).

    h = sqrt(PL) * ( sqrt(K/(K+1)) * h_LoS + sqrt(1/(K+1)) * h_NLoS )
    """
    K_linear = 10.0 ** (K / 10.0) if K < 100 else K
    scale_los = np.sqrt(K_linear / (K_linear + 1.0))
    scale_nlos = np.sqrt(1.0 / (K_linear + 1.0))

    if los_phases is None:
        los = np.exp(1j * np.random.uniform(0.0, 2.0 * np.pi, size=N_j))
    else:
        los = np.exp(1j * los_phases[:N_j])

    nlos = (
        np.random.normal(0.0, 1.0 / np.sqrt(2.0), size=N_j)
        + 1j * np.random.normal(0.0, 1.0 / np.sqrt(2.0), size=N_j)
    )

    h = np.sqrt(path_loss_factor) * (scale_los * los + scale_nlos * nlos)
    return h.reshape(1, N_j)


def isotropic_beamforming(N_j: int) -> np.ndarray:
    """Isotropic artificial noise beamformer.  Equal gain, random phase per antenna.

    Returns column vector w of shape (N_j, 1).
    """
    phases = np.random.uniform(0.0, 2.0 * np.pi, size=N_j)
    w = np.exp(1j * phases) / np.sqrt(N_j)
    return w.reshape(N_j, 1)


def mrt_beamforming(h_je: np.ndarray) -> np.ndarray:
    """Maximum-Ratio Transmission toward a specific eavesdropper.

    h_je : (1 x N_j)  MISO channel row vector
    Returns w of shape (N_j, 1).
    """
    w = h_je.conj().T
    norm = float(np.linalg.norm(w))
    if norm < 1e-15:
        return isotropic_beamforming(h_je.shape[1])
    return w / norm


def nullspace_beamforming(h_ju: np.ndarray, N_j: int) -> np.ndarray:
    """Null-space beamforming toward the user.

    Projects a random vector onto the nullspace of h_ju so that h_ju * w = 0.
    h_ju : (1 x N_j)
    Returns w of shape (N_j, 1).
    """
    w_rand = np.random.randn(N_j) + 1j * np.random.randn(N_j)
    w_rand = w_rand.reshape(N_j, 1)
    w_rand = w_rand / float(np.linalg.norm(w_rand))

    h_norm_sq = float(np.sum(np.abs(h_ju) ** 2))
    if h_norm_sq < 1e-15:
        return isotropic_beamforming(N_j)

    P = np.eye(N_j, dtype=complex) - (
        h_ju.conj().T @ h_ju
    ) / h_norm_sq

    w = P @ w_rand
    n = float(np.linalg.norm(w))
    if n < 1e-15:
        return isotropic_beamforming(N_j)
    return w / n


def compute_jammer_gain(h: np.ndarray, w: np.ndarray) -> float:
    """Compute |h @ w|^2 — the received jamming power gain.

    h : (1 x N_j)  MISO channel row
    w : (N_j x 1)  beamforming column
    """
    return float(np.abs(h @ w).item() ** 2)
