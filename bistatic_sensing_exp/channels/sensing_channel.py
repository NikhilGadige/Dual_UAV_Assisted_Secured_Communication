import numpy as np
from core.channel import path_loss


def _scalar_rician(
    K: float = 5.0,
    path_loss_factor: float = 1.0,
) -> complex:
    K_lin = 10.0 ** (K / 10.0) if K < 100 else K
    s_los = np.sqrt(K_lin / (K_lin + 1.0))
    s_nlos = np.sqrt(1.0 / (K_lin + 1.0))
    los = np.exp(1j * np.random.uniform(0.0, 2.0 * np.pi))
    nlos = np.random.normal(0.0, 1.0 / np.sqrt(2.0)) + 1j * np.random.normal(
        0.0, 1.0 / np.sqrt(2.0)
    )
    return complex(np.sqrt(path_loss_factor) * (s_los * los + s_nlos * nlos))


def compute_bistatic_distances(
    ris_pos: np.ndarray,
    vehicle_pos: np.ndarray,
) -> dict:
    """Compute TX, RX, and total bistatic distances.

    TX:  RIS-UAV -> vehicle
    RX:  vehicle -> RIS-UAV

    For monostatic reception at the RIS-UAV the two
    distances are identical.

    Returns dict with d_tx, d_rx, d_total (all floats).
    """
    d = float(np.linalg.norm(ris_pos[:2] - vehicle_pos[:2]))
    return {"d_tx": d, "d_rx": d, "d_total": 2.0 * d}


def compute_tx_distance(
    ris_pos: np.ndarray,
    vehicle_pos: np.ndarray,
) -> float:
    return float(np.linalg.norm(ris_pos[:2] - vehicle_pos[:2]))


def compute_rx_distance(
    ris_pos: np.ndarray,
    vehicle_pos: np.ndarray,
) -> float:
    return float(np.linalg.norm(ris_pos[:2] - vehicle_pos[:2]))


def compute_bistatic_distance(
    ris_pos: np.ndarray,
    vehicle_pos: np.ndarray,
) -> float:
    d = float(np.linalg.norm(ris_pos[:2] - vehicle_pos[:2]))
    return 2.0 * d


def generate_sensing_channel(
    d_tx: float,
    d_rx: float,
    sigma_rcs: float,
    K: float = 5.0,
    alpha: float = 2.0,
    beta0: float = 1.0,
) -> complex:
    """Generate the scalar bistatic sensing channel.

    h_sensing = h_UV * sqrt(sigma_rcs) * h_VU

    h_UV : RIS-UAV -> vehicle  (Rician with path loss over d_tx)
    h_VU : vehicle -> RIS-UAV  (Rician with path loss over d_rx)
    """
    pl_UV = path_loss(d_tx, alpha, beta0)
    pl_VU = path_loss(d_rx, alpha, beta0)

    h_UV = _scalar_rician(K, pl_UV)
    h_VU = _scalar_rician(K, pl_VU)

    return h_UV * np.sqrt(sigma_rcs) * h_VU


def compute_sensing_gain(h_sensing: complex) -> float:
    """Return |h_sensing|^2."""
    return float(np.abs(h_sensing) ** 2)


def compute_echo_signal(
    P_s: float,
    h_sensing: complex,
    s: complex = 1.0 + 0j,
    noise_power: float = 0.0,
) -> tuple:
    """Compute the received echo signal.

    y = sqrt(P_s) * h_sensing * s + n

    Returns (echo_signal, noise_realisation).
    """
    echo = np.sqrt(P_s) * h_sensing * s
    n = 0.0 + 0j
    if noise_power > 0.0:
        noise_std = np.sqrt(noise_power / 2.0)
        n = noise_std * np.random.normal(0.0, 1.0) + 1j * noise_std * np.random.normal(
            0.0, 1.0
        )
    return echo + n, n


def compute_sensing_snr(
    P_s: float,
    h_sensing_gain: float,
    noise_power: float,
) -> float:
    """Sensing SNR gamma = P_s * |h_sensing|^2 / noise_power."""
    if noise_power <= 0.0:
        return float("inf")
    return (P_s * h_sensing_gain) / noise_power
