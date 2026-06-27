import numpy as np


def ula_steering_vector(
    N: int,
    theta_deg: float,
    d: float = 0.5,
    wavelength: float = 1.0,
) -> np.ndarray:
    """ULA steering vector.

    a(theta) = [1, exp(-j*2*pi*d*sin(theta)/lambda), ...,
                exp(-j*2*pi*(N-1)*d*sin(theta)/lambda)]^T

    Args:
        N: Number of antennas.
        theta_deg: Angle of arrival/departure (degrees).
        d: Element spacing (same units as wavelength).
        wavelength: Signal wavelength.

    Returns:
        Complex steering vector of shape (N, 1).
    """
    theta_rad = np.radians(theta_deg)
    k = 2.0 * np.pi * d * np.sin(theta_rad) / wavelength
    n = np.arange(N)
    a = np.exp(-1j * k * n)
    return a.reshape(N, 1)


def target_response_matrix(
    a_rx: np.ndarray,
    a_tx: np.ndarray,
) -> np.ndarray:
    """Construct the rank-1 target response matrix.

    A = a_rx @ a_tx^H

    Args:
        a_rx: Receive steering vector (N_r x 1).
        a_tx: Transmit steering vector (N_t x 1).

    Returns:
        Target response matrix (N_r x N_t).
    """
    return a_rx @ a_tx.conj().T


def composite_sensing_channel(
    alpha_list: list[complex],
    A_list: list[np.ndarray],
) -> np.ndarray:
    """Construct the composite sensing channel.

    H_sense = sum_i alpha_i * A_i

    Args:
        alpha_list: Complex reflection coefficients for each target.
        A_list: Target response matrices for each target.

    Returns:
        Composite channel matrix (N_r x N_t).
    """
    if len(alpha_list) == 0:
        N_r, N_t = A_list[0].shape if A_list else (1, 1)
        return np.zeros((N_r, N_t), dtype=complex)

    H = np.zeros_like(A_list[0], dtype=complex)
    for alpha, A in zip(alpha_list, A_list):
        H += alpha * A
    return H


def compute_echo_matrix(
    H_sense: np.ndarray,
    X: np.ndarray,
    noise_power: float = 0.0,
) -> np.ndarray:
    """Compute the received echo matrix.

    Y = H_sense @ X + N

    Args:
        H_sense: Composite sensing channel (N_r x N_t).
        X: Pilot matrix (N_t x L).
        noise_power: Noise variance per element.

    Returns:
        Echo matrix Y (N_r x L).
    """
    Y = H_sense @ X
    if noise_power > 0.0:
        noise_std = np.sqrt(noise_power / 2.0)
        N_mat = noise_std * (
            np.random.randn(*Y.shape) + 1j * np.random.randn(*Y.shape)
        )
        Y += N_mat
    return Y


def compute_covariance_matrices(
    Y: np.ndarray,
    H_sense: np.ndarray,
    X: np.ndarray,
    noise_power: float,
) -> dict:
    """Compute signal, noise, and total covariance matrices.

    R_y    = (1/L) * Y @ Y^H            (sample covariance)
    R_sig  = H_sense @ R_x @ H_sense^H  (signal covariance)
    R_n    = noise_power * I             (noise covariance)

    where R_x = (1/L) * X @ X^H.

    Args:
        Y: Echo matrix (N_r x L).
        H_sense: Composite channel (N_r x N_t).
        X: Pilot matrix (N_t x L).
        noise_power: Noise variance.

    Returns:
        dict with keys: R_y, R_signal, R_noise, R_x, eigenvalues.
    """
    L = Y.shape[1]
    N_r = Y.shape[0]

    R_y = (Y @ Y.conj().T) / L
    R_x = (X @ X.conj().T) / L
    R_signal = H_sense @ R_x @ H_sense.conj().T
    R_noise = noise_power * np.eye(N_r, dtype=complex)

    eigvals = np.sort(np.linalg.eigvalsh(R_y))[::-1]

    return {
        "R_y": R_y,
        "R_signal": R_signal,
        "R_noise": R_noise,
        "R_x": R_x,
        "eigenvalues": eigvals,
    }
