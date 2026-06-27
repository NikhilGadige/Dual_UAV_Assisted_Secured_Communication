import numpy as np


def ula_steering_vector(
    N: int,
    theta_deg: float,
    d: float = 0.5,
    wavelength: float = 1.0,
) -> np.ndarray:
    """ULA steering vector a(theta).

    a = [1, exp(-j*2*pi*d*sin(theta)/lambda), ...,
         exp(-j*2*pi*(N-1)*d*sin(theta)/lambda)]^T
    """
    theta_rad = np.radians(theta_deg)
    k = 2.0 * np.pi * d * np.sin(theta_rad) / wavelength
    n = np.arange(N)
    a = np.exp(-1j * k * n)
    return a.reshape(N, 1)


def ula_steering_derivative(
    N: int,
    theta_deg: float,
    d: float = 0.5,
    wavelength: float = 1.0,
) -> np.ndarray:
    """First derivative of ULA steering vector w.r.t. theta.

    da_n/dtheta = (-j*2*pi*n*d*cos(theta)/lambda) * a_n
    """
    theta_rad = np.radians(theta_deg)
    k = 2.0 * np.pi * d * np.sin(theta_rad) / wavelength
    factor = -1j * 2.0 * np.pi * d * np.cos(theta_rad) / wavelength
    n = np.arange(N)
    a = np.exp(-1j * k * n)
    da = factor * n * a
    return da.reshape(N, 1)


def target_response_matrix(a: np.ndarray) -> np.ndarray:
    """Rank-1 target response matrix A = a @ a^H."""
    return a @ a.conj().T


def target_response_derivative(
    a: np.ndarray,
    da: np.ndarray,
) -> np.ndarray:
    """Derivative of target response matrix w.r.t. theta.

    dA/dtheta = (da/dtheta) @ a^H + a @ (da/dtheta)^H
    """
    return da @ a.conj().T + a @ da.conj().T


def composite_sensing_channel(
    alpha_list: list[complex],
    A_list: list[np.ndarray],
) -> np.ndarray:
    """H_sense = sum_i alpha_i * A_i."""
    if not A_list:
        return np.zeros((1, 1), dtype=complex)
    H = np.zeros_like(A_list[0], dtype=complex)
    for alpha, A in zip(alpha_list, A_list):
        H += alpha * A
    return H


def compute_channel_derivatives(
    alpha_list: list[complex],
    dA_list: list[np.ndarray],
) -> list[np.ndarray]:
    """dH/dtheta_i = alpha_i * dA_i/dtheta_i for each target."""
    return [alpha * dA for alpha, dA in zip(alpha_list, dA_list)]


def compute_fim(
    dH_derivs: list[np.ndarray],
    X: np.ndarray,
    noise_power: float,
) -> np.ndarray:
    """Fisher Information Matrix for angle estimation.

    J(i,j) = (2/sigma^2) * Re{ tr( X^H * (dH/dtheta_i)^H
                                       * (dH/dtheta_j) * X ) }

    Args:
        dH_derivs: List [dH/dtheta_0, ..., dH/dtheta_{K-1}],
                   each shape (N_r, N_t).
        X: Pilot matrix (N_t x L).
        noise_power: Noise variance sigma^2.

    Returns:
        FIM matrix (K x K).
    """
    K = len(dH_derivs)
    if K == 0:
        return np.zeros((0, 0))

    # X * X^H  (N_t x N_t)
    XXh = X @ X.conj().T
    factor = 2.0 / noise_power

    FIM = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(i, K):
            # M = (dH/dtheta_i)^H * (dH/dtheta_j)  (N_t x N_t)
            M = dH_derivs[i].conj().T @ dH_derivs[j]
            val = factor * np.real(np.trace(M @ XXh))
            FIM[i, j] = val
            FIM[j, i] = val  # symmetric
    return FIM


def compute_crb(FIM: np.ndarray) -> dict:
    """Compute CRB from FIM.

    CRB = J^{-1}

    Args:
        FIM: Fisher Information Matrix (K x K).

    Returns:
        dict with keys:
            crb_matrix:   J^{-1} (K x K)
            var_bound:    diag(J^{-1}) per-target variance bound
            rmse_bound:   sqrt(diag(J^{-1})) RMSE lower bound (degrees)
    """
    K = FIM.shape[0]
    if K == 0:
        return {
            "crb_matrix": np.zeros((0, 0)),
            "var_bound": np.array([]),
            "rmse_bound": np.array([]),
            "condition_number": np.inf,
            "fim_eigenvalues": np.array([]),
        }

    fim_eigvals = np.sort(np.linalg.eigvalsh(FIM))[::-1]
    cond = float(fim_eigvals[0] / fim_eigvals[-1]) if fim_eigvals[-1] > 0 else np.inf

    try:
        crb_mat = np.linalg.inv(FIM)
    except np.linalg.LinAlgError:
        crb_mat = np.full((K, K), np.inf)

    var_bound = np.diag(crb_mat).real
    rmse_bound = np.sqrt(var_bound)

    return {
        "crb_matrix": crb_mat,
        "var_bound": var_bound,
        "rmse_bound": rmse_bound,
        "condition_number": cond,
        "fim_eigenvalues": fim_eigvals,
    }
