"""Detection channel models: binary hypothesis, energy detector, GLRT."""

import numpy as np


def generate_h0(
    N_r: int,
    L: int,
    noise_power: float,
) -> np.ndarray:
    """Generate noise-only observation under H0.

    Y = N,  N ~ CN(0, noise_power * I)

    Returns: complex array (N_r x L).
    """
    std = np.sqrt(noise_power / 2.0)
    return std * (np.random.randn(N_r, L) + 1j * np.random.randn(N_r, L))


def generate_h1(
    H_sense: np.ndarray,
    X: np.ndarray,
    noise_power: float,
) -> np.ndarray:
    """Generate signal-plus-noise observation under H1.

    Y = H_sense @ X + N,  N ~ CN(0, noise_power * I)

    Returns: complex array (N_r x L).
    """
    N_r, _ = H_sense.shape
    L = X.shape[1]
    std = np.sqrt(noise_power / 2.0)
    N = std * (np.random.randn(N_r, L) + 1j * np.random.randn(N_r, L))
    return H_sense @ X + N


def energy_detector_statistic(Y: np.ndarray) -> float:
    """Energy detector test statistic.

    T(Y) = ||Y||_F^2
    """
    return float(np.linalg.norm(Y, "fro") ** 2)


def glrt_detector_statistic(
    Y: np.ndarray,
    X: np.ndarray,
    reg: float = 1e-12,
) -> float:
    """GLRT test statistic for unknown H_sense.

    Lambda(Y) = ||Y @ P||_F^2 / ||Y - Y @ P||_F^2

    where P = X^H @ (X @ X^H)^{-1} @ X is the projection onto row(X).

    Returns: Lambda >= 0. Large values favour H1.
    """
    N_t = X.shape[0]
    XXh = X @ X.conj().T  # (N_t x N_t)
    try:
        XXh_inv = np.linalg.inv(XXh + reg * np.eye(N_t, dtype=complex))
    except np.linalg.LinAlgError:
        return 0.0

    # Projection matrix P = X^H @ inv(XXh) @ X  (L x L)
    P = X.conj().T @ XXh_inv @ X

    # Project Y onto row space of X
    Y_proj = Y @ P
    Y_res = Y - Y_proj

    num = float(np.linalg.norm(Y_proj, "fro") ** 2)
    den = float(np.linalg.norm(Y_res, "fro") ** 2)

    if den < 1e-30:
        return 0.0
    return num / den


def detect(statistic: float, threshold: float) -> bool:
    """Compare test statistic against threshold.

    H1 if statistic > threshold, else H0.
    """
    return statistic > threshold


def monte_carlo_pd_pfa(
    gen_h0_func,
    gen_h1_func,
    detector_fn,
    threshold: float,
    num_mc: int = 500,
    return_stats: bool = False,
):
    """Estimate Pd and Pfa via Monte Carlo.

    Args:
        gen_h0_func: callable -> Y under H0 (call with no args).
        gen_h1_func: callable -> Y under H1 (call with no args).
        detector_fn: callable(Y) -> scalar test statistic.
        threshold: decision threshold.
        num_mc: number of Monte Carlo trials.

    Returns:
        (pd, pfa) or (pd, pfa, h0_stats, h1_stats) if return_stats.
    """
    h0_count = 0
    h1_count = 0
    h0_stats = []
    h1_stats = []

    for _ in range(num_mc):
        Y0 = gen_h0_func()
        T0 = detector_fn(Y0)
        h0_stats.append(T0)
        if detect(T0, threshold):
            h0_count += 1

        Y1 = gen_h1_func()
        T1 = detector_fn(Y1)
        h1_stats.append(T1)
        if detect(T1, threshold):
            h1_count += 1

    pfa = h0_count / num_mc
    pd = h1_count / num_mc

    if return_stats:
        return pd, pfa, np.array(h0_stats), np.array(h1_stats)
    return pd, pfa
