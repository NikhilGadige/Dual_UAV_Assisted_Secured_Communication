import numpy as np

EPSILON = 1e-10

def path_loss(d: float, alpha: float = 2.0, beta0: float = 1.0) -> float:
    d_safe = max(d, EPSILON)
    return beta0 * (d_safe ** -alpha)

def generate_rayleigh() -> float:
    h_real = np.random.normal(0, 1 / np.sqrt(2))
    h_imag = np.random.normal(0, 1 / np.sqrt(2))
    return h_real ** 2 + h_imag ** 2

def generate_rician(K: float = 5.0) -> float:
    s = np.sqrt(K / (K + 1))
    sigma = np.sqrt(1 / (2 * (K + 1)))
    h_real = s + np.random.normal(0, sigma)
    h_imag = np.random.normal(0, sigma)
    return h_real ** 2 + h_imag ** 2

def generate_fading(model: str = "rayleigh", K: float = 5.0) -> float:
    if model == "rician":
        return generate_rician(K)
    return generate_rayleigh()

def channel_gain(d: float, fading: float,
                 alpha: float = 2.0, beta0: float = 1.0) -> float:
    pl = path_loss(d, alpha, beta0)
    return pl * fading