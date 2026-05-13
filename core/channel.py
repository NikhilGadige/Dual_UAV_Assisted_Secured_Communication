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

def compute_elevation_angle(tx_pos: np.ndarray, rx_pos: np.ndarray) -> float:
    """Elevation angle (degrees) from rx to tx.  Vertical ÷ horizontal."""
    dx = tx_pos[0] - rx_pos[0]
    dy = tx_pos[1] - rx_pos[1]
    dz = tx_pos[2] - rx_pos[2]
    d_h = np.sqrt(dx * dx + dy * dy)
    if d_h < EPSILON:
        return 90.0 if abs(dz) > EPSILON else 0.0
    return float(np.degrees(np.arctan2(abs(dz), d_h)))


def los_probability(theta: float, los_a: float, los_b: float) -> float:
    """Sigmoid LoS probability  P(LoS) = 1 / (1 + a·exp(-b(θ - a)))."""
    return 1.0 / (1.0 + los_a * np.exp(-los_b * (theta - los_a)))


def channel_gain_los_aware(
    d: float,
    theta: float,
    fading: float,
    alpha_los: float,
    alpha_nlos: float,
    beta0: float = 1.0,
    los_a: float = 9.61,
    los_b: float = 0.16,
) -> float:
    p_los = los_probability(theta, los_a, los_b)
    gain_los = path_loss(d, alpha_los, beta0)
    gain_nlos = path_loss(d, alpha_nlos, beta0)
    blended = p_los * gain_los + (1.0 - p_los) * gain_nlos
    return blended * fading