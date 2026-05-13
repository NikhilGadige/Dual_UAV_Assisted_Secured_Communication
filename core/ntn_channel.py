import numpy as np

C = 299792458.0  # speed of light (m/s)
EPSILON = 1e-10


def free_space_path_loss(d: float, freq_hz: float) -> float:
    d_safe = max(d, EPSILON)
    return (C / (4.0 * np.pi * d_safe * freq_hz)) ** 2.0


def atmospheric_loss_db_to_linear(atmospheric_loss_db: float) -> float:
    """Convert atmospheric attenuation from dB to linear gain factor."""
    return 10.0 ** (-max(atmospheric_loss_db, 0.0) / 10.0)


def _rician_fading(k_db: float) -> float:
    K = 10.0 ** (k_db / 10.0)
    s = np.sqrt(K / (K + 1.0))
    sigma = np.sqrt(1.0 / (2.0 * (K + 1.0)))
    h_real = s + np.random.normal(0.0, sigma)
    h_imag = np.random.normal(0.0, sigma)
    return h_real ** 2 + h_imag ** 2


def satellite_channel_gain(
    d: float,
    freq_hz: float = 2e9,
    atmospheric_loss_db: float = 0.5,
    rician_k_db: float = 10.0,
) -> float:
    fspl = free_space_path_loss(d, freq_hz)
    latm = atmospheric_loss_db_to_linear(atmospheric_loss_db)
    fading = _rician_fading(rician_k_db)
    return fspl * latm * fading
