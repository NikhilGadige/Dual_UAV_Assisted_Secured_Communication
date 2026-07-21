"""
Elevation-Dependent Path Loss Model for Air-to-Ground (A2G) and Terrestrial Links.
Implements elevation angle calculation, sigmoid LoS probability, elevation-dependent 
path loss exponents/attenuation, and fading channels.
"""

import numpy as np

C_LIGHT = 299792458.0  # Speed of light in m/s
EPSILON = 1e-10

class ElevationPathLossModel:
    """
    Elevation-Dependent Path Loss Model.
    
    Parameters:
    -----------
    freq_hz : float
        Carrier frequency in Hertz (default: 2.4 GHz).
    los_a : float
        Environment constant 'a' for LoS probability sigmoid (default: 9.61 for Urban).
    los_b : float
        Environment constant 'b' for LoS probability sigmoid (default: 0.16 for Urban).
    eta_los_db : float
        Excessive path loss for LoS in dB (default: 1.0 dB).
    eta_nlos_db : float
        Excessive path loss for NLoS in dB (default: 20.0 dB).
    alpha_los : float
        Path loss exponent for LoS (default: 2.0).
    alpha_nlos : float
        Path loss exponent for NLoS (default: 3.5).
    """

    def __init__(
        self,
        freq_hz: float = 2.4e9,
        los_a: float = 9.61,
        los_b: float = 0.16,
        eta_los_db: float = 1.0,
        eta_nlos_db: float = 20.0,
        alpha_los: float = 2.0,
        alpha_nlos: float = 3.5,
    ):
        self.freq_hz = freq_hz
        self.los_a = los_a
        self.los_b = los_b
        self.eta_los_db = eta_los_db
        self.eta_nlos_db = eta_nlos_db
        self.alpha_los = alpha_los
        self.alpha_nlos = alpha_nlos

    @staticmethod
    def compute_distance_3d(pos_tx: np.ndarray, pos_rx: np.ndarray) -> float:
        """Computes 3D Euclidean distance between tx and rx positions."""
        pos_tx = np.asarray(pos_tx, dtype=float)
        pos_rx = np.asarray(pos_rx, dtype=float)
        return float(np.linalg.norm(pos_tx - pos_rx))

    @staticmethod
    def compute_elevation_angle(pos_tx: np.ndarray, pos_rx: np.ndarray) -> float:
        """
        Computes elevation angle in degrees (0 to 90 degrees) between TX and RX.
        """
        pos_tx = np.asarray(pos_tx, dtype=float)
        pos_rx = np.asarray(pos_rx, dtype=float)
        dx = pos_tx[0] - pos_rx[0]
        dy = pos_tx[1] - pos_rx[1]
        dz = abs(pos_tx[2] - pos_rx[2])
        
        d_2d = np.sqrt(dx * dx + dy * dy)
        if d_2d < EPSILON:
            return 90.0 if dz > EPSILON else 0.0
        
        theta_rad = np.arctan2(dz, d_2d)
        return float(np.degrees(theta_rad))

    def los_probability(self, theta_deg: float) -> float:
        """
        Sigmoidal Line-of-Sight (LoS) probability as a function of elevation angle theta.
        P_LoS(theta) = 1 / (1 + a * exp(-b * (theta - a)))
        """
        exponent = -self.los_b * (theta_deg - self.los_a)
        # Avoid overflow in exp
        exponent = np.clip(exponent, -50.0, 50.0)
        return float(1.0 / (1.0 + self.los_a * np.exp(exponent)))

    def free_space_path_loss_db(self, d: float) -> float:
        """Free space path loss in dB."""
        d_safe = max(d, EPSILON)
        fspl_linear = (4.0 * np.pi * d_safe * self.freq_hz / C_LIGHT) ** 2
        return float(10.0 * np.log10(fspl_linear))

    def compute_path_loss_db(self, pos_tx: np.ndarray, pos_rx: np.ndarray) -> float:
        """
        Computes elevation-dependent mean path loss in dB.
        PL_mean(d, theta) = P_LoS(theta) * PL_LoS(d) + (1 - P_LoS(theta)) * PL_NLoS(d)
        """
        d = self.compute_distance_3d(pos_tx, pos_rx)
        theta = self.compute_elevation_angle(pos_tx, pos_rx)
        
        p_los = self.los_probability(theta)
        fspl_db = self.free_space_path_loss_db(d)
        
        # LoS and NLoS path loss with excessive attenuation
        pl_los_db = fspl_db + self.eta_los_db
        pl_nlos_db = fspl_db + self.eta_nlos_db
        
        # Average path loss in dB
        pl_mean_db = p_los * pl_los_db + (1.0 - p_los) * pl_nlos_db
        return pl_mean_db

    def compute_channel_gain(
        self,
        pos_tx: np.ndarray,
        pos_rx: np.ndarray,
        fading_type: str = "rician",
    ) -> float:
        """
        Computes linear channel power gain |h|^2 incorporating elevation-dependent path loss 
        and small-scale fading (Rician or Rayleigh).
        """
        pl_db = self.compute_path_loss_db(pos_tx, pos_rx)
        pl_linear = 10.0 ** (-pl_db / 10.0)
        
        theta = self.compute_elevation_angle(pos_tx, pos_rx)
        
        if fading_type == "rician":
            # Elevation-dependent Rician K-factor: higher elevation -> higher K-factor
            k_factor_db = 3.0 + 0.15 * theta
            K = 10.0 ** (k_factor_db / 10.0)
            s = np.sqrt(K / (K + 1.0))
            sigma = np.sqrt(1.0 / (2.0 * (K + 1.0)))
            h_real = s + np.random.normal(0.0, sigma)
            h_imag = np.random.normal(0.0, sigma)
            fading_gain = h_real ** 2 + h_imag ** 2
        else: # rayleigh
            h_real = np.random.normal(0.0, 1.0 / np.sqrt(2.0))
            h_imag = np.random.normal(0.0, 1.0 / np.sqrt(2.0))
            fading_gain = h_real ** 2 + h_imag ** 2

        return float(pl_linear * fading_gain)
