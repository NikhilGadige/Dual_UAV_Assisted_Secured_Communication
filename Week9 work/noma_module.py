"""
Power-Domain NOMA (PD-NOMA) Module for Near and Distant (Far) Users.
Implements superposed signal power allocation, channel ordering, 
Successive Interference Cancellation (SIC) decoding, and SINR calculations.
"""

from typing import Dict, Tuple

EPSILON = 1e-12

class PowerDomainNOMA:
    """
    Power-Domain NOMA engine for paired Near and Distant (Far) users.
    
    Parameters:
    -----------
    p_tx : float
        Total transmit power at transmitter (BS / RIS) in Watts (default: 1.0 W = 30 dBm).
    power_alloc_far : float
        Power allocation factor for Far User a_F (default: 0.7, meaning a_N = 0.3).
    noise_power : float
        Noise power sigma^2 in Watts (default: 1e-10 W = -70 dBm).
    sic_threshold_db : float
        Minimum SINR threshold required at Near User to successfully cancel Far User signal via SIC.
    """

    def __init__(
        self,
        p_tx: float = 1.0,
        power_alloc_far: float = 0.7,
        noise_power: float = 1e-10,
        sic_threshold_db: float = 0.0,
    ):
        self.p_tx = p_tx
        self.power_alloc_far = max(0.5, min(0.99, power_alloc_far))
        self.power_alloc_near = 1.0 - self.power_alloc_far
        self.noise_power = noise_power
        self.sic_threshold_lin = 10.0 ** (sic_threshold_db / 10.0)

    def set_power_allocation(self, a_far: float):
        """Update power allocation ratio (a_far > a_near)."""
        assert 0.5 <= a_far < 1.0, "Far user power allocation coefficient must be in [0.5, 1.0)"
        self.power_alloc_far = a_far
        self.power_alloc_near = 1.0 - a_far

    def compute_sinr(
        self,
        g_near: float,
        g_far: float,
        jamming_power_near: float = 0.0,
        jamming_power_far: float = 0.0,
    ) -> Dict[str, float]:
        """
        Computes SINR for Far User and Near User (with SIC).
        
        Parameters:
        -----------
        g_near : float
            Effective linear channel gain for Near User.
        g_far : float
            Effective linear channel gain for Far User.
        jamming_power_near : float
            Received jamming interference power at Near User.
        jamming_power_far : float
            Received jamming interference power at Far User.
            
        Returns:
        --------
        dict containing:
            - 'sinr_far': SINR of Far User (decodes s_F directly)
            - 'sinr_near_sic': SINR at Near User when decoding Far User signal s_F
            - 'sic_successful': bool indicating whether SIC decoding succeeded
            - 'sinr_near': Final SINR of Near User for decoding its own signal s_N
        """
        # Ensure proper ordering (g_near >= g_far)
        if g_near < g_far:
            # Swap if user channel gains are reversed
            g_near, g_far = g_far, g_near
            jamming_power_near, jamming_power_far = jamming_power_far, jamming_power_near

        p_far = self.power_alloc_far * self.p_tx
        p_near = self.power_alloc_near * self.p_tx

        # 1. Far User decodes its signal s_F (treats s_N as interference)
        denom_far = p_near * g_far + jamming_power_far + self.noise_power
        sinr_far = (p_far * g_far) / max(denom_far, EPSILON)

        # 2. Near User decodes Far User signal s_F for SIC
        denom_near_sic = p_near * g_near + jamming_power_near + self.noise_power
        sinr_near_sic = (p_far * g_near) / max(denom_near_sic, EPSILON)

        # Check SIC condition
        sic_successful = sinr_near_sic >= self.sic_threshold_lin

        if sic_successful:
            # SIC succeeds: subtract s_F and decode s_N without NOMA intra-cell interference
            denom_near = jamming_power_near + self.noise_power
            sinr_near = (p_near * g_near) / max(denom_near, EPSILON)
        else:
            # SIC fails: near user cannot subtract s_F, treats it as interference
            denom_near = p_far * g_near + jamming_power_near + self.noise_power
            sinr_near = (p_near * g_near) / max(denom_near, EPSILON)

        return {
            "sinr_far": float(sinr_far),
            "sinr_near_sic": float(sinr_near_sic),
            "sic_successful": bool(sic_successful),
            "sinr_near": float(sinr_near),
            "g_near": float(g_near),
            "g_far": float(g_far),
        }
