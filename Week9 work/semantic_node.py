"""
Semantic-Aware Node and Metric Framework for ISAC System.
Models all nodes (BS, RIS-UAV, UAV Jammer, Vehicles, Mobile Users, Eavesdroppers) 
with semantic-aware communication and sensing capabilities.
"""

from dataclasses import dataclass, field
from enum import Enum
import numpy as np

class SemanticNodeType(Enum):
    BASE_STATION = "Base Station (B)"
    RIS_UAV = "RIS-mounted UAV (R)"
    UAV_JAMMER = "UAV Jammer (J)"
    NEAR_USER = "Target / Near User (T)"
    FAR_USER = "Mobile User / Distant User (U)"
    EAVESDROPPER = "Eavesdropper (E)"

@dataclass
class SemanticMetrics:
    """
    Semantic Communication & Sensing Performance Metrics.
    """
    sinr_db: float = 0.0
    semantic_similarity: float = 0.0      # S(SINR) in [0, 1]
    semantic_rate_suts: float = 0.0       # Semantic rate (semantic units / sec / Hz)
    semantic_distortion: float = 0.0     # D = 1 - S(SINR)
    sensing_accuracy: float = 0.0        # Semantic sensing target accuracy in [0, 1]
    crb: float = 0.0                     # Cramér-Rao Bound (CRB) for sensing estimation

class SemanticNode:
    """
    Represents a Semantic-Aware Node in the ISAC network.
    """

    def __init__(
        self,
        node_id: str,
        node_type: SemanticNodeType,
        position: np.ndarray,
        semantic_unit_length_m: int = 10,   # M: number of semantic concepts
        channel_symbol_length_k: int = 15,  # K: number of transmitted channel symbols
        lambda1: float = 0.3,               # Logistic shape param for semantic similarity
        lambda2: float = -0.5,              # Logistic threshold param for semantic similarity
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.position = np.asarray(position, dtype=float)
        self.m_concepts = semantic_unit_length_m
        self.k_symbols = channel_symbol_length_k
        self.compression_ratio = self.k_symbols / max(1, self.m_concepts)
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def update_position(self, new_position: np.ndarray):
        """Update 3D node coordinates."""
        self.position = np.asarray(new_position, dtype=float)

    def compute_semantic_similarity(self, sinr_linear: float) -> float:
        """
        Computes semantic similarity score S(SINR) in range [0, 1].
        S(SINR_dB) = 1 / (1 + exp(-(lambda1 * SINR_dB + lambda2)))
        """
        sinr_db = 10.0 * np.log10(max(sinr_linear, 1e-12))
        exponent = -(self.lambda1 * sinr_db + self.lambda2)
        exponent = np.clip(exponent, -50.0, 50.0)
        return float(1.0 / (1.0 + np.exp(exponent)))

    def compute_semantic_rate(
        self, sinr_linear: float, bandwidth_hz: float = 1.0e6
    ) -> float:
        """
        Computes Semantic Rate R_sem (suts/sec):
        R_sem = Bandwidth * (M / K) * S(SINR) * log2(1 + SINR)
        """
        s_score = self.compute_semantic_similarity(sinr_linear)
        shannon_rate = bandwidth_hz * np.log2(1.0 + max(sinr_linear, 0.0))
        semantic_rate = (self.m_concepts / self.k_symbols) * s_score * shannon_rate
        return float(semantic_rate)

    def compute_sensing_accuracy(self, snr_radar_linear: float) -> float:
        """
        Computes semantic sensing accuracy for target detection/feature extraction.
        A_sensing = 1 / (1 + exp(-0.4 * (SNR_dB - 2.0)))
        """
        snr_db = 10.0 * np.log10(max(snr_radar_linear, 1e-12))
        exponent = -0.4 * (snr_db - 2.0)
        exponent = np.clip(exponent, -50.0, 50.0)
        return float(1.0 / (1.0 + np.exp(exponent)))

    def compute_crb(self, snr_radar_linear: float) -> float:
        """
        Computes Cramér-Rao Bound (CRB) for location estimation.
        Modelled as inversely proportional to radar sensing SNR.
        """
        return float(1.0 / max(snr_radar_linear, 1e-12))

    def evaluate_metrics(
        self, sinr_linear: float, snr_radar_linear: float = 1.0, bandwidth_hz: float = 1.0e6
    ) -> SemanticMetrics:
        """Evaluates complete set of semantic metrics for the node."""
        sinr_db = float(10.0 * np.log10(max(sinr_linear, 1e-12)))
        s_score = self.compute_semantic_similarity(sinr_linear)
        s_rate = self.compute_semantic_rate(sinr_linear, bandwidth_hz)
        s_dist = 1.0 - s_score
        sense_acc = self.compute_sensing_accuracy(snr_radar_linear)
        crb_val = self.compute_crb(snr_radar_linear)

        return SemanticMetrics(
            sinr_db=sinr_db,
            semantic_similarity=s_score,
            semantic_rate_suts=s_rate,
            semantic_distortion=s_dist,
            sensing_accuracy=sense_acc,
            crb=crb_val,
        )
