"""
Week 9 System Model Integration: Semantic-Aware ISAC Network with PD-NOMA 
and Elevation-Dependent Path Loss.

Models:
- Semantic-aware nodes: BS, RIS-UAV (R), UAV Jammer (J), Near User/Vehicle (V), 
  Far User/Mobile User (U), Eavesdroppers (E1, E2, E3).
- Elevation-dependent path loss for all 3D air-to-ground & ground links.
- Power-Domain NOMA (PD-NOMA) with SIC for near/distant user differentiation.
- Semantic metrics evaluation for communication and sensing.
"""

from typing import Dict, Any, List
import numpy as np

try:
    from .elevation_path_loss import ElevationPathLossModel
    from .noma_module import PowerDomainNOMA
    from .semantic_node import SemanticNode, SemanticNodeType, SemanticMetrics
except ImportError:
    from elevation_path_loss import ElevationPathLossModel
    from noma_module import PowerDomainNOMA
    from semantic_node import SemanticNode, SemanticNodeType, SemanticMetrics

class Week9SystemModel:
    """
    Comprehensive System Model for Week 9 ISAC Network.
    """

    def __init__(
        self,
        p_bs_tx: float = 2.0,            # BS transmit power in Watts (33 dBm)
        p_jam_tx: float = 0.5,           # Jammer transmit power in Watts (27 dBm)
        n_ris_elements: int = 64,        # Number of RIS elements
        noma_a_far: float = 0.7,         # Power allocation factor for far user (a_F)
        freq_hz: float = 2.4e9,          # Carrier frequency 2.4 GHz
        bandwidth_hz: float = 1e6,       # Bandwidth 1 MHz
        noise_power: float = 1e-10,      # Thermal noise (-70 dBm)
    ):
        self.p_bs_tx = p_bs_tx
        self.p_jam_tx = p_jam_tx
        self.n_ris_elements = n_ris_elements
        self.bandwidth_hz = bandwidth_hz
        self.noise_power = noise_power

        # Modules
        self.path_loss_model = ElevationPathLossModel(freq_hz=freq_hz)
        self.noma_engine = PowerDomainNOMA(
            p_tx=p_bs_tx,
            power_alloc_far=noma_a_far,
            noise_power=noise_power,
        )

        # Initialize Semantic Nodes according to system architecture diagram
        self.nodes: Dict[str, SemanticNode] = {
            "BS": SemanticNode("BS", SemanticNodeType.BASE_STATION, np.array([0.0, 0.0, 15.0])),
            "RIS": SemanticNode("RIS", SemanticNodeType.RIS_UAV, np.array([50.0, 50.0, 80.0])),
            "Jammer": SemanticNode("Jammer", SemanticNodeType.UAV_JAMMER, np.array([100.0, 120.0, 60.0])),
            "Vehicle": SemanticNode("Vehicle", SemanticNodeType.NEAR_USER, np.array([40.0, 45.0, 1.5])),
            "MobileUser": SemanticNode("MobileUser", SemanticNodeType.FAR_USER, np.array([150.0, 180.0, 1.5])),
            "E1": SemanticNode("E1", SemanticNodeType.EAVESDROPPER, np.array([70.0, 80.0, 0.0])),
            "E2": SemanticNode("E2", SemanticNodeType.EAVESDROPPER, np.array([90.0, 100.0, 0.0])),
            "E3": SemanticNode("E3", SemanticNodeType.EAVESDROPPER, np.array([110.0, 110.0, 0.0])),
        }

    def compute_cascaded_gain(
        self, pos_tx: np.ndarray, pos_ris: np.ndarray, pos_rx: np.ndarray
    ) -> float:
        """
        Computes cascaded channel gain through RIS: BS -> RIS -> RX
        Gain_cascaded = g_{BS-RIS} * g_{RIS-RX} * N_{RIS}^2
        incorporating elevation-dependent path loss on both segments.
        """
        g_bs_ris = self.path_loss_model.compute_channel_gain(pos_tx, pos_ris)
        g_ris_rx = self.path_loss_model.compute_channel_gain(pos_ris, pos_rx)
        ris_array_gain = self.n_ris_elements ** 2
        return float(g_bs_ris * g_ris_rx * ris_array_gain)

    def evaluate_system(self) -> Dict[str, Any]:
        """
        Executes full system simulation:
        1. Computes elevation angles & path losses for all channels.
        2. Computes cascaded channel gains for Near (Vehicle) and Far (Mobile User) users.
        3. Computes jamming interference from UAV Jammer.
        4. Applies Power-Domain NOMA with SIC decoding.
        5. Evaluates semantic performance metrics (Semantic Rate, Similarity, Distortion).
        6. Evaluates monostatic/bistatic sensing performance at UAV Jammer / BS.
        """
        pos_bs = self.nodes["BS"].position
        pos_ris = self.nodes["RIS"].position
        pos_jam = self.nodes["Jammer"].position
        pos_veh = self.nodes["Vehicle"].position
        pos_user = self.nodes["MobileUser"].position

        # 1. Elevation angles
        elev_bs_ris = self.path_loss_model.compute_elevation_angle(pos_bs, pos_ris)
        elev_ris_veh = self.path_loss_model.compute_elevation_angle(pos_ris, pos_veh)
        elev_ris_user = self.path_loss_model.compute_elevation_angle(pos_ris, pos_user)
        elev_jam_veh = self.path_loss_model.compute_elevation_angle(pos_jam, pos_veh)
        elev_jam_user = self.path_loss_model.compute_elevation_angle(pos_jam, pos_user)

        # 2. Channel Gains
        g_cascaded_near = self.compute_cascaded_gain(pos_bs, pos_ris, pos_veh)
        g_cascaded_far = self.compute_cascaded_gain(pos_bs, pos_ris, pos_user)

        # Direct link gains (if LoS exists)
        g_direct_near = self.path_loss_model.compute_channel_gain(pos_bs, pos_veh)
        g_direct_far = self.path_loss_model.compute_channel_gain(pos_bs, pos_user)

        # Total combined channel gain for NOMA ordering
        g_total_near = g_cascaded_near + g_direct_near
        g_total_far = g_cascaded_far + g_direct_far

        # 3. Jamming Interference from UAV Jammer (J)
        g_jam_veh = self.path_loss_model.compute_channel_gain(pos_jam, pos_veh)
        g_jam_user = self.path_loss_model.compute_channel_gain(pos_jam, pos_user)

        jamming_power_near = self.p_jam_tx * g_jam_veh
        jamming_power_far = self.p_jam_tx * g_jam_user

        # 4. Power Domain NOMA Evaluation
        noma_results = self.noma_engine.compute_sinr(
            g_near=g_total_near,
            g_far=g_total_far,
            jamming_power_near=jamming_power_near,
            jamming_power_far=jamming_power_far,
        )

        # 5. Eavesdropper SINR evaluation (E1, E2, E3 intercepting signals)
        eavesdropper_results = {}
        for e_name in ["E1", "E2", "E3"]:
            pos_e = self.nodes[e_name].position
            g_cascaded_e = self.compute_cascaded_gain(pos_bs, pos_ris, pos_e)
            g_jam_e = self.path_loss_model.compute_channel_gain(pos_jam, pos_e)
            p_jam_e = self.p_jam_tx * g_jam_e
            
            # Eavesdropper tries to decode far user signal s_F
            sinr_e_far = (self.noma_engine.power_alloc_far * self.p_bs_tx * g_cascaded_e) / max(
                self.noma_engine.power_alloc_near * self.p_bs_tx * g_cascaded_e + p_jam_e + self.noise_power, 1e-12
            )
            e_metrics = self.nodes[e_name].evaluate_metrics(sinr_e_far, bandwidth_hz=self.bandwidth_hz)
            eavesdropper_results[e_name] = {
                "sinr_db": e_metrics.sinr_db,
                "semantic_similarity": e_metrics.semantic_similarity,
                "semantic_rate": e_metrics.semantic_rate_suts,
            }

        # 6. Evaluate Semantic Metrics for Downlink (BS -> RIS -> Vehicle & MobileUser)
        near_metrics = self.nodes["Vehicle"].evaluate_metrics(
            noma_results["sinr_near"], bandwidth_hz=self.bandwidth_hz
        )
        far_metrics = self.nodes["MobileUser"].evaluate_metrics(
            noma_results["sinr_far"], bandwidth_hz=self.bandwidth_hz
        )

        # 7. Vehicle Active Communication (Uplink V2I/V2A to BS via RIS, and V2X to Mobile User)
        p_veh_tx = 0.2  # Vehicle transmit power in Watts (23 dBm)
        
        # Vehicle -> RIS -> BS Uplink Cascaded Channel Gain
        g_veh_ris_bs = self.compute_cascaded_gain(pos_veh, pos_ris, pos_bs)
        g_jam_bs = self.path_loss_model.compute_channel_gain(pos_jam, pos_bs)
        jamming_power_bs = self.p_jam_tx * g_jam_bs
        
        sinr_veh_uplink = (p_veh_tx * g_veh_ris_bs) / max(jamming_power_bs + self.noise_power, 1e-12)
        veh_uplink_metrics = self.nodes["Vehicle"].evaluate_metrics(sinr_veh_uplink, bandwidth_hz=self.bandwidth_hz)

        # Vehicle -> Mobile User (V2X Direct Semantic Link)
        g_veh_user = self.path_loss_model.compute_channel_gain(pos_veh, pos_user)
        sinr_veh_v2x = (p_veh_tx * g_veh_user) / max(jamming_power_far + self.noise_power, 1e-12)
        veh_v2x_metrics = self.nodes["Vehicle"].evaluate_metrics(sinr_veh_v2x, bandwidth_hz=self.bandwidth_hz)

        # 8. Monostatic Reflection Sensing at UAV Jammer / Sensing SNR
        g_jam_monostatic = self.path_loss_model.compute_channel_gain(pos_jam, pos_ris)
        snr_radar_jam = (self.p_jam_tx * g_jam_monostatic) / self.noise_power
        sensing_acc_jam = self.nodes["Jammer"].compute_sensing_accuracy(snr_radar_jam)

        return {
            "elevation_angles": {
                "BS_to_RIS": elev_bs_ris,
                "RIS_to_Vehicle": elev_ris_veh,
                "RIS_to_MobileUser": elev_ris_user,
                "Jammer_to_Vehicle": elev_jam_veh,
                "Jammer_to_MobileUser": elev_jam_user,
            },
            "noma": noma_results,
            "semantic_performance": {
                "Downlink_Vehicle_NearUser": {
                    "sinr_db": near_metrics.sinr_db,
                    "semantic_similarity": near_metrics.semantic_similarity,
                    "semantic_rate_suts": near_metrics.semantic_rate_suts,
                    "semantic_distortion": near_metrics.semantic_distortion,
                },
                "Downlink_MobileUser_FarUser": {
                    "sinr_db": far_metrics.sinr_db,
                    "semantic_similarity": far_metrics.semantic_similarity,
                    "semantic_rate_suts": far_metrics.semantic_rate_suts,
                    "semantic_distortion": far_metrics.semantic_distortion,
                },
                "Uplink_Vehicle_to_BS": {
                    "sinr_db": veh_uplink_metrics.sinr_db,
                    "semantic_similarity": veh_uplink_metrics.semantic_similarity,
                    "semantic_rate_suts": veh_uplink_metrics.semantic_rate_suts,
                    "semantic_distortion": veh_uplink_metrics.semantic_distortion,
                },
                "V2X_Vehicle_to_MobileUser": {
                    "sinr_db": veh_v2x_metrics.sinr_db,
                    "semantic_similarity": veh_v2x_metrics.semantic_similarity,
                    "semantic_rate_suts": veh_v2x_metrics.semantic_rate_suts,
                    "semantic_distortion": veh_v2x_metrics.semantic_distortion,
                },
            },
            "sensing_performance": {
                "Jammer_Monostatic_Radar_SNR_dB": 10.0 * np.log10(max(snr_radar_jam, 1e-12)),
                "Jammer_Sensing_Accuracy": sensing_acc_jam,
            },
            "eavesdroppers": eavesdropper_results,
        }
