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
            sic_threshold_semantic=0.5,  # Semantic SIC similarity threshold
        )

        # Initialize Semantic Nodes according to system architecture diagram
        self.nodes: Dict[str, SemanticNode] = {
            "BS": SemanticNode("BS", SemanticNodeType.BASE_STATION, np.array([0.0, 0.0, 15.0])),
            "R": SemanticNode("R", SemanticNodeType.RIS_UAV, np.array([50.0, 50.0, 80.0])),
            "J": SemanticNode("J", SemanticNodeType.UAV_JAMMER, np.array([100.0, 120.0, 60.0])),
            "T": SemanticNode("T", SemanticNodeType.NEAR_USER, np.array([40.0, 45.0, 1.5])),
            "U": SemanticNode("U", SemanticNodeType.FAR_USER, np.array([150.0, 180.0, 1.5])),
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
        2. Computes cascaded channel gains for Near (Target) and Far (Mobile User) users.
        3. Computes jamming interference from UAV Jammer J.
        4. Applies Power-Domain NOMA with semantic-aware SIC decoding.
        5. Evaluates semantic performance metrics (Semantic Rate, Similarity, Distortion).
        6. Evaluates monostatic/bistatic sensing performance at UAV Jammer J / BS.
        """
        pos_bs = self.nodes["BS"].position
        pos_r = self.nodes["R"].position
        pos_j = self.nodes["J"].position
        pos_t = self.nodes["T"].position
        pos_u = self.nodes["U"].position

        # 1. Elevation angles
        elev_bs_r = self.path_loss_model.compute_elevation_angle(pos_bs, pos_r)
        elev_r_t = self.path_loss_model.compute_elevation_angle(pos_r, pos_t)
        elev_r_u = self.path_loss_model.compute_elevation_angle(pos_r, pos_u)
        elev_j_t = self.path_loss_model.compute_elevation_angle(pos_j, pos_t)
        elev_j_u = self.path_loss_model.compute_elevation_angle(pos_j, pos_u)

        # 2. Channel Gains
        g_cascaded_near = self.compute_cascaded_gain(pos_bs, pos_r, pos_t)
        g_cascaded_far = self.compute_cascaded_gain(pos_bs, pos_r, pos_u)

        # Total combined channel gain for NOMA ordering (only via RIS R as per system model)
        g_total_near = g_cascaded_near
        g_total_far = g_cascaded_far

        # 3. Jamming Interference from UAV Jammer (J)
        g_jam_t = self.path_loss_model.compute_channel_gain(pos_j, pos_t)
        g_jam_u = self.path_loss_model.compute_channel_gain(pos_j, pos_u)

        jamming_power_near = self.p_jam_tx * g_jam_t
        jamming_power_far = self.p_jam_tx * g_jam_u

        # 4. Power Domain NOMA Evaluation with semantic SIC parameters
        noma_results = self.noma_engine.compute_sinr(
            g_near=g_total_near,
            g_far=g_total_far,
            jamming_power_near=jamming_power_near,
            jamming_power_far=jamming_power_far,
        )

        # 5. Eavesdropper SINR evaluation (E1, E2, E3 intercepting signals)
        eavesdropper_results = {}
        sum_r_e_far = 0.0
        sum_r_e_near = 0.0
        for e_name in ["E1", "E2", "E3"]:
            pos_e = self.nodes[e_name].position
            g_cascaded_e = self.compute_cascaded_gain(pos_bs, pos_r, pos_e)
            g_jam_e = self.path_loss_model.compute_channel_gain(pos_j, pos_e)
            p_jam_e = self.p_jam_tx * g_jam_e
            
            # Eavesdropper tries to decode far user signal s_F (Mobile User U)
            sinr_e_far = (self.noma_engine.power_alloc_far * self.p_bs_tx * g_cascaded_e) / max(
                self.noma_engine.power_alloc_near * self.p_bs_tx * g_cascaded_e + p_jam_e + self.noise_power, 1e-12
            )
            # Eavesdropper tries to decode near user signal s_N (Target T, after perfect SIC)
            sinr_e_near = (self.noma_engine.power_alloc_near * self.p_bs_tx * g_cascaded_e) / max(
                p_jam_e + self.noise_power, 1e-12
            )
            
            r_e_far = self.nodes[e_name].compute_semantic_rate(sinr_e_far, bandwidth_hz=self.bandwidth_hz)
            r_e_near = self.nodes[e_name].compute_semantic_rate(sinr_e_near, bandwidth_hz=self.bandwidth_hz)
            
            sum_r_e_far += r_e_far
            sum_r_e_near += r_e_near
            
            eavesdropper_results[e_name] = {
                "sinr_db": float(10.0 * np.log10(max(sinr_e_far, 1e-12))),
                "semantic_similarity": self.nodes[e_name].compute_semantic_similarity(sinr_e_far),
                "semantic_rate": r_e_far,
                "sinr_near_db": float(10.0 * np.log10(max(sinr_e_near, 1e-12))),
                "semantic_similarity_near": self.nodes[e_name].compute_semantic_similarity(sinr_e_near),
                "semantic_rate_near": r_e_near,
            }

        # 6. Evaluate Semantic Metrics for Downlink (BS -> RIS R -> Target T & MobileUser U)
        near_metrics = self.nodes["T"].evaluate_metrics(
            noma_results["sinr_near"], bandwidth_hz=self.bandwidth_hz
        )
        far_metrics = self.nodes["U"].evaluate_metrics(
            noma_results["sinr_far"], bandwidth_hz=self.bandwidth_hz
        )

        # 6a. Average Secrecy Rate (ASR) Evaluation
        r_sec_t = max(near_metrics.semantic_rate_suts - sum_r_e_near, 0.0)
        r_sec_u = max(far_metrics.semantic_rate_suts - sum_r_e_far, 0.0)
        r_sec_total = r_sec_t + r_sec_u

        # 7. Target T Active Communication (Uplink V2I to BS via RIS R, and V2X to Mobile User U)
        p_t_tx = 0.2  # Target transmit power in Watts (23 dBm)
        
        # Target -> RIS R -> BS Uplink Cascaded Channel Gain
        g_t_ris_bs = self.compute_cascaded_gain(pos_t, pos_r, pos_bs)
        g_jam_bs = self.path_loss_model.compute_channel_gain(pos_j, pos_bs)
        jamming_power_bs = self.p_jam_tx * g_jam_bs
        
        sinr_t_uplink = (p_t_tx * g_t_ris_bs) / max(jamming_power_bs + self.noise_power, 1e-12)
        t_uplink_metrics = self.nodes["T"].evaluate_metrics(sinr_t_uplink, bandwidth_hz=self.bandwidth_hz)

        # Target -> Mobile User U (V2X Direct Semantic Link)
        g_t_user = self.path_loss_model.compute_channel_gain(pos_t, pos_u)
        sinr_t_v2x = (p_t_tx * g_t_user) / max(jamming_power_far + self.noise_power, 1e-12)
        t_v2x_metrics = self.nodes["T"].evaluate_metrics(sinr_t_v2x, bandwidth_hz=self.bandwidth_hz)

        # 8. Monostatic Radar Sensing at UAV Jammer J
        g_jt = self.path_loss_model.compute_channel_gain(pos_j, pos_t)
        snr_radar_t = (self.p_jam_tx * (g_jt ** 2)) / self.noise_power
        sensing_acc_t = self.nodes["J"].compute_sensing_accuracy(snr_radar_t)
        crb_t = self.nodes["J"].compute_crb(snr_radar_t)

        sensing_e_results = {}
        for e_name in ["E1", "E2", "E3"]:
            pos_e = self.nodes[e_name].position
            g_je = self.path_loss_model.compute_channel_gain(pos_j, pos_e)
            snr_radar_e = (self.p_jam_tx * (g_je ** 2)) / self.noise_power
            acc_e = self.nodes["J"].compute_sensing_accuracy(snr_radar_e)
            crb_e = self.nodes["J"].compute_crb(snr_radar_e)
            sensing_e_results[e_name] = {
                "snr_db": float(10.0 * np.log10(max(snr_radar_e, 1e-12))),
                "sensing_accuracy": float(acc_e),
                "crb": float(crb_e)
            }

        return {
            "elevation_angles": {
                "BS_to_RIS": elev_bs_r,
                "RIS_to_Vehicle": elev_r_t,
                "RIS_to_MobileUser": elev_r_u,
                "Jammer_to_Vehicle": elev_j_t,
                "Jammer_to_MobileUser": elev_j_u,
            },
            "noma": noma_results,
            "semantic_performance": {
                "Downlink_Target_NearUser": {
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
                "Uplink_Target_to_BS": {
                    "sinr_db": t_uplink_metrics.sinr_db,
                    "semantic_similarity": t_uplink_metrics.semantic_similarity,
                    "semantic_rate_suts": t_uplink_metrics.semantic_rate_suts,
                    "semantic_distortion": t_uplink_metrics.semantic_distortion,
                },
                "V2X_Target_to_MobileUser": {
                    "sinr_db": t_v2x_metrics.sinr_db,
                    "semantic_similarity": t_v2x_metrics.semantic_similarity,
                    "semantic_rate_suts": t_v2x_metrics.semantic_rate_suts,
                    "semantic_distortion": t_v2x_metrics.semantic_distortion,
                },
            },
            "sensing_performance": {
                "Target_T": {
                    "snr_db": float(10.0 * np.log10(max(snr_radar_t, 1e-12))),
                    "sensing_accuracy": float(sensing_acc_t),
                    "crb": float(crb_t),
                },
                "Eavesdroppers": sensing_e_results,
            },
            "secrecy_performance": {
                "secrecy_rate_target": r_sec_t,
                "secrecy_rate_user": r_sec_u,
                "secrecy_rate_total": r_sec_total,
            },
            "eavesdroppers": eavesdropper_results,
        }

    def run_simulation_episodes(
        self, 
        num_episodes: int = 100,
        w1: float = 0.5,
        w2: float = 0.5,
        w3: float = 0.5,
        w4: float = 0.5,
        delta_t: float = 1.0,
        delta_e: float = 1.0,
        pd_threshold_t: float = 0.5,
        pd_threshold_e: float = 0.5,
    ) -> Dict[str, float]:
        """
        Runs the system model for multiple episodes/independent fading realizations.
        Computes secrecy rates, detection probabilities (Pd), CRB metrics,
        normalizes the secrecy rates by the maximum secrecy rate achieved,
        and evaluates a multi-objective utility function with CRB and Pd constraints.
        """
        t_sec_rates = []
        u_sec_rates = []
        total_sec_rates = []
        
        pd_t_values = []
        pd_e_values = []
        pd_values = []
        
        crb_t_values = []
        crb_e_mean_values = []
        
        crb_satisfied_values = []
        pd_satisfied_values = []
        all_satisfied_values = []
        
        for _ in range(num_episodes):
            results = self.evaluate_system()
            
            # Secrecy rates
            sec = results["secrecy_performance"]
            t_sec = sec["secrecy_rate_target"]
            u_sec = sec["secrecy_rate_user"]
            tot_sec = sec["secrecy_rate_total"]
            
            t_sec_rates.append(t_sec)
            u_sec_rates.append(u_sec)
            total_sec_rates.append(tot_sec)
            
            # Detection probabilities
            pd_t = results["sensing_performance"]["Target_T"]["sensing_accuracy"]
            pd_eavesdroppers = [eve["sensing_accuracy"] for eve in results["sensing_performance"]["Eavesdroppers"].values()]
            pd_e = float(np.mean(pd_eavesdroppers)) if pd_eavesdroppers else 0.0
            pd_comb = w3 * pd_e + w4 * pd_t
            
            pd_t_values.append(pd_t)
            pd_e_values.append(pd_e)
            pd_values.append(pd_comb)
            
            # CRB values
            crb_t = results["sensing_performance"]["Target_T"]["crb"]
            crbs_e = [eve["crb"] for eve in results["sensing_performance"]["Eavesdroppers"].values()]
            crb_e_mean = float(np.mean(crbs_e)) if crbs_e else 0.0
            
            crb_t_values.append(crb_t)
            crb_e_mean_values.append(crb_e_mean)
            
            # Constraint satisfaction
            is_crb_t_satisfied = crb_t <= delta_t
            is_crb_e_satisfied = all(c <= delta_e for c in crbs_e)
            
            is_pd_t_satisfied = pd_t >= pd_threshold_t
            is_pd_e_satisfied = all(pe >= pd_threshold_e for pe in pd_eavesdroppers)
            
            crb_satisfied_values.append(1.0 if (is_crb_t_satisfied and is_crb_e_satisfied) else 0.0)
            pd_satisfied_values.append(1.0 if (is_pd_t_satisfied and is_pd_e_satisfied) else 0.0)
            all_satisfied_values.append(1.0 if (is_crb_t_satisfied and is_crb_e_satisfied and is_pd_t_satisfied and is_pd_e_satisfied) else 0.0)
            
        # Normalization over episodes
        max_sec_rate = max(total_sec_rates) if total_sec_rates else 0.0
        if max_sec_rate <= 1e-12:
            max_sec_rate = 1.0
            
        normalized_total_sec_rates = [rate / max_sec_rate for rate in total_sec_rates]
        
        # Utility calculation
        utilities = [w1 * r_norm + w2 * pd_val for r_norm, pd_val in zip(normalized_total_sec_rates, pd_values)]
        
        return {
            "avg_secrecy_rate_target": float(np.mean(t_sec_rates)),
            "avg_secrecy_rate_user": float(np.mean(u_sec_rates)),
            "avg_secrecy_rate_total": float(np.mean(total_sec_rates)),
            "avg_normalized_secrecy_rate": float(np.mean(normalized_total_sec_rates)),
            "avg_pd_target": float(np.mean(pd_t_values)),
            "avg_pd_eavesdroppers": float(np.mean(pd_e_values)),
            "avg_pd_combined": float(np.mean(pd_values)),
            "avg_crb_target": float(np.mean(crb_t_values)),
            "avg_crb_eavesdroppers": float(np.mean(crb_e_mean_values)),
            "avg_utility": float(np.mean(utilities)),
            "crb_constraint_satisfaction": float(np.mean(crb_satisfied_values)),
            "pd_constraint_satisfaction": float(np.mean(pd_satisfied_values)),
            "overall_constraint_satisfaction": float(np.mean(all_satisfied_values)),
            "max_secrecy_rate": float(max_sec_rate),
        }
