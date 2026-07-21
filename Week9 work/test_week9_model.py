"""
Test & Validation Script for Week 9 Work:
1. Elevation-Dependent Path Loss Model
2. Power-Domain NOMA Engine with SIC
3. Semantic-Aware Node Framework
4. Full Integrated System Model
"""

import sys
import os
import numpy as np

# Ensure path includes workspace directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from elevation_path_loss import ElevationPathLossModel
from noma_module import PowerDomainNOMA
from semantic_node import SemanticNode, SemanticNodeType
from system_model import Week9SystemModel

def test_elevation_path_loss():
    print("=== 1. Testing Elevation-Dependent Path Loss Model ===")
    model = ElevationPathLossModel(freq_hz=2.4e9)
    
    pos_bs = np.array([0.0, 0.0, 15.0])
    pos_low_elev = np.array([500.0, 0.0, 20.0])   # Low elevation angle
    pos_high_elev = np.array([10.0, 0.0, 150.0])  # High elevation angle
    
    elev_low = model.compute_elevation_angle(pos_bs, pos_low_elev)
    elev_high = model.compute_elevation_angle(pos_bs, pos_high_elev)
    
    pl_low = model.compute_path_loss_db(pos_bs, pos_low_elev)
    pl_high = model.compute_path_loss_db(pos_bs, pos_high_elev)
    
    p_los_low = model.los_probability(elev_low)
    p_los_high = model.los_probability(elev_high)
    
    print(f"Low Elevation Link: Angle = {elev_low:.2f} deg, P(LoS) = {p_los_low:.4f}, Path Loss = {pl_low:.2f} dB")
    print(f"High Elevation Link: Angle = {elev_high:.2f} deg, P(LoS) = {p_los_high:.4f}, Path Loss = {pl_high:.2f} dB")
    assert elev_high > elev_low
    assert p_los_high > p_los_low
    print("-> Elevation-Dependent Path Loss Test PASSED!\n")

def test_pd_noma():
    print("=== 2. Testing Power-Domain NOMA Engine ===")
    noma = PowerDomainNOMA(p_tx=1.0, power_alloc_far=0.7, noise_power=1e-10)
    
    # Near user channel gain > Far user channel gain
    g_near = 1e-6
    g_far = 1e-8
    
    res = noma.compute_sinr(g_near=g_near, g_far=g_far, jamming_power_near=1e-11, jamming_power_far=1e-11)
    
    print(f"Near User Channel Gain: {g_near:.2e}, Far User Channel Gain: {g_far:.2e}")
    print(f"Far User SINR: {10*np.log10(res['sinr_far']):.2f} dB")
    print(f"Near User (SIC phase) SINR: {10*np.log10(res['sinr_near_sic']):.2f} dB")
    print(f"SIC Successful: {res['sic_successful']}")
    print(f"Near User Final SINR: {10*np.log10(res['sinr_near']):.2f} dB")
    
    assert res['sic_successful'] is True
    assert res['sinr_near'] > res['sinr_far']
    print("-> Power-Domain NOMA Test PASSED!\n")

def test_semantic_nodes():
    print("=== 3. Testing Semantic-Aware Nodes ===")
    node = SemanticNode("V1", SemanticNodeType.NEAR_USER, np.array([10.0, 10.0, 1.5]))
    
    sinr_low = 0.5  # ~ -3 dB
    sinr_high = 100.0 # ~ 20 dB
    
    metrics_low = node.evaluate_metrics(sinr_low)
    metrics_high = node.evaluate_metrics(sinr_high)
    
    print(f"Low SINR ({metrics_low.sinr_db:.1f} dB): Semantic Similarity = {metrics_low.semantic_similarity:.4f}, Rate = {metrics_low.semantic_rate_suts:.2e} suts")
    print(f"High SINR ({metrics_high.sinr_db:.1f} dB): Semantic Similarity = {metrics_high.semantic_similarity:.4f}, Rate = {metrics_high.semantic_rate_suts:.2e} suts")
    
    assert metrics_high.semantic_similarity > metrics_low.semantic_similarity
    assert metrics_high.semantic_rate_suts > metrics_low.semantic_rate_suts
    print("-> Semantic-Aware Node Test PASSED!\n")

def test_system_model_integration():
    print("=== 4. Testing Integrated Week 9 System Model ===")
    sys_model = Week9SystemModel(
        p_bs_tx=2.0,
        p_jam_tx=0.5,
        n_ris_elements=64,
        noma_a_far=0.7,
    )
    
    results = sys_model.evaluate_system()
    
    print("\n--- Elevation Angles ---")
    for link, angle in results["elevation_angles"].items():
        print(f"  {link}: {angle:.2f} deg")
        
    print("\n--- Power-Domain NOMA & SIC Results ---")
    noma = results["noma"]
    print(f"  SIC Successful at Near User: {noma['sic_successful']}")
    print(f"  Near User SINR: {10*np.log10(noma['sinr_near']):.2f} dB")
    print(f"  Far User SINR: {10*np.log10(noma['sinr_far']):.2f} dB")
    
    print("\n--- Semantic Performance ---")
    sem = results["semantic_performance"]
    for user, metrics in sem.items():
        print(f"  {user}: SINR = {metrics['sinr_db']:.2f} dB | Similarity = {metrics['semantic_similarity']:.4f} | Semantic Rate = {metrics['semantic_rate_suts']:.2e} suts/s | Distortion = {metrics['semantic_distortion']:.4f}")
        
    print("\n--- Sensing Performance at UAV Jammer ---")
    sense = results["sensing_performance"]
    print(f"  Monostatic Radar SNR: {sense['Jammer_Monostatic_Radar_SNR_dB']:.2f} dB")
    print(f"  Sensing Accuracy: {sense['Jammer_Sensing_Accuracy']:.4f}")

    print("\n--- Eavesdropper Performance ---")
    for eve, metrics in results["eavesdroppers"].items():
        print(f"  {eve}: Intercept SINR = {metrics['sinr_db']:.2f} dB | Semantic Similarity = {metrics['semantic_similarity']:.4f}")
        
    print("\n-> Integrated System Model Test PASSED Successfully!\n")

if __name__ == "__main__":
    test_elevation_path_loss()
    test_pd_noma()
    test_semantic_nodes()
    test_system_model_integration()
