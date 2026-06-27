"""Validate bistatic sensing model for vehicle targets via RIS-UAV.

Tests:
1. Bistatic distance correctness (d_tx == d_rx == d_total/2)
2. Sensing channel dimensions (scalar complex)
3. Sensing gain positivity
4. Echo signal dimensions
5. Multi-target support
6. SNR positivity
7. Vehicle type impact (truck > car > motorcycle)

Plots:
1. vehicle_sensing_snr.png
2. sensing_gain_vs_time.png
3. bistatic_distance_vs_time.png
4. per_vehicle_echo_power.png
"""

import sys
from pathlib import Path

import numpy as np

from bistatic_sensing_exp.channels.sensing_channel import (
    compute_bistatic_distances,
    compute_tx_distance,
    compute_rx_distance,
    compute_bistatic_distance,
    generate_sensing_channel,
    compute_sensing_gain,
    compute_echo_signal,
    compute_sensing_snr,
)
from bistatic_sensing_exp.environments.bistatic_sensing_env import (
    BistaticSensingConfig,
    BistaticSensingEnvironment,
)

OUTPUT_DIR = Path("outputs/bistatic_sensing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_bistatic_distance():
    print("\n" + "=" * 60)
    print("TEST: Bistatic Distance Correctness")
    print("=" * 60)

    ris_pos = np.array([0.0, 0.0, 50.0])
    veh_pos = np.array([100.0, 200.0])

    d_tx = compute_tx_distance(ris_pos, veh_pos)
    d_rx = compute_rx_distance(ris_pos, veh_pos)
    d_total = compute_bistatic_distance(ris_pos, veh_pos)
    dists = compute_bistatic_distances(ris_pos, veh_pos)

    expected_d = float(np.linalg.norm(np.array([100.0, 200.0])))
    expected_total = 2.0 * expected_d

    print(f"  Vehicle at ({veh_pos[0]}, {veh_pos[1]})")
    print(f"  d_tx   = {d_tx:.4f}")
    print(f"  d_rx   = {d_rx:.4f}")
    print(f"  d_total= {d_total:.4f}")
    print(f"  expected d  = {expected_d:.4f}")
    print(f"  expected d_total = {expected_total:.4f}")

    assert abs(d_tx - expected_d) < 1e-10, "TX distance mismatch"
    assert abs(d_rx - expected_d) < 1e-10, "RX distance mismatch"
    assert abs(d_total - expected_total) < 1e-10, "Bistatic distance mismatch"
    assert abs(dists["d_total"] - expected_total) < 1e-10, "Dict distance mismatch"
    assert d_tx > 0, "Distance must be positive"
    assert d_total > d_tx, "Total > one-way"

    print(f"  TX=RX={d_tx:.4f}, Total={d_total:.4f}: PASS")
    print()
    return True


def test_sensing_channel_dimensions():
    print("=" * 60)
    print("TEST: Sensing Channel Dimensions")
    print("=" * 60)

    d_tx = 150.0
    d_rx = 150.0
    rcs = 10.0

    gains = []
    for _ in range(20):
        h = generate_sensing_channel(d_tx, d_rx, rcs, K=5.0)
        g = compute_sensing_gain(h)
        gains.append(g)

        assert isinstance(h, complex), "h_sensing must be complex scalar"
        assert isinstance(g, float), "gain must be float"
        assert np.isfinite(h), "h_sensing must be finite"
        assert np.isfinite(g), "gain must be finite"

    mean_g = float(np.mean(gains))
    print(f"  h_sensing type: complex scalar  PASS")
    print(f"  Mean gain over 20 trials: {mean_g:.6e}")
    print(f"  All finite: PASS")
    print()
    return True


def test_sensing_gain_positivity():
    print("=" * 60)
    print("TEST: Sensing Gain Positivity")
    print("=" * 60)

    positions = [(50, 50), (200, 300), (400, 100)]
    rcs_values = [10.0, 100.0, 3.16]

    ris_pos = np.array([0.0, 0.0, 50.0])
    all_positive = True
    for (x, y), rcs in zip(positions, rcs_values):
        vp = np.array([float(x), float(y)])
        d = float(np.linalg.norm(vp))
        h = generate_sensing_channel(d, d, rcs, K=5.0)
        g = compute_sensing_gain(h)
        pos = g > 0
        print(f"  Vehicle ({x},{y}), rcs={rcs:.1f}: gain={g:.6e}, positive={pos}")
        if not pos:
            all_positive = False

    assert all_positive, "Some gains were zero"
    print(f"  All gains strictly positive: PASS\n")
    return True


def test_echo_signal():
    print("=" * 60)
    print("TEST: Echo Signal")
    print("=" * 60)

    P_s = 1.0
    h = 0.001 + 0.002j
    noise_power = 1e-10
    s = 1.0 + 0j

    y, n = compute_echo_signal(P_s, h, s, noise_power)

    expected_echo = np.sqrt(P_s) * h * s
    print(f"  Expected echo: {expected_echo}")
    print(f"  Received echo: {y}")
    print(f"  Noise: {n}")
    assert isinstance(y, complex), "Echo must be complex"
    assert np.isfinite(y), "Echo must be finite"

    y_noiseless, _ = compute_echo_signal(P_s, h, s, 0.0)
    assert abs(y_noiseless - expected_echo) < 1e-15
    print(f"  Noiseless echo matches expected: PASS")

    print(f"  Echo signal valid: PASS\n")
    return True


def test_multi_target():
    print("=" * 60)
    print("TEST: Multi-Target Support")
    print("=" * 60)

    env = BistaticSensingEnvironment(
        BistaticSensingConfig(seed=42, num_vehicles=4)
    )
    env.reset()
    sensing = env._compute_sensing()

    print(f"  Vehicles: {env.config.num_vehicles}")
    print(f"  Per-target entries: {len(sensing['per_target'])}")
    assert len(sensing["per_target"]) == 4, "Expected 4 targets"
    assert sensing["num_vehicles"] == 4

    gains = [t["gain"] for t in sensing["per_target"]]
    total_gain = sum(gains)
    print(f"  Per-target gains: {[f'{g:.4e}' for g in gains]}")
    print(f"  Total gain: {total_gain:.6e}")
    print(f"  Reported total: {sensing['total_sensing_gain']:.6e}")
    assert abs(total_gain - sensing["total_sensing_gain"]) < 1e-15
    print(f"  Total gain consistency: PASS")

    for t in sensing["per_target"]:
        assert "echo" in t, "Missing echo in per-target"
        assert isinstance(t["echo"], complex)
        assert np.isfinite(t["echo"])
    print(f"  All per-target echoes valid: PASS")
    print()
    return True


def test_snr_positivity():
    print("=" * 60)
    print("TEST: Sensing SNR Positivity")
    print("=" * 60)

    P_s = 1.0
    noise_power = 1e-10

    gains = []
    for _ in range(10):
        h = generate_sensing_channel(100.0, 100.0, 10.0, K=5.0)
        g = compute_sensing_gain(h)
        gains.append(g)
        snr = compute_sensing_snr(P_s, g, noise_power)
        assert snr > 0, "SNR must be positive"
        assert np.isfinite(snr), "SNR must be finite"

    gains_arr = np.array(gains)
    min_snr = float(np.min(P_s * gains_arr / noise_power))
    max_snr = float(np.max(P_s * gains_arr / noise_power))
    total_snr = compute_sensing_snr(P_s, float(np.sum(gains_arr)), noise_power)
    print(f"  Min per-target SNR: {min_snr:.4f}")
    print(f"  Max per-target SNR: {max_snr:.4f}")
    print(f"  Total SNR (sum): {total_snr:.4f}")
    assert total_snr > min_snr, "Total SNR should exceed per-target"
    print(f"  All SNR positive: PASS\n")
    return True


def test_vehicle_type_impact():
    print("=" * 60)
    print("TEST: Vehicle Type Impact (truck > car > motorcycle)")
    print("=" * 60)

    d_tx = 100.0
    d_rx = 100.0
    K = 5.0

    types = {"truck": 100.0, "car": 10.0, "motorcycle": 3.16}
    mean_gains = {}

    for vtype, rcs in types.items():
        gains = []
        for _ in range(100):
            h = generate_sensing_channel(d_tx, d_rx, rcs, K)
            gains.append(compute_sensing_gain(h))
        mean_gains[vtype] = float(np.mean(gains))

    ratio_truck_car = mean_gains["truck"] / mean_gains["car"]
    ratio_car_moto = mean_gains["car"] / mean_gains["motorcycle"]

    print(f"  Mean gains over 100 trials:")
    for vtype, mg in mean_gains.items():
        print(f"    {vtype}: {mg:.6e}")
    print(f"  truck/car ratio: {ratio_truck_car:.4f}")
    print(f"  car/motorcycle ratio: {ratio_car_moto:.4f}")

    assert mean_gains["truck"] > mean_gains["car"], "truck < car"
    assert mean_gains["car"] > mean_gains["motorcycle"], "car < motorcycle"
    print(f"  truck > car > motorcycle: PASS\n")
    return True


def generate_plots():
    print("=" * 60)
    print("PLOT GENERATION")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10})
    except Exception:
        print("  matplotlib not available, skipping plots")
        return False

    env = BistaticSensingEnvironment(
        BistaticSensingConfig(
            seed=42, num_vehicles=3, vehicle_mobility_mode="straight_road"
        )
    )
    env.reset()

    snr_history = {v.vehicle_id: [] for v in env.vehicles}
    gain_history = {v.vehicle_id: [] for v in env.vehicles}
    dist_history = {v.vehicle_id: [] for v in env.vehicles}
    echo_power_history = {v.vehicle_id: [] for v in env.vehicles}
    time_steps = list(range(100))

    for _ in time_steps:
        env._update_vehicles()
        sensing = env._compute_sensing()
        for t in sensing["per_target"]:
            vid = t["vehicle_id"]
            snr_history[vid].append(t["snr"])
            gain_history[vid].append(t["gain"])
            dist_history[vid].append(t["d_total"])
            echo_power_history[vid].append(float(np.abs(t["echo"]) ** 2))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for vid in snr_history:
        c = colors[vid % len(colors)]
        ax1.semilogy(time_steps, snr_history[vid], "-", color=c,
                     linewidth=1.0, label=f"V{vid}", alpha=0.8)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Sensing SNR (linear)")
    ax1.set_title("Vehicle Sensing SNR over Time")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.15)
    fig1.tight_layout()
    fig1.savefig(str(OUTPUT_DIR / "vehicle_sensing_snr.png"), dpi=150)
    plt.close(fig1)
    print(f"  Saved: vehicle_sensing_snr.png")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for vid in gain_history:
        c = colors[vid % len(colors)]
        ax2.semilogy(time_steps, gain_history[vid], "-", color=c,
                     linewidth=1.0, label=f"V{vid}", alpha=0.8)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("|h_sensing|^2")
    ax2.set_title("Sensing Channel Gain over Time")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.15)
    fig2.tight_layout()
    fig2.savefig(str(OUTPUT_DIR / "sensing_gain_vs_time.png"), dpi=150)
    plt.close(fig2)
    print(f"  Saved: sensing_gain_vs_time.png")

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for vid in dist_history:
        c = colors[vid % len(colors)]
        ax3.plot(time_steps, dist_history[vid], "-", color=c,
                 linewidth=1.0, label=f"V{vid}", alpha=0.8)
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Bistatic Distance (m)")
    ax3.set_title("Bistatic Distance (RIS -> V -> RIS) over Time")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.15)
    fig3.tight_layout()
    fig3.savefig(str(OUTPUT_DIR / "bistatic_distance_vs_time.png"), dpi=150)
    plt.close(fig3)
    print(f"  Saved: bistatic_distance_vs_time.png")

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    for vid in echo_power_history:
        c = colors[vid % len(colors)]
        ax4.semilogy(time_steps, echo_power_history[vid], "-", color=c,
                     linewidth=1.0, label=f"V{vid}", alpha=0.8)
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Echo Power |y|^2")
    ax4.set_title("Per-Vehicle Echo Power over Time")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.15)
    fig4.tight_layout()
    fig4.savefig(str(OUTPUT_DIR / "per_vehicle_echo_power.png"), dpi=150)
    plt.close(fig4)
    print(f"  Saved: per_vehicle_echo_power.png")

    with open(OUTPUT_DIR / "validation_summary.txt", "w") as f:
        f.write("Bistatic Sensing Validation Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Num vehicles: {env.config.num_vehicles}\n")
        f.write(f"Sensing power: {env.config.sensing_power} W\n")
        f.write(f"Noise PSD: {env.config.noise_psd}\n")
        f.write(f"Rician K: {env.config.rician_k}\n\n")
        f.write("Final sensing snapshot:\n")
        s = env._compute_sensing()
        for t in s["per_target"]:
            f.write(
                f"  V{t['vehicle_id']} ({t['vehicle_type']}, "
                f"rcs={t['rcs']:.1f}m^2):\n"
            )
            f.write(f"    d_total={t['d_total']:.1f}m, "
                    f"gain={t['gain']:.4e}, "
                    f"SNR={t['snr']:.4f}\n")
        f.write(f"\nTotal sensing gain: {s['total_sensing_gain']:.6e}\n")
        f.write(f"Total SNR: {s['total_snr']:.4f}\n")
    print(f"  Saved: validation_summary.txt\n")
    return True


def run_validation():
    print("\n" + "=" * 60)
    print("BISTATIC SENSING VALIDATION SUITE")
    print("=" * 60)

    results = {}
    results["bistatic_distance"] = test_bistatic_distance()
    results["channel_dims"] = test_sensing_channel_dimensions()
    results["gain_positivity"] = test_sensing_gain_positivity()
    results["echo_signal"] = test_echo_signal()
    results["multi_target"] = test_multi_target()
    results["snr"] = test_snr_positivity()
    results["vehicle_type"] = test_vehicle_type_impact()

    plots_ok = generate_plots()
    results["plots"] = plots_ok

    print("=" * 60)
    all_ok = all(v for v in results.values())
    if all_ok:
        print("ALL VALIDATION CHECKS PASSED")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"SOME CHECKS FAILED: {failed}")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 60 + "\n")
    return all_ok


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
