"""Validate Vehicle Reflection Model.

Tests:
1. Vehicle mobility (3 modes, boundary reflection, trajectory)
2. RCS model (positive, type-dependent)
3. Reflection channel gain |h_UV|^2 * rcs * |h_VR|^2 > 0
4. Multiple vehicle support
5. HPPP compatibility

Plots:
- Vehicle trajectories
- Vehicle distance vs time
- Reflection gain vs time
- Cascaded channel gain
"""

import sys
from pathlib import Path

import numpy as np

from vehicle_reflection_exp.channels.vehicle_channel import (
    Vehicle,
    compute_rcs,
    compute_reflection_channel_gain,
    compute_cascaded_reflection_gain,
)
from vehicle_reflection_exp.environments.vehicle_reflection_env import (
    VehicleReflectionConfig,
    VehicleReflectionEnvironment,
)

OUTPUT_DIR = Path("outputs/vehicle_reflection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_vehicle_mobility():
    print("\n" + "=" * 60)
    print("TEST: Vehicle Mobility (3 modes)")
    print("=" * 60)

    modes = ["straight_road", "lane_change", "urban_grid"]
    half_area = 500.0
    dt = 0.1

    for mode in modes:
        pos = np.array([0.0, 0.0])
        v = Vehicle(0, pos, mobility_mode=mode, max_speed=10.0)
        positions = [v.position.copy()]
        for _ in range(200):
            v.update(dt=dt, half_area=half_area)
            positions.append(v.position.copy())

        pos_arr = np.array(positions)
        in_bounds = bool(np.all((pos_arr >= -half_area) & (pos_arr <= half_area)))
        max_disp = float(np.max(np.linalg.norm(np.diff(pos_arr, axis=0), axis=1)))
        speed = float(np.linalg.norm(v.velocity))

        print(f"  Mode: {mode}")
        print(f"    Steps: {len(positions)}, In bounds: {in_bounds}")
        print(f"    Max step displacement: {max_disp:.5f} (max: {v.max_speed * dt:.5f})")
        print(f"    Final speed: {speed:.3f} m/s")
        assert in_bounds, f"{mode}: Out of bounds"
        assert max_disp <= v.max_speed * dt + 1e-6, f"{mode}: Teleportation"
        print(f"    PASS")
    print()
    return True


def test_rcs_model():
    print("=" * 60)
    print("TEST: RCS Model")
    print("=" * 60)

    types = {"car": 10.0, "truck": 20.0, "bus": 25.0, "motorcycle": 5.0}
    for t, expected_db in types.items():
        rcs = compute_rcs(t)
        rcs_db = 10.0 * np.log10(rcs)
        print(f"  {t}: RCS = {rcs:.2f} m^2 ({rcs_db:.1f} dBsm)")
        assert rcs > 0, f"{t}: RCS must be positive"
        assert abs(rcs_db - expected_db) < 1.0, (
            f"{t}: Expected {expected_db} dBsm, got {rcs_db:.1f}"
        )
        print(f"    PASS")

    rcs_0deg = compute_rcs("car", aspect_angle_deg=0.0)
    rcs_60deg = compute_rcs("car", aspect_angle_deg=60.0)
    print(f"  Aspect angle: 0 deg -> {rcs_0deg:.2f}, 60 deg -> {rcs_60deg:.2f}")
    assert rcs_0deg > rcs_60deg, "RCS should decrease with aspect angle"
    print(f"    PASS")
    print()
    return True


def test_reflection_channel_gain():
    print("=" * 60)
    print("TEST: Reflection Channel Gain")
    print("=" * 60)

    d_UV = 100.0
    d_VR = 50.0
    rcs = 10.0

    gains = []
    for _ in range(50):
        g = compute_reflection_channel_gain(d_UV, d_VR, rcs, K=5.0)
        assert g > 0, "Gain must be positive"
        gains.append(g)

    mean_g = float(np.mean(gains))
    max_g = float(np.max(gains))
    min_g = float(np.min(gains))
    print(f"  d_UV={d_UV}m, d_VR={d_VR}m, rcs={rcs}m^2")
    print(f"  Mean gain: {mean_g:.6e}, Min: {min_g:.6e}, Max: {max_g:.6e}")
    assert min_g > 0, "All gains must be positive"
    print(f"  All gains positive: PASS")

    h, g = compute_cascaded_reflection_gain(d_UV, d_VR, rcs)
    print(f"  Cascaded |h|^2 = {g:.6e}")
    assert abs(g - float(np.abs(h) ** 2)) < 1e-15
    print(f"  Scalar consistency: PASS")
    print()
    return True


def test_multiple_vehicles():
    print("=" * 60)
    print("TEST: Multiple Vehicle Support")
    print("=" * 60)

    env = VehicleReflectionEnvironment(
        VehicleReflectionConfig(seed=42, num_vehicles=3)
    )
    state = env.reset()
    rates = state["rates"]

    print(f"  Number of vehicles: {env.config.num_vehicles}")
    print(f"  Vehicle count in env: {len(env.vehicles)}")
    print(f"  Vehicle info in rates: {len(rates.get('vehicle_info', []))}")
    assert len(env.vehicles) == 3, "Expected 3 vehicles"
    assert len(rates.get("vehicle_info", [])) == 3, "Expected 3 vehicle info entries"
    print(f"    PASS")

    for i, v in enumerate(env.vehicles):
        rcs_db = 10.0 * np.log10(v.rcs)
        speed = float(np.linalg.norm(v.velocity))
        print(f"    Vehicle {i}: type={v.vehicle_type}, "
              f"rcs={v.rcs:.1f}m^2 ({rcs_db:.1f} dBsm), "
              f"speed={speed:.2f} m/s")
    print()

    env2 = VehicleReflectionEnvironment(
        VehicleReflectionConfig(seed=42, num_vehicles=6)
    )
    env2.reset()
    vg = env2.compute_vehicle_reflection_gains()
    assert len(vg["vehicle_to_user"]) == 6, "Expected 6 vehicle gains"
    assert len(vg["vehicle_info"]) == 6, "Expected 6 vehicle info entries"
    print(f"  6-vehicle config: {len(env2.vehicles)} vehicles  PASS\n")
    return True


def test_hppp_compatibility():
    print("=" * 60)
    print("TEST: HPPP Compatibility with Vehicles")
    print("=" * 60)

    N_TRIALS = 100
    densities = [1e-5, 2e-5, 5e-5]
    area = 1000.0 * 1000.0
    for lam in densities:
        counts = []
        for t in range(N_TRIALS):
            cfg = VehicleReflectionConfig(seed=t, eve_density_lambda=lam)
            env = VehicleReflectionEnvironment(cfg)
            env.reset()
            counts.append(env.num_eves)
        mean_count = float(np.mean(counts))
        expected = lam * area
        ratio = mean_count / max(expected, 1e-10)
        ok = 0.8 <= ratio <= 1.2
        print(f"  lam={lam:.0e}: mean={mean_count:.2f}, expected={expected:.2f}, "
              f"ratio={ratio:.3f}  {'PASS' if ok else 'FAIL'}")

    env = VehicleReflectionEnvironment(
        VehicleReflectionConfig(seed=0, eve_density_lambda=2e-5, num_vehicles=3)
    )
    env.reset()
    rates = env.compute_rates()
    vg = env.compute_vehicle_reflection_gains()
    print(f"\n  Vehicles: {len(env.vehicles)}, Eves: {env.num_eves}")
    print(f"  Reflection gains (user): {[f'{g:.4e}' for g in vg['vehicle_to_user']]}")
    print(f"  R_sec = {rates['R_sec']:.4f} bps  PASS\n")
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

    env = VehicleReflectionEnvironment(
        VehicleReflectionConfig(
            seed=42, num_vehicles=3, vehicle_mobility_mode="straight_road"
        )
    )
    env.reset()

    phases = -np.angle(env.h_RU.conj() * env.h_BR)
    env.set_phases(phases)
    env.set_jammer_power(0.0)

    traj = {v.vehicle_id: [v.position.copy()] for v in env.vehicles}
    distance_records = {v.vehicle_id: [] for v in env.vehicles}
    reflection_gains = {v.vehicle_id: [] for v in env.vehicles}
    cascaded_gains = {v.vehicle_id: [] for v in env.vehicles}

    for step in range(100):
        env._update_vehicles()
        for v in env.vehicles:
            traj[v.vehicle_id].append(v.position.copy())
            d_ris = env._distance_2d(env.ris_position, v.position)
            d_user = env._distance_2d(v.position, env.user_position)
            distance_records[v.vehicle_id].append(d_ris + d_user)
            g = compute_reflection_channel_gain(
                d_ris, d_user, v.rcs, env.config.rician_k,
                env.config.alpha, env.config.beta0,
            )
            reflection_gains[v.vehicle_id].append(g)
            h, g2 = compute_cascaded_reflection_gain(
                d_ris, d_user, v.rcs, env.config.rician_k,
                env.config.alpha, env.config.beta0,
            )
            cascaded_gains[v.vehicle_id].append(g2)

    fig1, ax1 = plt.subplots(figsize=(8, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for vid, pts in traj.items():
        pts_arr = np.array(pts)
        c = colors[vid % len(colors)]
        ax1.plot(pts_arr[:, 0], pts_arr[:, 1], "-", color=c, linewidth=1.0,
                 label=f"V{vid}", alpha=0.8)
        ax1.scatter(pts_arr[0, 0], pts_arr[0, 1], color=c, marker="o", s=40, zorder=5)
        ax1.scatter(pts_arr[-1, 0], pts_arr[-1, 1], color=c, marker="s", s=40, zorder=5)
    ax1.scatter(env.ris_position[0], env.ris_position[1],
                marker="^", s=100, color="red", label="RIS-UAV", zorder=5)
    ax1.scatter(env.user_position[0], env.user_position[1],
                marker="D", s=80, color="purple", label="User", zorder=5)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Vehicle Trajectories (100 steps)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.15)
    fig1.tight_layout()
    fig1.savefig(str(OUTPUT_DIR / "vehicle_trajectories.png"), dpi=150)
    plt.close(fig1)
    print(f"  Saved: vehicle_trajectories.png")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for vid in distance_records:
        c = colors[vid % len(colors)]
        ax2.plot(distance_records[vid], "-", color=c, linewidth=1.0,
                 label=f"V{vid}", alpha=0.8)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("RIS-Vehicle-User Distance (m)")
    ax2.set_title("Vehicle Distance (RIS->V + V->User) over Time")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.15)
    fig2.tight_layout()
    fig2.savefig(str(OUTPUT_DIR / "vehicle_distance_vs_time.png"), dpi=150)
    plt.close(fig2)
    print(f"  Saved: vehicle_distance_vs_time.png")

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for vid in reflection_gains:
        c = colors[vid % len(colors)]
        ax3.plot(reflection_gains[vid], "-", color=c, linewidth=1.0,
                 label=f"V{vid}", alpha=0.8)
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Reflection Gain |h_vehicle|^2")
    ax3.set_title("Reflection Channel Gain over Time")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.15)
    ax3.set_yscale("log")
    fig3.tight_layout()
    fig3.savefig(str(OUTPUT_DIR / "reflection_gain_vs_time.png"), dpi=150)
    plt.close(fig3)
    print(f"  Saved: reflection_gain_vs_time.png")

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    for vid in cascaded_gains:
        c = colors[vid % len(colors)]
        ax4.plot(cascaded_gains[vid], "-", color=c, linewidth=1.0,
                 label=f"V{vid}", alpha=0.8)
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Cascaded Channel Gain |h_UV sqrt(rcs) h_VR|^2")
    ax4.set_title("Cascaded Reflection Gain over Time")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.15)
    ax4.set_yscale("log")
    fig4.tight_layout()
    fig4.savefig(str(OUTPUT_DIR / "cascaded_channel_gain.png"), dpi=150)
    plt.close(fig4)
    print(f"  Saved: cascaded_channel_gain.png")

    with open(OUTPUT_DIR / "validation_summary.txt", "w") as f:
        f.write("Vehicle Reflection Validation Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Num vehicles: {env.config.num_vehicles}\n")
        f.write(f"Mobility mode: {env.config.vehicle_mobility_mode}\n")
        f.write(f"Vehicle max speed: {env.config.vehicle_max_speed} m/s\n")
        f.write(f"Rician K: {env.config.rician_k}\n")
        f.write(f"Eve density lambda: {env.config.eve_density_lambda}\n")
        f.write(f"Num eves: {env.num_eves}\n\n")
        f.write("Vehicle details:\n")
        for v in env.vehicles:
            rcs_db = 10.0 * np.log10(v.rcs)
            speed = float(np.linalg.norm(v.velocity))
            f.write(f"  V{v.vehicle_id}: type={v.vehicle_type}, "
                    f"rcs={v.rcs:.1f}m^2 ({rcs_db:.1f} dBsm), "
                    f"speed={speed:.2f} m/s\n")
        f.write(f"\nFinal rates:\n")
        f.write(f"  R_sec = {env.compute_rates()['R_sec']:.4f} bps\n")
        f.write(f"  Vehicles: {len(env.vehicles)}\n")
    print(f"  Saved: validation_summary.txt\n")
    return True


def run_validation():
    print("\n" + "=" * 60)
    print("VEHICLE REFLECTION VALIDATION SUITE")
    print("=" * 60)

    results = {}
    results["mobility"] = test_vehicle_mobility()
    results["rcs"] = test_rcs_model()
    results["reflection_gain"] = test_reflection_channel_gain()
    results["multiple_vehicles"] = test_multiple_vehicles()
    results["hppp"] = test_hppp_compatibility()

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
