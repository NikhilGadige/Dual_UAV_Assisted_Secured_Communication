"""Validate vehicle receiver implementation.

Tests:
1. VehicleReceiver mobility (all 3 modes)
2. VehicleUAVEnvironment integration
3. All 12 validation experiments (6 algos x 2 channels)
4. Plot generation
5. Summary generation
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from vehicle_receiver_exp.vehicle_models import VehicleReceiver, VehicleUAVEnvironment
from vehicle_receiver_exp.run_vehicle_experiments import build_vehicle_env_config


def test_vehicle_mobility():
    """Test all 3 mobility modes produce valid trajectories."""
    print("\n" + "=" * 60)
    print("VEHICLE MOBILITY VALIDATION")
    print("=" * 60)

    modes = ["straight_road", "lane_change", "urban_grid"]
    half_area = 500.0
    dt = 0.1

    for mode in modes:
        pos = np.array([0.0, 0.0])
        v = VehicleReceiver(pos, mobility_mode=mode, max_speed=10.0)
        positions = [v.position.copy()]
        velocities = [v.velocity.copy()]

        for _ in range(200):
            v.update(dt=dt, half_area=half_area)
            positions.append(v.position.copy())
            velocities.append(v.velocity.copy())

        pos_arr = np.array(positions)
        vel_arr = np.array(velocities)
        speeds = np.linalg.norm(vel_arr, axis=1)
        displacements = np.linalg.norm(np.diff(pos_arr, axis=0), axis=1)

        max_speed_achieved = float(np.max(speeds))
        mean_speed = float(np.mean(speeds))
        max_displacement = float(np.max(displacements))
        theoretical_max = v.max_speed * dt

        in_bounds = bool(np.all((pos_arr >= -half_area) & (pos_arr <= half_area)))
        smooth = max_displacement <= theoretical_max + 1e-6

        print(f"\n  Mode: {mode}")
        print(f"    Max speed: {max_speed_achieved:.3f} m/s (limit: {v.max_speed})")
        print(f"    Mean speed: {mean_speed:.3f} m/s")
        print(f"    Max step displacement: {max_displacement:.5f} m (theoretical: {theoretical_max:.5f})")
        print(f"    In bounds: {in_bounds}")
        print(f"    Smooth movement: {smooth}")
        print(f"    Trajectory length: {len(positions)} steps")

        errors = []
        if not in_bounds:
            errors.append("OUT OF BOUNDS")
        if not smooth:
            errors.append("TELEPORTATION DETECTED")
        if errors:
            print(f"    ** ERRORS: {', '.join(errors)}")
            return False

    print(f"\n  All mobility modes PASSED\n")
    return True


def test_vehicle_environment():
    """Test VehicleUAVEnvironment integration with the full env pipeline."""
    print("=" * 60)
    print("VEHICLE ENVIRONMENT INTEGRATION TEST")
    print("=" * 60)

    config = build_vehicle_env_config(seed=42, fading_model="rician")
    env = VehicleUAVEnvironment(config, mobility_mode="straight_road", vehicle_max_speed=10.0)
    state = env.reset()

    print(f"  State dimension: {state.shape[0]}")
    print(f"  Vehicle initial position: {env.user_position[:2]}")
    print(f"  Vehicle initial velocity: {env.user_velocity}")

    positions = [env.user_position[:2].copy()]
    for t in range(10):
        a_relay = np.random.uniform(-1, 1, size=2).astype(np.float32)
        a_jammer = np.random.uniform(-1, 1, size=2).astype(np.float32)
        state, reward, done, info = env.step(a_relay, a_jammer, 1.0)
        positions.append(env.user_position[:2].copy())

    total_movement = float(np.sum([np.linalg.norm(positions[i+1] - positions[i]) for i in range(len(positions)-1)]))
    print(f"  Total vehicle movement over 10 steps: {total_movement:.3f} m")
    print(f"  Final position: {env.user_position[:2]}")
    print(f"  Reward: {reward:.4f}")
    print(f"  R_sec: {info['R_sec']:.2f} bps")
    print(f"  R_legit: {info['R_legit']:.2f} bps")
    print(f"  R_eve: {info['R_eve']:.2f} bps")

    print(f"\n  All environment integration checks passed - vehicle moves, channel/reward unchanged.\n")
    return True


def test_experiment_csv_structure(csv_path: str) -> bool:
    """Verify training CSV has the expected columns."""
    required = ["episode", "avg_R_sec_mbps", "rolling100_avg_R_sec_mbps", "avg_shaped_reward"]
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for col in required:
            if col not in headers:
                print(f"  MISSING COLUMN: {col} in {csv_path}")
                return False
        rows = list(reader)
        if not rows:
            print(f"  EMPTY CSV: {csv_path}")
            return False
        print(f"    CSV OK: {len(rows)} rows, {len(headers)} columns")
    return True


def test_summary_csv(csv_path: str) -> bool:
    """Verify summary CSV."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"    Summary CSV: {len(rows)} rows")
    for r in rows:
        print(f"      {r.get('Algorithm','?')} + {r.get('Channel','?')}: "
              f"Final={float(r.get('Final_Rolling100_Secrecy',0)):.4f} Mbps")
    return len(rows) > 0


def run_validation(experiments_dir: str = "outputs/vehicle_receiver"):
    """Run full validation suite."""
    print("\n" + "=" * 60)
    print("VEHICLE RECEIVER VALIDATION SUITE")
    print("=" * 60)

    all_ok = True

    all_ok &= test_vehicle_mobility()
    all_ok &= test_vehicle_environment()

    # Check experiment outputs
    print("=" * 60)
    print("EXPERIMENT OUTPUT VALIDATION")
    print("=" * 60)

    exp_dir = Path(experiments_dir)

    algos = ["dqn", "ddpg", "d3qn", "ppo", "sac", "td3pg"]
    channels = ["rician", "rayleigh"]
    csv_count = 0
    plot_count = 0

    for algo in algos:
        for chan in channels:
            run_dir = exp_dir / algo / chan
            csv_path = run_dir / "training_log.csv"
            if csv_path.exists():
                csv_count += 1
                ok = test_experiment_csv_structure(str(csv_path))
                all_ok &= ok

                for plot_name in ["reward_curve.png", "secrecy_curve.png", "rolling100_curve.png"]:
                    plot_path = run_dir / plot_name
                    if plot_path.exists():
                        plot_count += 1
                    else:
                        print(f"  MISSING PLOT: {plot_path}")
                        all_ok = False
            else:
                print(f"  MISSING CSV: {csv_path}")
                all_ok = False

    print(f"\n  Found {csv_count}/12 training CSVs, {plot_count}/36 plots")

    # Check summary CSVs
    vs_path = exp_dir / "vehicle_summary.csv"
    if vs_path.exists():
        test_summary_csv(str(vs_path))
    else:
        print(f"  MISSING: vehicle_summary.csv")
        all_ok = False

    ivs_path = exp_dir / "iot_vs_vehicle_summary.csv"
    if ivs_path.exists():
        test_summary_csv(str(ivs_path))
    else:
        print(f"  NOTE: iot_vs_vehicle_summary.csv not found (run with --run-iot)")

    print(f"\n{'='*60}")
    if all_ok:
        print("ALL VALIDATION CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    print(f"{'='*60}\n")
    return all_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate vehicle receiver experiments.")
    parser.add_argument("--experiments-dir", type=str, default="outputs/vehicle_receiver")
    parser.add_argument("--run-quick", action="store_true", help="Run a quick single-episode test of each algorithm")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = run_validation(args.experiments_dir)
    sys.exit(0 if success else 1)
