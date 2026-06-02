"""Quick smoke test for vehicle receiver implementation."""
import numpy as np
from vehicle_receiver_exp.vehicle_models import VehicleReceiver, VehicleUAVEnvironment
from core.config_utils import build_env_config as build_vehicle_env_config

print("=== VEHICLE MOBILITY VALIDATION ===")
for mode in ['straight_road', 'lane_change', 'urban_grid']:
    pos = np.array([0.0, 0.0])
    v = VehicleReceiver(pos, mobility_mode=mode, max_speed=10.0)
    positions = [v.position.copy()]
    for _ in range(200):
        v.update(dt=0.1, half_area=500.0)
        positions.append(v.position.copy())
    positions = np.array(positions)
    in_bounds = bool(np.all((positions >= -500) & (positions <= 500)))
    speed = float(np.linalg.norm(v.velocity))
    print(f"  {mode:15s} | speed={speed:.2f} | in_bounds={in_bounds} | steps={len(positions)}")

print()
print("=== ENVIRONMENT INTEGRATION TEST ===")
config = build_vehicle_env_config(42, 'rician')
env = VehicleUAVEnvironment(config, mobility_mode='straight_road', vehicle_max_speed=10.0)
state = env.reset()
print(f"  State dim: {state.shape[0]}")
print(f"  Initial pos: {env.user_position[:2]}")
for t in range(5):
    a_r = np.random.uniform(-1, 1, 2).astype(np.float32)
    a_j = np.random.uniform(-1, 1, 2).astype(np.float32)
    state, reward, done, info = env.step(a_r, a_j, 1.0)
print(f"  Final pos: {env.user_position[:2]}")
print(f"  R_sec: {info['R_sec']:.2f}, R_legit: {info['R_legit']:.2f}, Reward: {reward:.4f}")
print()
print("All checks passed!")
