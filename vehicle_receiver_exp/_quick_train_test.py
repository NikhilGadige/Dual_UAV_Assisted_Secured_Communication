"""Quick validation: run one experiment for 5 episodes to verify training infra."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vehicle_receiver_exp.run_vehicle_experiments import train_vehicle_dqn

result = train_vehicle_dqn("rician", "outputs/vehicle_receiver/_test", episodes=5, seed=42)
print(f"\nDQN test result: {result.keys()}")
print(f"CSV exists: {os.path.exists(result['training_log_csv'])}")
print(f"Plots: reward={result.get('reward_curve','')}, secrecy={result.get('secrecy_curve','')}")

# Also test DDPG
from vehicle_receiver_exp.run_vehicle_experiments import train_vehicle_ddpg
result2 = train_vehicle_ddpg("rician", "outputs/vehicle_receiver/_test_ddpg", episodes=5, seed=42)
print(f"\nDDPG test result: CSV exists: {os.path.exists(result2['training_log_csv'])}")
print("Quick training test PASSED")
