"""Quick validation: test D3QN, PPO, SAC, TD3PG with 3 episodes each."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vehicle_receiver_exp.run_vehicle_experiments import (
    train_vehicle_d3qn, train_vehicle_ppo, train_vehicle_sac, train_vehicle_td3pg
)

for name, fn in [("D3QN", train_vehicle_d3qn), ("PPO", train_vehicle_ppo),
                  ("SAC", train_vehicle_sac), ("TD3PG", train_vehicle_td3pg)]:
    print(f"\n--- Testing {name} ---")
    result = fn("rician", f"outputs/vehicle_receiver/_test_{name.lower()}", episodes=3, seed=42)
    assert os.path.exists(result['training_log_csv']), f"{name} CSV missing"
    print(f"  {name}: CSV OK, final_roll100={result['final_roll100']:.3f}")

print("\nAll training functions validated!")
