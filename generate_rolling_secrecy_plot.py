import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "/Users/divyanshughosh/Documents/Summer_Internship_2026/output_final"

algos = {
    "MAPPO (Proposed)": "mappo_history.csv",
    "MATD3PG": "matd3pg_history.csv",
    "SAC": "sac_history.csv",
    "Single Agent TD3PG": "td3pg_single_history.csv"
}

plt.figure(figsize=(10, 6))

for name, filename in algos.items():
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Secrecy total is in suts/s, convert to M-suts/s
        secrecy = df["secrecy_total"] / 1.0e6
        
        # Calculate rolling 100 average
        rolling_window = 100
        rolling_secrecy = secrecy.rolling(window=rolling_window, min_periods=1).mean()
        
        plt.plot(df["episode"], rolling_secrecy, label=name, linewidth=2)
    else:
        print(f"Warning: {filepath} not found.")

plt.title("Convergence Comparison: Rolling 100 Average Secrecy Rate")
plt.xlabel("Episode")
plt.ylabel("Rolling Average Secrecy Rate (M-suts/s)")
plt.grid(True)
plt.legend()

plot_path = os.path.join(OUTPUT_DIR, "rolling_secrecy_comparison.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Successfully generated combined rolling average secrecy rate plot at: {plot_path}")
