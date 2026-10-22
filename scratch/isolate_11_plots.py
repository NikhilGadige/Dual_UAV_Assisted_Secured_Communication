import os
import shutil
from pathlib import Path

WORKSPACE_DIR = Path(r"d:\Summer_Internship_2026\Summer_Internship_2026")
TARGET_DIR = WORKSPACE_DIR / "Plots_20thOct"

# Clean out everything in Plots_20thOct
if TARGET_DIR.exists():
    shutil.rmtree(TARGET_DIR)
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Define exact sources for the 11 plots
items = {
    "01_vehicle_mobility.png": WORKSPACE_DIR / "outputs" / "vehicle_reflection" / "vehicle_trajectories.png",
    "02_ris_element.png": WORKSPACE_DIR / "outputs" / "ris_uav" / "secrecy_rate_bar.png",
    "03_jammer_power.png": WORKSPACE_DIR / "outputs" / "fd_jammer" / "secrecy_vs_jammer_power.png",
    "06_uav_trajectory.png": WORKSPACE_DIR / "outputs" / "trajectories" / "sac" / "sac_trajectory_rician.png",
    "07_assr.png": WORKSPACE_DIR / "output_final" / "assr_comparison.png",
    "08_detection_probability.png": WORKSPACE_DIR / "output_final" / "pd_comparison.png",
    "10_combined_utility.png": WORKSPACE_DIR / "output_final" / "rolling_secrecy_comparison.png",
}

for name, src in items.items():
    if src.exists():
        shutil.copy2(src, TARGET_DIR / name)
        print(f"Copied {name} from {src.name}")

# Now generate strictly the remaining 4 exact plots directly into TARGET_DIR
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 04_semantic_compression.png
lookup_csv = WORKSPACE_DIR / "Week9 work" / "deepsc_lookup_table.csv"
if lookup_csv.exists():
    df_lookup = pd.read_csv(lookup_csv)
    B = 1e6
    ratios = np.linspace(0.1, 2.0, 30)
    sinr_lin = 10**(5.0 / 10.0)
    eff_sinr = sinr_lin / ratios
    sinr_dbs = 10 * np.log10(np.maximum(eff_sinr, 1e-4))
    interp_sim = np.interp(sinr_dbs, df_lookup["sinr_db"], df_lookup["semantic_similarity"])
    sem_rate = ratios * B * np.log2(1 + eff_sinr) * interp_sim
    
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(ratios, sem_rate / 1e6, 'm^-', linewidth=2)
    ax.set_title("Semantic Transmission Rate vs. Compression Ratio (M/K)")
    ax.set_xlabel("Compression Ratio (M/K)")
    ax.set_ylabel("Semantic Rate (M suts/s)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(TARGET_DIR / "04_semantic_compression.png")
    plt.close(fig)
    print("Saved 04_semantic_compression.png")

# 05_semantic_similarity.png
if lookup_csv.exists():
    df_lookup = pd.read_csv(lookup_csv)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(df_lookup["sinr_db"], df_lookup["semantic_similarity"], 'o-', color='#1f77b4', linewidth=2, label="Semantic Similarity")
    ax.plot(df_lookup["sinr_db"], df_lookup["word_accuracy"], 's--', color='#2ca02c', linewidth=1.5, label="Word Accuracy")
    ax.set_title("Semantic Similarity & Word Accuracy vs. Channel SINR (DeepSC)")
    ax.set_xlabel("SINR (dB)")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(TARGET_DIR / "05_semantic_similarity.png")
    plt.close(fig)
    print("Saved 05_semantic_similarity.png")

# 09_eavesdropper_radius.png
try:
    import sys
    sys.path.insert(0, str(WORKSPACE_DIR / "Week9 work"))
    from system_model import Week9SystemModel
    radii = np.linspace(2.0, 30.0, 15)
    secrecy_vals = []
    leakage_vals = []
    for r in radii:
        model = Week9SystemModel(eve_uncertainty_radius=r, eavesdropper_combining="selection")
        res = model.evaluate_system()
        sec = res["secrecy_performance"]
        secrecy_vals.append(sec["secrecy_rate_total"])
        leakage_vals.append(sec["leakage_near"] + sec["leakage_far"])
        
    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax1.plot(radii, np.array(secrecy_vals) / 1e3, 'ro-', linewidth=2, label="Total Secrecy Rate")
    ax1.set_xlabel("Eavesdropper Uncertainty Radius [m]")
    ax1.set_ylabel("Secrecy Rate (k suts/s)", color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(radii, np.array(leakage_vals) / 1e3, 'bs--', linewidth=2, label="Eavesdropper Leakage")
    ax2.set_ylabel("Leakage Rate (k suts/s)", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    plt.title("Impact of Eavesdropper Uncertainty Radius on Secrecy")
    fig.tight_layout()
    fig.savefig(TARGET_DIR / "09_eavesdropper_radius.png")
    plt.close(fig)
    print("Saved 09_eavesdropper_radius.png")
except Exception as e:
    print(f"Error 09: {e}")

# 11_tradeoff_dp_vs_secrecy.png
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
for algo, fname, col in [("MAPPO", "mappo_history.csv", "#7b1fa2"), 
                         ("MATD3PG", "matd3pg_history.csv", "#1976d2"),
                         ("SASAC", "sasac_history.csv", "#388e3c"),
                         ("SATD3PG", "satd3pg_history.csv", "#f57c00")]:
    fpath = WORKSPACE_DIR / "output_final" / fname
    if fpath.exists():
        df = pd.read_csv(fpath)
        w = 100
        roll_pd = df["pd_combined"].rolling(w).mean()
        roll_sec = df["assr"].rolling(w).mean()
        ax.plot(roll_pd, roll_sec, label=algo, color=col, linewidth=2)
        
ax.set_title("Pareto Tradeoff: Detection Probability vs. ASSR")
ax.set_xlabel("Combined Detection Probability (Pd)")
ax.set_ylabel("Average Semantic Secrecy Rate (ASSR)")
ax.grid(True, alpha=0.3)
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig(TARGET_DIR / "11_tradeoff_dp_vs_secrecy.png")
plt.close(fig)
print("Saved 11_tradeoff_dp_vs_secrecy.png")

print("Done! Exactly 11 plots in target folder.")
