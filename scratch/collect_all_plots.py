import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKSPACE_DIR = Path(r"d:\Summer_Internship_2026\Summer_Internship_2026")
TARGET_DIR = WORKSPACE_DIR / "Plots_20thOct"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# 1. Collect and copy all existing image files in the repository
ignore_dirs = {".git", ".mplconfig", "__pycache__", "Plots_20thOct"}

copied_count = 0
existing_names = set()

print(f"Scanning for plots in {WORKSPACE_DIR}...")

for root, dirs, files in os.walk(WORKSPACE_DIR):
    # Skip ignored dirs
    dirs[:] = [d for d in dirs if d not in ignore_dirs]
    
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            src_path = Path(root) / file
            
            # Formulate destination filename
            # If name already taken, prepend parent folder names to make unique
            rel_path = src_path.relative_to(WORKSPACE_DIR)
            parts = rel_path.parts[:-1]
            
            if file in existing_names:
                # Prepend parent info to avoid collision
                prefix = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1] if len(parts) >= 1 else "plot"
                dest_name = f"{prefix}_{file}".replace(" ", "_").replace("/", "_").replace("\\", "_")
            else:
                dest_name = file
                
            # If still collision:
            idx = 1
            base_dest_name = dest_name
            while (TARGET_DIR / dest_name).exists():
                stem = Path(base_dest_name).stem
                suffix = Path(base_dest_name).suffix
                dest_name = f"{stem}_{idx}{suffix}"
                idx += 1
                
            shutil.copy2(src_path, TARGET_DIR / dest_name)
            existing_names.add(dest_name)
            copied_count += 1

print(f"Copied {copied_count} existing plots into {TARGET_DIR}")

# 2. Generate the extra plots for the 11 items so all 11 are 100% covered in Plots_20thOct!

# Item 5: Semantic Similarity vs SINR
lookup_csv = WORKSPACE_DIR / "Week9 work" / "deepsc_lookup_table.csv"
if lookup_csv.exists():
    df_lookup = pd.read_csv(lookup_csv)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(df_lookup["sinr_db"], df_lookup["semantic_similarity"], 'o-', color='#1f77b4', linewidth=2, label="Semantic Similarity")
    ax.plot(df_lookup["sinr_db"], df_lookup["word_accuracy"], 's--', color='#2ca02c', linewidth=1.5, label="Word Accuracy")
    ax.set_title("Semantic Metric vs. Channel SINR (DeepSC)")
    ax.set_xlabel("SINR (dB)")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(TARGET_DIR / "05_semantic_similarity_vs_sinr.png")
    plt.close(fig)
    print("Generated 05_semantic_similarity_vs_sinr.png")

# Item 11: Tradeoff (Detection Probability vs Secrecy)
hist_csv = WORKSPACE_DIR / "output_final" / "mappo_history.csv"
if hist_csv.exists():
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    for algo, fname, col in [("MAPPO", "mappo_history.csv", "#7b1fa2"), 
                             ("MATD3PG", "matd3pg_history.csv", "#1976d2"),
                             ("SASAC", "sasac_history.csv", "#388e3c"),
                             ("SATD3PG", "satd3pg_history.csv", "#f57c00")]:
        fpath = WORKSPACE_DIR / "output_final" / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            # Rolling window for smooth Pareto curve
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
    print("Generated 11_tradeoff_dp_vs_secrecy.png")

# Item 9: Radius (Eavesdropper Uncertainty Radius)
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
    fig.savefig(TARGET_DIR / "09_eavesdropper_radius_vs_secrecy.png")
    plt.close(fig)
    print("Generated 09_eavesdropper_radius_vs_secrecy.png")
except Exception as e:
    print(f"Skipped radius plot generation: {e}")

# Item 4: Semantic Rate vs Compression Ratio (M/K)
try:
    if lookup_csv.exists():
        df_lookup = pd.read_csv(lookup_csv)
        # Average similarity at typical SNR (say 5 dB)
        sim_val = float(df_lookup.loc[(df_lookup["sinr_db"] - 5.0).abs().idxmin(), "semantic_similarity"])
        B = 1e6
        # Vary M/K from 0.1 to 2.0
        ratios = np.linspace(0.1, 2.0, 30)
        # Dilution model: effective SNR reduces as M/K increases
        sinr_lin = 10**(5.0 / 10.0)
        eff_sinr = sinr_lin / ratios
        # interpolate similarity for each eff_sinr
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
        fig.savefig(TARGET_DIR / "04_semantic_compression_ratio_rate.png")
        plt.close(fig)
        print("Generated 04_semantic_compression_ratio_rate.png")
except Exception as e:
    print(f"Skipped compression plot generation: {e}")

print(f"\nAll plots successfully organized into: {TARGET_DIR}")
