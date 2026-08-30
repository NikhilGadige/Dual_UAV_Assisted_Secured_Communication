"""
Regenerates output_final/rolling_secrecy_comparison.png from the already
logged per-episode training history CSVs (no retraining required).

This is the figure referenced by report_week9.tex (Problem Formulation /
Convergence Analysis). It must reflect the joint secrecy + sensing
optimization objective, not raw secrecy rate alone:
    ASSR^(e) = R_sec_total^(e) / max_j R_sec_total^(j)   (secrecy normalized
               by the maximum secrecy achieved anywhere in the run)
    P_d^(e)  = w3 * pd_eaves^(e) + w4 * pd_target^(e)     (combined detection
               probability)
    U^(e)    = lambda1 * ASSR^(e) + lambda2 * P_d^(e)     (plotted, rolling
               100-episode average)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_final")

LAMBDA1_ASSR = 0.5
LAMBDA2_PD = 0.5
W3_PD_EAVES = 0.5
W4_PD_TARGET = 0.5
ROLLING_WINDOW = 100
CRB_THRESHOLD = 1.0e8

algos = {
    "MAPPO (Proposed)": ("mappo_history.csv", "purple"),
    "MATD3PG": ("matd3pg_history.csv", "blue"),
    "SASAC": ("sasac_history.csv", "green"),
    "SATD3PG": ("satd3pg_history.csv", "orange"),
}


def robust_max_reference(values: pd.Series) -> float:
    """Return the maximum non-outlier secrecy value for stable ASSR scaling."""
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0

    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    inliers = arr[arr <= upper_fence]
    ref = float(np.max(inliers)) if inliers.size else float(np.max(arr))
    return ref if ref > 1e-12 else 1.0

plt.figure(figsize=(10, 6))

for name, (filename, color) in algos.items():
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        continue

    df = pd.read_csv(filepath)

    r_ref = robust_max_reference(df["secrecy_total"])
    assr = (df["secrecy_total"] / r_ref).clip(lower=0.0, upper=1.0)

    pd_combined = W3_PD_EAVES * df["pd_eaves"] + W4_PD_TARGET * df["pd_target"]
    crb_feasible = (df["crb_target"] <= CRB_THRESHOLD).astype(float)
    utility = (LAMBDA1_ASSR * assr + LAMBDA2_PD * pd_combined) * crb_feasible

    rolling_utility = utility.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    plt.plot(df["episode"], rolling_utility, label=name, color=color, linewidth=2)

plt.title("Convergence Comparison: Rolling 100 Average Multi-Objective Utility")
plt.xlabel("Episode")
plt.ylabel(r"Rolling Avg Utility (($\lambda_1$ ASSR $+\ \lambda_2\ P_d$) $\times$ CRB-feasible)")
plt.grid(True)
plt.legend()

plot_path = os.path.join(OUTPUT_DIR, "rolling_secrecy_comparison.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Successfully generated combined rolling-average utility plot at: {plot_path}")
