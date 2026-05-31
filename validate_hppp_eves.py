import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("outputs/hppp_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_REALIZATIONS = 1000

# Default HPPP parameters from Phase 7
EVE_DENSITY_LAMBDA = 2e-5
REGION_XMIN = 0.0
REGION_XMAX = 1000.0
REGION_YMIN = 0.0
REGION_YMAX = 1000.0

AREA = (REGION_XMAX - REGION_XMIN) * (REGION_YMAX - REGION_YMIN)
EXPECTED_MEAN = EVE_DENSITY_LAMBDA * AREA


def generate_hppp_realization() -> np.ndarray:
    n_eve = np.random.poisson(EVE_DENSITY_LAMBDA * AREA)
    if n_eve == 0:
        return np.empty((0, 2), dtype=float)
    xs = np.random.uniform(REGION_XMIN, REGION_XMAX, size=n_eve)
    ys = np.random.uniform(REGION_YMIN, REGION_YMAX, size=n_eve)
    return np.column_stack([xs, ys])


def run_validation():
    print("=" * 60)
    print("HPPP EAVESDROPPER VALIDATION")
    print("=" * 60)
    print(f"  Lambda        : {EVE_DENSITY_LAMBDA}")
    print(f"  Region        : [{REGION_XMIN}, {REGION_XMAX}] x [{REGION_YMIN}, {REGION_YMAX}]")
    print(f"  Area          : {AREA:.0f} m^2")
    print(f"  Expected mean : {EXPECTED_MEAN:.2f} Eves/realization")
    print(f"  Realizations  : {N_REALIZATIONS}")
    print()

    counts = []
    all_positions = []

    for i in range(N_REALIZATIONS):
        pos = generate_hppp_realization()
        counts.append(pos.shape[0])
        if pos.shape[0] > 0:
            all_positions.append(pos)

    counts_arr = np.array(counts)
    observed_mean = float(np.mean(counts_arr))
    observed_var = float(np.var(counts_arr))
    theoretical_var = EXPECTED_MEAN

    print("RESULTS")
    print("-" * 40)
    print(f"  Observed mean Eve count   : {observed_mean:.4f}")
    print(f"  Expected mean             : {EXPECTED_MEAN:.4f}")
    print(f"  Ratio (obs/exp)           : {observed_mean / max(EXPECTED_MEAN, 1e-10):.4f}")
    print(f"  Observed variance         : {observed_var:.4f}")
    print(f"  Theoretical variance      : {theoretical_var:.4f}  (Poisson: mean = var)")
    print(f"  Min Eves observed         : {int(counts_arr.min())}")
    print(f"  Max Eves observed         : {int(counts_arr.max())}")
    print(f"  P(zero Eves)              : {float(np.mean(counts_arr == 0)):.4f}")

    # Verify mean ≈ lambda × area (within 5% due to MC noise)
    relative_error = abs(observed_mean - EXPECTED_MEAN) / max(EXPECTED_MEAN, 1e-10)
    mean_ok = relative_error < 0.05
    print(f"  Mean accuracy within 5%  : {'PASS' if mean_ok else 'FAIL'}  (error={relative_error:.4f})")

    # Verify variance ≈ mean (Poisson property)
    var_ratio = observed_var / max(theoretical_var, 1e-10)
    var_ok = 0.8 < var_ratio < 1.2
    print(f"  Variance ~ mean (Poisson): {'PASS' if var_ok else 'FAIL'}  (ratio={var_ratio:.4f})")

    print()

    # --- Plotting ---
    print("Generating plots...")
    _safe_import_matplotlib()

    _plot_eve_distribution(all_positions)
    _plot_eve_count_histogram(counts_arr)

    print(f"  Plots saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 60)

    return {
        "observed_mean": observed_mean,
        "expected_mean": EXPECTED_MEAN,
        "observed_var": observed_var,
        "theoretical_var": theoretical_var,
        "mean_accuracy_pass": mean_ok,
        "variance_pass": var_ok,
        "total_realizations": N_REALIZATIONS,
    }


def _safe_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10})
        # Force import to register backends
        import matplotlib.figure
        return plt
    except Exception:
        return None


def _plot_eve_distribution(all_positions):
    plt = _safe_import_matplotlib()
    if plt is None or not all_positions:
        return

    stacked = np.concatenate(all_positions, axis=0)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(stacked[:, 0], stacked[:, 1], s=1, alpha=0.3, color="#ff7f0e")
    ax.set_xlim(REGION_XMIN, REGION_XMAX)
    ax.set_ylim(REGION_YMIN, REGION_YMAX)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title(f"HPPP Eve Distribution ({N_REALIZATIONS} realizations)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "eve_distribution.png"), dpi=150)
    plt.close(fig)


def _plot_eve_count_histogram(counts):
    plt = _safe_import_matplotlib()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts, bins=min(50, counts.max() - counts.min() + 1),
            density=True, alpha=0.7, color="#1f77b4", edgecolor="white", linewidth=0.5)
    # Overlay Poisson PMF
    from scipy.stats import poisson
    k = np.arange(0, counts.max() + 1)
    ax.plot(k, poisson.pmf(k, EXPECTED_MEAN), "ro-", markersize=3, label="Poisson PMF (theoretical)")
    ax.axvline(EXPECTED_MEAN, color="green", linestyle="--", label=f"Expected mean ({EXPECTED_MEAN:.1f})")
    ax.axvline(float(np.mean(counts)), color="red", linestyle=":", label=f"Observed mean ({np.mean(counts):.2f})")
    ax.set_xlabel("Number of Eavesdroppers")
    ax.set_ylabel("Probability")
    ax.set_title("Eve Count Distribution (HPPP)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "eve_count_histogram.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_validation()
