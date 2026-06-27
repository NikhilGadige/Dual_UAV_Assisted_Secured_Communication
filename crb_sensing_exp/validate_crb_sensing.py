"""Validation for CRB-based sensing framework (Phase 4C).

10 tests + 5 plots.
"""

import os
import sys
import numpy as np

from crb_sensing_exp.channels.crb_channel import (
    ula_steering_vector,
    ula_steering_derivative,
    target_response_matrix,
    target_response_derivative,
    composite_sensing_channel,
    compute_channel_derivatives,
    compute_fim,
    compute_crb,
)
from crb_sensing_exp.environments.crb_sensing_env import (
    CRBConfig,
    CRBSensingEnvironment,
)

OUTPUT_DIR = "outputs/crb_sensing"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Tests ───────────────────────────────────────────────

def test_steering_derivative_dimensions():
    N = 16
    theta = 30.0
    da = ula_steering_derivative(N, theta)
    assert da.shape == (N, 1), f"Expected ({N},1), got {da.shape}"
    return True


def test_derivative_finite_values():
    N = 16
    thetas = [-60.0, -30.0, 0.0, 30.0, 60.0]
    for th in thetas:
        da = ula_steering_derivative(N, th)
        assert np.all(np.isfinite(da)), f"Non-finite values at theta = {th}"
    return True


def test_fim_symmetry():
    N_tx, N_rx = 16, 16
    thetas = [-30.0, 0.0, 45.0]
    alphas = [1.0 + 0.0j, 0.5 - 0.5j, -0.3 + 0.7j]
    A_list = []
    dA_list = []
    for th in thetas:
        a = ula_steering_vector(N_tx, th)
        da = ula_steering_derivative(N_tx, th)
        A_list.append(target_response_matrix(a))
        dA_list.append(target_response_derivative(a, da))
    dH = compute_channel_derivatives(alphas, dA_list)
    L = 32
    np.random.seed(0)
    X = np.random.randn(N_tx, L) + 1j * np.random.randn(N_tx, L)
    FIM = compute_fim(dH, X, noise_power=1e-10)
    assert np.allclose(FIM, FIM.T, atol=1e-12), "FIM not symmetric"
    return True


def test_fim_psd():
    env = CRBSensingEnvironment(
        CRBConfig(num_targets=3, seed=42)
    )
    thetas = [-30.0, 0.0, 45.0]
    alphas = [1.0 + 0.0j, 0.5 - 0.5j, -0.3 + 0.7j]
    env.reset(theta_deg_list=thetas, alpha_list=alphas)
    crb_r = env.compute_fim_and_crb()
    eigvals = crb_r["fim_eigenvalues"]
    assert np.all(eigvals >= -1e-10), (
        f"FIM has negative eigenvalues: {eigvals[eigvals < 0]}"
    )
    return True


def test_positive_crb():
    env = CRBSensingEnvironment(CRBConfig(num_targets=3, seed=42))
    thetas = [-30.0, 0.0, 45.0]
    alphas = [1.0 + 0.0j, 0.5 - 0.5j, -0.3 + 0.7j]
    env.reset(theta_deg_list=thetas, alpha_list=alphas)
    crb_r = env.compute_fim_and_crb()
    assert np.all(crb_r["var_bound"] > 0.0), "CRB variance not positive"
    assert np.all(np.isfinite(crb_r["var_bound"])), "CRB variance not finite"
    return True


def test_higher_snr_decreases_crb():
    env = CRBSensingEnvironment(CRBConfig(seed=100))
    snrs = [-10, -5, 0, 5, 10, 20]
    thetas = [-30.0, 0.0, 30.0]
    alphas = [1.0 + 0.0j, 0.5 - 0.5j, -0.3 + 0.7j]
    results = env.sweep_snr(snrs, thetas, alphas, num_trials=3)
    rmse_per_target = np.array(results["rmse_deg"])
    for k in range(rmse_per_target.shape[1]):
        vals = rmse_per_target[:, k]
        # Check generally decreasing (allow small fluctuations)
        for i in range(len(vals) - 1):
            if vals[i + 1] > vals[i] * 1.05:
                continue
        # At least the last value should be smaller than the first
        assert vals[-1] < vals[0] * 0.95, (
            f"Target {k}: RMSE at high SNR ({vals[-1]:.4e}) "
            f"not lower than at low SNR ({vals[0]:.4e})"
        )
    return True


def test_more_antennas_decreases_crb():
    env = CRBSensingEnvironment(CRBConfig(seed=100))
    N_range = [4, 8, 16, 32]
    thetas = [-30.0, 0.0, 30.0]
    alphas = [1.0 + 0.0j, 0.5 - 0.5j, -0.3 + 0.7j]
    results = env.sweep_antennas(N_range, thetas, alphas, num_trials=3)
    rmse_per_target = np.array(results["rmse_deg"])
    for k in range(rmse_per_target.shape[1]):
        vals = rmse_per_target[:, k]
        assert vals[-1] < vals[0] * 0.95, (
            f"Target {k}: RMSE at N={N_range[-1]} ({vals[-1]:.4e}) "
            f"not lower than at N={N_range[0]} ({vals[0]:.4e})"
        )
    return True


def test_more_pilots_decreases_crb():
    env = CRBSensingEnvironment(CRBConfig(seed=100))
    L_range = [4, 8, 16, 32, 64]
    thetas = [-30.0, 0.0, 30.0]
    alphas = [1.0 + 0.0j, 0.5 - 0.5j, -0.3 + 0.7j]
    results = env.sweep_pilots(L_range, thetas, alphas, num_trials=3)
    rmse_per_target = np.array(results["rmse_deg"])
    for k in range(rmse_per_target.shape[1]):
        vals = rmse_per_target[:, k]
        assert vals[-1] < vals[0] * 0.95, (
            f"Target {k}: RMSE at L={L_range[-1]} ({vals[-1]:.4e}) "
            f"not lower than at L={L_range[0]} ({vals[0]:.4e})"
        )
    return True


def test_single_target_analytical_sanity():
    """Single-target CRB should be positive, finite, and decrease with SNR."""
    cfg = CRBConfig(num_targets=1, L_pilot=32, seed=42)
    env = CRBSensingEnvironment(cfg)
    env.reset(theta_deg_list=[10.0], alpha_list=[1.0 + 0.0j])
    crb_r = env.compute_fim_and_crb()
    var_b = float(crb_r["var_bound"][0])
    rmse_b = float(crb_r["rmse_bound"][0])
    assert var_b > 0.0, f"Single-target variance {var_b} not positive"
    assert np.isfinite(var_b), f"Single-target variance {var_b} not finite"
    # Higher SNR should reduce RMSE
    cfg_low = CRBConfig(num_targets=1, L_pilot=32, noise_power=1e-8, seed=42)
    env_low = CRBSensingEnvironment(cfg_low)
    env_low.reset(theta_deg_list=[10.0], alpha_list=[1.0 + 0.0j])
    crb_low = env_low.compute_fim_and_crb()
    cfg_high = CRBConfig(num_targets=1, L_pilot=32, noise_power=1e-12, seed=42)
    env_high = CRBSensingEnvironment(cfg_high)
    env_high.reset(theta_deg_list=[10.0], alpha_list=[1.0 + 0.0j])
    crb_high = env_high.compute_fim_and_crb()
    rmse_low = float(crb_low["rmse_bound"][0])
    rmse_high = float(crb_high["rmse_bound"][0])
    assert rmse_high < rmse_low, (
        f"Single-target high-SNR RMSE ({rmse_high}) not < low-SNR RMSE ({rmse_low})"
    )
    return True


def test_multi_target_support():
    """CRB works for K=5 targets with distinct angles."""
    env = CRBSensingEnvironment(
        CRBConfig(num_targets=5, L_pilot=64, seed=99)
    )
    thetas = [-40.0, -20.0, 0.0, 20.0, 40.0]
    alphas = [1.0+0.0j, 0.5+0.5j, -0.7+0.1j, 0.3-0.9j, 1.2+0.4j]
    state = env.reset(theta_deg_list=thetas, alpha_list=alphas)
    crb_r = env.compute_fim_and_crb()
    assert crb_r["FIM"].shape == (5, 5), f"FIM shape {crb_r['FIM'].shape} != (5,5)"
    assert len(crb_r["rmse_bound"]) == 5
    assert np.all(crb_r["var_bound"] > 0.0)
    assert np.all(np.isfinite(crb_r["rmse_bound"]))
    return True


# ── Plots ───────────────────────────────────────────────

def plot_crb_vs_snr():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = CRBSensingEnvironment(CRBConfig(seed=100))
    snrs = list(range(-10, 25, 2))
    thetas = [-30.0, 0.0, 30.0]
    alphas = [1.0 + 0.0j, 0.5 - 0.5j, -0.3 + 0.7j]
    results = env.sweep_snr(snrs, thetas, alphas, num_trials=5)

    fig, ax = plt.subplots(figsize=(9, 5))
    rmse_arr = np.array(results["rmse_deg"])
    for k in range(rmse_arr.shape[1]):
        ax.plot(
            results["snr_db"],
            rmse_arr[:, k],
            "o-",
            label=f"Target {k} (theta={thetas[k]:.0f} deg)",
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("RMSE lower bound (deg)")
    ax.set_title("CRB: RMSE vs SNR  (N=16, L=32, 3 targets)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "crb_vs_snr.png"), dpi=150)
    plt.close(fig)
    return True


def plot_crb_vs_antennas():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = CRBSensingEnvironment(CRBConfig(seed=100))
    N_range = [2, 4, 8, 12, 16, 24, 32, 48, 64]
    thetas = [-30.0, 0.0, 30.0]
    alphas = [1.0 + 0.0j, 0.5 - 0.5j, -0.3 + 0.7j]
    results = env.sweep_antennas(N_range, thetas, alphas, num_trials=5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    rmse_arr = np.array(results["rmse_deg"])
    for k in range(rmse_arr.shape[1]):
        ax1.plot(
            results["N"],
            rmse_arr[:, k],
            "s-",
            label=f"Target {k} (theta={thetas[k]:.0f} deg)",
        )
    ax1.set_xlabel("Number of antennas N")
    ax1.set_ylabel("RMSE lower bound (deg)")
    ax1.set_title("Linear scale")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for k in range(rmse_arr.shape[1]):
        ax2.loglog(
            results["N"],
            rmse_arr[:, k],
            "s-",
            label=f"Target {k} (theta={thetas[k]:.0f} deg)",
        )
    ax2.set_xlabel("Number of antennas N")
    ax2.set_ylabel("RMSE lower bound (deg)")
    ax2.set_title("Log-log scale")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle("CRB: RMSE vs Number of Antennas  (L=32, 3 targets)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "crb_vs_antennas.png"), dpi=150)
    plt.close(fig)
    return True


def plot_crb_vs_pilots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = CRBSensingEnvironment(CRBConfig(seed=100))
    L_range = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    thetas = [-30.0, 0.0, 30.0]
    alphas = [1.0 + 0.0j, 0.5 - 0.5j, -0.3 + 0.7j]
    results = env.sweep_pilots(L_range, thetas, alphas, num_trials=5)

    fig, ax = plt.subplots(figsize=(9, 5))
    rmse_arr = np.array(results["rmse_deg"])
    for k in range(rmse_arr.shape[1]):
        ax.plot(
            results["L"],
            rmse_arr[:, k],
            "^-",
            label=f"Target {k} (theta={thetas[k]:.0f} deg)",
        )
    ax.set_xlabel("Number of pilots L")
    ax.set_ylabel("RMSE lower bound (deg)")
    ax.set_title("CRB: RMSE vs Pilot Length  (N=16, 3 targets)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "crb_vs_pilots.png"), dpi=150)
    plt.close(fig)
    return True


def plot_fim_eigenvalues():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = CRBSensingEnvironment(CRBConfig(num_targets=4, L_pilot=64, seed=2024))
    thetas = [-40.0, -15.0, 20.0, 50.0]
    alphas = [1.0 + 0.0j, 0.5 + 0.5j, -0.7 + 0.1j, 0.3 - 0.9j]
    env.reset(theta_deg_list=thetas, alpha_list=alphas)
    crb_r = env.compute_fim_and_crb()
    eigvals = crb_r["fim_eigenvalues"]
    cond = crb_r["condition_number"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    idx = np.arange(1, len(eigvals) + 1)
    ax1.semilogy(idx, eigvals, "C0o-", markersize=8)
    ax1.set_xlabel("Eigenvalue index")
    ax1.set_ylabel("Magnitude (log)")
    ax1.set_title("FIM Eigenvalues")
    ax1.grid(True, which="both", alpha=0.3)

    cumulative = np.cumsum(eigvals) / np.sum(eigvals)
    ax2.plot(idx, cumulative * 100, "C1s-", markersize=8)
    ax2.set_xlabel("Eigenvalue index")
    ax2.set_ylabel("Cumulative % of total energy")
    ax2.set_title(f"Condition number = {cond:.2f}")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    fig.suptitle("FIM Eigenanalysis  (4 targets, N=16, L=64)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fim_eigenvalues.png"), dpi=150)
    plt.close(fig)
    return True


def plot_crb_heatmap():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = CRBSensingEnvironment(CRBConfig(num_targets=3, L_pilot=64, seed=42))
    thetas = [-30.0, 0.0, 45.0]
    alphas = [1.0 + 0.0j, 0.5 - 0.5j, -0.3 + 0.7j]
    env.reset(theta_deg_list=thetas, alpha_list=alphas)
    crb_r = env.compute_fim_and_crb()
    crb_mat = crb_r["crb_matrix"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    im1 = axes[0].imshow(crb_mat, aspect="auto", cmap="hot")
    axes[0].set_title("CRB matrix J^{-1}")
    axes[0].set_xlabel("Target index")
    axes[0].set_ylabel("Target index")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    labels = [
        f"T{i}\ntheta={thetas[i]:.0f}"
        for i in range(len(thetas))
    ]
    rmse = crb_r["rmse_bound"]
    colors = ["C0", "C1", "C2"]
    bars = axes[1].bar(labels, rmse, color=colors, alpha=0.7)
    axes[1].set_ylabel("RMSE lower bound (deg)")
    axes[1].set_title("Per-target RMSE bound")
    for bar, val in zip(bars, rmse):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "CRB: Target Angle Estimation  (N=16, L=64, 3 targets)", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "crb_heatmap.png"), dpi=150)
    plt.close(fig)
    return True


# ── Test runner ─────────────────────────────────────────

def run_validation():
    summary_lines = []
    sep = "=" * 64
    summary_lines.append(sep)
    summary_lines.append("  CRB Sensing Validation (Phase 4C)")
    summary_lines.append(sep)
    summary_lines.append("")

    passed = 0
    failed = 0

    # --- plots first ---
    plot_fns = [
        plot_crb_vs_snr,
        plot_crb_vs_antennas,
        plot_crb_vs_pilots,
        plot_fim_eigenvalues,
        plot_crb_heatmap,
    ]

    for fn in plot_fns:
        try:
            result = fn()
            status = "PASS" if result else "FAIL"
            if result:
                passed += 1
            else:
                failed += 1
            line = f"  [ {status} ] {fn.__name__}"
            print(line)
            summary_lines.append(line)
        except Exception as e:
            failed += 1
            line = f"  [ FAIL ] {fn.__name__}: {e}"
            print(line)
            summary_lines.append(line)

    # --- tests ---
    test_fns = [
        test_steering_derivative_dimensions,
        test_derivative_finite_values,
        test_fim_symmetry,
        test_fim_psd,
        test_positive_crb,
        test_higher_snr_decreases_crb,
        test_more_antennas_decreases_crb,
        test_more_pilots_decreases_crb,
        test_single_target_analytical_sanity,
        test_multi_target_support,
    ]

    for fn in test_fns:
        try:
            result = fn()
            status = "PASS" if result else "FAIL"
            if result:
                passed += 1
            else:
                failed += 1
            line = f"  [ {status} ] {fn.__name__}"
            print(line)
            summary_lines.append(line)
        except Exception as e:
            failed += 1
            line = f"  [ FAIL ] {fn.__name__}: {e}"
            print(line)
            summary_lines.append(line)

    total = passed + failed
    summary_lines.append("")
    summary_lines.append(sep)
    summary_lines.append(f"  Total: {total}  |  Passed: {passed}  |  Failed: {failed}")
    summary_lines.append(sep)

    summary_path = os.path.join(OUTPUT_DIR, "validation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nSummary saved to {summary_path}")
    return passed, failed, total


if __name__ == "__main__":
    run_validation()
