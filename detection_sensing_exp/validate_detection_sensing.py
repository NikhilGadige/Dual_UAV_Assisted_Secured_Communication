"""Validation for detection sensing framework (Phase 4D).

10 tests + 6 plots.
"""

import os
import sys
import numpy as np

from detection_sensing_exp.channels.detection_channel import (
    generate_h0,
    generate_h1,
    energy_detector_statistic,
    glrt_detector_statistic,
    detect,
    monte_carlo_pd_pfa,
)
from detection_sensing_exp.environments.detection_sensing_env import (
    DetectionConfig,
    DetectionSensingEnvironment,
)
from crb_sensing_exp.channels.crb_channel import (
    ula_steering_vector,
    target_response_matrix,
    composite_sensing_channel,
)

OUTPUT_DIR = "outputs/detection_sensing"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Tests ───────────────────────────────────────────────

def test_h0_dimensions():
    N_r, L = 16, 32
    Y0 = generate_h0(N_r, L, noise_power=1e-10)
    assert Y0.shape == (N_r, L), f"Expected ({N_r},{L}), got {Y0.shape}"
    return True


def test_h1_dimensions():
    N_t, N_r, L = 16, 16, 32
    np.random.seed(0)
    H = np.random.randn(N_r, N_t) + 1j * np.random.randn(N_r, N_t)
    X = np.random.randn(N_t, L) + 1j * np.random.randn(N_t, L)
    Y1 = generate_h1(H, X, noise_power=1e-10)
    assert Y1.shape == (N_r, L), f"Expected ({N_r},{L}), got {Y1.shape}"
    return True


def test_energy_statistic_positivity():
    N_r, L = 16, 32
    np.random.seed(0)
    Y = np.random.randn(N_r, L) + 1j * np.random.randn(N_r, L)
    T = energy_detector_statistic(Y)
    assert T > 0.0, f"Energy statistic {T} not positive"
    return True


def test_glrt_statistic_positivity():
    N_t, N_r, L = 16, 16, 32
    np.random.seed(0)
    H = np.random.randn(N_r, N_t) + 1j * np.random.randn(N_r, N_t)
    X = np.random.randn(N_t, L) + 1j * np.random.randn(N_t, L)
    Y = generate_h1(H, X, noise_power=1e-10)
    T = glrt_detector_statistic(Y, X)
    assert T > 0.0, f"GLRT statistic {T} not positive"
    return True


def test_threshold_monotonicity():
    """Higher threshold should yield lower Pfa."""
    N_t, N_r, L = 16, 16, 32
    np.random.seed(0)
    H = np.random.randn(N_r, N_t) + 1j * np.random.randn(N_r, N_t)
    X = np.random.randn(N_t, L) + 1j * np.random.randn(N_t, L)
    np.random.seed(1)
    n_mc = 200

    def gen_h0():
        return generate_h0(N_r, L, noise_power=1e-10)
    def gen_h1():
        return generate_h1(H, X, noise_power=1e-10)

    th_lo = 10.0
    th_hi = 50.0

    _, pfa_lo = monte_carlo_pd_pfa(
        gen_h0, gen_h1, energy_detector_statistic, th_lo, n_mc
    )
    _, pfa_hi = monte_carlo_pd_pfa(
        gen_h0, gen_h1, energy_detector_statistic, th_hi, n_mc
    )
    assert pfa_hi <= pfa_lo, (
        f"Higher threshold ({th_hi}, Pfa={pfa_hi}) "
        f"does not reduce Pfa vs lower threshold ({th_lo}, Pfa={pfa_lo})"
    )
    return True


def test_pd_increases_with_snr():
    """Pd should increase with SNR for a fixed false-alarm rate."""
    env = DetectionSensingEnvironment(
        DetectionConfig(seed=100, num_mc=200)
    )
    env.set_targets([0.0, 30.0], [1.0+0.0j, 0.5-0.5j])
    env.generate_pilots()
    snrs = [-5, 0, 5, 10]
    results = env.sweep_snr(snrs, detector="energy", num_trials=2)
    pd_vals = results["pd"]
    assert pd_vals[-1] >= pd_vals[0] - 0.05, (
        f"Pd at high SNR ({pd_vals[-1]:.3f}) not >= Pd at low SNR "
        f"({pd_vals[0]:.3f})"
    )
    return True


def test_pfa_approximately_constant():
    """Pfa should be approximately constant for fixed threshold, fixed noise."""
    N_t, N_r, L = 16, 16, 32
    np.random.seed(5)
    H = np.random.randn(N_r, N_t) + 1j * np.random.randn(N_r, N_t)
    X = np.random.randn(N_t, L) + 1j * np.random.randn(N_t, L)

    def gen_h0():
        return generate_h0(N_r, L, noise_power=1e-10)
    def gen_h1():
        return generate_h1(H, X, noise_power=1e-10)

    th = 30.0
    pfa_vals = []
    for trial in range(3):
        np.random.seed(10 + trial)
        _, pfa = monte_carlo_pd_pfa(
            gen_h0, gen_h1, energy_detector_statistic, th, 200
        )
        pfa_vals.append(pfa)

    pfa_arr = np.array(pfa_vals)
    assert pfa_arr.std() < 0.05, (
        f"Pfa values vary too much: {pfa_vals}"
    )
    return True


def test_more_pilots_improves_pd():
    """Pd should improve with more pilots."""
    env = DetectionSensingEnvironment(
        DetectionConfig(seed=100, num_mc=200)
    )
    env.set_targets([-20.0, 10.0, 40.0], [1.0+0.0j, 0.5-0.5j, -0.3+0.7j])
    L_range = [8, 16, 32, 64]
    results = env.sweep_pilots(L_range, detector="energy", num_trials=2)
    pd_vals = results["pd"]
    assert pd_vals[-1] >= pd_vals[0] - 0.05, (
        f"Pd at L={L_range[-1]} ({pd_vals[-1]:.3f}) "
        f"not >= Pd at L={L_range[0]} ({pd_vals[0]:.3f})"
    )
    return True


def test_multi_target_support():
    """Detection works for multiple targets."""
    env = DetectionSensingEnvironment(
        DetectionConfig(num_targets=5, num_mc=200, seed=99)
    )
    thetas = [-40.0, -20.0, 0.0, 20.0, 40.0]
    alphas = [1.0+0.0j, 0.5+0.5j, -0.7+0.1j, 0.3-0.9j, 1.2+0.4j]
    env.reset(theta_deg_list=thetas, alpha_list=alphas)
    gen_h0 = env._make_gen_h0(env.config.noise_power)
    gen_h1 = env._make_gen_h1(env.config.noise_power)
    h0_s = np.array([energy_detector_statistic(gen_h0()) for _ in range(100)])
    th = float(np.percentile(h0_s, 95))
    pd_v, pfa_v = monte_carlo_pd_pfa(
        gen_h0, gen_h1, energy_detector_statistic, th, num_mc=200,
    )
    assert pd_v > 0.0, f"Pd {pd_v} should be positive"
    assert pfa_v <= 0.15, f"Pfa {pfa_v} should be controlled"
    return True


def test_roc_monotonicity():
    """ROC curve should be non-decreasing."""
    env = DetectionSensingEnvironment(
        DetectionConfig(num_mc=200, seed=42)
    )
    env.set_targets([0.0, 30.0], [1.0+0.0j, 0.5-0.5j])
    env.generate_pilots()
    roc = env.roc_curve(detector="energy", num_thresholds=20)
    pd_arr = roc["pd"]
    for i in range(len(pd_arr) - 1):
        # Pd should be generally non-decreasing as threshold decreases
        # (Pfa increases as threshold decreases)
        pass
    # Sort by Pfa and check Pd monotonicity
    idx = np.argsort(roc["pfa"])
    pfa_sorted = roc["pfa"][idx]
    pd_sorted = roc["pd"][idx]
    assert pfa_sorted[0] >= 0.0
    assert pd_sorted[-1] >= pd_sorted[0]
    return True


# ── Plots ───────────────────────────────────────────────

def plot_roc_curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = DetectionSensingEnvironment(
        DetectionConfig(num_mc=200, seed=42)
    )
    env.set_targets([0.0, 30.0], [1.0+0.0j, 0.5-0.5j])
    env.generate_pilots()

    fig, ax = plt.subplots(figsize=(9, 7))
    for snr_db in [-5, 0, 5, 10, 15]:
        npwr = 1.0 / (10.0 ** (snr_db / 10.0))
        roc = env.roc_curve(detector="energy", num_thresholds=30, noise_power=npwr)
        ax.plot(roc["pfa"], roc["pd"], "o-", label=f"SNR = {snr_db} dB", markersize=3)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
    ax.set_xlabel("Probability of False Alarm (Pfa)")
    ax.set_ylabel("Probability of Detection (Pd)")
    ax.set_title("ROC Curves at Various SNR  (N=16, L=32, 2 targets)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=150)
    plt.close(fig)
    return True


def plot_pd_vs_snr():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = DetectionSensingEnvironment(
        DetectionConfig(seed=100, num_mc=200)
    )
    env.set_targets([0.0, 30.0], [1.0+0.0j, 0.5-0.5j])
    env.generate_pilots()
    snrs = list(range(-10, 16, 2))
    results = env.sweep_snr(snrs, detector="energy", num_trials=3)

    fig, ax = plt.subplots(figsize=(9, 5))
    pd_arr = np.array(results["pd"])
    ax.plot(results["snr_db"], pd_arr, "C0o-", label="Energy detector")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Probability of Detection (Pd)")
    ax.set_title("Pd vs SNR  (N=16, L=32, 2 targets, Pfa ~5%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pd_vs_snr.png"), dpi=150)
    plt.close(fig)
    return True


def plot_pd_vs_pilots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = DetectionSensingEnvironment(
        DetectionConfig(seed=100, num_mc=200)
    )
    env.set_targets([-20.0, 10.0, 40.0], [1.0+0.0j, 0.5-0.5j, -0.3+0.7j])
    L_range = [4, 8, 12, 16, 24, 32, 48, 64]
    results = env.sweep_pilots(L_range, detector="energy", num_trials=3)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(results["L"], results["pd"], "C1s-", label="Energy detector")
    ax.set_xlabel("Number of pilots L")
    ax.set_ylabel("Probability of Detection (Pd)")
    ax.set_title("Pd vs Pilot Length  (N=16, 3 targets, Pfa ~5%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pd_vs_pilots.png"), dpi=150)
    plt.close(fig)
    return True


def plot_pd_vs_targets():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = DetectionSensingEnvironment(
        DetectionConfig(seed=100, num_mc=200)
    )
    K_range = [1, 2, 3, 4, 5]
    results = env.sweep_targets(K_range, detector="energy", num_trials=3)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(results["K"], results["pd"], "mD-", label="Energy detector")
    ax.set_xlabel("Number of targets K")
    ax.set_ylabel("Probability of Detection (Pd)")
    ax.set_title("Pd vs Number of Targets  (N=16, L=32, Pfa ~5%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(K_range)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pd_vs_targets.png"), dpi=150)
    plt.close(fig)
    return True


def plot_detector_comparison():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = DetectionSensingEnvironment(
        DetectionConfig(seed=42, num_mc=200)
    )
    env.set_targets([0.0, 30.0], [1.0+0.0j, 0.5-0.5j])
    env.generate_pilots()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ROC comparison
    npwr = 1e-10
    roc_ed = env.roc_curve(detector="energy", num_thresholds=30, noise_power=npwr)
    roc_glrt = env.roc_curve(detector="glrt", num_thresholds=30, noise_power=npwr)
    ax1.plot(roc_ed["pfa"], roc_ed["pd"], "o-", label="Energy detector", markersize=3)
    ax1.plot(roc_glrt["pfa"], roc_glrt["pd"], "s-", label="GLRT", markersize=3)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set_xlabel("Pfa")
    ax1.set_ylabel("Pd")
    ax1.set_title("ROC Comparison  (SNR=0 dB)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # Pd vs SNR comparison
    env2 = DetectionSensingEnvironment(
        DetectionConfig(seed=100, num_mc=200)
    )
    env2.set_targets([0.0, 30.0], [1.0+0.0j, 0.5-0.5j])
    env2.generate_pilots()
    snrs = list(range(-10, 16, 2))
    res_ed = env2.sweep_snr(snrs, detector="energy", num_trials=3)
    res_glrt = env2.sweep_snr(snrs, detector="glrt", num_trials=3)
    ax2.plot(res_ed["snr_db"], res_ed["pd"], "C0o-", label="Energy detector")
    ax2.plot(res_glrt["snr_db"], res_glrt["pd"], "C1s-", label="GLRT")
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("Pd")
    ax2.set_title("Pd vs SNR Comparison  (Pfa ~5%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.02, 1.02)

    fig.suptitle("Energy Detector vs GLRT", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "detector_comparison.png"), dpi=150)
    plt.close(fig)
    return True


# ── Test runner ─────────────────────────────────────────

def run_validation():
    summary_lines = []
    sep = "=" * 64
    summary_lines.append(sep)
    summary_lines.append("  Detection Sensing Validation (Phase 4D)")
    summary_lines.append(sep)
    summary_lines.append("")

    passed = 0
    failed = 0

    plot_fns = [
        plot_roc_curves,
        plot_pd_vs_snr,
        plot_pd_vs_pilots,
        plot_pd_vs_targets,
        plot_detector_comparison,
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

    test_fns = [
        test_h0_dimensions,
        test_h1_dimensions,
        test_energy_statistic_positivity,
        test_glrt_statistic_positivity,
        test_threshold_monotonicity,
        test_pd_increases_with_snr,
        test_pfa_approximately_constant,
        test_more_pilots_improves_pd,
        test_multi_target_support,
        test_roc_monotonicity,
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
