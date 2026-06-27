"""Validation for sensing matrix framework (Phase 4B).

8 tests + 5 plots.
"""

import os
import sys
import numpy as np

from sensing_matrix_exp.channels.sensing_matrix_channel import (
    ula_steering_vector,
    target_response_matrix,
    composite_sensing_channel,
    compute_echo_matrix,
    compute_covariance_matrices,
)
from sensing_matrix_exp.environments.sensing_matrix_env import (
    SensingMatrixConfig,
    SensingMatrixEnvironment,
)

OUTPUT_DIR = "outputs/sensing_matrix"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_steering_vector_dimensions():
    N = 16
    theta = 30.0
    a = ula_steering_vector(N, theta, d=0.5, wavelength=1.0)
    assert a.shape == (N, 1), f"Expected ({N},1), got {a.shape}"
    return True


def test_steering_vector_unit_norm():
    N = 16
    theta = 45.0
    a = ula_steering_vector(N, theta, d=0.5, wavelength=1.0)
    norm = float(np.abs(a[0, 0]))
    assert np.isclose(
        norm, 1.0
    ), f"First element magnitude {norm} != 1"
    norm_vec = float(np.linalg.norm(a))
    assert np.isclose(
        norm_vec, np.sqrt(N), atol=1e-10
    ), f"Vector norm {norm_vec} != sqrt({N}) = {np.sqrt(N)}"
    return True


def test_target_response_matrix_rank_one():
    N_tx, N_rx = 16, 16
    theta = 30.0
    a_tx = ula_steering_vector(N_tx, theta)
    a_rx = ula_steering_vector(N_rx, theta)
    A = target_response_matrix(a_rx, a_tx)
    rank = int(np.linalg.matrix_rank(A))
    assert rank == 1, f"Target response matrix rank {rank} != 1"
    assert A.shape == (N_rx, N_tx)
    return True


def test_composite_sensing_matrix_dimensions():
    N_tx, N_rx = 16, 16
    angles = [-30.0, 0.0, 45.0]
    alphas = [1.0 + 0.0j, 0.5 - 0.5j, -0.3 + 0.7j]
    A_list = []
    for th in angles:
        a_tx = ula_steering_vector(N_tx, th)
        a_rx = ula_steering_vector(N_rx, th)
        A_list.append(target_response_matrix(a_rx, a_tx))
    H = composite_sensing_channel(alphas, A_list)
    assert H.shape == (N_rx, N_tx), f"Expected ({N_rx},{N_tx}), got {H.shape}"
    return True


def test_echo_matrix_dimensions():
    N_tx, N_rx = 16, 16
    L = 32
    np.random.seed(0)
    H = np.random.randn(N_rx, N_tx) + 1j * np.random.randn(N_rx, N_tx)
    X = np.random.randn(N_tx, L) + 1j * np.random.randn(N_tx, L)
    Y = compute_echo_matrix(H, X, noise_power=1e-10)
    assert Y.shape == (N_rx, L), f"Expected ({N_rx},{L}), got {Y.shape}"
    return True


def test_covariance_psd():
    N_tx, N_rx = 16, 16
    L = 32
    np.random.seed(1)
    H = np.random.randn(N_rx, N_tx) + 1j * np.random.randn(N_rx, N_tx)
    X = np.random.randn(N_tx, L) + 1j * np.random.randn(N_tx, L)
    Y = compute_echo_matrix(H, X, noise_power=1e-10)
    cov = compute_covariance_matrices(Y, H, X, noise_power=1e-10)
    R_y = cov["R_y"]
    eigvals = np.linalg.eigvalsh(R_y)
    assert np.all(eigvals >= -1e-10), f"Negative eigenvalues: {eigvals[eigvals < 0]}"
    return True


def test_multi_target_support():
    env = SensingMatrixEnvironment(
        SensingMatrixConfig(
            num_targets=4, L_pilot=64, seed=99
        )
    )
    thetas = [-45.0, -15.0, 20.0, 50.0]
    alphas = [1.0 + 0.0j, 0.5 + 0.5j, -0.7 + 0.1j, 0.3 - 0.9j]
    state = env.reset(theta_deg_list=thetas, alpha_list=alphas)
    rank = state["H_sense"]["rank"]
    frob = state["H_sense"]["frobenius_norm"]
    assert len(env.A_list) == 4, f"Expected 4 response matrices, got {len(env.A_list)}"
    assert rank >= 1, f"Rank {rank} < 1"
    assert frob > 0.0, f"Frobenius norm {frob} <= 0"
    assert env.Y.shape == (env.N_rx, env.L)
    return True


def test_frobenius_norm_positivity():
    env = SensingMatrixEnvironment(
        SensingMatrixConfig(num_targets=3, seed=123)
    )
    thetas = [0.0, 30.0, -30.0]
    alphas = [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]
    state = env.reset(theta_deg_list=thetas, alpha_list=alphas)
    frob = state["H_sense"]["frobenius_norm"]
    assert frob > 0.0, f"Frobenius norm {frob} <= 0"
    # With stronger alphas, norm should be larger than weakest case
    weak_alphas = [0.01 + 0.0j, 0.01 + 0.0j, 0.01 + 0.0j]
    env_weak = SensingMatrixEnvironment(
        SensingMatrixConfig(num_targets=3, seed=123)
    )
    sw = env_weak.reset(theta_deg_list=thetas, alpha_list=weak_alphas)
    assert frob > sw["H_sense"]["frobenius_norm"]
    return True


# ── Plotting ─────────────────────────────────────────────
def plot_steering_vector_magnitude():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = 16
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    angles = [0.0, 30.0, -45.0, 60.0]
    for ax, theta in zip(axes.flatten(), angles):
        a = ula_steering_vector(N, theta, d=0.5, wavelength=1.0)
        mag = np.abs(a.flatten())
        phase = np.angle(a.flatten(), deg=True)
        ax.stem(range(N), mag, basefmt=" ", linefmt="C0-", markerfmt="C0o")
        ax.set_title(f"theta = {theta:.0f} deg")
        ax.set_xlabel("Antenna index n")
        ax.set_ylabel("|a_n|")
        ax.set_ylim(0, 1.1)

    fig.suptitle("Steering Vector Magnitude (ULA, N=16, d=0.5 lambda)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "steering_vector_magnitude.png"), dpi=150)
    plt.close(fig)
    return True


def plot_sensing_matrix_heatmap():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N_tx, N_rx = 16, 16
    angles = [-30.0, 0.0, 45.0]
    alphas = [1.0 + 0.0j, 0.5 - 0.5j, -0.3 + 0.7j]
    A_list = []
    for th in angles:
        a_tx = ula_steering_vector(N_tx, th)
        a_rx = ula_steering_vector(N_rx, th)
        A_list.append(target_response_matrix(a_rx, a_tx))
    H = composite_sensing_channel(alphas, A_list)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = ["Real", "Imag", "Magnitude"]
    data = [H.real, H.imag, np.abs(H)]
    for ax, d, lbl in zip(axes, data, labels):
        im = ax.imshow(d, aspect="auto", cmap="viridis")
        ax.set_title(lbl)
        ax.set_xlabel("TX antenna index")
        ax.set_ylabel("RX antenna index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Composite Sensing Channel H_sense  (N_tx=N_rx=16, 3 targets)", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "sensing_matrix_heatmap.png"), dpi=150)
    plt.close(fig)
    return True


def plot_covariance_eigenvalues():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = SensingMatrixEnvironment(
        SensingMatrixConfig(num_targets=4, L_pilot=64, seed=2024)
    )
    thetas = [-45.0, -15.0, 20.0, 50.0]
    alphas = [1.0 + 0.0j, 0.5 + 0.5j, -0.7 + 0.1j, 0.3 - 0.9j]
    env.reset(theta_deg_list=thetas, alpha_list=alphas)
    cov = env.compute_covariances()
    eigvals = cov["eigenvalues"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    idx = np.arange(1, len(eigvals) + 1)
    ax1.semilogy(idx, eigvals, "C0o-", markersize=5)
    ax1.set_xlabel("Eigenvalue index")
    ax1.set_ylabel("Magnitude (log)")
    ax1.set_title("Eigenvalues of R_y (sample covariance)")
    ax1.grid(True, which="both", alpha=0.3)

    cumulative = np.cumsum(eigvals) / np.sum(eigvals)
    ax2.plot(idx, cumulative * 100, "C1s-", markersize=5)
    ax2.set_xlabel("Eigenvalue index")
    ax2.set_ylabel("Cumulative % of total energy")
    ax2.set_title("Cumulative energy fraction")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    fig.suptitle("Covariance Eigenanalysis  (4 targets, 64 pilots)", fontsize=13)
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "covariance_eigenvalues.png"), dpi=150
    )
    plt.close(fig)
    return True


def plot_target_matrix_rank():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    configs = [
        (1, [0.0]),
        (2, [0.0, 30.0]),
        (3, [0.0, 30.0, -45.0]),
        (4, [0.0, 30.0, -45.0, 60.0]),
        (5, [0.0, 30.0, -45.0, 60.0, -20.0]),
    ]
    for idx, (ax, (n_targets, thetas)) in enumerate(zip(axes.flatten(), configs)):
        alphas = [
            complex(np.cos(i), np.sin(i)) for i in range(n_targets)
        ]
        A_list = []
        for th in thetas:
            a_tx = ula_steering_vector(16, th)
            a_rx = ula_steering_vector(16, th)
            A_list.append(target_response_matrix(a_rx, a_tx))
        H = composite_sensing_channel(alphas, A_list)
        rank = int(np.linalg.matrix_rank(H))

        im = ax.imshow(np.abs(H), aspect="auto", cmap="viridis")
        ax.set_title(
            f"K={n_targets} target(s), rank(H_sense)={rank}"
        )
        ax.set_xlabel("TX antenna index")
        ax.set_ylabel("RX antenna index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 6th subplot: rank vs number of targets
    ax_last = axes.flatten()[5]
    n_targets_range = range(1, 7)
    ranks = []
    for k in n_targets_range:
        thetas_k = np.linspace(-60.0, 60.0, k).tolist()
        alphas_k = [
            complex(np.cos(i), np.sin(i)) for i in range(k)
        ]
        A_list_k = []
        for th in thetas_k:
            a_tx_k = ula_steering_vector(16, th)
            a_rx_k = ula_steering_vector(16, th)
            A_list_k.append(target_response_matrix(a_rx_k, a_tx_k))
        H_k = composite_sensing_channel(alphas_k, A_list_k)
        ranks.append(int(np.linalg.matrix_rank(H_k)))
    ax_last.plot(
        list(n_targets_range), ranks, "C2o-", markersize=8
    )
    ax_last.set_xlabel("Number of targets K")
    ax_last.set_ylabel("Rank(H_sense)")
    ax_last.set_title("Rank vs # of targets (ULA, N=16)")
    ax_last.set_xticks(list(n_targets_range))
    ax_last.grid(True, alpha=0.3)

    fig.suptitle(
        "Target Response Matrices: |H_sense| and Rank Analysis", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "target_matrix_rank.png"), dpi=150
    )
    plt.close(fig)
    return True


# ── Test runner ───────────────────────────────────────────
RUN_ALL = os.environ.get("RUN_ALL", "1") == "1"
TEST_PREFIX = "test_"


def run_validation():
    summary_lines = []
    sep = "=" * 64
    summary_lines.append(sep)
    summary_lines.append("  Sensing Matrix Validation (Phase 4B)")
    summary_lines.append(sep)
    summary_lines.append("")

    passed = 0
    failed = 0

    plot_fns = [
        plot_steering_vector_magnitude,
        plot_sensing_matrix_heatmap,
        plot_covariance_eigenvalues,
        plot_target_matrix_rank,
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
        test_steering_vector_dimensions,
        test_steering_vector_unit_norm,
        test_target_response_matrix_rank_one,
        test_composite_sensing_matrix_dimensions,
        test_echo_matrix_dimensions,
        test_covariance_psd,
        test_multi_target_support,
        test_frobenius_norm_positivity,
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
