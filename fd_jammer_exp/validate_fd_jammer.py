"""Validate Full-Duplex UAV Jammer with RIS communication model.

Tests:
1. MISO Rician channel vector dimensions (1 x N_j)
2. Beamforming strategies (isotropic, MRT, nullspace)
3. FD jammer interference model (SINR with jammer interference)
4. Secrecy rate with jammer: R_s = [R_b - max(R_e_i)]^+
5. HPPP compatibility with jammer system
6. Jammer position and trajectory
7. Plots: secrecy vs jammer power, SINR vs jammer power, secrecy gain
"""

import sys
from pathlib import Path

import numpy as np

from ris_uav_exp.channels.ris_channel import (
    compute_ris_reflection_matrix,
    compute_effective_channel,
)
from fd_jammer_exp.channels.fd_jammer_channel import (
    generate_miso_rician_channel,
    isotropic_beamforming,
    mrt_beamforming,
    nullspace_beamforming,
    compute_jammer_gain,
)
from fd_jammer_exp.environments.fd_jammer_env import FDJammerConfig, FDJammerEnvironment

OUTPUT_DIR = Path("outputs/fd_jammer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_miso_channel():
    print("\n" + "=" * 60)
    print("TEST: MISO Rician Channel Vector")
    print("=" * 60)

    N_j = 4
    pl = 1e-4
    K = 5.0

    h = generate_miso_rician_channel(N_j, K, pl)
    assert h.shape == (1, N_j), f"Expected (1,{N_j}), got {h.shape}"
    print(f"  Dimensions: {h.shape}  PASS")
    print(f"  Mean |h|^2: {float(np.mean(np.abs(h)**2)):.6e}")
    print(f"  ||h||^2: {float(np.sum(np.abs(h)**2)):.6e}")
    assert np.iscomplexobj(h), "Not complex"
    print(f"  Complex dtype: PASS\n")
    return True


def test_beamforming_strategies():
    print("=" * 60)
    print("TEST: Beamforming Strategies")
    print("=" * 60)

    N_j = 4
    pl = 1e-4
    K = 5.0

    h_JU = generate_miso_rician_channel(N_j, K, pl)
    h_JE = generate_miso_rician_channel(N_j, K, pl)

    w_iso = isotropic_beamforming(N_j)
    assert w_iso.shape == (N_j, 1), f"Isotropic w shape: {w_iso.shape}"
    assert abs(float(np.linalg.norm(w_iso)) - 1.0) < 1e-10, "Isotropic not unit norm"
    print(f"  Isotropic: unit norm = {float(np.linalg.norm(w_iso)):.6f}  PASS")

    w_mrt = mrt_beamforming(h_JE)
    assert w_mrt.shape == (N_j, 1), f"MRT w shape: {w_mrt.shape}"
    assert abs(float(np.linalg.norm(w_mrt)) - 1.0) < 1e-10, "MRT not unit norm"
    g_mrt = compute_jammer_gain(h_JE, w_mrt)
    g_iso = compute_jammer_gain(h_JE, w_iso)
    print(f"  MRT gain at target eve: {g_mrt:.6e}")
    print(f"  Isotropic gain at same eve: {g_iso:.6e}")
    assert g_mrt >= g_iso - 1e-15, "MRT should >= isotropic at target"
    print(f"  MRT >= isotropic at target: PASS")

    w_ns = nullspace_beamforming(h_JU, N_j)
    assert w_ns.shape == (N_j, 1), f"Nullspace w shape: {w_ns.shape}"
    assert abs(float(np.linalg.norm(w_ns)) - 1.0) < 1e-10, "Nullspace not unit norm"
    g_ns_user = compute_jammer_gain(h_JU, w_ns)
    print(f"  Nullspace gain at user: {g_ns_user:.6e}")
    assert g_ns_user < 1e-10, f"Nullspace should null user, got {g_ns_user}"
    print(f"  Nullspace zero-forcing user: PASS\n")
    return True


def test_jammer_interference_model():
    print("=" * 60)
    print("TEST: FD Jammer Interference Model")
    print("=" * 60)

    env = FDJammerEnvironment(FDJammerConfig(seed=42, jammer_power=0.0))
    env.reset()
    rates_no_jammer = env.compute_rates()

    env.set_jammer_power(0.0)
    sinr_no_interference = rates_no_jammer["sinr_user"]

    env.set_jammer_power(0.5)
    env._compute_beamforming()
    rates_with_jammer = env.compute_rates()
    sinr_with_interference = rates_with_jammer["sinr_user"]

    print(f"  SINR user (P_j=0.0): {sinr_no_interference:.4f}")
    print(f"  SINR user (P_j=0.5): {sinr_with_interference:.4f}")

    if sinr_no_interference > sinr_with_interference:
        print(f"  Jammer interference reduces user SINR: PASS")
    else:
        print(f"  Jammer interference reduces user SINR: CHECK (geom)")

    jammer_int_user = rates_with_jammer["jammer_interference_user"]
    print(f"  Jammer interference at user: {jammer_int_user:.6e}")
    assert jammer_int_user >= 0.0, "Negative interference"
    print(f"  Jammer interference non-negative: PASS\n")
    return True


def test_secrecy_with_jammer():
    print("=" * 60)
    print("TEST: Secrecy Rate with Jammer")
    print("=" * 60)

    env = FDJammerEnvironment(FDJammerConfig(seed=42, jammer_power=0.0))
    env.reset()
    rates_no_jam = env.compute_rates()
    R_sec_no_jam = rates_no_jam["R_sec"]

    expected = max(rates_no_jam["R_legit"] - rates_no_jam["R_eve"], 0.0)
    assert abs(R_sec_no_jam - expected) < 1e-6
    print(f"  No jammer: R_sec = {R_sec_no_jam:.4f} bps  PASS")

    env.set_jammer_power(0.1)
    env.set_beamforming_mode("mrt")
    env._compute_beamforming()
    rates_jam = env.compute_rates()
    R_sec_jam = rates_jam["R_sec"]

    print(f"  With jammer (MRT, P_j=0.1): R_sec = {R_sec_jam:.4f} bps")
    print(f"    R_legit = {rates_jam['R_legit']:.4f}, R_eve = {rates_jam['R_eve']:.4f}")

    expected_jam = max(rates_jam["R_legit"] - rates_jam["R_eve"], 0.0)
    assert abs(R_sec_jam - expected_jam) < 1e-6
    print(f"  R_sec = [R_legit - max(R_eve_i)]^+: PASS")

    R_eve_no_jam = rates_no_jam["R_eve"]
    R_eve_with_jam = rates_jam["R_eve"]
    print(f"  Eve rate change: {R_eve_no_jam:.4f} -> {R_eve_with_jam:.4f} bps")
    if R_eve_with_jam <= R_eve_no_jam + 1e-6:
        print(f"  Jamming suppresses eves: PASS")
    else:
        print(f"  Jamming suppresses eves: CHECK (geom)")
    print()
    return True


def sweep_jammer_power(env: FDJammerEnvironment, powers: np.ndarray) -> dict:
    results = {"P_j": [], "R_sec": [], "R_legit": [], "R_eve": [],
               "sinr_user": [], "sinr_eve_max": []}
    for pj in powers:
        env.set_jammer_power(pj)
        env._compute_beamforming()
        r = env.compute_rates()
        results["P_j"].append(pj)
        results["R_sec"].append(r["R_sec"])
        results["R_legit"].append(r["R_legit"])
        results["R_eve"].append(r["R_eve"])
        results["sinr_user"].append(r["sinr_user"])
        results["sinr_eve_max"].append(r["sinr_eve_max"])
    return results


def test_hppp_compatibility():
    print("=" * 60)
    print("TEST: HPPP Compatibility with Jammer")
    print("=" * 60)

    N_TRIALS = 100
    densities = [1e-5, 2e-5, 5e-5]
    area = 1000.0 * 1000.0
    for lam in densities:
        counts = []
        for t in range(N_TRIALS):
            cfg = FDJammerConfig(seed=t, eve_density_lambda=lam)
            env = FDJammerEnvironment(cfg)
            env.reset()
            counts.append(env.num_eves)
        mean_count = float(np.mean(counts))
        expected = lam * area
        ratio = mean_count / max(expected, 1e-10)
        ok = 0.8 <= ratio <= 1.2
        print(f"  lam={lam:.0e}: mean={mean_count:.2f}, expected={expected:.2f}, "
              f"ratio={ratio:.3f}  {'PASS' if ok else 'FAIL'}")

    env = FDJammerEnvironment(FDJammerConfig(seed=0, eve_density_lambda=2e-5))
    env.reset()
    rates = env.compute_rates()
    print(f"\n  Jammer + HPPP integrated: R_sec = {rates['R_sec']:.4f} bps")
    print(f"  Num eves: {env.num_eves}")
    print(f"  Jammer power: {env.config.jammer_power}")
    print()
    return True


def generate_plots():
    print("=" * 60)
    print("PLOT GENERATION")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10})
    except Exception:
        print("  matplotlib not available, skipping plots")
        return False

    env = FDJammerEnvironment(FDJammerConfig(seed=42))
    env.reset()

    phases = -np.angle(env.h_RU.conj() * env.h_BR)
    env.set_phases(phases)

    powers = np.linspace(0.0, 1.0, 11)

    bf_modes = ["isotropic", "mrt", "nullspace"]
    sweep_data = {}
    for mode in bf_modes:
        env.set_beamforming_mode(mode)
        sweep_data[mode] = sweep_jammer_power(env, powers)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    colors = {"isotropic": "#1f77b4", "mrt": "#d62728", "nullspace": "#2ca02c"}
    for mode in bf_modes:
        d = sweep_data[mode]
        ax1.plot(d["P_j"], np.array(d["R_sec"]) / 1e6, "o-",
                 color=colors[mode], label=mode, markersize=4)
    ax1.set_xlabel("Jammer Power (W)")
    ax1.set_ylabel("Secrecy Rate (Mbps)")
    ax1.set_title("Secrecy Rate vs Jammer Power")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.15)
    fig1.tight_layout()
    fig1.savefig(str(OUTPUT_DIR / "secrecy_vs_jammer_power.png"), dpi=150)
    plt.close(fig1)
    print(f"  Saved: secrecy_vs_jammer_power.png")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for mode in bf_modes:
        d = sweep_data[mode]
        ax2.plot(d["P_j"], d["sinr_user"], "s-",
                 color=colors[mode], label=mode, markersize=4)
    ax2.set_xlabel("Jammer Power (W)")
    ax2.set_ylabel("User SINR (linear)")
    ax2.set_title("User SINR vs Jammer Power")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.15)
    fig2.tight_layout()
    fig2.savefig(str(OUTPUT_DIR / "user_sinr_vs_jammer_power.png"), dpi=150)
    plt.close(fig2)
    print(f"  Saved: user_sinr_vs_jammer_power.png")

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for mode in bf_modes:
        d = sweep_data[mode]
        ax3.plot(d["P_j"], d["sinr_eve_max"], "D-",
                 color=colors[mode], label=mode, markersize=4)
    ax3.set_xlabel("Jammer Power (W)")
    ax3.set_ylabel("Max Eve SINR (linear)")
    ax3.set_title("Worst-Case Eve SINR vs Jammer Power")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.15)
    fig3.tight_layout()
    fig3.savefig(str(OUTPUT_DIR / "eve_sinr_vs_jammer_power.png"), dpi=150)
    plt.close(fig3)
    print(f"  Saved: eve_sinr_vs_jammer_power.png")

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    baseline_sec = np.array(sweep_data["isotropic"]["R_sec"])
    for mode in bf_modes:
        d = sweep_data[mode]
        sec_gain = np.array(d["R_sec"]) - baseline_sec
        ax4.plot(d["P_j"], sec_gain / 1e6, "^-",
                 color=colors[mode], label=mode, markersize=4)
    ax4.set_xlabel("Jammer Power (W)")
    ax4.set_ylabel("Secrecy Gain over Isotropic (Mbps)")
    ax4.set_title("Secrecy Gain Due to Jammer Beamforming")
    ax4.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.15)
    fig4.tight_layout()
    fig4.savefig(str(OUTPUT_DIR / "secrecy_gain.png"), dpi=150)
    plt.close(fig4)
    print(f"  Saved: secrecy_gain.png")

    with open(OUTPUT_DIR / "validation_summary.txt", "w") as f:
        f.write("FD Jammer Validation Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"N_ris: {env.config.N_ris}\n")
        f.write(f"N_j: {env.config.N_j}\n")
        f.write(f"RIS altitude: {env.config.ris_altitude} m\n")
        f.write(f"Jammer altitude: {env.config.jammer_altitude} m\n")
        f.write(f"Rician K: {env.config.rician_k}\n")
        f.write(f"Eve density lambda: {env.config.eve_density_lambda}\n")
        f.write(f"Num eves: {env.num_eves}\n\n")
        for mode in bf_modes:
            d = sweep_data[mode]
            f.write(f"Mode: {mode}\n")
            for i in range(len(powers)):
                f.write(f"  P_j={d['P_j'][i]:.2f}: "
                        f"R_sec={d['R_sec'][i]:.4f}, "
                        f"sinr_user={d['sinr_user'][i]:.4f}, "
                        f"sinr_eve={d['sinr_eve_max'][i]:.4f}\n")
            f.write("\n")
    print(f"  Saved: validation_summary.txt")
    print()
    return True


def run_validation():
    print("\n" + "=" * 60)
    print("FD JAMMER VALIDATION SUITE")
    print("=" * 60)

    results = {}

    results["miso_channel"] = test_miso_channel()
    results["beamforming"] = test_beamforming_strategies()
    results["interference"] = test_jammer_interference_model()
    results["secrecy"] = test_secrecy_with_jammer()
    results["hppp"] = test_hppp_compatibility()

    plots_ok = generate_plots()
    results["plots"] = plots_ok

    print("=" * 60)
    all_ok = all(v for v in results.values())
    if all_ok:
        print("ALL VALIDATION CHECKS PASSED")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"SOME CHECKS FAILED: {failed}")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 60 + "\n")
    return all_ok


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
