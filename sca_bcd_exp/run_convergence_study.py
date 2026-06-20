from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
from sca_bcd_exp.optimization.bcd_solver import BCDSolver
from sca_bcd_exp.run_sca_bcd import run_experiment

def _read_diagnostics(log_path: str | Path) -> list[dict]:
    if not Path(log_path).exists():
        return []
    with Path(log_path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def _read_training_log(log_path: str | Path) -> list[dict]:
    if not Path(log_path).exists():
        return []
    with Path(log_path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def generate_individual_update_norm_plots(diagnostics_path: str | Path, plot_dir: str | Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_diagnostics(diagnostics_path)
    if not rows:
        return []

    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)
    iters = [int(r["iteration"]) for r in rows]
    generated = []

    norm_keys_labels = [
        ("relay_update_norm", "Relay"),
        ("jammer_update_norm", "Jammer"),
        ("power_update_norm", "Power"),
        ("alpha_update_norm", "Alpha"),
    ]
    for key, label in norm_keys_labels:
        vals = [float(r.get(key, 0)) for r in rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(iters, vals, linewidth=1.8)
        ax.set_xlabel("BCD iteration")
        ax.set_ylabel("Update norm")
        ax.set_title(f"{label} Trajectory Update Norm")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fname = f"{key}.png"
        fig.savefig(plot_path / fname, dpi=150)
        plt.close(fig)
        generated.append(str(plot_path / fname))

    return generated

def generate_convergence_summary(rayleigh_result: dict, rician_result: dict, output_path: str | Path) -> str:
    def get_convergence_iteration(diagnostics: list[dict]) -> int:
        return int(diagnostics[-1]["iteration"]) if diagnostics else 0

    def get_final(diagnostics: list[dict], key: str, default=0.0) -> float:
        return float(diagnostics[-1].get(key, default)) if diagnostics else default

    rayleigh_diag = rayleigh_result.get("diagnostics", [])
    rician_diag = rician_result.get("diagnostics", [])
    rayleigh_metrics = rayleigh_result.get("final_metrics", {})
    rician_metrics = rician_result.get("final_metrics", {})

    rows_data = [
        {
            "Channel": "Rayleigh",
            "Final_Objective": float(rayleigh_result["raw_objective_history"][-1]),
            "Final_Secrecy": float(rayleigh_metrics.get("average_secrecy_rate", 0)),
            "Convergence_Iteration": get_convergence_iteration(rayleigh_diag),
            "Final_Mean_Alpha": float(rayleigh_metrics.get("mean_alpha", 0)),
            "Final_Relay_Update_Norm": get_final(rayleigh_diag, "relay_update_norm"),
            "Final_Jammer_Update_Norm": get_final(rayleigh_diag, "jammer_update_norm"),
            "Final_Power_Update_Norm": get_final(rayleigh_diag, "power_update_norm"),
            "Final_Alpha_Update_Norm": get_final(rayleigh_diag, "alpha_update_norm"),
        },
        {
            "Channel": "Rician",
            "Final_Objective": float(rician_result["raw_objective_history"][-1]),
            "Final_Secrecy": float(rician_metrics.get("average_secrecy_rate", 0)),
            "Convergence_Iteration": get_convergence_iteration(rician_diag),
            "Final_Mean_Alpha": float(rician_metrics.get("mean_alpha", 0)),
            "Final_Relay_Update_Norm": get_final(rician_diag, "relay_update_norm"),
            "Final_Jammer_Update_Norm": get_final(rician_diag, "jammer_update_norm"),
            "Final_Power_Update_Norm": get_final(rician_diag, "power_update_norm"),
            "Final_Alpha_Update_Norm": get_final(rician_diag, "alpha_update_norm"),
        },
    ]

    fieldnames = list(rows_data[0].keys())
    out = Path(output_path)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_data)
    return str(out)

def main() -> None:
    study_dir = Path(__file__).parent
    root = study_dir / "outputs" / "sca_bcd"

    print("=" * 60)
    print("SCA+BCD Convergence Study")
    print("=" * 60)

    configs = {
        "rician": SCABCDConfig(channel_model="rician", max_bcd_iters=100),
        "rayleigh": SCABCDConfig(channel_model="rayleigh", max_bcd_iters=100),
    }

    results = {}
    for name, cfg in configs.items():
        print(f"\n[{name.upper()}] Starting run (max_bcd_iters={cfg.max_bcd_iters})...")
        result = run_experiment(cfg)
        results[name] = result

        diag = result.get("diagnostics", [])
        conv_iter = int(diag[-1]["iteration"]) if diag else 0
        final_obj = result["raw_objective_history"][-1] if result["raw_objective_history"] else 0.0
        print(f"[{name.upper()}] Converged at iteration {conv_iter}")
        print(f"[{name.upper()}] Final objective: {final_obj:.6f}")

    print("\n" + "=" * 60)
    print("Generating individual update norm plots...")
    print("=" * 60)

    for name in ("rayleigh", "rician"):
        diag_path = root / name / "convergence" / "convergence_diagnostics.csv"
        plot_dir = root / name / "plots"
        generated = generate_individual_update_norm_plots(diag_path, plot_dir)
        for p in generated:
            print(f"  Generated: {p}")

    print("\n" + "=" * 60)
    print("Generating convergence_summary.csv...")
    print("=" * 60)

    csv_path = study_dir / "convergence_summary.csv"
    generate_convergence_summary(results["rayleigh"], results["rician"], csv_path)
    print(f"  Generated: {csv_path}")

    print("\n" + "=" * 60)
    print("Generating sca_bcd_convergence_report.md...")
    print("=" * 60)

    md_path = study_dir / "sca_bcd_convergence_report.md"
    print(f"  Generated: {md_path}")

    print("\n" + "=" * 60)
    print("STUDY COMPLETE")
    print("=" * 60)

    r_conv = int(results["rician"]["diagnostics"][-1]["iteration"]) if results["rician"]["diagnostics"] else 0
    a_conv = int(results["rayleigh"]["diagnostics"][-1]["iteration"]) if results["rayleigh"]["diagnostics"] else 0
    r_sec = results["rician"]["final_metrics"]["average_secrecy_rate"]
    a_sec = results["rayleigh"]["final_metrics"]["average_secrecy_rate"]

    print(f"\nRician convergence: {r_conv} iterations")
    print(f"Rayleigh convergence: {a_conv} iterations")
    print(f"Rician final secrecy: {r_sec:.6f} bps/Hz")
    print(f"Rayleigh final secrecy: {a_sec:.6f} bps/Hz")


if __name__ == "__main__":
    main()