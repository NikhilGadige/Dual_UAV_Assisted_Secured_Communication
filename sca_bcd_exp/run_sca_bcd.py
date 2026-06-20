from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict

import numpy as np

from sca_bcd_exp.analysis.plotting import plot_alpha_convergence, plot_convergence, plot_diagnostics, plot_mean_alpha_vs_iteration
from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
from sca_bcd_exp.optimization.bcd_solver import BCDSolver

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper-based SCA + BCD optimization.")
    parser.add_argument("--channel-model", choices=["rician", "rayleigh"], default="rician")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--max-bcd-iters", type=int, default=100)
    parser.add_argument("--min-bcd-iters", type=int, default=20)
    parser.add_argument("--max-sca-iters", type=int, default=8)
    parser.add_argument("--bcd-abs-tol", type=float, default=1e-3)
    parser.add_argument("--bcd-rel-tol", type=float, default=5e-4)
    parser.add_argument("--bcd-patience", type=int, default=8)
    return parser

def run_experiment(config: SCABCDConfig) -> dict:
    dirs = config.ensure_output_dirs()
    env = SCABCDEnvironment(config)
    solver = BCDSolver(config)
    result = solver.solve(env)

    training_log = dirs["convergence"] / "training_log.csv"
    with training_log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                "objective",
                "average_secrecy_rate",
                "average_num_eves",
                "average_nearest_eve_distance",
                "average_max_eve_capacity",
                "mean_alpha",
                "min_alpha",
                "max_alpha",
            ],
        )
        writer.writeheader()
        writer.writerows(result.iteration_metrics)

    diagnostics_log = dirs["convergence"] / "convergence_diagnostics.csv"
    with diagnostics_log.open("w", newline="", encoding="utf-8") as handle:
        if result.diagnostics:
            writer = csv.DictWriter(handle, fieldnames=list(result.diagnostics[0].keys()))
            writer.writeheader()
            writer.writerows(result.diagnostics)

    checkpoint_path = dirs["checkpoints"] / "latest_solution.npz"
    np.savez(
        checkpoint_path,
        relay_trajectory=result.solution.relay_trajectory,
        jammer_trajectory=result.solution.jammer_trajectory,
        source_power=result.solution.source_power,
        relay_power=result.solution.relay_power,
        jammer_power=result.solution.jammer_power,
        alpha_trajectory=result.solution.alpha_trajectory,
        raw_objective_history=np.asarray(result.raw_objective_history, dtype=float),
        display_objective_history=np.asarray(result.display_objective_history, dtype=float),
    )

    report = {
        "config": asdict(config),
        "raw_objective_history": result.raw_objective_history,
        "display_objective_history": result.display_objective_history,
        "sca_objective_history": result.sca_objective_history,
        "final_metrics": result.iteration_metrics[-1],
        "checkpoint": str(checkpoint_path),
        "diagnostics": result.diagnostics,
    }
    report_path = dirs["reports"] / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    plot_paths = plot_convergence(str(training_log), str(dirs["plots"]))
    diag_paths = plot_diagnostics(str(diagnostics_log), str(dirs["plots"]))
    alpha_paths = plot_alpha_convergence(str(diagnostics_log), str(dirs["plots"]))
    mean_alpha_paths = plot_mean_alpha_vs_iteration(str(training_log), str(dirs["plots"]))

    return {
        "output_dirs": {key: str(value) for key, value in dirs.items()},
        "training_log": str(training_log),
        "diagnostics_log": str(diagnostics_log),
        "report": str(report_path),
        "plots": {**plot_paths, **diag_paths, **alpha_paths, **mean_alpha_paths},
        "final_metrics": result.iteration_metrics[-1],
        "raw_objective_history": result.raw_objective_history,
        "display_objective_history": result.display_objective_history,
        "sca_objective_history": result.sca_objective_history,
        "diagnostics": result.diagnostics,
    }

def main() -> None:
    args = build_parser().parse_args()
    config = SCABCDConfig(
        channel_model=args.channel_model,
        seed=args.seed,
        horizon=args.horizon,
        max_bcd_iters=args.max_bcd_iters,
        min_bcd_iters=args.min_bcd_iters,
        max_sca_iters=args.max_sca_iters,
        bcd_abs_tolerance=args.bcd_abs_tol,
        bcd_rel_tolerance=args.bcd_rel_tol,
        bcd_patience=args.bcd_patience,
    )
    result = run_experiment(config)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()