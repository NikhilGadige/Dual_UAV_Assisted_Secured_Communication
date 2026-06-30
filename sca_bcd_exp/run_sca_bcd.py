from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sca_bcd_exp.analysis.plotting import save_all_plots
from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
from sca_bcd_exp.optimization.bcd_solver import BCDSolver


def _write_summary(output_dir: str, result, config: SCABCDConfig):
    lines = [
        "SCA-BCD Solver Summary",
        "=======================\n",
        f"Channel model: {config.channel_model}",
        f"Seed: {config.seed}",
        f"BCD iterations: {result.n_iters}",
        f"Converged: {result.converged}\n",
        f"Initial objective: {result.objective_history[0]:.6f}",
        f"Final objective:   {result.objective_history[-1]:.6f}",
        f"Initial secrecy:   {result.secrecy_history[0]:.6f}",
        f"Final secrecy:     {result.secrecy_history[-1]:.6f}",
        f"Initial sensing:   {result.sensing_history[0]:.6f}",
        f"Final sensing:     {result.sensing_history[-1]:.6f}\n",
        "Final constraint violations:",
    ]
    final_viol = result.violation_history[-1] if result.violation_history else {}
    for k, v in final_viol.items():
        lines.append(f"  {k}: {v:.6e}")
    lines.append("")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, "solver_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def run_sca_bcd(config: SCABCDConfig) -> dict:
    dirs = config.ensure_output_dirs()
    env = SCABCDEnvironment(config)
    solver = BCDSolver(config)
    result = solver.solve(env)

    _write_summary(str(dirs["root"]), result, config)

    plot_paths = save_all_plots(
        output_dir=str(dirs["root"]),
        objective_history=result.objective_history,
        constraint_history=result.violation_history,
        secrecy_history=result.secrecy_history,
        sensing_history=result.sensing_history,
    )

    return {
        "output_dir": str(dirs["root"]),
        "plots": plot_paths,
        "n_iters": result.n_iters,
        "converged": result.converged,
        "initial_objective": float(result.objective_history[0]),
        "final_objective": float(result.objective_history[-1]),
        "initial_secrecy": float(result.secrecy_history[0]),
        "final_secrecy": float(result.secrecy_history[-1]),
        "initial_sensing": float(result.sensing_history[0]),
        "final_sensing": float(result.sensing_history[-1]),
        "final_violations": {
            k: float(v) for k, v in result.violation_history[-1].items()
        } if result.violation_history else {},
    }


def main():
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel-model", choices=["rician", "rayleigh"], default="rician")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-bcd-iters", type=int, default=50)
    parser.add_argument("--max-sca-iters", type=int, default=20)
    parser.add_argument("--tol-obj", type=float, default=1e-4)
    parser.add_argument("--tol-var", type=float, default=1e-4)
    args = parser.parse_args()

    config = SCABCDConfig(
        channel_model=args.channel_model,
        seed=args.seed,
        max_bcd_iters=args.max_bcd_iters,
        max_sca_iters=args.max_sca_iters,
        tol_obj=args.tol_obj,
        tol_var=args.tol_var,
    )
    result = run_sca_bcd(config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
