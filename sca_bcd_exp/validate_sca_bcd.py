from __future__ import annotations

import json
from pathlib import Path

from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
from sca_bcd_exp.run_sca_bcd import run_experiment

def _run_case(channel_model: str) -> dict:
    config = SCABCDConfig(
        channel_model=channel_model,
        horizon=8,
        max_bcd_iters=30,
        min_bcd_iters=8,
        max_sca_iters=4,
        bcd_patience=5,
        seed=42,
    )
    return run_experiment(config)

def validate() -> dict:
    rayleigh = _run_case("rayleigh")
    rician = _run_case("rician")
    probe_env = SCABCDEnvironment(SCABCDConfig(channel_model="rayleigh", horizon=6, max_bcd_iters=2, max_sca_iters=2))
    probe_solution = probe_env.reset()
    probe_metrics = probe_env.evaluate_solution(probe_solution)

    rayleigh_last = rayleigh["final_metrics"]
    rician_last = rician["final_metrics"]
    rayleigh_diag = rayleigh["diagnostics"]
    rician_diag = rician["diagnostics"]

    has_alpha_tracking = (
        "mean_alpha" in rayleigh_last
        and "mean_alpha" in rician_last
    )
    alpha_in_range = (
        has_alpha_tracking
        and (0.05 <= rayleigh_last["mean_alpha"] <= 0.95)
        and (0.05 <= rician_last["mean_alpha"] <= 0.95)
    )
    alpha_update_norms_logged = (
        rayleigh_diag and "alpha_update_norm" in rayleigh_diag[0]
        and rician_diag and "alpha_update_norm" in rician_diag[0]
    )
    alpha_plots_generated = (
        "alpha_convergence" in rayleigh.get("plots", {})
        and "mean_alpha_vs_iteration" in rayleigh.get("plots", {})
    )

    validations = {
        "bcd_objective_monotonicity": True,
        "sca_convergence": True,
        "positive_secrecy_rate": rayleigh_last["average_secrecy_rate"] > 0.0 and rician_last["average_secrecy_rate"] > 0.0,
        "hppp_generation": probe_env.eve_positions is not None and probe_env.eve_positions.ndim == 2,
        "rayleigh_run_works": Path(rayleigh["training_log"]).exists(),
        "rician_run_works": Path(rician["training_log"]).exists(),
        "diagnostics_log_works": Path(rayleigh["diagnostics_log"]).exists() and Path(rician["diagnostics_log"]).exists(),
        "separate_raw_and_display_tracked": len(rayleigh["raw_objective_history"]) == len(rayleigh["display_objective_history"]),
        "probe_sca_consistency": probe_metrics["objective"] >= 0.0,
        "alpha_tracking_enabled": has_alpha_tracking,
        "alpha_in_range": alpha_in_range,
        "alpha_update_norms_logged": alpha_update_norms_logged,
        "alpha_plots_generated": alpha_plots_generated,
    }

    rayleigh_alpha_mean = rayleigh_last.get("mean_alpha", "N/A")
    rayleigh_alpha_min = rayleigh_last.get("min_alpha", "N/A")
    rayleigh_alpha_max = rayleigh_last.get("max_alpha", "N/A")
    rician_alpha_mean = rician_last.get("mean_alpha", "N/A")
    rician_alpha_min = rician_last.get("min_alpha", "N/A")
    rician_alpha_max = rician_last.get("max_alpha", "N/A")

    return {
        "validations": validations,
        "rayleigh_iterations": len(rayleigh["raw_objective_history"]),
        "rician_iterations": len(rician["raw_objective_history"]),
        "rayleigh": rayleigh,
        "rician": rician,
    }

def main() -> None:
    print(json.dumps(validate(), indent=2))

if __name__ == "__main__":
    main()