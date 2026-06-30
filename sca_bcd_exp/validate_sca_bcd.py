from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sca_bcd_exp.analysis.convergence_analysis import (
    has_converged,
    is_objective_finite,
    violation_decreasing,
    objective_non_decreasing,
    check_no_nan_inf,
)
from sca_bcd_exp.complex_gradient_audit import (
    gradient_real_fd,
    gradient_complex_fd,
    gradient_wirtinger,
    chain_rule_error_real,
    chain_rule_error_complex_perturbation,
    chain_rule_error_wirtinger,
)
from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
from sca_bcd_exp.run_sca_bcd import run_sca_bcd


def _multi_seed_test(config: SCABCDConfig, seeds: list[int]) -> dict:
    results = {}
    for s in seeds:
        cfg = SCABCDConfig(**{**config.__dict__, "seed": s})
        results[s] = run_sca_bcd(cfg)
    return results


def validate() -> dict:
    config = SCABCDConfig(
        channel_model="rician",
        seed=0,
        max_bcd_iters=10,
        max_sca_iters=5,
        tol_obj=1e-4,
        tol_var=1e-4,
    )
    result = run_sca_bcd(config)

    obj_hist = result.get("objective_history", result.get("initial_objective", []))
    if not isinstance(obj_hist, list):
        obj_hist = [result.get("initial_objective", 0.0), result.get("final_objective", 0.0)]

    viol_hist = result.get("violation_history", [])
    if not isinstance(viol_hist, list):
        viol_hist = []

    checks = {
        "objective_finite_every_iteration": is_objective_finite(obj_hist),
        "constraint_violation_decreases": violation_decreasing(viol_hist),
        "objective_non_decreasing": objective_non_decreasing(obj_hist),
        "solver_converged_before_max_iter": result.get("converged", False) or len(obj_hist) < config.max_bcd_iters,
        "final_solution_satisfies_constraints": _check_final_constraints(result),
        "no_nan_or_inf": check_no_nan_inf(obj_hist),
    }

    multi_seed_results = _multi_seed_test(config, [1, 2, 3])
    seeds_stable = all(
        r.get("converged", False) and np.isfinite(r.get("final_objective", -np.inf))
        for r in multi_seed_results.values()
    )
    checks["multiple_random_seeds_stable"] = bool(seeds_stable)

    report_lines = [
        "# Phase 5B Validation Report\n",
        "## Checks",
    ]
    for k, v in checks.items():
        status = "PASS" if v else "FAIL"
        report_lines.append(f"- {k}: **{status}**")

    report_lines += [
        "",
        "## Metrics",
        f"- BCD iterations: {result.get('n_iters', 'N/A')}",
        f"- Initial objective: {result.get('initial_objective', 'N/A'):.6f}",
        f"- Final objective: {result.get('final_objective', 'N/A'):.6f}",
        f"- Initial secrecy: {result.get('initial_secrecy', 'N/A'):.6f}",
        f"- Final secrecy: {result.get('final_secrecy', 'N/A'):.6f}",
        f"- Initial sensing: {result.get('initial_sensing', 'N/A'):.6f}",
        f"- Final sensing: {result.get('final_sensing', 'N/A'):.6f}",
        "",
        "## Multi-Seed",
    ]
    for s, r in multi_seed_results.items():
        report_lines.append(f"- Seed {s}: converged={r.get('converged', False)}, final_obj={r.get('final_objective', 'N/A'):.6f}")

    report_lines.append("")
    Path("outputs/optimization/mimo_analysis/validation_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {"checks": checks, "multi_seed": multi_seed_results, "report": "outputs/optimization/mimo_analysis/validation_report.md"}


def _check_final_constraints(result: dict) -> bool:
    final_viol = result.get("final_violations", {})
    if not final_viol:
        return False
    total = sum(float(v) for v in final_viol.values())
    return total < 1e-6


# ── Complex-gradient audit tests ─────────────────────────────────────


def test_power_chain_rule_accuracy(
    config: SCABCDConfig | None = None,
    n_perturbations: int = 100,
) -> dict:
    """Validate that the real-FD power-block gradient satisfies the
    chain rule to within ``median < 1e-2`` and ``max < 1e-1``."""
    if config is None:
        config = SCABCDConfig(
            channel_model="rician", seed=0,
            max_bcd_iters=2, max_sca_iters=2,
        )
    env = SCABCDEnvironment(config)
    solution = env.reset()
    blocks = env.block_slices()
    grad_real = gradient_real_fd(env, solution, blocks["power"], h=config.fd_h)
    err = chain_rule_error_real(env, solution, grad_real, blocks["power"], n_perturbations)
    result = {
        "test": "test_power_chain_rule_accuracy",
        "median_chain_rel_err": err["median_rel_err"],
        "max_chain_rel_err": err["max_rel_err"],
        "passed_median": err["median_rel_err"] < 1e-2,
        "passed_max": err["max_rel_err"] < 1e-1,
    }
    print(f"  [test_power_chain_rule_accuracy] median={err['median_rel_err']:.3e}  max={err['max_rel_err']:.3e}  "
          f"PASS={result['passed_median'] and result['passed_max']}")
    return result


def test_complex_gradient_consistency(
    config: SCABCDConfig | None = None,
    n_perturbations: int = 100,
) -> dict:
    """Verify that the real-FD gradient correctly predicts objective
    changes under complex perturbations of w_bs (using the Re/Im split
    chain rule)."""
    if config is None:
        config = SCABCDConfig(
            channel_model="rician", seed=0,
            max_bcd_iters=2, max_sca_iters=2,
        )
    env = SCABCDEnvironment(config)
    solution = env.reset()
    blocks = env.block_slices()
    grad_real = gradient_real_fd(env, solution, blocks["power"], h=config.fd_h)
    err = chain_rule_error_complex_perturbation(
        env, solution, grad_real, n_perturbations,
    )
    result = {
        "test": "test_complex_gradient_consistency",
        "median_chain_rel_err": err["median_rel_err"],
        "max_chain_rel_err": err["max_rel_err"],
        "passed_median": err["median_rel_err"] < 1e-2,
        "passed_max": err["max_rel_err"] < 1e-1,
    }
    print(f"  [test_complex_gradient_consistency] median={err['median_rel_err']:.3e}  max={err['max_rel_err']:.3e}  "
          f"PASS={result['passed_median'] and result['passed_max']}")
    return result


def test_wirtinger_fd_stability(
    config: SCABCDConfig | None = None,
    n_perturbations: int = 100,
) -> dict:
    """Check that the Wirtinger gradient (derived from real FD)
    predicts objective changes via the Wirtinger chain rule."""
    if config is None:
        config = SCABCDConfig(
            channel_model="rician", seed=0,
            max_bcd_iters=2, max_sca_iters=2,
        )
    env = SCABCDEnvironment(config)
    solution = env.reset()
    blocks = env.block_slices()
    grad_real = gradient_real_fd(env, solution, blocks["power"], h=config.fd_h)
    err = chain_rule_error_wirtinger(env, solution, grad_real, n_perturbations)
    result = {
        "test": "test_wirtinger_fd_stability",
        "median_chain_rel_err": err["median_rel_err"],
        "max_chain_rel_err": err["max_rel_err"],
        "passed_median": err["median_rel_err"] < 1e-2,
        "passed_max": err["max_rel_err"] < 1e-1,
        "df_dw_wirtinger_conj_symmetry_violation": float(
            np.max(np.abs(
                gradient_wirtinger(grad_real, config.N_time)[1]
                - np.conj(gradient_wirtinger(grad_real, config.N_time)[0])
            ))
        ),
    }
    print(f"  [test_wirtinger_fd_stability] median={err['median_rel_err']:.3e}  max={err['max_rel_err']:.3e}  "
          f"PASS={result['passed_median'] and result['passed_max']}")
    return result


def main():
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(json.dumps(validate(), indent=2, default=str))


if __name__ == "__main__":
    main()
