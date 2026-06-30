"""SCA-BCD benchmark evaluation package.

Phase-5C: comprehensive baselines, Monte-Carlo evaluation, ablation
studies, complexity analysis, Pareto-front characterisation,
validation, and audit.
"""

from sca_bcd_benchmark_exp.baselines import (
    BaselineMethod,
    BaselineResult,
    run_baseline,
)
from sca_bcd_benchmark_exp.configs import BenchmarkConfig
from sca_bcd_benchmark_exp.evaluation import (
    MCSummary,
    aggregate,
    evaluate_baseline_mc,
    run_mc_evaluation,
)
from sca_bcd_benchmark_exp.plotting import save_ablation_plots
from sca_bcd_benchmark_exp.complexity import run_complexity_study
from sca_bcd_benchmark_exp.pareto import run_pareto_sweep
from sca_bcd_benchmark_exp.validate import run_all_validations
from sca_bcd_benchmark_exp.final_audit import (
    run_benchmark_ranking_sanity,
    run_statistical_significance,
    run_pareto_quality_audit,
    run_complexity_scaling_audit,
    run_constraint_activity_analysis,
    run_block_activity_analysis,
    run_local_optimum_sensitivity,
    run_reproducibility_check,
    generate_final_audit_report,
    run_final_audit,
)
from sca_bcd_benchmark_exp.jammer_diagnosis import (
    diagnosis_jammer_mode_override,
    diagnosis_gradient_flatness,
    diagnosis_power_projection_bug,
    diagnosis_trust_region_analysis,
    diagnosis_sca_solver_output,
    diagnosis_corrected_jammer_sensitivity,
    run_jammer_diagnosis,
)
from sca_bcd_benchmark_exp.jammer_fix_verification import (
    run_bcd_with_details,
    verify_acceptance_criteria,
    run_jammer_fix_verification,
)
from sca_bcd_benchmark_exp.sensing_audit import run_sensing_audit
from sca_bcd_benchmark_exp.sensing_fix import run_sensing_fix

__all__ = [
    "BaselineMethod",
    "BaselineResult",
    "run_baseline",
    "BenchmarkConfig",
    "MCSummary",
    "aggregate",
    "evaluate_baseline_mc",
    "run_mc_evaluation",
    "save_ablation_plots",
    "run_complexity_study",
    "run_pareto_sweep",
    "run_all_validations",
    "run_final_audit",
    "generate_final_audit_report",
    "run_benchmark_ranking_sanity",
    "run_statistical_significance",
    "run_pareto_quality_audit",
    "run_complexity_scaling_audit",
    "run_constraint_activity_analysis",
    "run_block_activity_analysis",
    "run_local_optimum_sensitivity",
    "run_reproducibility_check",
    "diagnosis_jammer_mode_override",
    "diagnosis_gradient_flatness",
    "diagnosis_power_projection_bug",
    "diagnosis_trust_region_analysis",
    "diagnosis_sca_solver_output",
    "diagnosis_corrected_jammer_sensitivity",
    "run_jammer_diagnosis",
    "run_bcd_with_details",
    "verify_acceptance_criteria",
    "run_jammer_fix_verification",
    "run_sensing_audit",
    "run_sensing_fix",
]
