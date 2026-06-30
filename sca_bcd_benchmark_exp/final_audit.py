"""Phase 5A+5B+5C final audit.

Performs:
  1. Benchmark ranking sanity (N_mc=100)
  2. Statistical significance (paired t-test, Wilcoxon, Cohen's d)
  3. Pareto front quality (monotonicity, hypervolume, spacing, spread)
  4. Complexity scaling audit (R^2 of power-law fits)
  5. Constraint activity analysis
  6. BCD block activity analysis
  7. Local optimum sensitivity (20 random inits)
  8. Reproducibility verification
  9. Final audit report
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

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


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _import_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# Part 1: Benchmark Ranking Sanity
# ═══════════════════════════════════════════════════════════════


def run_benchmark_ranking_sanity(
    cfg: BenchmarkConfig,
    output_dir: str,
    methods: list[BaselineMethod] | None = None,
) -> dict:
    """Run N_mc baselines, verify expected ordering, flag violations."""
    out = _ensure_dir(output_dir)
    if methods is None:
        methods = [
            BaselineMethod.RANDOM_FEASIBLE,
            BaselineMethod.POWER_ONLY,
            BaselineMethod.TRAJECTORY_ONLY,
            BaselineMethod.JAMMER_ONLY,
            BaselineMethod.NO_RIS,
            BaselineMethod.NO_JAMMER,
            BaselineMethod.NO_SECRECY,
            BaselineMethod.NO_SENSING,
            BaselineMethod.SCA_BCD_FULL,
        ]

    print("  Running MC evaluation for ranking sanity...")
    summaries = run_mc_evaluation(cfg, methods=[m for m in methods], quiet=False)

    order = [m.value for m in methods]
    ordered = [summaries.get(m.value) for m in methods if m.value in summaries]

    ranking_flags = []
    full_summary = summaries.get(BaselineMethod.SCA_BCD_FULL.value)

    for s in ordered:
        if s is None or full_summary is None:
            continue
        name = s.method
        # Check: full > random
        if name == BaselineMethod.RANDOM_FEASIBLE.value:
            if full_summary.objective_mean < s.objective_mean - 1e-8:
                ranking_flags.append(f"FLAG: mean(full)={full_summary.objective_mean:.4f} < mean({name})={s.objective_mean:.4f}")
        # Check: full > single-block
        if name in ("power_only", "trajectory_only", "jammer_only"):
            if full_summary.objective_mean < s.objective_mean - 1e-8:
                ranking_flags.append(f"FLAG: mean(full)={full_summary.objective_mean:.4f} < mean({name})={s.objective_mean:.4f}")
        # Check: ablation variants <= full
        if name in ("no_ris", "no_jammer"):
            if s.objective_mean > full_summary.objective_mean + 1e-8:
                ranking_flags.append(f"FLAG: mean({name})={s.objective_mean:.4f} > mean(full)={full_summary.objective_mean:.4f}")

    # Write CSV
    csv_path = out / "benchmark_ranking.csv"
    fieldnames = [
        "method", "N_mc", "objective_mean", "objective_std", "objective_median",
        "secrecy_mean", "secrecy_std", "sensing_mean", "sensing_std",
        "runtime_mean", "n_iters_mean", "violation_mean", "converged_fraction",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for s in ordered:
            d = asdict(s)
            for k in list(d.keys()):
                if k.startswith("raw_"):
                    del d[k]
            w.writerow(d)

    # Write report
    report_lines = [
        "# Benchmark Ranking Sanity Report",
        "",
        f"Config: N_mc={cfg.N_mc}, max_bcd_iters={cfg.max_bcd_iters}, max_sca_iters={cfg.max_sca_iters}",
        "",
        "## Rankings (mean objective)",
        "",
    ]
    for s in ordered:
        if s is None:
            continue
        report_lines.append(
            f"  {s.method:25s}  {s.objective_mean:.6f} ± {s.objective_std:.6f}"
        )

    report_lines += ["", "## Expected Ordering", "",
                      "  Random < Single-block < Ablation variants < Full SCA-BCD", ""]

    if ranking_flags:
        report_lines += ["## FLAGS", ""]
        report_lines += [f"  {f}" for f in ranking_flags]
    else:
        report_lines += ["## No flags raised. Ordering is as expected.", ""]

    report_path = out / "benchmark_ranking_report.md"
    Path(report_path).write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  Ranking report: {report_path}")

    return {
        "summaries": summaries,
        "ranking_flags": ranking_flags,
        "report_path": str(report_path),
    }


# ═══════════════════════════════════════════════════════════════
# Part 2: Statistical Significance
# ═══════════════════════════════════════════════════════════════


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    n1, n2 = len(x), len(y)
    s1, s2 = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if sp < 1e-15:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / sp)


def run_statistical_significance(
    cfg: BenchmarkConfig,
    output_dir: str,
    target_method: BaselineMethod = BaselineMethod.SCA_BCD_FULL,
    other_methods: list[BaselineMethod] | None = None,
) -> dict:
    """Paired tests: Full SCA-BCD vs each baseline."""
    out = _ensure_dir(output_dir)

    if other_methods is None:
        other_methods = [
            BaselineMethod.RANDOM_FEASIBLE,
            BaselineMethod.POWER_ONLY,
            BaselineMethod.TRAJECTORY_ONLY,
            BaselineMethod.JAMMER_ONLY,
            BaselineMethod.NO_RIS,
            BaselineMethod.NO_JAMMER,
            BaselineMethod.NO_SECRECY,
            BaselineMethod.NO_SENSING,
        ]

    rows = []
    N_mc = cfg.N_mc

    print(f"  Running {N_mc} seeds for {target_method.value}...")
    target_results = []
    for s in range(N_mc):
        try:
            r = run_baseline(target_method, cfg, seed=s)
            target_results.append(r)
        except Exception:
            target_results.append(None)

    for meth in other_methods:
        label = f"{target_method.value}_vs_{meth.value}"
        print(f"  Testing {label}...")
        other_results = []
        for s in range(N_mc):
            try:
                r = run_baseline(meth, cfg, seed=s)
                other_results.append(r)
            except Exception:
                other_results.append(None)

        # Paired: only use seeds where both succeeded
        pairs = [(t, o) for t, o in zip(target_results, other_results)
                 if t is not None and o is not None]
        if len(pairs) < 3:
            rows.append({
                "comparison": label,
                "N_valid_pairs": len(pairs),
                "t_statistic": float("nan"),
                "t_p_value": float("nan"),
                "wilcoxon_statistic": float("nan"),
                "wilcoxon_p_value": float("nan"),
                "cohens_d": float("nan"),
                "mean_diff": float("nan"),
                "target_mean": float("nan"),
                "other_mean": float("nan"),
            })
            continue

        t_vals = np.array([p[0].objective for p in pairs])
        o_vals = np.array([p[1].objective for p in pairs])
        diff = t_vals - o_vals

        try:
            from scipy import stats
            t_stat, t_p = stats.ttest_rel(t_vals, o_vals)
            w_stat, w_p = stats.wilcoxon(diff, alternative="two-sided")
        except Exception:
            t_stat, t_p = float("nan"), float("nan")
            w_stat, w_p = float("nan"), float("nan")

        d = _cohens_d(t_vals, o_vals)

        rows.append({
            "comparison": label,
            "N_valid_pairs": len(pairs),
            "t_statistic": float(t_stat),
            "t_p_value": float(t_p),
            "wilcoxon_statistic": float(w_stat),
            "wilcoxon_p_value": float(w_p),
            "cohens_d": float(d),
            "mean_diff": float(np.mean(diff)),
            "target_mean": float(np.mean(t_vals)),
            "other_mean": float(np.mean(o_vals)),
        })

    csv_path = out / "statistical_significance.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Report
    report_lines = [
        "# Statistical Significance Report",
        "",
        f"Target method: {target_method.value} vs each baseline (N_mc={N_mc})",
        "",
        "| Comparison | N_pairs | t_stat | t_pval | wilcoxon_stat | wilcoxon_pval | Cohen's d | Mean diff |",
        "|-----------|---------|--------|--------|---------------|----------------|-----------|-----------|",
    ]
    for r in rows:
        report_lines.append(
            f"| {r['comparison']} | {r['N_valid_pairs']} | "
            f"{r['t_statistic']:.4f} | {r['t_p_value']:.4e} | "
            f"{r['wilcoxon_statistic']:.4f} | {r['wilcoxon_p_value']:.4e} | "
            f"{r['cohens_d']:.4f} | {r['mean_diff']:.4f} |"
        )

    report_lines += [
        "",
        "## Interpretation",
        "",
        "- p < 0.05: statistically significant difference",
        "- |Cohen's d| > 0.8: large effect size",
        "- |Cohen's d| > 0.5: medium effect size",
        "- |Cohen's d| > 0.2: small effect size",
        "",
    ]

    report_path = out / "statistical_report.md"
    Path(report_path).write_text("\n".join(report_lines), encoding="utf-8")

    return {"rows": rows, "csv_path": str(csv_path), "report_path": str(report_path)}


# ═══════════════════════════════════════════════════════════════
# Part 3: Pareto Front Quality
# ═══════════════════════════════════════════════════════════════


def _hypervolume(pts: np.ndarray, ref: np.ndarray) -> float:
    """Compute hypervolume for 2D Pareto front."""
    order = np.argsort(pts[:, 0])[::-1]
    pts = pts[order]
    hv = 0.0
    prev_y = ref[1]
    for p in pts:
        if p[1] < prev_y:
            hv += (prev_y - p[1]) * (p[0] - ref[0])
            prev_y = p[1]
    return float(hv)


def _spacing_metric(pts: np.ndarray) -> float:
    """Spacing metric: std of distances between consecutive Pareto points."""
    order = np.argsort(pts[:, 0])
    sorted_pts = pts[order]
    dists = np.array([
        float(np.linalg.norm(sorted_pts[i + 1] - sorted_pts[i]))
        for i in range(len(sorted_pts) - 1)
    ])
    if len(dists) < 2:
        return 0.0
    return float(np.std(dists, ddof=1))


def _spread_metric(pts: np.ndarray, ref: np.ndarray) -> float:
    """Spread metric: how well points cover the extents."""
    if len(pts) < 2:
        return 0.0
    extents = np.max(pts, axis=0) - np.min(pts, axis=0)
    ref_extents = ref - np.min(pts, axis=0)
    return float(np.mean(extents / np.maximum(ref_extents, 1e-15)))


def run_pareto_quality_audit(
    cfg: BenchmarkConfig,
    output_dir: str,
    alpha_vals: list[float] | None = None,
) -> dict:
    """Audit Pareto front quality."""
    out = _ensure_dir(output_dir)

    if alpha_vals is None:
        alpha_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("  Computing Pareto front...")
    rows_list = []
    for alpha in alpha_vals:
        mod = BenchmarkConfig(**{**cfg.__dict__, "alpha": alpha, "seed": cfg.seed})
        r = run_baseline(BaselineMethod.SCA_BCD_FULL, mod, seed=cfg.seed)
        rows_list.append({
            "alpha": alpha,
            "secrecy_rate": r.secrecy_rate,
            "sensing_utility": r.sensing_utility,
            "objective": r.objective,
        })

    secs = np.array([r["secrecy_rate"] for r in rows_list])
    sens = np.array([r["sensing_utility"] for r in rows_list])
    alphas = np.array([r["alpha"] for r in rows_list])

    # Monotonicity: secrecy increases with alpha, sensing decreases with alpha
    sec_nondec = all(secs[i] <= secs[i + 1] + 1e-10 for i in range(len(secs) - 1))
    sens_noninc = all(sens[i] >= sens[i + 1] - 1e-10 for i in range(len(sens) - 1))

    # Dominance check
    dominated = []
    for i in range(len(rows_list)):
        for j in range(len(rows_list)):
            if i == j:
                continue
            if (secs[i] >= secs[j] + 1e-10 and sens[i] >= sens[j] + 1e-10):
                dominated.append((i, j, rows_list[i]["alpha"], rows_list[j]["alpha"]))
    n_dominated_pairs = len(dominated)

    # Convexity (approximate): check second differences
    pts = np.column_stack([secs, sens])
    ref_pt = np.array([float(np.min(secs) - 1.0), float(np.min(sens) - 1.0)])
    hv = _hypervolume(pts, ref_pt)
    sp = _spacing_metric(pts)
    spr = _spread_metric(pts, np.array([float(np.max(secs)), float(np.max(sens))]))

    # Convexity heuristic: check if the frontier is concave or convex
    convex_scores = []
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        cross = float(np.cross(v1, v2))
        convex_scores.append(cross)
    is_convex = np.mean(convex_scores) > 0 if convex_scores else None

    # CSV
    csv_path = out / "pareto_quality.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "secrecy_rate", "sensing_utility", "objective"])
        w.writeheader()
        w.writerows(rows_list)

    # Report
    report_lines = [
        "# Pareto Front Quality Report",
        "",
        "## Monotonicity",
        f"  Secrecy non-decreasing with alpha: {sec_nondec}",
        f"  Sensing non-increasing with alpha:  {sens_noninc}",
        "",
        "## Dominance",
        f"  Dominated pairs: {n_dominated_pairs}",
    ]
    if dominated:
        report_lines.append("  Dominated points:")
        for i, j, ai, aj in dominated[:10]:
            report_lines.append(f"    alpha={ai:.1f} dominates alpha={aj:.1f}")

    report_lines += [
        "",
        "## Quality Metrics",
        f"  Hypervolume (ref={ref_pt}): {hv:.6f}",
        f"  Spacing metric:            {sp:.6f}",
        f"  Spread metric:             {spr:.6f}",
        f"  Convex (approx):           {is_convex}",
        "",
        "## Interpretation",
        "",
    ]

    if sec_nondec:
        report_lines.append("  PASS: Secrecy monotonic (non-decreasing with alpha).")
    else:
        report_lines.append("  FAIL: Secrecy is not monotonic with alpha.")
    if sens_noninc:
        report_lines.append("  PASS: Sensing monotonic (non-increasing with alpha).")
    else:
        report_lines.append("  FAIL: Sensing is not monotonic with alpha.")

    report_path = out / "pareto_quality_report.md"
    Path(report_path).write_text("\n".join(report_lines), encoding="utf-8")

    # Plot
    plt = _import_plt()
    if plt:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ax = axes[0]
        ax.plot(alphas, secs, "o-", color="coral")
        ax.set_xlabel("alpha"); ax.set_ylabel("Secrecy rate")
        ax.set_title("Secrecy vs alpha"); ax.grid(alpha=0.2)

        ax = axes[1]
        ax.plot(alphas, sens, "s-", color="mediumpurple")
        ax.set_xlabel("alpha"); ax.set_ylabel("Sensing utility")
        ax.set_title("Sensing vs alpha"); ax.grid(alpha=0.2)

        ax = axes[2]
        sc = ax.scatter(secs, sens, c=alphas, cmap="viridis", s=60, edgecolor="k")
        ax.plot(secs, sens, "--", color="gray", alpha=0.5)
        for i, a in enumerate(alphas):
            ax.annotate(f"{a:.1f}", (secs[i], sens[i]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        plt.colorbar(sc, ax=ax, label="alpha")
        ax.set_xlabel("Secrecy rate"); ax.set_ylabel("Sensing utility")
        ax.set_title("Pareto frontier"); ax.grid(alpha=0.2)

        fig.tight_layout()
        fig.savefig(str(out / "pareto_quality.png"), dpi=150)
        plt.close(fig)

    return {
        "rows": rows_list,
        "sec_noninc": bool(sec_nondec),
        "sens_nondec": bool(sens_noninc),
        "hypervolume": hv,
        "spacing": sp,
        "spread": spr,
        "is_convex": is_convex,
        "report_path": str(report_path),
    }


# ═══════════════════════════════════════════════════════════════
# Part 4: Complexity Scaling Audit
# ═══════════════════════════════════════════════════════════════


def run_complexity_scaling_audit(
    cfg: BenchmarkConfig,
    output_dir: str,
) -> dict:
    """Audit complexity scaling fits and flag R^2 < 0.9."""
    from sca_bcd_benchmark_exp.complexity import run_complexity_study

    out = _ensure_dir(output_dir)
    print("  Running complexity study...")
    res = run_complexity_study(cfg, str(out), seed=cfg.seed)

    fits = res["power_law_fits"]
    flags = []
    for param, fit in fits.items():
        r2 = fit.get("R2", 0.0)
        if r2 < 0.9:
            flags.append(f"FLAG: {param} has R^2={r2:.4f} < 0.9")

    report_lines = [
        "# Complexity Scaling Audit",
        "",
        "## Power-Law Fits",
        "",
        "| Parameter | c | p | R^2 | T = c * N^p |",
        "|-----------|---|----|-----|-------------|",
    ]
    for param, fit in fits.items():
        report_lines.append(
            f"| {param} | {fit['c']:.4e} | {fit['p']:.4f} | "
            f"{fit.get('R2', 0):.4f} | T = {fit['c']:.4e} * N^{fit['p']:.4f} |"
        )

    if flags:
        report_lines += ["", "## FLAGS", ""]
        report_lines += [f"  {f}" for f in flags]
    else:
        report_lines += ["", "## All R^2 >= 0.9. No flags raised.", ""]

    report_lines += [
        "",
        "## Interpretation",
        "",
        "- p close to 1: linear scaling",
        "- p close to 2: quadratic scaling",
        "- p > 2: super-quadratic (may need optimization)",
        "",
    ]

    report_path = out / "complexity_scaling_audit.md"
    Path(report_path).write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "fits": fits,
        "flags": flags,
        "report_path": str(report_path),
        "complexity_result": res,
    }


# ═══════════════════════════════════════════════════════════════
# Part 5: Constraint Activity Analysis
# ═══════════════════════════════════════════════════════════════


def _make_scfg_from_cfg(cfg: BenchmarkConfig, seed: int):
    """Construct SCABCDConfig from BenchmarkConfig with proper defaults."""
    from sca_bcd_exp.configs import SCABCDConfig

    base = {k: v for k, v in asdict(cfg).items()
            if k in SCABCDConfig.__dataclass_fields__}
    base["seed"] = seed
    for missing in ("trust_region_weight", "sca_candidate_step_sizes"):
        if missing not in base:
            base[missing] = SCABCDConfig.__dataclass_fields__[missing].default
    return SCABCDConfig(**base)


def _extract_constraint_violations(
    cfg: BenchmarkConfig, seed: int,
) -> dict | None:
    """Run SCA-BCD and return constraint violations."""
    try:
        from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
        from sca_bcd_exp.optimization.bcd_solver import BCDSolver

        scfg = _make_scfg_from_cfg(cfg, seed)
        env = SCABCDEnvironment(scfg)
        solver = BCDSolver(scfg)
        bcd_res = solver.solve(env)
        r = env.evaluate(bcd_res.solution)
        return {
            **r["violations"],
            "converged": bcd_res.converged,
            "n_iters": bcd_res.n_iters,
        }
    except Exception as e:
        return None


def run_constraint_activity_analysis(
    cfg: BenchmarkConfig,
    output_dir: str,
) -> dict:
    """Across N_mc seeds, compute activation frequency of each constraint."""
    out = _ensure_dir(output_dir)
    N_mc = cfg.N_mc
    activation_tol = 1e-6

    all_violations = []
    print(f"  Running {N_mc} seeds for constraint activity...")
    for s in range(N_mc):
        v = _extract_constraint_violations(cfg, s)
        if v is not None:
            all_violations.append(v)

    if not all_violations:
        return {"error": "No successful runs", "report_path": ""}

    # Compute activation frequency
    constraint_keys = [
        "bs_power_excess", "bs_power_negative", "jammer_power_excess",
        "uav_speed_excess", "uav_boundary_violation",
        "secrecy_rate_shortfall", "sensing_utility_shortfall",
    ]
    rows = []
    for key in constraint_keys:
        vals = np.array([v.get(key, 0.0) for v in all_violations])
        active_count = int(np.sum(vals > activation_tol))
        rows.append({
            "constraint": key,
            "active_count": active_count,
            "total_runs": len(all_violations),
            "activation_frequency": active_count / len(all_violations),
            "mean_violation": float(np.mean(vals)),
            "max_violation": float(np.max(vals)),
            "std_violation": float(np.std(vals)),
        })

    # Convergence stats
    conv_rates = [v.get("converged", False) for v in all_violations]
    n_iters_arr = [v.get("n_iters", 0) for v in all_violations]
    converged_fraction = float(np.mean(conv_rates))

    csv_path = out / "constraint_activity.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    report_lines = [
        "# Constraint Activity Analysis",
        "",
        f"Total runs: {len(all_violations)}, "
        f"Converged: {converged_fraction:.1%}, "
        f"Mean iters: {float(np.mean(n_iters_arr)):.2f}",
        "",
        "| Constraint | Active count | Frequency | Mean viol | Max viol |",
        "|------------|-------------|-----------|-----------|----------|",
    ]
    for r in rows:
        report_lines.append(
            f"| {r['constraint']} | {r['active_count']}/{r['total_runs']} | "
            f"{r['activation_frequency']:.2%} | "
            f"{r['mean_violation']:.4e} | {r['max_violation']:.4e} |"
        )

    # Flag dead constraints
    dead = [r for r in rows if r["activation_frequency"] == 0.0]
    active_all = [r for r in rows if r["activation_frequency"] == 1.0]
    report_lines += [
        "",
        "## Dead Constraints (never active)",
    ]
    if dead:
        for r in dead:
            report_lines.append(f"  {r['constraint']}")
    else:
        report_lines.append("  None — all constraints activate at least occasionally.")

    report_lines += [
        "",
        "## Always Active Constraints",
    ]
    if active_all:
        for r in active_all:
            report_lines.append(f"  {r['constraint']} (freq={r['activation_frequency']:.0%})")
    else:
        report_lines.append("  None.")

    report_lines += [
        "",
        "## Interpretation",
        "",
        "- Dead constraints (0% activation): can potentially be removed",
        "- Always active (100%): likely binding at optimum",
        "- Low activation (<5%): check if constraint is needed",
        "",
    ]

    report_path = out / "constraint_activity_report.md"
    Path(report_path).write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "rows": rows,
        "converged_fraction": converged_fraction,
        "csv_path": str(csv_path),
        "report_path": str(report_path),
    }


# ═══════════════════════════════════════════════════════════════
# Part 6: BCD Block Activity Analysis
# ═══════════════════════════════════════════════════════════════


def _run_sca_bcd_with_block_info(
    cfg: BenchmarkConfig, seed: int,
) -> dict | None:
    """Run SCA-BCD and return objective history + block contributions."""
    try:
        from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
        from sca_bcd_exp.optimization.bcd_solver import BCDSolver

        scfg = _make_scfg_from_cfg(cfg, seed)
        env = SCABCDEnvironment(scfg)
        solver = BCDSolver(scfg)
        bcd_res = solver.solve(env)
        return {
            "objective_history": bcd_res.objective_history,
            "block_contributions": bcd_res.block_contributions,
            "n_iters": bcd_res.n_iters,
            "converged": bcd_res.converged,
        }
    except Exception as e:
        return None


def run_block_activity_analysis(
    cfg: BenchmarkConfig,
    output_dir: str,
) -> dict:
    """Across N_mc seeds, measure average improvement per block."""
    out = _ensure_dir(output_dir)
    N_mc = cfg.N_mc

    block_names = ["power", "trajectory", "jammer"]
    all_contribs = {b: [] for b in block_names}
    all_objs = []

    print(f"  Running {N_mc} seeds for block activity...")
    for s in range(N_mc):
        res = _run_sca_bcd_with_block_info(cfg, s)
        if res is None:
            continue
        contribs = res["block_contributions"]
        for b in block_names:
            if b in contribs:
                all_contribs[b].extend(contribs[b])
        all_objs.append(res["objective_history"])

    # Average improvement per block (summed across iterations, then averaged across seeds)
    block_summary = {}
    for b in block_names:
        vals = np.array(all_contribs[b])
        total_improvement = float(np.sum(vals)) if len(vals) > 0 else 0.0
        mean_impr = float(np.mean(vals)) if len(vals) > 0 else 0.0
        block_summary[b] = {
            "total_improvement": total_improvement,
            "mean_per_iteration": mean_impr,
            "n_observations": len(vals),
        }

    total_all = sum(block_summary[b]["total_improvement"] for b in block_names)
    for b in block_names:
        pct = (block_summary[b]["total_improvement"] / total_all * 100
               if abs(total_all) > 1e-15 else 0.0)
        block_summary[b]["percent_of_total"] = pct

    flags = []
    for b in block_names:
        if block_summary[b]["percent_of_total"] < 1.0:
            flags.append(f"FLAG: {b} contributes only {block_summary[b]['percent_of_total']:.2f}% of total improvement")

    csv_path = out / "block_activity.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["block", "total_improvement",
                                           "mean_per_iteration",
                                           "percent_of_total", "n_observations"])
        w.writeheader()
        for b in block_names:
            w.writerow({"block": b, **block_summary[b]})

    report_lines = [
        "# BCD Block Activity Report",
        "",
        f"Total runs: {len(all_objs)}",
        "",
        "| Block | Total improvement | Mean/iteration | % of total | Observations |",
        "|-------|-------------------|----------------|------------|--------------|",
    ]
    for b in block_names:
        bs = block_summary[b]
        report_lines.append(
            f"| {b} | {bs['total_improvement']:.6f} | "
            f"{bs['mean_per_iteration']:.6f} | "
            f"{bs['percent_of_total']:.2f}% | "
            f"{bs['n_observations']} |"
        )

    if flags:
        report_lines += ["", "## FLAGS", ""]
        report_lines += [f"  {f}" for f in flags]
    else:
        report_lines += ["", "## All blocks contribute >= 1% of total improvement.", ""]

    report_lines += [
        "",
        "## Interpretation",
        "",
        "- Blocks with < 1% contribution may be candidates for removal or reduced effort.",
        "- If a block contributes negatively, check monotonicity of BCD.",
        "",
    ]

    report_path = out / "block_activity_report.md"
    Path(report_path).write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "block_summary": block_summary,
        "flags": flags,
        "csv_path": str(csv_path),
        "report_path": str(report_path),
    }


# ═══════════════════════════════════════════════════════════════
# Part 7: Local Optimum Sensitivity
# ═══════════════════════════════════════════════════════════════


def run_local_optimum_sensitivity(
    cfg: BenchmarkConfig,
    output_dir: str,
    n_random_inits: int = 20,
) -> dict:
    """Run SCA-BCD with different random seeds, measure objective variance."""
    out = _ensure_dir(output_dir)

    objectives = []
    secrecies = []
    sensings = []
    runtimes = []

    print(f"  Running SCA-BCD with {n_random_inits} random seeds...")
    for s in range(n_random_inits):
        t0 = time.perf_counter()
        try:
            r = run_baseline(BaselineMethod.SCA_BCD_FULL, cfg, seed=s)
            dt = time.perf_counter() - t0
            objectives.append(r.objective)
            secrecies.append(r.secrecy_rate)
            sensings.append(r.sensing_utility)
            runtimes.append(dt)
        except Exception as e:
            print(f"    seed {s} failed: {e}")

    objs = np.array(objectives)
    secs = np.array(secrecies)
    sens = np.array(sensings)

    stats = {
        "objective_mean": float(np.mean(objs)),
        "objective_std": float(np.std(objs)),
        "objective_cv": float(np.std(objs) / max(abs(np.mean(objs)), 1e-15)),
        "objective_min": float(np.min(objs)),
        "objective_max": float(np.max(objs)),
        "objective_range": float(np.max(objs) - np.min(objs)),
        "secrecy_mean": float(np.mean(secs)),
        "secrecy_std": float(np.std(secs)),
        "sensing_mean": float(np.mean(sens)),
        "sensing_std": float(np.std(sens)),
        "n_success": len(objectives),
        "n_requested": n_random_inits,
    }

    # CSV
    csv_path = out / "local_optima.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "objective", "secrecy", "sensing", "runtime"])
        w.writeheader()
        for i in range(len(objectives)):
            w.writerow({"seed": i, "objective": objectives[i],
                        "secrecy": secrecies[i], "sensing": sensings[i],
                        "runtime": runtimes[i]})

    # Histogram
    plt = _import_plt()
    if plt:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        ax = axes[0]
        ax.hist(objectives, bins=min(10, len(set(objectives))), color="steelblue", edgecolor="k")
        ax.axvline(float(np.mean(objs)), color="red", linestyle="--", label=f"mean={np.mean(objs):.4f}")
        ax.set_xlabel("Objective"); ax.set_ylabel("Count"); ax.set_title("Objective distribution")
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

        ax = axes[1]
        ax.hist(secrecies, bins=min(10, len(set(secrecies))), color="coral", edgecolor="k")
        ax.axvline(float(np.mean(secs)), color="red", linestyle="--", label=f"mean={np.mean(secs):.4f}")
        ax.set_xlabel("Secrecy rate"); ax.set_ylabel("Count"); ax.set_title("Secrecy distribution")
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

        ax = axes[2]
        ax.hist(sensings, bins=min(10, len(set(sensings))), color="mediumpurple", edgecolor="k")
        ax.axvline(float(np.mean(sens)), color="red", linestyle="--", label=f"mean={np.mean(sens):.4f}")
        ax.set_xlabel("Sensing utility"); ax.set_ylabel("Count"); ax.set_title("Sensing distribution")
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

        fig.suptitle(f"Local optimum sensitivity ({n_random_inits} random init)", fontsize=13)
        fig.tight_layout()
        fig.savefig(str(out / "local_optima_histograms.png"), dpi=150)
        plt.close(fig)

    report_lines = [
        "# Local Optimum Sensitivity Report",
        "",
        f"Random initializations: {n_random_inits} (successful: {len(objectives)})",
        "",
        "## Objective",
        f"  Mean: {stats['objective_mean']:.6f}",
        f"  Std:  {stats['objective_std']:.6f}",
        f"  CV:   {stats['objective_cv']:.4f}",
        f"  Range: [{stats['objective_min']:.6f}, {stats['objective_max']:.6f}]",
        "",
        "## Secrecy Rate",
        f"  Mean: {stats['secrecy_mean']:.6f}",
        f"  Std:  {stats['secrecy_std']:.6f}",
        "",
        "## Sensing Utility",
        f"  Mean: {stats['sensing_mean']:.6f}",
        f"  Std:  {stats['sensing_std']:.6f}",
        "",
        "## Interpretation",
        "",
    ]
    if stats["objective_cv"] > 0.1:
        report_lines.append(
            f"  CV={stats['objective_cv']:.2%} > 10%: "
            "High sensitivity to initialization. Landscape is highly nonconvex."
        )
    else:
        report_lines.append(
            f"  CV={stats['objective_cv']:.2%} <= 10%: "
            "Low sensitivity. Landscape is relatively well-behaved."
        )
    report_lines.append("")

    report_path = out / "local_optima_report.md"
    Path(report_path).write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "stats": stats,
        "objectives": objectives,
        "secrecies": secrecies,
        "sensings": sensings,
        "csv_path": str(csv_path),
        "report_path": str(report_path),
    }


# ═══════════════════════════════════════════════════════════════
# Part 8: Reproducibility
# ═══════════════════════════════════════════════════════════════


def run_reproducibility_check(
    cfg: BenchmarkConfig,
    output_dir: str,
    n_repeats: int = 3,
) -> dict:
    """Verify same seed => identical, different seeds => variation."""
    out = _ensure_dir(output_dir)

    # Same seed repeated
    print("  Checking same-seed reproducibility...")
    same_seed_results = []
    for _ in range(n_repeats):
        r = run_baseline(BaselineMethod.SCA_BCD_FULL, cfg, seed=cfg.seed)
        same_seed_results.append(r)

    objs_same = [r.objective for r in same_seed_results]
    secs_same = [r.secrecy_rate for r in same_seed_results]
    sens_same = [r.sensing_utility for r in same_seed_results]

    same_seed_identical = (
        len(set(objs_same)) == 1 and
        len(set(secs_same)) == 1 and
        len(set(sens_same)) == 1
    )

    # Different seeds
    print("  Checking different-seed variation...")
    diff_seed_results = []
    for s in range(n_repeats):
        r = run_baseline(BaselineMethod.SCA_BCD_FULL, cfg, seed=cfg.seed + s + 1)
        diff_seed_results.append(r)

    objs_diff = [r.objective for r in diff_seed_results]
    diff_seed_varied = len(set([f"{x:.10f}" for x in objs_diff])) > 1

    csv_path = out / "reproducibility.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run_type", "run_index", "seed",
                                           "objective", "secrecy", "sensing"])
        w.writeheader()
        for i, r in enumerate(same_seed_results):
            w.writerow({"run_type": "same_seed", "run_index": i,
                        "seed": cfg.seed, "objective": r.objective,
                        "secrecy": r.secrecy_rate, "sensing": r.sensing_utility})
        for i, r in enumerate(diff_seed_results):
            w.writerow({"run_type": "diff_seed", "run_index": i,
                        "seed": cfg.seed + i + 1, "objective": r.objective,
                        "secrecy": r.secrecy_rate, "sensing": r.sensing_utility})

    report_lines = [
        "# Reproducibility Report",
        "",
        f"Same seed ({cfg.seed}) repeated {n_repeats}x:",
        f"  Objectives: {objs_same}",
        f"  Secrecies:  {secs_same}",
        f"  Sensings:   {sens_same}",
        f"  Identical:  {same_seed_identical}",
        "",
        f"Different seeds ({cfg.seed+1}..{cfg.seed+n_repeats}):",
        f"  Objectives: {objs_diff}",
        f"  Varied:     {diff_seed_varied}",
        "",
    ]
    if same_seed_identical:
        report_lines.append("PASS: Same seed produces identical results.")
    else:
        report_lines.append("FAIL: Same seed produces DIFFERENT results (non-deterministic).")
    if diff_seed_varied:
        report_lines.append("PASS: Different seeds produce varied results.")
    else:
        report_lines.append("FAIL: Different seeds produce identical results (unexpected).")
    report_lines.append("")

    report_path = out / "reproducibility_report.md"
    Path(report_path).write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "same_seed_identical": same_seed_identical,
        "diff_seed_varied": diff_seed_varied,
        "same_seed_results": same_seed_results,
        "diff_seed_results": diff_seed_results,
        "csv_path": str(csv_path),
        "report_path": str(report_path),
    }


# ═══════════════════════════════════════════════════════════════
# Part 9: Final Audit Report
# ═══════════════════════════════════════════════════════════════


def generate_final_audit_report(
    output_dir: str,
    ranking_result: dict | None = None,
    statsig_result: dict | None = None,
    pareto_result: dict | None = None,
    complexity_result: dict | None = None,
    constraint_result: dict | None = None,
    block_result: dict | None = None,
    local_opt_result: dict | None = None,
    repro_result: dict | None = None,
) -> str:
    """Generate the comprehensive final audit report."""
    out = _ensure_dir(output_dir)

    lines = [
        "# Phase 5 Final Audit Report",
        "",
        "## 1. Numerical Robustness",
        "",
    ]

    # 1. Numerical robustness
    num_issues = []
    if ranking_result:
        flags = ranking_result.get("ranking_flags", [])
        if flags:
            num_issues.extend(flags)
    if pareto_result:
        if not pareto_result.get("sec_noninc", True):
            num_issues.append("Pareto secrecy is not monotonic.")
        if not pareto_result.get("sens_nondec", True):
            num_issues.append("Pareto sensing is not monotonic.")

    if num_issues:
        lines += ["  Issues found:"] + [f"  - {x}" for x in num_issues]
    else:
        lines.append("  No numerical issues detected.")

    # 2. Statistical robustness
    lines += ["", "## 2. Statistical Robustness", ""]
    if statsig_result:
        rows = statsig_result.get("rows", [])
        sig_count = sum(1 for r in rows if r.get("t_p_value", 1) < 0.05)
        large_effect = sum(1 for r in rows if abs(r.get("cohens_d", 0)) > 0.8)
        lines.append(f"  {sig_count}/{len(rows)} comparisons statistically significant (p<0.05)")
        lines.append(f"  {large_effect}/{len(rows)} with large effect size (|d|>0.8)")

    # 3. Benchmark validity
    lines += ["", "## 3. Benchmark Validity", ""]
    if ranking_result:
        flags = ranking_result.get("ranking_flags", [])
        if flags:
            lines.append("  WARNING: Invalid ranking detected:")
            for f in flags:
                lines.append(f"    {f}")
        else:
            lines.append("  Expected ordering: Random < Single-block < Ablation < Full SCA-BCD")
            lines.append("  VALID: Ranking is as expected.")

    # 4. Constraint activity
    lines += ["", "## 4. Constraint Activity", ""]
    if constraint_result:
        rows = constraint_result.get("rows", [])
        dead = [r for r in rows if r.get("activation_frequency", 1) == 0.0]
        always = [r for r in rows if r.get("activation_frequency", 0) == 1.0]
        lines.append(f"  Dead constraints (0% activation): {[r['constraint'] for r in dead]}")
        lines.append(f"  Always active (100%): {[r['constraint'] for r in always]}")
        lines.append(f"  Converged fraction: {constraint_result.get('converged_fraction', 0):.1%}")

    # 5. Block activity
    lines += ["", "## 5. BCD Block Activity", ""]
    if block_result:
        flags = block_result.get("flags", [])
        summary = block_result.get("block_summary", {})
        for b, s in summary.items():
            lines.append(f"  {b}: {s.get('percent_of_total', 0):.1f}% of total improvement")
        if flags:
            lines.append("  WARNING: Inactive blocks:")
            for f in flags:
                lines.append(f"    {f}")
        else:
            lines.append("  All blocks contribute >= 1%.")

    # 6. Recommendation
    lines += ["", "## 6. Recommendation", ""]

    all_flags = []
    if ranking_result:
        all_flags.extend(ranking_result.get("ranking_flags", []))
    if pareto_result:
        if not pareto_result.get("sec_noninc", True):
            all_flags.append("Pareto secrecy non-monotonic")
        if not pareto_result.get("sens_nondec", True):
            all_flags.append("Pareto sensing non-monotonic")
    if complexity_result:
        all_flags.extend(complexity_result.get("flags", []))
    if block_result:
        all_flags.extend(block_result.get("flags", []))
    if constraint_result:
        rows = constraint_result.get("rows", [])
        high_viol = [r for r in rows if r.get("max_violation", 0) > 1.0]
        for r in high_viol:
            all_flags.append(f"High violation: {r['constraint']} max={r['max_violation']:.4e}")
    if repro_result:
        if not repro_result.get("same_seed_identical", True):
            all_flags.append("Non-deterministic: same seed gives different results")
        if not repro_result.get("diff_seed_varied", True):
            all_flags.append("Different seeds give identical results")

    if all_flags:
        lines += ["  Issues requiring attention:"]
        for f in all_flags:
            lines.append(f"  - {f}")
        lines.append("")
        lines.append("  Recommendation: Fix issues before proceeding to Phase 5D.")
    else:
        lines.append("  All checks passed.")
        lines.append("")
        lines.append("  Recommendation: Proceed to Phase 5D (multi-antenna BS upgrade).")

    lines.append("")

    report_path = out / "phase5_final_audit.md"
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"  Final audit report: {report_path}")
    return str(report_path)


# ═══════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════


def run_final_audit(
    cfg: BenchmarkConfig | None = None,
    output_dir: str = "outputs/optimization/reports/final_audit",
    skip_parts: list[str] | None = None,
) -> dict:
    """Run all audit parts."""
    if cfg is None:
        cfg = BenchmarkConfig(
            seed=0, max_bcd_iters=5, max_sca_iters=3, N_mc=20,
        )
    if skip_parts is None:
        skip_parts = []

    out = _ensure_dir(output_dir)
    results = {}

    total_start = time.perf_counter()

    # Part 1
    if "ranking" not in skip_parts:
        print("\n=== Part 1: Benchmark Ranking Sanity ===")
        results["ranking"] = run_benchmark_ranking_sanity(cfg, str(out / "ranking"))

    # Part 2
    if "statsig" not in skip_parts:
        print("\n=== Part 2: Statistical Significance ===")
        results["statsig"] = run_statistical_significance(cfg, str(out / "statsig"))

    # Part 3
    if "pareto" not in skip_parts:
        print("\n=== Part 3: Pareto Front Quality ===")
        results["pareto"] = run_pareto_quality_audit(cfg, str(out / "pareto"))

    # Part 4
    if "complexity" not in skip_parts:
        print("\n=== Part 4: Complexity Scaling Audit ===")
        results["complexity"] = run_complexity_scaling_audit(cfg, str(out / "complexity"))

    # Part 5
    if "constraint" not in skip_parts:
        print("\n=== Part 5: Constraint Activity ===")
        results["constraint"] = run_constraint_activity_analysis(cfg, str(out / "constraint"))

    # Part 6
    if "block" not in skip_parts:
        print("\n=== Part 6: BCD Block Activity ===")
        results["block"] = run_block_activity_analysis(cfg, str(out / "block"))

    # Part 7
    if "local_optima" not in skip_parts:
        print("\n=== Part 7: Local Optimum Sensitivity ===")
        results["local_optima"] = run_local_optimum_sensitivity(cfg, str(out / "local_optima"))

    # Part 8
    if "reproducibility" not in skip_parts:
        print("\n=== Part 8: Reproducibility ===")
        results["reproducibility"] = run_reproducibility_check(cfg, str(out / "reproducibility"))

    # Part 9
    print("\n=== Part 9: Final Audit Report ===")
    results["report_path"] = generate_final_audit_report(
        str(out),
        ranking_result=results.get("ranking"),
        statsig_result=results.get("statsig"),
        pareto_result=results.get("pareto"),
        complexity_result=results.get("complexity"),
        constraint_result=results.get("constraint"),
        block_result=results.get("block"),
        local_opt_result=results.get("local_optima"),
        repro_result=results.get("reproducibility"),
    )

    total_elapsed = time.perf_counter() - total_start
    print(f"\nTotal audit time: {total_elapsed:.1f}s")
    print(f"Output directory: {out}")

    results["elapsed_s"] = total_elapsed
    results["output_dir"] = str(out)
    return results


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Phase 5 Final Audit")
    parser.add_argument("--N_mc", type=int, default=20, help="MC seeds (default: 20)")
    parser.add_argument("--max_bcd_iters", type=int, default=5)
    parser.add_argument("--max_sca_iters", type=int, default=3)
    parser.add_argument("--output", type=str, default="outputs/optimization/reports/final_audit")
    parser.add_argument("--skip", type=str, nargs="+", default=[])
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        seed=0,
        max_bcd_iters=args.max_bcd_iters,
        max_sca_iters=args.max_sca_iters,
        N_mc=args.N_mc,
    )
    run_final_audit(cfg, output_dir=args.output, skip_parts=args.skip)


if __name__ == "__main__":
    main()
