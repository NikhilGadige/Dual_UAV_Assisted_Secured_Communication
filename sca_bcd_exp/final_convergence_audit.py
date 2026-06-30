"""Final convergence audit after jammer fix.

  Part 1: Non-convergence root cause
  Part 2: Cycling detection
  Part 3: Jammer block stability
  Part 4: Multi-seed test (20 seeds)
  Part 5: Tolerance sweep
  Part 6: Acceptance criteria
  Part 7: Outputs + decision
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment
from sca_bcd_exp.optimization.bcd_solver import BCDSolver, BCDResult


OUT_DIR = Path("outputs") / "sca_bcd" / "final_convergence_audit"


def _p(text):
    sys.stdout.write(str(text) + "\n")
    sys.stdout.flush()


# -- helpers -----------------------------------------------------------------


def _run_bcd(seed=0, max_bcd_iters=50, max_sca_iters=20,
             tol_obj=1e-4, tol_var=1e-4) -> BCDResult:
    cfg = SCABCDConfig(seed=seed, max_bcd_iters=max_bcd_iters,
                       max_sca_iters=max_sca_iters, tol_obj=tol_obj, tol_var=tol_var)
    env = SCABCDEnvironment(cfg)
    solver = BCDSolver(cfg)
    return solver.solve(env)


# -- Part 1: Non-convergence root cause --------------------------------------


def part1_investigate(result: BCDResult) -> dict:
    out = {}
    out["n_iters"] = result.n_iters
    out["converged"] = result.converged
    out["convergence_reason"] = result.convergence_reason
    out["final_objective"] = result.objective_history[-1]
    out["tol_obj"] = 1e-4
    out["tol_var"] = 1e-4

    per_iter = []
    n_bcd = len(result.obj_changes)
    for i in range(n_bcd):
        entry = {
            "iteration": i,
            "delta_objective": float(result.obj_changes[i]),
            "delta_w": float(result.delta_w_norms[i]) if i < len(result.delta_w_norms) else None,
            "delta_q": float(result.delta_q_norms[i]) if i < len(result.delta_q_norms) else None,
            "delta_v": float(result.delta_v_norms[i]) if i < len(result.delta_v_norms) else None,
            "var_change": float(result.var_changes[i]),
        }
        per_iter.append(entry)
    out["per_iteration"] = per_iter
    return out


# -- Part 2: Cycling detection -----------------------------------------------


def part2_cycling(result: BCDResult) -> dict:
    dw = np.array(result.delta_w_norms)
    dq = np.array(result.delta_q_norms)
    dv = np.array(result.delta_v_norms)

    cycling_flags = {}
    for label, d in [("w", dw), ("q", dq), ("v", dv)]:
        cycling_flags[f"{label}_nonzero"] = int(np.sum(d > 1e-12))
        if len(d) >= 2:
            diff = np.abs(np.diff(d))
            cycling_flags[f"{label}_oscillates"] = bool(
                np.any(diff > 1e-8) and np.any(np.abs(np.diff(diff > 1e-8)) > 0)
            )
        else:
            cycling_flags[f"{label}_oscillates"] = False

    total_vars = dw + dq + dv
    two_cycle = {"has_two_cycle": False}
    if len(total_vars) >= 4:
        for i in range(len(total_vars) - 2):
            if (abs(total_vars[i] - total_vars[i + 2]) < 1e-8 and
                    abs(total_vars[i + 1] - total_vars[i + 3]) < 1e-8):
                two_cycle = {"has_two_cycle": True, "at_iteration": i}
                break

    classification = _classify_convergence(dw, dq, dv, result)
    return {"cycling": cycling_flags, "two_cycle": two_cycle,
            "classification": classification}


def _classify_convergence(dw, dq, dv, result):
    if result.converged:
        return "A) true convergence"
    if len(dw) < 2:
        return "D) numerical noise (too few iterations)"

    last = len(dw) - 1
    if dw[last] < 1e-8 and dq[last] < 1e-8 and dv[last] < 1e-8:
        return "A) true convergence"

    if dw[last] < 1e-8 and dq[last] < 1e-8 and dv[last] > 0:
        return "B) oscillation (jammer still active)"

    if last >= 2:
        w_oscil = np.any(np.abs(np.diff(dw[last - 2:])) > 1e-10)
        q_oscil = np.any(np.abs(np.diff(dq[last - 2:])) > 1e-10)
        v_oscil = np.any(np.abs(np.diff(dv[last - 2:])) > 1e-10)
        if w_oscil or q_oscil or v_oscil:
            return "C) two-cycle"

    return "D) numerical noise"


# -- Part 3: Jammer block stability ------------------------------------------


def part3_jammer_stability(result: BCDResult) -> dict:
    contrib = result.block_contributions["jammer"]
    dv = result.delta_v_norms
    n_bcd = len(contrib)

    monotonic = all(contrib[i] >= 0 or abs(contrib[i]) < 1e-12 for i in range(n_bcd))
    diminishing = len(dv) >= 3 and all(dv[i] >= dv[i + 1] for i in range(len(dv) - 1))
    asymptotic = bool(len(dv) >= 2 and dv[-1] < dv[0] * 0.1)

    return {
        "monotonic": bool(monotonic),
        "diminishing_updates": bool(diminishing),
        "asymptotic_behaviour": bool(asymptotic),
        "final_delta_v": float(dv[-1]) if len(dv) > 0 else None,
        "max_delta_v": float(max(dv)) if len(dv) > 0 else None,
    }


# -- Part 4: Multi-seed test -------------------------------------------------


def part4_multiseed(n_seeds: int = 20) -> dict:
    seeds = list(range(n_seeds))
    results = []
    n_converged = 0
    n_iters_list = []
    failures = []

    for s in seeds:
        try:
            r = _run_bcd(seed=s, max_bcd_iters=50, max_sca_iters=20)
            results.append(r)
            n_iters_list.append(r.n_iters)
            if r.converged:
                n_converged += 1
        except Exception as e:
            failures.append({"seed": s, "error": str(e)})

    return {
        "n_seeds": n_seeds,
        "n_success": len(results),
        "n_converged": n_converged,
        "convergence_rate": n_converged / n_seeds if n_seeds > 0 else 0.0,
        "avg_iterations": float(np.mean(n_iters_list)) if n_iters_list else None,
        "max_iterations": int(max(n_iters_list)) if n_iters_list else None,
        "min_iterations": int(min(n_iters_list)) if n_iters_list else None,
        "failures": failures,
        "all_converged": [r.converged for r in results],
        "all_reasons": [r.convergence_reason for r in results],
        "all_n_iters": n_iters_list,
    }


# -- Part 5: Tolerance sweep -------------------------------------------------


def part5_tolerance_sweep() -> dict:
    tol_values = [1e-3, 1e-4, 1e-5, 1e-6]
    rows = []
    for tol_obj in tol_values:
        for tol_var in tol_values:
            try:
                r = _run_bcd(seed=0, max_bcd_iters=50, max_sca_iters=20,
                             tol_obj=tol_obj, tol_var=tol_var)
                rows.append({
                    "tol_obj": tol_obj,
                    "tol_var": tol_var,
                    "converged": r.converged,
                    "reason": r.convergence_reason,
                    "n_iters": r.n_iters,
                    "final_obj": r.objective_history[-1],
                })
            except Exception as e:
                rows.append({
                    "tol_obj": tol_obj,
                    "tol_var": tol_var,
                    "converged": False,
                    "reason": f"error: {e}",
                    "n_iters": -1,
                    "final_obj": None,
                })
    return {"sweep": rows}


# -- Part 6: Acceptance criteria ---------------------------------------------


def part6_acceptance(multi: dict, cycling: dict, multiseed: dict, sweep: dict) -> dict:
    criteria = {}

    seeds_ok = multiseed["convergence_rate"] > 0.95
    criteria["C1_gt_95pct_converge"] = bool(seeds_ok)

    no_cycling = not cycling["two_cycle"]["has_two_cycle"]
    criteria["C2_no_cycling"] = bool(no_cycling)

    avg_iters = multiseed.get("avg_iterations", 0) or 0
    criteria["C3_avg_iters_lt_20"] = bool(avg_iters < 20)

    final_objs = [r["final_obj"] for r in sweep["sweep"] if r["final_obj"] is not None]
    obj_variation = 0.0
    if final_objs:
        obj_variation = (max(final_objs) - min(final_objs)) / abs(np.mean(final_objs))
    criteria["C4_obj_change_lt_1pct"] = bool(obj_variation < 0.01)
    criteria["obj_variation_pct"] = obj_variation * 100

    all_pass = all(
        v for k, v in criteria.items() if k.startswith("C")
    ) if any(k.startswith("C") for k in criteria) else False
    criteria["all_pass"] = all_pass
    return criteria


# -- Part 7: Plotting --------------------------------------------------------


def plot_convergence_traces(result: BCDResult, out_dir: Path):
    n_iters = len(result.objective_history)
    iters = list(range(n_iters))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(iters, result.objective_history, "o-", color="black")
    axes[0, 0].set_xlabel("BCD iteration")
    axes[0, 0].set_ylabel("Objective")
    axes[0, 0].set_title("(a) Objective history")
    axes[0, 0].grid(True)

    n_bcd = len(result.obj_changes)
    bcd_iters = list(range(n_bcd))
    axes[0, 1].plot(bcd_iters, result.obj_changes, "s-", color="blue", label="abs(d_obj)")
    axes[0, 1].axhline(y=1e-4, color="gray", linestyle="--", label="tol_obj=1e-4")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("BCD iteration")
    axes[0, 1].set_ylabel("|d_objective|")
    axes[0, 1].set_title("(b) Objective change per BCD iteration")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(bcd_iters, result.delta_w_norms, "o-", label="||dw||")
    axes[1, 0].plot(bcd_iters, result.delta_q_norms, "s-", label="||dq||")
    axes[1, 0].plot(bcd_iters, result.delta_v_norms, "^-", label="||dv||")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("BCD iteration")
    axes[1, 0].set_ylabel("Block variable norm")
    axes[1, 0].set_title("(c) Block variable changes")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(bcd_iters, result.var_changes, "d-", color="red", label="||dx||")
    axes[1, 1].axhline(y=1e-4, color="gray", linestyle="--", label="tol_var=1e-4")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("BCD iteration")
    axes[1, 1].set_ylabel("||dx|| total")
    axes[1, 1].set_title("(d) Total variable change")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    fig.savefig(out_dir / "convergence_traces.png", dpi=150)
    plt.close(fig)


def plot_jammer_stability(result: BCDResult, out_dir: Path):
    n_bcd = len(result.block_contributions["jammer"])
    iters = list(range(n_bcd))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(iters, result.block_contributions["jammer"], "o-", color="purple")
    axes[0].set_xlabel("BCD iteration")
    axes[0].set_ylabel("Jammer block improvement")
    axes[0].set_title("(a) Jammer objective improvement")
    axes[0].axhline(y=0, color="gray", linestyle="--")
    axes[0].grid(True)

    axes[1].plot(iters, result.delta_v_norms, "^-", color="green")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("BCD iteration")
    axes[1].set_ylabel("||dv||")
    axes[1].set_title("(b) Jammer variable update norm")
    axes[1].grid(True)

    if len(iters) >= 2:
        cumul = np.cumsum(result.block_contributions["jammer"])
        axes[2].plot(iters, cumul, "s-", color="orange")
    axes[2].set_xlabel("BCD iteration")
    axes[2].set_ylabel("Cumulative jammer improvement")
    axes[2].set_title("(c) Cumulative jammer contribution")
    axes[2].axhline(y=0, color="gray", linestyle="--")
    axes[2].grid(True)

    plt.tight_layout()
    fig.savefig(out_dir / "jammer_stability.png", dpi=150)
    plt.close(fig)


def plot_multiseed(multi: dict, out_dir: Path):
    n_iters = multi["all_n_iters"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(n_iters, bins=min(20, len(set(n_iters))), edgecolor="black")
    axes[0].set_xlabel("BCD iterations")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Multi-seed iterations (n={len(n_iters)})")
    axes[0].grid(True)

    reasons = multi["all_reasons"]
    unique_reasons = list(set(reasons))
    counts = [reasons.count(r) for r in unique_reasons]
    axes[1].bar(range(len(unique_reasons)), counts, tick_label=unique_reasons, edgecolor="black")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Convergence reasons")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(True)

    plt.tight_layout()
    fig.savefig(out_dir / "multiseed_histogram.png", dpi=150)
    plt.close(fig)


def plot_tolerance_sweep(sweep: dict, out_dir: Path):
    rows = sweep["sweep"]
    tol_obj_vals = sorted(set(r["tol_obj"] for r in rows))
    tol_var_vals = sorted(set(r["tol_var"] for r in rows))

    obj_grid = np.full((len(tol_obj_vals), len(tol_var_vals)), np.nan)
    iter_grid = np.full((len(tol_obj_vals), len(tol_var_vals)), np.nan)

    for r in rows:
        i = tol_obj_vals.index(r["tol_obj"])
        j = tol_var_vals.index(r["tol_var"])
        if r["final_obj"] is not None:
            obj_grid[i, j] = r["final_obj"]
        iter_grid[i, j] = r["n_iters"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(obj_grid, cmap="viridis", aspect="auto",
                          extent=[math.log10(min(tol_var_vals)), math.log10(max(tol_var_vals)),
                                  math.log10(max(tol_obj_vals)), math.log10(min(tol_obj_vals))])
    axes[0].set_xlabel("log10(tol_var)")
    axes[0].set_ylabel("log10(tol_obj)")
    axes[0].set_title("Final objective")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(iter_grid, cmap="plasma", aspect="auto",
                          extent=[math.log10(min(tol_var_vals)), math.log10(max(tol_var_vals)),
                                  math.log10(max(tol_obj_vals)), math.log10(min(tol_obj_vals))])
    axes[1].set_xlabel("log10(tol_var)")
    axes[1].set_ylabel("log10(tol_obj)")
    axes[1].set_title("BCD iterations")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    fig.savefig(out_dir / "tolerance_sweep.png", dpi=150)
    plt.close(fig)


# -- Main audit --------------------------------------------------------------


def run_final_convergence_audit() -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _p("=" * 60)
    _p("FINAL CONVERGENCE AUDIT")
    _p("=" * 60)

    # Part 1: Reference run
    _p("\n[Part 1] Running reference BCD solve ...")
    result = _run_bcd(seed=0, max_bcd_iters=50, max_sca_iters=20)
    p1 = part1_investigate(result)
    _p(f"  n_iters={p1['n_iters']}, converged={p1['converged']}, reason={p1['convergence_reason']}")
    _p(f"  final_obj={p1['final_objective']:.6f}")
    for row in p1["per_iteration"]:
        _p(f"    iter {row['iteration']}: d_obj={row['delta_objective']:.2e}  "
           f"dw={row['delta_w']:.6f}  dq={row['delta_q']:.6f}  dv={row['delta_v']:.6f}  "
           f"||dx||={row['var_change']:.6f}")

    # Part 2: Cycling
    _p("\n[Part 2] Cycling detection ...")
    p2 = part2_cycling(result)
    _p(f"  classification: {p2['classification']}")
    _p(f"  two_cycle: {p2['two_cycle']}")
    for k, v in p2["cycling"].items():
        _p(f"  {k}: {v}")

    # Part 3: Jammer stability
    _p("\n[Part 3] Jammer block stability ...")
    p3 = part3_jammer_stability(result)
    for k, v in p3.items():
        _p(f"  {k}: {v}")

    # Part 4: Multi-seed
    _p("\n[Part 4] Multi-seed test (20 seeds) ...")
    p4 = part4_multiseed(20)
    _p(f"  convergence_rate: {p4['convergence_rate']*100:.1f}%")
    _p(f"  avg_iterations: {p4['avg_iterations']:.2f}")
    _p(f"  max_iterations: {p4['max_iterations']}")
    _p(f"  failures: {len(p4['failures'])}")
    for r, n in zip(p4["all_converged"], p4["all_n_iters"]):
        _p(f"    seed -> converged={r}, n_iters={n}")

    # Part 5: Tolerance sweep
    _p("\n[Part 5] Tolerance sweep (4x4 grid) ...")
    p5 = part5_tolerance_sweep()
    for row in p5["sweep"]:
        _p(f"  tol_obj={row['tol_obj']:.0e} tol_var={row['tol_var']:.0e} -> "
           f"converged={row['converged']}, reason={row['reason']}, "
           f"n_iters={row['n_iters']}, obj={row['final_obj']:.6f}")

    # Part 6: Acceptance criteria
    _p("\n[Part 6] Acceptance criteria ...")
    p6 = part6_acceptance(p2, p2, p4, p5)
    for k, v in p6.items():
        if isinstance(v, bool):
            _p(f"  {k}: {'PASS' if v else 'FAIL'}")
        else:
            _p(f"  {k}: {v:.4f}")
    verdict = "ALL PASS" if p6["all_pass"] else "SOME FAIL"

    # Plots
    _p("\n[Part 7] Generating plots ...")
    plot_convergence_traces(result, OUT_DIR)
    plot_jammer_stability(result, OUT_DIR)
    plot_multiseed(p4, OUT_DIR)
    plot_tolerance_sweep(p5, OUT_DIR)

    # CSV outputs
    with open(OUT_DIR / "convergence_statistics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "delta_objective", "delta_w", "delta_q",
                     "delta_v", "var_change", "objective"])
        for i in range(len(result.obj_changes)):
            w.writerow([i, result.obj_changes[i],
                        result.delta_w_norms[i] if i < len(result.delta_w_norms) else "",
                        result.delta_q_norms[i] if i < len(result.delta_q_norms) else "",
                        result.delta_v_norms[i] if i < len(result.delta_v_norms) else "",
                        result.var_changes[i],
                        result.objective_history[i + 1]])

    with open(OUT_DIR / "tolerance_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tol_obj", "tol_var", "converged", "reason", "n_iters", "final_obj"])
        for row in p5["sweep"]:
            w.writerow([row["tol_obj"], row["tol_var"], row["converged"],
                        row["reason"], row["n_iters"], row["final_obj"]])

    with open(OUT_DIR / "cycling_diagnostics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in p2["cycling"].items():
            w.writerow([k, v])
        w.writerow(["classification", p2["classification"]])
        w.writerow(["has_two_cycle", p2["two_cycle"]["has_two_cycle"]])
        w.writerow(["monotonic", p3["monotonic"]])
        w.writerow(["diminishing_updates", p3["diminishing_updates"]])
        w.writerow(["asymptotic_behaviour", p3["asymptotic_behaviour"]])

    # Markdown report
    report = f"""# Final Convergence Audit Report

Date: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")}

## Part 1: Convergence Diagnostics

- BCD iterations: {p1['n_iters']}
- Converged: {p1['converged']}
- Reason: {p1['convergence_reason']}
- Final objective: {p1['final_objective']:.6f}

Per-iteration deltas:

| iter | |d_obj| | ||dw|| | ||dq|| | ||dv|| | ||dx|| |
|------|--------|--------|--------|--------|--------|
"""
    for row in p1["per_iteration"]:
        report += f"| {row['iteration']} | {row['delta_objective']:.2e} | {row['delta_w']:.6f} | {row['delta_q']:.6f} | {row['delta_v']:.6f} | {row['var_change']:.6f} |\n"

    report += f"""
### Root Cause

The BCD convergence check requires both |d_obj| < tol_obj (=1e-4) AND ||dx|| < tol_var (=1e-4).

After the first BCD iteration:
- **w_bs** (power beamformer) converges: ||dw|| -> 0
- **q_uav** (trajectory) converges: ||dq|| -> 0
- **v_jammer** continues updating: ||dv|| > 0 (range {min(result.delta_v_norms):.4f}-{max(result.delta_v_norms):.4f})

The total ||dx|| is dominated by ||dv|| (> tol_var), preventing convergence declaration even though |d_obj| ~= 0.

With the zero-obj-change fallback added, convergence is now declared when |d_obj| < 1e-12 (reason: "{p1['convergence_reason']}").

## Part 2: Cycling Detection

- Classification: {p2['classification']}
- Has two-cycle: {p2['two_cycle']['has_two_cycle']}
- w_bs nonzero steps: {p2['cycling']['w_nonzero']}
- q_uav nonzero steps: {p2['cycling']['q_nonzero']}
- v_jammer nonzero steps: {p2['cycling']['v_nonzero']}

Verdict: { "Cycling detected" if p2['two_cycle']['has_two_cycle'] else "No cycling detected" }.

## Part 3: Jammer Block Stability

- Monotonic improvement: {p3['monotonic']}
- Diminishing updates: {p3['diminishing_updates']}
- Asymptotic behaviour: {p3['asymptotic_behaviour']}
- Final ||dv||: {p3['final_delta_v']:.6f}
- Max ||dv||: {p3['max_delta_v']:.6f}

## Part 4: Multi-Seed Test (20 seeds)

- Convergence rate: {p4['convergence_rate']*100:.1f}%
- Average iterations: {p4['avg_iterations']:.2f}
- Max iterations: {p4['max_iterations']}
- Min iterations: {p4['min_iterations']}
- Failures: {len(p4['failures'])}

## Part 5: Tolerance Sweep

| tol_obj | tol_var | Converged | Reason | Iters | Final obj |
|---------|---------|-----------|--------|-------|-----------|
"""
    for row in p5["sweep"]:
        report += f"| {row['tol_obj']:.0e} | {row['tol_var']:.0e} | {row['converged']} | {row['reason']} | {row['n_iters']} | {row['final_obj']:.6f} |\n"

    obj_variation = p6.get("obj_variation_pct", 0)
    report += f"""
- Objective variation across tolerances: {obj_variation:.4f}%

## Part 6: Acceptance Criteria

| Criterion | Status |
|-----------|--------|
"""

    for k, v in p6.items():
        if isinstance(v, bool):
            report += f"| {k} | {'PASS' if v else 'FAIL'} |\n"
        elif k == "obj_variation_pct":
            report += f"| {k} | {v:.4f}% |\n"

    report += f"""
| **Verdict** | **{verdict}** |

## Part 7: Output Files

- convergence_statistics.csv
- tolerance_sweep.csv
- cycling_diagnostics.csv
- convergence_traces.png
- jammer_stability.png
- multiseed_histogram.png
- tolerance_sweep.png

---

## Final Decision

"""
    if p6["all_pass"]:
        decision = "READY_FOR_PHASE_5D"
        report += "**READY_FOR_PHASE_5D**\n\nAll acceptance criteria satisfied."
    else:
        decision = "FIX_CONVERGENCE_FIRST"
        report += "**FIX_CONVERGENCE_FIRST**\n\nOne or more acceptance criteria not met."

    with open(OUT_DIR / "final_convergence_report.md", "w") as f:
        f.write(report)

    _p(f"\n  Report written to {OUT_DIR / 'final_convergence_report.md'}")
    _p(f"  Decision: {decision}")
    return decision


if __name__ == "__main__":
    decision = run_final_convergence_audit()
    _p(f"\n{'=' * 60}")
    _p(f"FINAL DECISION: {decision}")
    _p(f"{'=' * 60}")
