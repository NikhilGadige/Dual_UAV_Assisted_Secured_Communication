from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from sca_bcd_benchmark_exp.baselines import (
    BaselineMethod,
    BaselineResult,
    run_baseline,
)
from sca_bcd_benchmark_exp.configs import BenchmarkConfig


@dataclass
class MCSummary:
    method: str
    N_mc: int
    objective_mean: float
    objective_std: float
    objective_median: float
    objective_p5: float
    objective_p95: float
    secrecy_mean: float
    secrecy_std: float
    secrecy_median: float
    secrecy_p5: float
    secrecy_p95: float
    sensing_mean: float
    sensing_std: float
    sensing_median: float
    sensing_p5: float
    sensing_p95: float
    runtime_mean: float
    runtime_std: float
    n_iters_mean: float
    violation_mean: float
    converged_fraction: float
    raw_objectives: list[float] = field(default_factory=list)
    raw_secrecies: list[float] = field(default_factory=list)
    raw_sensing: list[float] = field(default_factory=list)
    raw_runtimes: list[float] = field(default_factory=list)


def _percentile(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p))


def aggregate(results: list[BaselineResult]) -> MCSummary:
    objs = np.array([r.objective for r in results])
    secs = np.array([r.secrecy_rate for r in results])
    sens = np.array([r.sensing_utility for r in results])
    rt = np.array([r.runtime_s for r in results])
    iters = np.array([r.n_iterations for r in results])
    viols = np.array([r.violation_total for r in results])
    conv = np.mean([1.0 if r.converged else 0.0 for r in results])
    method = results[0].method if results else ""

    return MCSummary(
        method=method,
        N_mc=len(results),
        objective_mean=float(np.mean(objs)),
        objective_std=float(np.std(objs)),
        objective_median=float(np.median(objs)),
        objective_p5=_percentile(objs, 5),
        objective_p95=_percentile(objs, 95),
        secrecy_mean=float(np.mean(secs)),
        secrecy_std=float(np.std(secs)),
        secrecy_median=float(np.median(secs)),
        secrecy_p5=_percentile(secs, 5),
        secrecy_p95=_percentile(secs, 95),
        sensing_mean=float(np.mean(sens)),
        sensing_std=float(np.std(sens)),
        sensing_median=float(np.median(sens)),
        sensing_p5=_percentile(sens, 5),
        sensing_p95=_percentile(sens, 95),
        runtime_mean=float(np.mean(rt)),
        runtime_std=float(np.std(rt)),
        n_iters_mean=float(np.mean(iters)),
        violation_mean=float(np.mean(viols)),
        converged_fraction=float(conv),
        raw_objectives=objs.tolist(),
        raw_secrecies=secs.tolist(),
        raw_sensing=sens.tolist(),
        raw_runtimes=rt.tolist(),
    )


def evaluate_baseline_mc(
    method: BaselineMethod | str,
    cfg: BenchmarkConfig,
    N_mc: int | None = None,
    quiet: bool = False,
) -> MCSummary:
    if N_mc is None:
        N_mc = cfg.N_mc
    results = []
    seeds = list(range(N_mc))
    for s in seeds:
        if not quiet:
            print(f"    seed {s + 1}/{N_mc}", end="\r")
        try:
            r = run_baseline(method, cfg, seed=s)
            results.append(r)
        except Exception as e:
            if not quiet:
                print(f"    seed {s} FAILED: {e}")
            continue
    if not quiet:
        print()
    return aggregate(results)


def run_mc_evaluation(
    cfg: BenchmarkConfig,
    methods: list[BaselineMethod | str] | None = None,
    quiet: bool = False,
) -> dict[str, MCSummary]:
    if methods is None:
        methods = list(BaselineMethod)

    summaries = {}
    for meth in methods:
        if isinstance(meth, str):
            meth = BaselineMethod(meth)
        label = meth.value
        if not quiet:
            print(f"\n[{label}] Running {cfg.N_mc} MC seeds...")
        summaries[label] = evaluate_baseline_mc(meth, cfg, quiet=quiet)

    return summaries
