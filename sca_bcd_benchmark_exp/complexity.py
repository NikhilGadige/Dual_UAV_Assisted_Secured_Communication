from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from sca_bcd_benchmark_exp.baselines import run_baseline, BaselineMethod
from sca_bcd_benchmark_exp.configs import BenchmarkConfig


def _fit_power_law(Ns: list[int], times: list[float]) -> tuple[float, float, float]:
    """Fit T(N) = c * N^p via log-log regression. Returns (c, p, r_squared)."""
    logN = np.log(np.maximum(Ns, 1))
    logT = np.log(np.maximum(times, 1e-15))
    A = np.vstack([logN, np.ones_like(logN)]).T
    coeffs, residuals, *_ = np.linalg.lstsq(A, logT, rcond=None)
    p = float(coeffs[0])
    ln_c = float(coeffs[1])
    c = float(np.exp(ln_c))
    # R²
    logT_mean = float(np.mean(logT))
    ss_tot = float(np.sum((logT - logT_mean) ** 2))
    ss_res = float(residuals[0]) if len(residuals) > 0 else 0.0
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return c, p, r_sq


def run_complexity_study(
    cfg: BenchmarkConfig,
    output_dir: str,
    seed: int = 0,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── N_RIS scaling ────────────────────────────────────────────
    ris_vals = [8, 16, 32, 64]
    ris_times = []
    ris_iters = []
    ris_mem = []
    for n_ris in ris_vals:
        try:
            mod = BenchmarkConfig(**{**cfg.__dict__, "N_ris": n_ris, "seed": seed})
            t0 = time.perf_counter()
            r = run_baseline(BaselineMethod.SCA_BCD_FULL, mod, seed=seed)
            dt = time.perf_counter() - t0
            ris_times.append(dt)
            ris_iters.append(r.n_bcd_iters)
            ris_mem.append(0.0)
        except Exception as e:
            ris_times.append(float("nan"))
            ris_iters.append(0)
            ris_mem.append(0.0)

    c_ris, p_ris, r2_ris = _fit_power_law(ris_vals, ris_times)

    # ── N_time scaling ───────────────────────────────────────────
    time_vals = [5, 10, 20]
    time_times = []
    time_iters = []
    time_mem = []
    for n_time in time_vals:
        try:
            mod = BenchmarkConfig(**{**cfg.__dict__, "N_time": n_time, "seed": seed})
            t0 = time.perf_counter()
            r = run_baseline(BaselineMethod.SCA_BCD_FULL, mod, seed=seed)
            dt = time.perf_counter() - t0
            time_times.append(dt)
            time_iters.append(r.n_bcd_iters)
            time_mem.append(0.0)
        except Exception as e:
            time_times.append(float("nan"))
            time_iters.append(0)
            time_mem.append(0.0)

    c_time, p_time, r2_time = _fit_power_law(time_vals, time_times)

    # ── N_eve scaling ────────────────────────────────────────────
    eve_vals = [1, 2, 3, 5]
    n_eve_tuple_map = {1: ((100.0, 80.0, 1.5),),
                       2: ((100.0, 80.0, 1.5), (200.0, 120.0, 1.5)),
                       3: ((200.0, 150.0, 1.5), (150.0, -130.0, 1.5), (300.0, 80.0, 1.5)),
                       5: ((200.0, 150.0, 1.5), (150.0, -130.0, 1.5), (300.0, 80.0, 1.5),
                           (100.0, 50.0, 1.5), (250.0, -50.0, 1.5))}
    eve_times = []
    eve_iters = []
    for n_eve in eve_vals:
        try:
            mod = BenchmarkConfig(**{**cfg.__dict__, "q_eves": n_eve_tuple_map[n_eve],
                                      "seed": seed})
            t0 = time.perf_counter()
            r = run_baseline(BaselineMethod.SCA_BCD_FULL, mod, seed=seed)
            dt = time.perf_counter() - t0
            eve_times.append(dt)
            eve_iters.append(r.n_bcd_iters)
        except Exception as e:
            eve_times.append(float("nan"))
            eve_iters.append(0)

    c_eve, p_eve, r2_eve = _fit_power_law(eve_vals, eve_times)

    # ── N_vehicle scaling ────────────────────────────────────────
    veh_vals = [1, 2, 3, 5]
    n_veh_tuple_map = {1: ((200.0, 80.0, 0.0),),
                       2: ((200.0, 80.0, 0.0), (250.0, -60.0, 0.0)),
                       3: ((200.0, 80.0, 0.0), (250.0, -60.0, 0.0), (180.0, -100.0, 0.0)),
                       5: ((200.0, 80.0, 0.0), (250.0, -60.0, 0.0), (180.0, -100.0, 0.0),
                           (150.0, 50.0, 0.0), (300.0, -30.0, 0.0))}
    veh_types_map = {1: ("car",),
                     2: ("car", "truck"),
                     3: ("car", "truck", "motorcycle"),
                     5: ("car", "truck", "motorcycle", "car", "truck")}
    veh_times = []
    veh_iters = []
    for n_veh in veh_vals:
        try:
            mod = BenchmarkConfig(**{**cfg.__dict__,
                                      "q_vehicles": n_veh_tuple_map[n_veh],
                                      "vehicle_types": veh_types_map[n_veh],
                                      "seed": seed})
            t0 = time.perf_counter()
            r = run_baseline(BaselineMethod.SCA_BCD_FULL, mod, seed=seed)
            dt = time.perf_counter() - t0
            veh_times.append(dt)
            veh_iters.append(r.n_bcd_iters)
        except Exception as e:
            veh_times.append(float("nan"))
            veh_iters.append(0)

    c_veh, p_veh, r2_veh = _fit_power_law(veh_vals, veh_times)

    # ── Build output ─────────────────────────────────────────────
    rows = []
    for nv, t, it in zip(ris_vals, ris_times, ris_iters):
        rows.append({"parameter": "N_RIS", "value": nv, "runtime_s": f"{t:.4f}",
                      "bcd_iters": it, "mem_mb": "N/A"})
    for nv, t, it in zip(time_vals, time_times, time_iters):
        rows.append({"parameter": "N_time", "value": nv, "runtime_s": f"{t:.4f}",
                      "bcd_iters": it, "mem_mb": "N/A"})
    for nv, t, it in zip(eve_vals, eve_times, eve_iters):
        rows.append({"parameter": "N_eve", "value": nv, "runtime_s": f"{t:.4f}",
                      "bcd_iters": it, "mem_mb": "N/A"})
    for nv, t, it in zip(veh_vals, veh_times, veh_iters):
        rows.append({"parameter": "N_veh", "value": nv, "runtime_s": f"{t:.4f}",
                      "bcd_iters": it, "mem_mb": "N/A"})

    power_law_fits = {
        "N_RIS": {"c": c_ris, "p": p_ris, "R2": r2_ris},
        "N_time": {"c": c_time, "p": p_time, "R2": r2_time},
        "N_eve": {"c": c_eve, "p": p_eve, "R2": r2_eve},
        "N_veh": {"c": c_veh, "p": p_veh, "R2": r2_veh},
    }

    # ── Write CSV ────────────────────────────────────────────────
    import csv

    csv_path = out / "complexity_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["parameter", "value", "runtime_s", "bcd_iters", "mem_mb"])
        w.writeheader()
        w.writerows(rows)

    # ── Write fit summary ────────────────────────────────────────
    fit_lines = ["# Complexity Scaling: Power-Law Fits\n"]
    for param, fit in power_law_fits.items():
        fit_lines.append(f"| {param} | c = {fit['c']:.4e}, p = {fit['p']:.4f}, R2 = {fit.get('R2', 0):.4f} |")
    (out / "complexity_fits.txt").write_text("\n".join(fit_lines), encoding="utf-8")

    # ── Plot if possible ─────────────────────────────────────────
    _plot_complexity(ris_vals, ris_times, time_vals, time_times,
                     eve_vals, eve_times, veh_vals, veh_times,
                     power_law_fits, str(out))

    return {
        "rows": rows,
        "power_law_fits": power_law_fits,
        "csv_path": str(csv_path),
    }


def _plot_complexity(
    ris_vals, ris_times,
    time_vals, time_times,
    eve_vals, eve_times,
    veh_vals, veh_times,
    fits,
    output_dir: str,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    out = Path(output_dir)
    datasets = [
        (ris_vals, ris_times, "N_RIS", "N_RIS"),
        (time_vals, time_times, "N_time", "N_time"),
        (eve_vals, eve_times, "N_eve", "N_eve"),
        (veh_vals, veh_times, "N_veh", "N_veh"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for ax, (xs, ys, xlabel, key) in zip(axes, datasets):
        ax.plot(xs, ys, "o-", linewidth=1.5, color="steelblue")
        fit = fits.get(key, {})
        if "p" in fit:
            x_fit = np.linspace(min(xs), max(xs), 100)
            y_fit = fit["c"] * (x_fit ** fit["p"])
            ax.plot(x_fit, y_fit, "--", color="coral", linewidth=1.0,
                    label=f"T ∝ N^{fit['p']:.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Runtime (s)")
        ax.set_title(f"Scaling with {key}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle("Complexity scaling analysis", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out / "complexity_scaling.png"), dpi=150)
    plt.close(fig)
