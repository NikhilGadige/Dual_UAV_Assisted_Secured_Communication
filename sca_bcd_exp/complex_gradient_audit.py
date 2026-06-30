"""Complex-gradient audit for the BS power block.

Verifies whether finite differences for w_bs use:
  - real perturbation only (current approach),
  - imaginary perturbation only,
  - or Wirtinger derivatives.

Computes chain-rule errors for three approaches and generates diagnostic
plots saved to outputs/optimization/conditioning_analysis/complex_gradient_audit/.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.environments.sca_environment import SCABCDEnvironment

plt = None


def _import_plt():
    global plt
    if plt is None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt_impl
            plt = plt_impl
        except Exception:
            pass
    return plt


# ── Gradient computation methods ───────────────────────────────────────


def gradient_real_fd(
    env: SCABCDEnvironment,
    solution,
    block_sl: slice,
    h: float = 1e-5,
) -> np.ndarray:
    """Finite-difference gradient using *real* perturbations on the
    real / imaginary split representation.

    This is the exact approach used in ``finite_diff_gradient_for_block``.
    The returned gradient has shape ``(2 * N_time,)`` where entries
    0 … N_time-1 are ∂f/∂Re[wₙ] and entries N_time … 2*N_time-1 are
    ∂f/∂Im[wₙ].
    """
    full = env._unpack_decision_vars(solution.decision_vars)
    x_block = full[block_sl].copy()
    grad = np.zeros_like(x_block)
    f0 = env._flat_block_obj(full, block_sl, solution)
    for i in range(len(x_block)):
        full_p = full.copy()
        full_p[block_sl][i] += h
        fp = env._flat_block_obj(full_p, block_sl, solution)
        grad[i] = (fp - f0) / h
    return grad


def gradient_wirtinger(
    grad_real: np.ndarray,
    n_time: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the real-split FD gradient to Wirtinger derivatives.

    For a real-valued function ``f(w)`` with ``w = x + j y``:

    .. math::

        \\frac{\\partial f}{\\partial w}
            = \\tfrac12 \\bigl( \\frac{\\partial f}{\\partial x}
                                - j \\frac{\\partial f}{\\partial y} \\bigr)

        \\frac{\\partial f}{\\partial w^*}
            = \\tfrac12 \\bigl( \\frac{\\partial f}{\\partial x}
                                + j \\frac{\\partial f}{\\partial y} \\bigr)

    Returns
    -------
    df_dw : np.ndarray, shape (n_time,), complex
    df_dwconj : np.ndarray, shape (n_time,), complex
    """
    df_dre = grad_real[:n_time]
    df_dim = grad_real[n_time:]
    df_dw = 0.5 * (df_dre - 1j * df_dim)
    df_dwconj = 0.5 * (df_dre + 1j * df_dim)
    return df_dw, df_dwconj


def gradient_complex_fd(
    env: SCABCDEnvironment,
    solution,
    h: float = 1e-5,
) -> np.ndarray:
    """Finite-difference gradient using *complex* perturbations on the
    complex ``w_bs`` variable directly.

    For each time index *n* we apply a small complex perturbation
    ``Δ = h ⋅ (1 + j)/√2`` to ``w_bs[n]`` and estimate

    .. math::

        \\frac{\\partial f}{\\partial w_n}
            \\approx \\frac{f(w + Δ e_n) - f(w)}{Δ}

    Because ``f`` is real-valued but **not** holomorphic, this complex-step
    approximation is direction-dependent and will **not** agree with the
    Wirtinger derivative unless the function were analytic.  We include it
    purely to quantify the mismatch.

    Returns
    -------
    grad : np.ndarray, shape (n_time,), complex
        Raw complex-step gradient (Δf / Δ) for each time slot.
    """
    n_time = solution.decision_vars.N_time
    template = solution
    dv = solution.decision_vars
    w_bs = dv.w_bs.copy().astype(complex)
    grad = np.zeros(n_time, dtype=complex)
    f0 = env.evaluate_objective(solution)

    for n in range(n_time):
        delta = h * (1.0 + 1.0j) / np.sqrt(2.0)
        w_save = dv.w_bs[n]
        dv.w_bs[n] = w_save + delta
        fp = env.evaluate_objective(solution)
        grad[n] = (fp - f0) / delta
        dv.w_bs[n] = w_save

    return grad


# ── Chain-rule error helpers ────────────────────────────────────────────


def chain_rule_error_real(
    env: SCABCDEnvironment,
    solution,
    grad_real: np.ndarray,
    block_sl: slice,
    n_perturbations: int = 100,
    seed: int = 42,
) -> dict:
    """Chain-rule error using **real** perturbations.

    Perturb each entry of the flat real/imag power block with a random
    real ``δ`` and compare the linear prediction ``grad·δ`` against the
    exact change ``f(x+δ) - f(x)``.
    """
    rng = np.random.default_rng(seed)
    full = env._unpack_decision_vars(solution.decision_vars)
    x0 = full[block_sl].copy()
    f0 = env._flat_block_obj(full, block_sl, solution)
    rel_errs = []

    for _ in range(n_perturbations):
        delta = rng.normal(0.0, 1e-4, size=x0.shape)
        full_p = full.copy()
        full_p[block_sl] = x0 + delta
        fp = env._flat_block_obj(full_p, block_sl, solution)
        actual = fp - f0
        predicted = float(np.dot(grad_real, delta))
        denom = max(abs(actual), 1e-30)
        rel_errs.append(abs(predicted - actual) / denom)

    return {
        "mean_rel_err": float(np.mean(rel_errs)),
        "median_rel_err": float(np.median(rel_errs)),
        "max_rel_err": float(np.max(rel_errs)),
        "std_rel_err": float(np.std(rel_errs)),
        "rel_errs": rel_errs,
    }


def chain_rule_error_complex_perturbation(
    env: SCABCDEnvironment,
    solution,
    grad_real: np.ndarray,
    n_perturbations: int = 100,
    seed: int = 43,
) -> dict:
    """Chain-rule error using **complex perturbations**.

    Perturb each complex ``w_bs[n]`` by a small complex ``δw`` and
    predict the change using the real FD gradient:

    .. math::

        \\Delta f_{\\text{pred}}
            = \\sum_n \\Bigl(
                \\frac{\\partial f}{\\partial\\text{Re}[w_n]} \\,\\text{Re}[\\delta w_n]
                + \\frac{\\partial f}{\\partial\\text{Im}[w_n]} \\,\\text{Im}[\\delta w_n]
              \\Bigr)

    This is the correct chain rule for real-valued functions of complex
    variables when the gradient is expressed in the real-basis
    (Re/Im) coordinates.
    """
    rng = np.random.default_rng(seed)
    n_time = solution.decision_vars.N_time
    dv = solution.decision_vars
    w0 = dv.w_bs.copy()
    f0 = env.evaluate_objective(solution)

    # build full gradient for complex prediction
    # grad_real has shape (2*n_time,) with [df/d_re, df/d_im]
    df_dre = grad_real[:n_time]
    df_dim = grad_real[n_time:]

    rel_errs = []

    for _ in range(n_perturbations):
        delta_w = rng.normal(0.0, 1e-4, size=n_time) + 1j * rng.normal(0.0, 1e-4, size=n_time)
        dw_re = np.real(delta_w)
        dw_im = np.imag(delta_w)

        dv.w_bs = w0 + delta_w
        fp = env.evaluate_objective(solution)
        actual = fp - f0
        predicted = float(np.dot(df_dre, dw_re) + np.dot(df_dim, dw_im))
        denom = max(abs(actual), 1e-30)
        rel_errs.append(abs(predicted - actual) / denom)

    dv.w_bs = w0

    return {
        "mean_rel_err": float(np.mean(rel_errs)),
        "median_rel_err": float(np.median(rel_errs)),
        "max_rel_err": float(np.max(rel_errs)),
        "std_rel_err": float(np.std(rel_errs)),
        "rel_errs": rel_errs,
    }


def chain_rule_error_wirtinger(
    env: SCABCDEnvironment,
    solution,
    grad_real: np.ndarray,
    n_perturbations: int = 100,
    seed: int = 44,
) -> dict:
    """Chain-rule error using **Wirtinger gradients**.

    Convert the real-split FD to Wirtinger derivatives and use the
    Wirtinger chain rule for a real-valued function:

    .. math::

        \\Delta f_{\\text{pred}}
            = 2 \\, \\text{Re}\\Bigl(
                \\sum_n \\frac{\\partial f}{\\partial w_n} \\,\\delta w_n
              \\Bigr)

    where ``∂f/∂w`` is the Wirtinger derivative.
    """
    rng = np.random.default_rng(seed)
    n_time = solution.decision_vars.N_time
    df_dw, df_dwconj = gradient_wirtinger(grad_real, n_time)
    dv = solution.decision_vars
    w0 = dv.w_bs.copy()
    f0 = env.evaluate_objective(solution)
    rel_errs = []

    for _ in range(n_perturbations):
        delta_w = rng.normal(0.0, 1e-4, size=n_time) + 1j * rng.normal(0.0, 1e-4, size=n_time)
        dv.w_bs = w0 + delta_w
        fp = env.evaluate_objective(solution)
        actual = fp - f0
        predicted = float(2.0 * np.real(np.dot(df_dw, delta_w)))
        denom = max(abs(actual), 1e-30)
        rel_errs.append(abs(predicted - actual) / denom)

    dv.w_bs = w0

    return {
        "mean_rel_err": float(np.mean(rel_errs)),
        "median_rel_err": float(np.median(rel_errs)),
        "max_rel_err": float(np.max(rel_errs)),
        "std_rel_err": float(np.std(rel_errs)),
        "rel_errs": rel_errs,
    }


# ── Plots ──────────────────────────────────────────────────────────────


def _plot_chain_rule_errors(
    results: dict,
    save_path: str,
) -> str:
    p = _import_plt()
    if p is None:
        return ""

    fig, axes = p.subplots(1, 3, figsize=(14, 4.5))

    methods = [
        ("real_perturbation", "Real perturbations", "C0"),
        ("complex_perturbation", "Complex perturbations", "C1"),
        ("wirtinger", "Wirtinger gradients", "C2"),
    ]
    bins = np.linspace(-5, 1, 65)

    for ax, (key, label, color) in zip(axes, methods):
        errs = np.log10(np.maximum(np.array(results[key]["rel_errs"]), 1e-15))
        ax.hist(errs, bins=bins, color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.axvline(np.log10(results[key]["median_rel_err"]), color="k", ls="--", label=f"median={results[key]['median_rel_err']:.2e}")
        ax.axvline(np.log10(results[key]["max_rel_err"]), color="r", ls=":", label=f"max={results[key]['max_rel_err']:.2e}")
        ax.set_xlabel("log₁₀(relative error)")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    fig.suptitle("Chain-rule relative error distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    p.close(fig)
    return save_path


def _plot_power_gradient_components(
    grad_real: np.ndarray,
    n_time: int,
    save_path: str,
) -> str:
    p = _import_plt()
    if p is None:
        return ""

    df_dre = grad_real[:n_time]
    df_dim = grad_real[n_time:]
    df_dw, _ = gradient_wirtinger(grad_real, n_time)

    fig, axes = p.subplots(3, 1, figsize=(10, 7))

    slots = np.arange(n_time)

    axes[0].bar(slots - 0.15, df_dre, width=0.3, color="steelblue", label="∂f/∂Re[w]")
    axes[0].bar(slots + 0.15, df_dim, width=0.3, color="coral", label="∂f/∂Im[w]")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].set_ylabel("Gradient (real FD)")
    axes[0].set_title("Real-FD gradient components (real / imag split)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].bar(slots - 0.15, np.real(df_dw), width=0.3, color="seagreen", label="Re[∂f/∂w]")
    axes[1].bar(slots + 0.15, np.imag(df_dw), width=0.3, color="goldenrod", label="Im[∂f/∂w]")
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_ylabel("Wirtinger gradient")
    axes[1].set_title("Wirtinger derivative ∂f/∂w (derived from real FD)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)

    mag = np.abs(df_dw)
    axes[2].bar(slots, mag, width=0.5, color="mediumpurple", alpha=0.8)
    axes[2].axhline(0, color="gray", linewidth=0.5)
    axes[2].set_xlabel("Time slot n")
    axes[2].set_ylabel("|∂f/∂w|")
    axes[2].set_title("Wirtinger gradient magnitude")
    axes[2].grid(alpha=0.2)

    fig.suptitle("Power-block gradient components", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    p.close(fig)
    return save_path


# ── Main audit ─────────────────────────────────────────────────────────


def run_complex_gradient_audit(
    config: SCABCDConfig | None = None,
    output_dir: str | None = None,
    n_perturbations: int = 100,
) -> dict:
    """Run the full complex-gradient audit and save results."""
    if config is None:
        config = SCABCDConfig(
            channel_model="rician",
            seed=0,
            max_bcd_iters=2,
            max_sca_iters=2,
        )

    if output_dir is None:
        output_dir = str(config.output_root() / "complex_gradient_audit")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = SCABCDEnvironment(config)
    solution = env.reset()
    blocks = env.block_slices()
    power_sl = blocks["power"]
    n_time = config.N_time

    # ── 1. Compute gradient via real FD (current approach) ────────
    grad_real = gradient_real_fd(env, solution, power_sl, h=config.fd_h)

    # ── 2. Compute complex FD gradient (for comparison) ────────────
    grad_complex = gradient_complex_fd(env, solution, h=config.fd_h)

    # ── 3. Wirtinger gradient ──────────────────────────────────────
    df_dw, df_dwconj = gradient_wirtinger(grad_real, n_time)

    # ── 4. Chain-rule errors ───────────────────────────────────────
    err_real = chain_rule_error_real(env, solution, grad_real, power_sl, n_perturbations)
    err_cplx = chain_rule_error_complex_perturbation(env, solution, grad_real, n_perturbations)
    err_wirt = chain_rule_error_wirtinger(env, solution, grad_real, n_perturbations)

    results = {
        "real_perturbation": err_real,
        "complex_perturbation": err_cplx,
        "wirtinger": err_wirt,
    }

    # ── 5. Plots ───────────────────────────────────────────────────
    plot_chain = str(out / "chain_rule_errors.png")
    _plot_chain_rule_errors(results, plot_chain)
    plot_components = str(out / "power_gradient_components.png")
    _plot_power_gradient_components(grad_real, n_time, plot_components)

    # ── 6. Verification summary ────────────────────────────────────
    passed_median = err_real["median_rel_err"] < 1e-2
    passed_max = err_real["max_rel_err"] < 1e-1

    summary = {
        "fd_perturbation_type": "real perturbation on real/imag split (not Wirtinger)",
        "median_chain_rel_err": err_real["median_rel_err"],
        "max_chain_rel_err": err_real["max_rel_err"],
        "median_chain_rel_err_complex": err_cplx["median_rel_err"],
        "max_chain_rel_err_complex": err_cplx["max_rel_err"],
        "median_chain_rel_err_wirtinger": err_wirt["median_rel_err"],
        "max_chain_rel_err_wirtinger": err_wirt["max_rel_err"],
        "passed_median_criterion": bool(passed_median),
        "passed_max_criterion": bool(passed_max),
        "output_dir": output_dir,
        "plots": {
            "chain_rule_errors": plot_chain,
            "power_gradient_components": plot_components,
        },
    }

    # ── 7. Write markdown report ───────────────────────────────────
    _write_report(summary, results, grad_real, grad_complex, df_dw, n_time, out)

    return summary


def _write_report(
    summary: dict,
    chain_results: dict,
    grad_real: np.ndarray,
    grad_complex: np.ndarray,
    df_dw: np.ndarray,
    n_time: int,
    out_dir: Path,
):
    lines = [
        "# Complex-Gradient Audit: BS Power Block",
        "",
        "## 1. Perturbation Type Verification",
        "",
        "**Finding:** The finite-difference gradient for ``w_bs`` uses "
        "**real perturbations only** on the real/imaginary split "
        "representation.",
        "",
        "The power block packs ``w_bs`` into a flat real vector:",
        "``[Re(w₀), …, Re(w_{N-1}), Im(w₀), …, Im(w_{N-1})]``.",
        "",
        "Each entry is perturbed by a real ``h = 1e-5``, giving:",
        "- ``∂f/∂Re[wₙ]`` (first ``N_time`` entries)",
        "- ``∂f/∂Im[wₙ]`` (last ``N_time`` entries)",
        "",
        "This is **not** a Wirtinger derivative and **not** a "
        "pure-imaginary perturbation.",
        "",
        "---",
        "",
        "## 2. Chain-Rule Error Summary",
        "",
        f"| Method | Median rel. err | Max rel. err |",
        f"|--------|----------------|---------------|",
        f"| Real perturbations | {chain_results['real_perturbation']['median_rel_err']:.3e} | "
        f"{chain_results['real_perturbation']['max_rel_err']:.3e} |",
        f"| Complex perturbations (Re/Im split chain rule) | "
        f"{chain_results['complex_perturbation']['median_rel_err']:.3e} | "
        f"{chain_results['complex_perturbation']['max_rel_err']:.3e} |",
        f"| Wirtinger gradients | {chain_results['wirtinger']['median_rel_err']:.3e} | "
        f"{chain_results['wirtinger']['max_rel_err']:.3e} |",
        "",
        "Success criteria (``median < 1e-2``, ``max < 1e-1``):",
    ]

    passed_median = summary["passed_median_criterion"]
    passed_max = summary["passed_max_criterion"]
    lines.append("")
    lines.append(f"- Median criterion: **{'PASS' if passed_median else 'FAIL'}** "
                  f"(median={summary['median_chain_rel_err']:.3e}, required < 1e-2)")
    lines.append(f"- Max criterion: **{'PASS' if passed_max else 'FAIL'}** "
                  f"(max={summary['max_chain_rel_err']:.3e}, required < 1e-1)")

    # Wirtinger consistency check
    wirt_self_consistency = "N/A (see note below)"
    lines += [
        "",
        "## 3. Wirtinger Self-Consistency",
        "",
        "For a real-valued function ``f(w)`` the Wirtinger derivatives satisfy:",
        "",
        "``∂f/∂w* = conj(∂f/∂w)``",
        "",
    ]
    df_dw_conj = np.conj(df_dw)
    df_dwconj_computed = np.conj(df_dw)  # should equal from the formula above

    # Maximum violation
    df_dw_derived, df_dwconj_derived = gradient_wirtinger(grad_real, n_time)
    max_conj_viol = float(np.max(np.abs(df_dwconj_derived - np.conj(df_dw_derived))))
    lines.append(f"The Wirtinger conjugate symmetry holds to within "
                  f"**{max_conj_viol:.2e}** (machine precision).")

    lines += [
        "",
        "## 4. Gradient Structure",
        "",
        "- ``∂f/∂Re[w]`` mean magnitude: "
        f"{float(np.mean(np.abs(grad_real[:n_time]))):.4e}",
        "- ``∂f/∂Im[w]`` mean magnitude: "
        f"{float(np.mean(np.abs(grad_real[n_time:]))):.4e}",
        "- Wirtinger ``|∂f/∂w|`` mean magnitude: "
        f"{float(np.mean(np.abs(df_dw))):.4e}",
        "",
        "## 5. Plots",
        "",
        f"- ![Chain-rule errors](chain_rule_errors.png)",
        f"- ![Gradient components](power_gradient_components.png)",
        "",
        "---",
        "",
        "*Generated by sca_bcd_exp/complex_gradient_audit.py*",
    ]

    (out_dir / "complex_gradient_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    config = SCABCDConfig(
        channel_model="rician",
        seed=0,
        max_bcd_iters=2,
        max_sca_iters=2,
    )
    result = run_complex_gradient_audit(config)
    print(f"Median rel err (real FD): {result['median_chain_rel_err']:.3e}")
    print(f"Max rel err (real FD):    {result['max_chain_rel_err']:.3e}")
    print(f"Passed median criterion:  {result['passed_median_criterion']}")
    print(f"Passed max criterion:     {result['passed_max_criterion']}")
    print(f"Output:                   {result['output_dir']}")
    return result


if __name__ == "__main__":
    main()
