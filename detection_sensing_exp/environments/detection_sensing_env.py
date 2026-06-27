"""Detection environment: ROC curves, parameter sweeps."""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from detection_sensing_exp.channels.detection_channel import (
    generate_h0,
    generate_h1,
    energy_detector_statistic,
    glrt_detector_statistic,
    monte_carlo_pd_pfa,
)
from crb_sensing_exp.channels.crb_channel import (
    ula_steering_vector,
    ula_steering_derivative,
    target_response_matrix,
    target_response_derivative,
    composite_sensing_channel,
    compute_channel_derivatives,
)


@dataclass
class DetectionConfig:
    N_tx: int = 16
    N_rx: int = 16
    antenna_spacing: float = 0.5
    wavelength: float = 1.0
    L_pilot: int = 32
    noise_power: float = 1e-10
    num_mc: int = 500
    seed: int | None = 42
    num_targets: int = 3
    output_root: str = "outputs/detection_sensing"


class DetectionSensingEnvironment:
    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        self.N_tx = self.config.N_tx
        self.N_rx = self.config.N_rx
        self.L = self.config.L_pilot
        self.num_mc = self.config.num_mc

        self.target_thetas: list[float] = []
        self.target_alphas: list[complex] = []
        self.H_sense: np.ndarray = np.zeros(
            (self.N_rx, self.N_tx), dtype=complex
        )
        self.X: np.ndarray = np.zeros(
            (self.N_tx, self.L), dtype=complex
        )

    def set_targets(
        self,
        theta_deg_list: list[float],
        alpha_list: list[complex] | None = None,
    ) -> None:
        """Set up vehicle targets using steering vectors from CRB phase."""
        self.target_thetas = theta_deg_list
        A_list = []

        for theta in theta_deg_list:
            a = ula_steering_vector(
                self.N_tx, theta,
                self.config.antenna_spacing,
                self.config.wavelength,
            )
            a_rx = a
            a_tx = a
            if self.N_rx != self.N_tx:
                a_rx = ula_steering_vector(
                    self.N_rx, theta,
                    self.config.antenna_spacing,
                    self.config.wavelength,
                )
            A_list.append(target_response_matrix(a_rx))

        if alpha_list is None:
            self.target_alphas = [
                complex(np.random.randn() + 1j * np.random.randn())
                for _ in theta_deg_list
            ]
        else:
            self.target_alphas = alpha_list

        self.H_sense = composite_sensing_channel(
            self.target_alphas, A_list
        )

    def generate_pilots(self) -> np.ndarray:
        """Unit-norm column pilot matrix (N_t x L)."""
        X_raw = np.random.randn(self.N_tx, self.L) + 1j * np.random.randn(
            self.N_tx, self.L
        )
        for col in range(self.L):
            nrm = float(np.linalg.norm(X_raw[:, col]))
            if nrm > 0.0:
                X_raw[:, col] /= nrm
        self.X = X_raw
        return self.X

    def _make_gen_h0(self, noise_power: float) -> Callable:
        """Return H0 generator with captured noise_power."""
        def _gen():
            return generate_h0(self.N_rx, self.L, noise_power)
        return _gen

    def _make_gen_h1(self, noise_power: float) -> Callable:
        """Return H1 generator with captured H_sense, X, noise_power."""
        def _gen():
            return generate_h1(self.H_sense, self.X, noise_power)
        return _gen

    # ── ROC ────────────────────────────────────────────

    def roc_curve(
        self,
        detector: str = "energy",
        num_thresholds: int = 50,
        noise_power: float | None = None,
    ) -> dict:
        """Generate ROC curve (Pd vs Pfa) for a range of thresholds."""
        if noise_power is None:
            noise_power = self.config.noise_power

        np.random.seed(self.config.seed)

        gen_h0 = self._make_gen_h0(noise_power)
        gen_h1 = self._make_gen_h1(noise_power)

        if detector == "energy":
            detector_fn = energy_detector_statistic
        elif detector == "glrt":
            X_fixed = self.X.copy()
            detector_fn = lambda Y: glrt_detector_statistic(Y, X_fixed)
        else:
            raise ValueError(f"Unknown detector: {detector}")

        # Determine threshold range from H0 statistics
        _, _, h0_s, h1_s = monte_carlo_pd_pfa(
            gen_h0, gen_h1, detector_fn,
            threshold=0.0, num_mc=min(200, self.num_mc),
            return_stats=True,
        )
        lo = float(min(h0_s.min(), h1_s.min()))
        hi = float(max(h0_s.max(), h1_s.max()))
        thresholds = np.linspace(lo, hi, num_thresholds)

        pfa_list = []
        pd_list = []
        for gamma in thresholds:
            pd_val, pfa_val = monte_carlo_pd_pfa(
                gen_h0, gen_h1, detector_fn,
                threshold=float(gamma),
                num_mc=self.num_mc,
            )
            pfa_list.append(pfa_val)
            pd_list.append(pd_val)

        return {
            "thresholds": thresholds,
            "pfa": np.array(pfa_list),
            "pd": np.array(pd_list),
            "detector": detector,
            "noise_power": noise_power,
        }

    # ── Sweeps ─────────────────────────────────────────

    def sweep_snr(
        self,
        snr_db_range: list[float],
        detector: str = "energy",
        threshold: float | None = None,
        num_trials: int = 3,
    ) -> dict:
        """Sweep Pd over SNR."""
        results = {"snr_db": [], "pd": [], "pfa": []}
        for snr_db in snr_db_range:
            snr_lin = 10.0 ** (snr_db / 10.0)
            noise_power = 1.0 / snr_lin
            gen_h0 = self._make_gen_h0(noise_power)
            gen_h1 = self._make_gen_h1(noise_power)
            if detector == "energy":
                det_fn = energy_detector_statistic
            elif detector == "glrt":
                Xf = self.X.copy()
                det_fn = lambda Y: glrt_detector_statistic(Y, Xf)
            else:
                raise ValueError(f"Unknown detector: {detector}")

            if threshold is None:
                # auto-threshold from H0 at this noise power
                h0_s = np.array([det_fn(gen_h0()) for _ in range(100)])
                th = float(np.percentile(h0_s, 95))
            else:
                th = threshold

            pd_vals, pfa_vals = [], []
            for _ in range(num_trials):
                pd_v, pfa_v = monte_carlo_pd_pfa(
                    gen_h0, gen_h1, det_fn, th, num_mc=self.num_mc,
                )
                pd_vals.append(pd_v)
                pfa_vals.append(pfa_v)

            results["snr_db"].append(snr_db)
            results["pd"].append(np.mean(pd_vals))
            results["pfa"].append(np.mean(pfa_vals))
        return results

    def sweep_pilots(
        self,
        L_range: list[int],
        detector: str = "energy",
        num_trials: int = 3,
    ) -> dict:
        """Sweep Pd over pilot length."""
        results = {"L": [], "pd": []}
        for L in L_range:
            cfg = DetectionConfig(
                N_tx=self.N_tx, N_rx=self.N_rx, L_pilot=L,
                noise_power=self.config.noise_power,
                num_mc=self.num_mc,
                seed=self.config.seed,
            )
            pd_vals = []
            for _ in range(num_trials):
                env = DetectionSensingEnvironment(cfg)
                env.set_targets(self.target_thetas, self.target_alphas)
                env.generate_pilots()
                gen_h0 = env._make_gen_h0(cfg.noise_power)
                gen_h1 = env._make_gen_h1(cfg.noise_power)
                if detector == "energy":
                    det_fn = energy_detector_statistic
                elif detector == "glrt":
                    Xf = env.X.copy()
                    det_fn = lambda Y: glrt_detector_statistic(Y, Xf)
                else:
                    raise ValueError(f"Unknown detector: {detector}")

                h0_s = np.array([det_fn(gen_h0()) for _ in range(100)])
                th = float(np.percentile(h0_s, 95))

                pd_v, _ = monte_carlo_pd_pfa(
                    gen_h0, gen_h1, det_fn, th, num_mc=self.num_mc,
                )
                pd_vals.append(pd_v)
            results["L"].append(L)
            results["pd"].append(np.mean(pd_vals))
        return results

    def sweep_targets(
        self,
        K_range: list[int],
        detector: str = "energy",
        num_trials: int = 3,
    ) -> dict:
        """Sweep Pd over number of targets."""
        results = {"K": [], "pd": []}
        for K in K_range:
            cfg = DetectionConfig(
                N_tx=self.N_tx, N_rx=self.N_rx,
                L_pilot=self.L,
                noise_power=self.config.noise_power,
                num_mc=self.num_mc,
                seed=self.config.seed,
            )
            thetas = np.linspace(-50.0, 50.0, K).tolist()
            alphas = [
                complex(np.random.randn(), np.random.randn())
                for _ in range(K)
            ]
            pd_vals = []
            for _ in range(num_trials):
                env = DetectionSensingEnvironment(cfg)
                env.set_targets(thetas, alphas)
                env.generate_pilots()
                gen_h0 = env._make_gen_h0(cfg.noise_power)
                gen_h1 = env._make_gen_h1(cfg.noise_power)

                if detector == "energy":
                    det_fn = energy_detector_statistic
                elif detector == "glrt":
                    Xf = env.X.copy()
                    det_fn = lambda Y: glrt_detector_statistic(Y, Xf)
                else:
                    raise ValueError(f"Unknown detector: {detector}")

                h0_s = np.array([det_fn(gen_h0()) for _ in range(100)])
                th = float(np.percentile(h0_s, 95))

                pd_v, _ = monte_carlo_pd_pfa(
                    gen_h0, gen_h1, det_fn, th, num_mc=self.num_mc,
                )
                pd_vals.append(pd_v)
            results["K"].append(K)
            results["pd"].append(np.mean(pd_vals))
        return results

    def reset(
        self,
        theta_deg_list: list[float] | None = None,
        alpha_list: list[complex] | None = None,
    ) -> dict:
        """Full reset: targets, pilots."""
        if theta_deg_list is None:
            thetas = np.random.uniform(
                -60.0, 60.0, size=self.config.num_targets
            ).tolist()
            self.set_targets(thetas, alpha_list)
        else:
            self.set_targets(theta_deg_list, alpha_list)
        self.generate_pilots()
        return {
            "H_sense_fro": float(np.linalg.norm(self.H_sense, "fro")),
            "targets": self.target_thetas,
        }
