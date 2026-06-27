import numpy as np
from dataclasses import dataclass
from typing import Tuple

from crb_sensing_exp.channels.crb_channel import (
    ula_steering_vector,
    ula_steering_derivative,
    target_response_matrix,
    target_response_derivative,
    composite_sensing_channel,
    compute_channel_derivatives,
    compute_fim,
    compute_crb,
)


@dataclass
class CRBConfig:
    N_tx: int = 16
    N_rx: int = 16
    antenna_spacing: float = 0.5
    wavelength: float = 1.0
    L_pilot: int = 32
    noise_power: float = 1e-10
    seed: int | None = 42
    num_targets: int = 3
    output_root: str = "outputs/crb_sensing"


class CRBSensingEnvironment:
    def __init__(self, config: CRBConfig | None = None):
        self.config = config or CRBConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        self.N_tx = self.config.N_tx
        self.N_rx = self.config.N_rx
        self.L = self.config.L_pilot

        self.target_thetas: list[float] = []
        self.target_alphas: list[complex] = []
        self.A_list: list[np.ndarray] = []
        self.dA_list: list[np.ndarray] = []
        self.dH_derivs: list[np.ndarray] = []
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
        """Set up vehicle targets with steering vectors and derivatives."""
        self.target_thetas = theta_deg_list
        self.A_list = []
        self.dA_list = []

        for theta in theta_deg_list:
            a = ula_steering_vector(
                self.N_tx, theta,
                self.config.antenna_spacing,
                self.config.wavelength,
            )
            # For monostatic, rx = tx
            a_rx = a
            a_tx = a
            da = ula_steering_derivative(
                self.N_tx, theta,
                self.config.antenna_spacing,
                self.config.wavelength,
            )
            A = target_response_matrix(a)
            dA = target_response_derivative(a, da)

            # Ensure correct dimensions for rectangular case
            if self.N_rx != self.N_tx:
                a_rx = ula_steering_vector(
                    self.N_rx, theta,
                    self.config.antenna_spacing,
                    self.config.wavelength,
                )
                A = target_response_matrix(a_rx)
                da_rx = ula_steering_derivative(
                    self.N_rx, theta,
                    self.config.antenna_spacing,
                    self.config.wavelength,
                )
                dA = target_response_derivative(a_rx, da_rx)

            self.A_list.append(A)
            self.dA_list.append(dA)

        if alpha_list is None:
            self.target_alphas = [
                complex(np.random.randn() + 1j * np.random.randn())
                for _ in theta_deg_list
            ]
        else:
            self.target_alphas = alpha_list

        self.H_sense = composite_sensing_channel(
            self.target_alphas, self.A_list
        )
        self.dH_derivs = compute_channel_derivatives(
            self.target_alphas, self.dA_list
        )

    def generate_pilots(self) -> np.ndarray:
        """Generate unit-norm column pilot matrix (N_t x L)."""
        X_raw = np.random.randn(self.N_tx, self.L) + 1j * np.random.randn(
            self.N_tx, self.L
        )
        for col in range(self.L):
            nrm = float(np.linalg.norm(X_raw[:, col]))
            if nrm > 0.0:
                X_raw[:, col] /= nrm
        self.X = X_raw
        return self.X

    def compute_fim_and_crb(self) -> dict:
        """Compute FIM and CRB from current state."""
        FIM = compute_fim(self.dH_derivs, self.X, self.config.noise_power)
        crb_result = compute_crb(FIM)
        crb_result["FIM"] = FIM
        return crb_result

    def reset(
        self,
        theta_deg_list: list[float] | None = None,
        alpha_list: list[complex] | None = None,
    ) -> dict:
        """Full reset: targets, pilots, FIM, CRB."""
        if theta_deg_list is None:
            thetas = np.random.uniform(-60.0, 60.0, size=self.config.num_targets)
            self.set_targets(thetas.tolist(), alpha_list)
        else:
            self.set_targets(theta_deg_list, alpha_list)

        self.generate_pilots()
        return self.get_state()

    def get_state(self) -> dict:
        crb_result = self.compute_fim_and_crb()
        return {
            "config": {
                "N_tx": self.N_tx,
                "N_rx": self.N_rx,
                "L": self.L,
            },
            "targets": {
                "theta_deg": self.target_thetas,
                "alpha": self.target_alphas,
            },
            "H_sense": {
                "matrix": self.H_sense,
                "frobenius_norm": float(np.linalg.norm(self.H_sense, "fro")),
            },
            "FIM": {
                "matrix": crb_result["FIM"],
                "eigenvalues": crb_result["fim_eigenvalues"],
                "condition_number": crb_result["condition_number"],
            },
            "CRB": {
                "matrix": crb_result["crb_matrix"],
                "var_bound": crb_result["var_bound"],
                "rmse_bound": crb_result["rmse_bound"],
            },
        }

    # ── Parameter sweeps ───────────────────────────────

    def sweep_snr(
        self,
        snr_db_range: list[float],
        theta_deg_list: list[float] | None = None,
        alpha_list: list[complex] | None = None,
        num_trials: int = 5,
    ) -> dict:
        """Sweep CRB over SNR."""
        results = {"snr_db": [], "rmse_deg": [], "crb_diag": []}
        if theta_deg_list is None:
            theta_deg_list = [-30.0, 0.0, 30.0]
            alpha_list = [1.0 + 0.0j, 0.5 - 0.5j, -0.3 + 0.7j]

        for snr_db in snr_db_range:
            snr_linear = 10.0 ** (snr_db / 10.0)
            # noise_power = signal_power / snr  (signal power ~ 1.0 from pilots)
            noise_power = 1.0 / snr_linear
            cfg = CRBConfig(
                N_tx=self.N_tx,
                N_rx=self.N_rx,
                antenna_spacing=self.config.antenna_spacing,
                wavelength=self.config.wavelength,
                L_pilot=self.L,
                noise_power=noise_power,
                seed=self.config.seed,
            )
            rmse_list = []
            crb_diag_list = []
            for _ in range(num_trials):
                env = CRBSensingEnvironment(cfg)
                env.reset(theta_deg_list=theta_deg_list, alpha_list=alpha_list)
                crb_r = env.compute_fim_and_crb()
                rmse_list.append(crb_r["rmse_bound"])
                crb_diag_list.append(crb_r["var_bound"])
            rmse_avg = np.mean(rmse_list, axis=0)
            crb_diag_avg = np.mean(crb_diag_list, axis=0)
            results["snr_db"].append(snr_db)
            results["rmse_deg"].append(rmse_avg)
            results["crb_diag"].append(crb_diag_avg)
        return results

    def sweep_antennas(
        self,
        N_range: list[int],
        theta_deg_list: list[float] | None = None,
        alpha_list: list[complex] | None = None,
        num_trials: int = 5,
    ) -> dict:
        """Sweep CRB over number of antennas."""
        results = {"N": [], "rmse_deg": [], "crb_diag": []}
        if theta_deg_list is None:
            theta_deg_list = [-30.0, 0.0, 30.0]
            alpha_list = [1.0 + 0.0j, 0.5 - 0.5j, -0.3 + 0.7j]

        for N in N_range:
            cfg = CRBConfig(
                N_tx=N,
                N_rx=N,
                antenna_spacing=self.config.antenna_spacing,
                wavelength=self.config.wavelength,
                L_pilot=self.L,
                noise_power=self.config.noise_power,
                seed=self.config.seed,
            )
            rmse_list = []
            crb_diag_list = []
            for _ in range(num_trials):
                env = CRBSensingEnvironment(cfg)
                env.reset(theta_deg_list=theta_deg_list, alpha_list=alpha_list)
                crb_r = env.compute_fim_and_crb()
                rmse_list.append(crb_r["rmse_bound"])
                crb_diag_list.append(crb_r["var_bound"])
            rmse_avg = np.mean(rmse_list, axis=0)
            crb_diag_avg = np.mean(crb_diag_list, axis=0)
            results["N"].append(N)
            results["rmse_deg"].append(rmse_avg)
            results["crb_diag"].append(crb_diag_avg)
        return results

    def sweep_pilots(
        self,
        L_range: list[int],
        theta_deg_list: list[float] | None = None,
        alpha_list: list[complex] | None = None,
        num_trials: int = 5,
    ) -> dict:
        """Sweep CRB over pilot length."""
        results = {"L": [], "rmse_deg": [], "crb_diag": []}
        if theta_deg_list is None:
            theta_deg_list = [-30.0, 0.0, 30.0]
            alpha_list = [1.0 + 0.0j, 0.5 - 0.5j, -0.3 + 0.7j]

        for L in L_range:
            cfg = CRBConfig(
                N_tx=self.N_tx,
                N_rx=self.N_rx,
                antenna_spacing=self.config.antenna_spacing,
                wavelength=self.config.wavelength,
                L_pilot=L,
                noise_power=self.config.noise_power,
                seed=self.config.seed,
            )
            rmse_list = []
            crb_diag_list = []
            for _ in range(num_trials):
                env = CRBSensingEnvironment(cfg)
                env.reset(theta_deg_list=theta_deg_list, alpha_list=alpha_list)
                crb_r = env.compute_fim_and_crb()
                rmse_list.append(crb_r["rmse_bound"])
                crb_diag_list.append(crb_r["var_bound"])
            rmse_avg = np.mean(rmse_list, axis=0)
            crb_diag_avg = np.mean(crb_diag_list, axis=0)
            results["L"].append(L)
            results["rmse_deg"].append(rmse_avg)
            results["crb_diag"].append(crb_diag_avg)
        return results

    def sweep_targets(
        self,
        K_range: list[int],
        num_trials: int = 5,
    ) -> dict:
        """Sweep CRB over number of targets."""
        results = {"K": [], "rmse_deg_mean": [], "condition_number": []}
        for K in K_range:
            cfg = CRBConfig(
                N_tx=self.N_tx,
                N_rx=self.N_rx,
                L_pilot=self.L,
                noise_power=self.config.noise_power,
                seed=self.config.seed,
                num_targets=K,
            )
            rmse_mean_list = []
            cond_list = []
            for _ in range(num_trials):
                env = CRBSensingEnvironment(cfg)
                thetas = np.linspace(-50.0, 50.0, K).tolist()
                alphas = [
                    complex(np.random.randn(), np.random.randn())
                    for _ in range(K)
                ]
                env.reset(theta_deg_list=thetas, alpha_list=alphas)
                crb_r = env.compute_fim_and_crb()
                rmse_mean_list.append(float(np.mean(crb_r["rmse_bound"])))
                cond_list.append(crb_r["condition_number"])
            results["K"].append(K)
            results["rmse_deg_mean"].append(np.mean(rmse_mean_list))
            results["condition_number"].append(np.mean(cond_list))
        return results
