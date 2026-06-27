import numpy as np
from dataclasses import dataclass
from typing import Tuple

from sensing_matrix_exp.channels.sensing_matrix_channel import (
    ula_steering_vector,
    target_response_matrix,
    composite_sensing_channel,
    compute_echo_matrix,
    compute_covariance_matrices,
)


@dataclass
class SensingMatrixConfig:
    N_tx: int = 16
    N_rx: int = 16
    antenna_spacing: float = 0.5
    wavelength: float = 1.0
    L_pilot: int = 32
    noise_power: float = 1e-10
    sensing_power: float = 1.0
    seed: int | None = 42
    num_targets: int = 3
    output_root: str = "outputs/sensing_matrix"


class SensingMatrixEnvironment:
    def __init__(self, config: SensingMatrixConfig | None = None):
        self.config = config or SensingMatrixConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        self.N_tx = self.config.N_tx
        self.N_rx = self.config.N_rx
        self.L = self.config.L_pilot

        self.target_thetas: list[float] = []
        self.target_alphas: list[complex] = []
        self.A_list: list[np.ndarray] = []
        self.H_sense: np.ndarray = np.zeros(
            (self.N_rx, self.N_tx), dtype=complex
        )
        self.X: np.ndarray = np.zeros(
            (self.N_tx, self.L), dtype=complex
        )
        self.Y: np.ndarray = np.zeros(
            (self.N_rx, self.L), dtype=complex
        )

    def set_targets(
        self,
        theta_deg_list: list[float],
        alpha_list: list[complex] | None = None,
    ) -> None:
        """Configure vehicle target directions and reflection coefficients."""
        self.target_thetas = theta_deg_list
        self.A_list = []

        for i, theta in enumerate(theta_deg_list):
            a_tx = ula_steering_vector(
                self.N_tx, theta,
                self.config.antenna_spacing,
                self.config.wavelength,
            )
            a_rx = ula_steering_vector(
                self.N_rx, theta,
                self.config.antenna_spacing,
                self.config.wavelength,
            )
            A = target_response_matrix(a_rx, a_tx)
            self.A_list.append(A)

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

    def generate_pilots(self) -> np.ndarray:
        """Generate random orthogonal pilot matrix X (N_t x L).

        Pilots are unit-norm per column.
        """
        X_raw = np.random.randn(self.N_tx, self.L) + 1j * np.random.randn(
            self.N_tx, self.L
        )
        for col in range(self.L):
            norm = float(np.linalg.norm(X_raw[:, col]))
            if norm > 0.0:
                X_raw[:, col] /= norm
        self.X = X_raw
        return self.X

    def transmit_pilots(self) -> np.ndarray:
        """Compute echo matrix Y = H_sense @ X + N."""
        self.Y = compute_echo_matrix(
            self.H_sense, self.X, self.config.noise_power
        )
        return self.Y

    def compute_covariances(self) -> dict:
        """Compute all covariance matrices."""
        return compute_covariance_matrices(
            self.Y, self.H_sense, self.X, self.config.noise_power
        )

    def reset(
        self,
        theta_deg_list: list[float] | None = None,
        alpha_list: list[complex] | None = None,
    ) -> dict:
        """Full reset: set targets, generate pilots, transmit, compute covariances."""
        if theta_deg_list is None:
            thetas = np.random.uniform(-60.0, 60.0, size=self.config.num_targets)
            self.set_targets(thetas.tolist(), alpha_list)
        else:
            self.set_targets(theta_deg_list, alpha_list)

        self.generate_pilots()
        self.transmit_pilots()
        cov = self.compute_covariances()

        return self.get_state(cov)

    def get_state(self, cov: dict | None = None) -> dict:
        if cov is None:
            cov = self.compute_covariances()
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
                "rank": int(np.linalg.matrix_rank(self.H_sense)),
                "frobenius_norm": float(np.linalg.norm(self.H_sense, "fro")),
            },
            "covariance": {
                "eigenvalues": cov["eigenvalues"],
            },
        }

    def step(
        self,
        theta_deg_list: list[float] | None = None,
        alpha_list: list[complex] | None = None,
    ) -> Tuple[dict, dict, bool, dict]:
        """Advance one step: re-generate pilots and re-transmit."""
        if theta_deg_list is not None:
            self.set_targets(theta_deg_list, alpha_list)

        self.generate_pilots()
        self.transmit_pilots()
        cov = self.compute_covariances()
        done = False
        return self.get_state(cov), cov, done, cov
