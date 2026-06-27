import numpy as np
from dataclasses import dataclass
from typing import Tuple

from core.channel import path_loss
from ris_uav_exp.channels.ris_channel import (
    generate_ris_rician_channel,
    compute_ris_reflection_matrix,
    compute_effective_channel,
    compute_effective_channel_gain,
)


@dataclass
class RISUAVConfig:
    area_size: float = 1000.0
    ris_altitude: float = 50.0
    dt: float = 0.1
    seed: int | None = 42
    max_steps: int = 200
    bandwidth: float = 1e6
    noise_psd: float = 10 ** (-17.4)
    bs_power: float = 0.5
    alpha: float = 2.0
    beta0: float = 1.0
    N_ris: int = 16
    rician_k: float = 5.0
    output_root: str = "outputs/ris_uav"

    eve_density_lambda: float = 2e-5
    eve_region_xmin: float = 0.0
    eve_region_xmax: float = 1000.0
    eve_region_ymin: float = 0.0
    eve_region_ymax: float = 1000.0


class RISUAVEnvironment:
    def __init__(self, config: RISUAVConfig | None = None):
        self.config = config or RISUAVConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        self.half_area = self.config.area_size / 2.0

        self.bs_position = np.array([0.0, 0.0, 0.0])

        self.ris_position = np.zeros(3, dtype=float)
        self.user_position = np.zeros(3, dtype=float)
        self.eve_positions = np.empty((0, 2), dtype=float)
        self.num_eves = 0

        self.phases = np.zeros(self.config.N_ris, dtype=float)
        self.Phi = np.eye(self.config.N_ris, dtype=complex)

        self.h_BR = np.ones(self.config.N_ris, dtype=complex)
        self.h_RU = np.ones(self.config.N_ris, dtype=complex)
        self.h_RE = np.empty((0, self.config.N_ris), dtype=complex)

        self._step_counter = 0

    def _random_xy(self) -> np.ndarray:
        return np.random.uniform(-self.half_area, self.half_area, size=2)

    def _distance_3d(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def _generate_hppp_eves(self) -> np.ndarray:
        area = (
            (self.config.eve_region_xmax - self.config.eve_region_xmin)
            * (self.config.eve_region_ymax - self.config.eve_region_ymin)
        )
        n_eve = np.random.poisson(self.config.eve_density_lambda * area)
        if n_eve == 0:
            return np.empty((0, 2), dtype=float)
        xs = np.random.uniform(
            self.config.eve_region_xmin, self.config.eve_region_xmax, size=n_eve
        )
        ys = np.random.uniform(
            self.config.eve_region_ymin, self.config.eve_region_ymax, size=n_eve
        )
        return np.column_stack([xs, ys])

    def _reset_positions(self) -> None:
        ris_xy = self._random_xy()
        self.ris_position = np.array([ris_xy[0], ris_xy[1], self.config.ris_altitude])

        user_xy = self._random_xy()
        self.user_position = np.array([user_xy[0], user_xy[1], 0.0])

        self.eve_positions = self._generate_hppp_eves()
        self.num_eves = self.eve_positions.shape[0]

    def _generate_channels(self) -> None:
        pl_BR = path_loss(
            self._distance_3d(self.bs_position, self.ris_position),
            self.config.alpha,
            self.config.beta0,
        )
        pl_RU = path_loss(
            self._distance_3d(self.ris_position, self.user_position),
            self.config.alpha,
            self.config.beta0,
        )

        self.h_BR = generate_ris_rician_channel(
            self.config.N_ris, self.config.rician_k, pl_BR
        )
        self.h_RU = generate_ris_rician_channel(
            self.config.N_ris, self.config.rician_k, pl_RU
        )

        if self.num_eves > 0:
            h_RE_list = []
            for i in range(self.num_eves):
                ep = self.eve_positions[i]
                ep_3d = np.array([ep[0], ep[1], 0.0])
                pl_RE = path_loss(
                    self._distance_3d(self.ris_position, ep_3d),
                    self.config.alpha,
                    self.config.beta0,
                )
                h_RE_list.append(
                    generate_ris_rician_channel(
                        self.config.N_ris, self.config.rician_k, pl_RE
                    )
                )
            self.h_RE = np.array(h_RE_list)
        else:
            self.h_RE = np.empty((0, self.config.N_ris), dtype=complex)

    def reset(self) -> dict:
        self._step_counter = 0
        self._reset_positions()
        self._generate_channels()
        self.phases = np.random.uniform(0.0, 2.0 * np.pi, size=self.config.N_ris)
        self.Phi = compute_ris_reflection_matrix(self.phases)
        return self.get_state()

    def set_phases(self, phases: np.ndarray) -> None:
        self.phases = phases.copy()
        self.Phi = compute_ris_reflection_matrix(self.phases)

    def compute_effective_channel_gain(
        self, h_rx: np.ndarray, Phi: np.ndarray, h_tx: np.ndarray
    ) -> float:
        h_eff = compute_effective_channel(h_rx, Phi, h_tx)
        return compute_effective_channel_gain(h_eff)

    def compute_rates(self) -> dict:
        noise_power = self.config.noise_psd * self.config.bandwidth

        g_user = self.compute_effective_channel_gain(
            self.h_RU, self.Phi, self.h_BR
        )
        gamma_b = (self.config.bs_power * g_user) / noise_power
        R_legit = self.config.bandwidth * np.log2(1.0 + gamma_b)

        if self.num_eves > 0:
            g_eve_arr = np.array([
                self.compute_effective_channel_gain(
                    self.h_RE[i], self.Phi, self.h_BR
                )
                for i in range(self.num_eves)
            ])
            gamma_e_arr = (self.config.bs_power * g_eve_arr) / noise_power
            R_eve_arr = self.config.bandwidth * np.log2(1.0 + gamma_e_arr)
            max_eve_idx = int(np.argmax(R_eve_arr))
            R_eve = float(R_eve_arr[max_eve_idx])
            max_eve_capacity = R_eve
        else:
            g_eve_arr = np.array([0.0])
            R_eve = 0.0
            max_eve_capacity = 0.0

        R_sec = max(R_legit - R_eve, 0.0)

        return {
            "g_user": float(g_user),
            "g_eve_max": float(np.max(g_eve_arr)) if self.num_eves > 0 else 0.0,
            "gamma_b": float(gamma_b),
            "R_legit": float(R_legit),
            "R_eve": float(R_eve),
            "R_sec": float(R_sec),
            "max_eve_capacity": float(max_eve_capacity),
            "num_eves": self.num_eves,
            "noise_power": float(noise_power),
        }

    def get_state(self) -> dict:
        rates = self.compute_rates()
        return {
            "positions": {
                "bs": self.bs_position.copy(),
                "ris": self.ris_position.copy(),
                "user": self.user_position.copy(),
                "eves": self.eve_positions.copy(),
            },
            "phases": self.phases.copy(),
            "channels": {
                "h_BR": self.h_BR.copy(),
                "h_RU": self.h_RU.copy(),
                "h_RE": self.h_RE.copy(),
            },
            "rates": rates,
        }

    def step(self, action_phases: np.ndarray | None = None) -> Tuple[dict, float, bool, dict]:
        self._step_counter += 1

        if action_phases is not None:
            self.set_phases(np.clip(action_phases, 0.0, 2.0 * np.pi))
        else:
            self.phases = np.random.uniform(0.0, 2.0 * np.pi, size=self.config.N_ris)
            self.Phi = compute_ris_reflection_matrix(self.phases)

        self._generate_channels()
        rates = self.compute_rates()

        reward = rates["R_sec"]
        done = self._step_counter >= self.config.max_steps

        return self.get_state(), reward, done, rates
