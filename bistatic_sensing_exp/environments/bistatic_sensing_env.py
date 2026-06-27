import numpy as np
from dataclasses import dataclass
from typing import Tuple

from core.channel import path_loss
from ris_uav_exp.channels.ris_channel import (
    generate_ris_rician_channel,
    compute_ris_reflection_matrix,
    compute_effective_channel,
)
from fd_jammer_exp.channels.fd_jammer_channel import (
    generate_miso_rician_channel,
    isotropic_beamforming,
    compute_jammer_gain,
)
from vehicle_reflection_exp.channels.vehicle_channel import Vehicle
from bistatic_sensing_exp.channels.sensing_channel import (
    compute_bistatic_distances,
    compute_tx_distance,
    compute_rx_distance,
    compute_bistatic_distance,
    generate_sensing_channel,
    compute_sensing_gain,
    compute_echo_signal,
    compute_sensing_snr,
)


@dataclass
class BistaticSensingConfig:
    area_size: float = 1000.0
    ris_altitude: float = 50.0
    jammer_altitude: float = 50.0
    dt: float = 0.1
    seed: int | None = 42
    max_steps: int = 200
    bandwidth: float = 1e6
    noise_psd: float = 10 ** (-17.4)
    bs_power: float = 0.5
    jammer_power: float = 0.2
    jammer_power_min: float = 0.0
    jammer_power_max: float = 1.0
    sensing_power: float = 1.0
    alpha: float = 2.0
    beta0: float = 1.0
    N_ris: int = 16
    N_j: int = 4
    rician_k: float = 5.0
    output_root: str = "outputs/bistatic_sensing"
    eve_density_lambda: float = 2e-5
    eve_region_xmin: float = 0.0
    eve_region_xmax: float = 1000.0
    eve_region_ymin: float = 0.0
    eve_region_ymax: float = 1000.0
    num_vehicles: int = 3
    vehicle_max_speed: float = 10.0
    vehicle_mobility_mode: str = "straight_road"
    vehicle_types: tuple = ("car", "truck", "motorcycle")


class BistaticSensingEnvironment:
    def __init__(self, config: BistaticSensingConfig | None = None):
        self.config = config or BistaticSensingConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        self.half_area = self.config.area_size / 2.0

        self.bs_position = np.array([0.0, 0.0, 0.0])
        self.ris_position = np.zeros(3, dtype=float)
        self.user_position = np.zeros(3, dtype=float)
        self.jammer_position = np.zeros(3, dtype=float)
        self.jammer_velocity = np.zeros(2, dtype=float)
        self.eve_positions = np.empty((0, 2), dtype=float)
        self.num_eves = 0
        self.vehicles: list[Vehicle] = []

        self.phases = np.zeros(self.config.N_ris, dtype=float)
        self.Phi = np.eye(self.config.N_ris, dtype=complex)
        self.w = np.ones((self.config.N_j, 1), dtype=complex) / np.sqrt(
            self.config.N_j
        )

        self._step_counter = 0

    def _random_xy(self) -> np.ndarray:
        return np.random.uniform(-self.half_area, self.half_area, size=2)

    def _distance_3d(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def _distance_2d(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1[:2] - p2[:2]))

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
        self.ris_position = np.array(
            [ris_xy[0], ris_xy[1], self.config.ris_altitude]
        )
        jammer_xy = self._random_xy()
        self.jammer_position = np.array(
            [jammer_xy[0], jammer_xy[1], self.config.jammer_altitude]
        )
        self.jammer_velocity = np.zeros(2, dtype=float)
        user_xy = self._random_xy()
        self.user_position = np.array([user_xy[0], user_xy[1], 0.0])
        self.eve_positions = self._generate_hppp_eves()
        self.num_eves = self.eve_positions.shape[0]

        n_types = len(self.config.vehicle_types)
        self.vehicles = []
        for i in range(self.config.num_vehicles):
            v_xy = self._random_xy()
            v_type = self.config.vehicle_types[i % n_types]
            v = Vehicle(
                vehicle_id=i,
                position=np.array([v_xy[0], v_xy[1]]),
                mobility_mode=self.config.vehicle_mobility_mode,
                max_speed=float(
                    np.random.uniform(5.0, self.config.vehicle_max_speed)
                ),
                vehicle_type=v_type,
            )
            self.vehicles.append(v)

    def _update_vehicles(self) -> None:
        for v in self.vehicles:
            v.update(dt=self.config.dt, half_area=self.half_area)

    def _compute_sensing(self) -> dict:
        """Compute sensing channels, distances, echo, and SNR for all vehicles."""
        noise_power = self.config.noise_psd * self.config.bandwidth

        per_target = []
        total_sensing_gain = 0.0
        total_snr = 0.0
        combined_echo = 0.0 + 0.0j

        s_symbol = 1.0 + 0j

        for v in self.vehicles:
            dists = compute_bistatic_distances(self.ris_position, v.position)
            h_sense = generate_sensing_channel(
                dists["d_tx"],
                dists["d_rx"],
                v.rcs,
                self.config.rician_k,
                self.config.alpha,
                self.config.beta0,
            )
            gain = compute_sensing_gain(h_sense)
            snr = compute_sensing_snr(
                self.config.sensing_power, gain, noise_power
            )
            echo, _ = compute_echo_signal(
                self.config.sensing_power, h_sense, s_symbol, noise_power
            )

            per_target.append({
                "vehicle_id": v.vehicle_id,
                "vehicle_type": v.vehicle_type,
                "rcs": v.rcs,
                "d_tx": dists["d_tx"],
                "d_rx": dists["d_rx"],
                "d_total": dists["d_total"],
                "h_sensing": h_sense,
                "gain": gain,
                "snr": snr,
                "echo": echo,
            })

            total_sensing_gain += gain
            total_snr += snr
            combined_echo += echo

        return {
            "per_target": per_target,
            "total_sensing_gain": total_sensing_gain,
            "total_snr": total_snr,
            "combined_echo": combined_echo,
            "noise_power": noise_power,
            "sensing_power": self.config.sensing_power,
            "num_vehicles": len(self.vehicles),
        }

    def reset(self) -> dict:
        self._step_counter = 0
        self._reset_positions()
        self.phases = np.random.uniform(0.0, 2.0 * np.pi, size=self.config.N_ris)
        self.Phi = compute_ris_reflection_matrix(self.phases)
        sensing = self._compute_sensing()
        return self.get_state(sensing)

    def get_state(self, sensing: dict | None = None) -> dict:
        if sensing is None:
            sensing = self._compute_sensing()
        return {
            "positions": {
                "bs": self.bs_position.copy(),
                "ris": self.ris_position.copy(),
                "user": self.user_position.copy(),
                "jammer": self.jammer_position.copy(),
                "eves": self.eve_positions.copy(),
            },
            "sensing": sensing,
        }

    def step(
        self,
        action_phases: np.ndarray | None = None,
    ) -> Tuple[dict, dict, bool, dict]:
        self._step_counter += 1
        if action_phases is not None:
            phases = np.clip(action_phases, 0.0, 2.0 * np.pi)
            self.phases = phases.copy()
            self.Phi = compute_ris_reflection_matrix(self.phases)
        else:
            self.phases = np.random.uniform(
                0.0, 2.0 * np.pi, size=self.config.N_ris
            )
            self.Phi = compute_ris_reflection_matrix(self.phases)

        self._update_vehicles()
        sensing = self._compute_sensing()
        done = self._step_counter >= self.config.max_steps
        return self.get_state(sensing), sensing, done, sensing
