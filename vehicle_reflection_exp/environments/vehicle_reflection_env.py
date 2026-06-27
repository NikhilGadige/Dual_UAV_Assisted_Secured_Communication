import numpy as np
from dataclasses import dataclass, field
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
    mrt_beamforming,
    nullspace_beamforming,
    compute_jammer_gain,
)
from vehicle_reflection_exp.channels.vehicle_channel import (
    Vehicle,
    compute_rcs,
    compute_reflection_channel_gain,
)


@dataclass
class VehicleReflectionConfig:
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
    alpha: float = 2.0
    beta0: float = 1.0
    N_ris: int = 16
    N_j: int = 4
    rician_k: float = 5.0
    output_root: str = "outputs/vehicle_reflection"
    beamforming_mode: str = "isotropic"
    eve_density_lambda: float = 2e-5
    eve_region_xmin: float = 0.0
    eve_region_xmax: float = 1000.0
    eve_region_ymin: float = 0.0
    eve_region_ymax: float = 1000.0

    num_vehicles: int = 3
    vehicle_max_speed: float = 10.0
    vehicle_mobility_mode: str = "straight_road"
    vehicle_types: tuple = ("car", "truck", "motorcycle")


class VehicleReflectionEnvironment:
    def __init__(self, config: VehicleReflectionConfig | None = None):
        self.config = config or VehicleReflectionConfig()
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

        self.h_BR = np.ones(self.config.N_ris, dtype=complex)
        self.h_RU = np.ones(self.config.N_ris, dtype=complex)
        self.h_RE = np.empty((0, self.config.N_ris), dtype=complex)
        self.h_JU = np.ones((1, self.config.N_j), dtype=complex)
        self.h_JE = np.empty((0, self.config.N_j), dtype=complex)

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

        self.vehicles = []
        n_types = len(self.config.vehicle_types)
        for i in range(self.config.num_vehicles):
            v_xy = self._random_xy()
            v_type = self.config.vehicle_types[i % n_types]
            v = Vehicle(
                vehicle_id=i,
                position=np.array([v_xy[0], v_xy[1]]),
                mobility_mode=self.config.vehicle_mobility_mode,
                max_speed=float(np.random.uniform(5.0, self.config.vehicle_max_speed)),
                vehicle_type=v_type,
            )
            self.vehicles.append(v)

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
        pl_JU = path_loss(
            self._distance_3d(self.jammer_position, self.user_position),
            self.config.alpha,
            self.config.beta0,
        )

        self.h_BR = generate_ris_rician_channel(
            self.config.N_ris, self.config.rician_k, pl_BR
        )
        self.h_RU = generate_ris_rician_channel(
            self.config.N_ris, self.config.rician_k, pl_RU
        )
        self.h_JU = generate_miso_rician_channel(
            self.config.N_j, self.config.rician_k, pl_JU
        )

        if self.num_eves > 0:
            h_RE_list = []
            h_JE_list = []
            for i in range(self.num_eves):
                ep = self.eve_positions[i]
                ep_3d = np.array([ep[0], ep[1], 0.0])
                pl_RE = path_loss(
                    self._distance_3d(self.ris_position, ep_3d),
                    self.config.alpha,
                    self.config.beta0,
                )
                pl_JE = path_loss(
                    self._distance_3d(self.jammer_position, ep_3d),
                    self.config.alpha,
                    self.config.beta0,
                )
                h_RE_list.append(
                    generate_ris_rician_channel(
                        self.config.N_ris, self.config.rician_k, pl_RE
                    )
                )
                h_JE_list.append(
                    generate_miso_rician_channel(
                        self.config.N_j, self.config.rician_k, pl_JE
                    )
                )
            self.h_RE = np.array(h_RE_list)
            self.h_JE = np.array(h_JE_list)
        else:
            self.h_RE = np.empty((0, self.config.N_ris), dtype=complex)
            self.h_JE = np.empty((0, self.config.N_j), dtype=complex)

    def _compute_beamforming(self) -> None:
        mode = self.config.beamforming_mode
        if self.num_eves == 0:
            self.w = isotropic_beamforming(self.config.N_j)
            return
        if mode == "isotropic":
            self.w = isotropic_beamforming(self.config.N_j)
        elif mode == "mrt":
            noise_power = self.config.noise_psd * self.config.bandwidth
            phases = np.zeros(self.config.N_ris)
            Phi_I = compute_ris_reflection_matrix(phases)
            g_eve_arr = np.array([
                float(
                    np.abs(
                        compute_effective_channel(self.h_RE[i], Phi_I, self.h_BR)
                    )
                    ** 2
                )
                for i in range(self.num_eves)
            ])
            gamma_e_arr = (self.config.bs_power * g_eve_arr) / noise_power
            R_eve_arr = self.config.bandwidth * np.log2(1.0 + gamma_e_arr)
            strongest_idx = int(np.argmax(R_eve_arr))
            self.w = mrt_beamforming(self.h_JE[strongest_idx])
        elif mode == "nullspace":
            self.w = nullspace_beamforming(self.h_JU, self.config.N_j)
        else:
            self.w = isotropic_beamforming(self.config.N_j)

    def _update_jammer_position(self) -> None:
        speed = np.linalg.norm(self.jammer_velocity)
        if speed < 1e-12:
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(2.0, 8.0)
            self.jammer_velocity = speed * np.array(
                [np.cos(angle), np.sin(angle)]
            )
        else:
            current_angle = np.arctan2(
                self.jammer_velocity[1], self.jammer_velocity[0]
            )
            new_angle = current_angle + np.random.uniform(-np.pi / 6, np.pi / 6)
            speed_jitter = np.random.uniform(0.9, 1.1)
            speed = np.clip(speed * speed_jitter, 0.0, 10.0)
            self.jammer_velocity = speed * np.array(
                [np.cos(new_angle), np.sin(new_angle)]
            )
        new_xy = self.jammer_position[:2] + self.jammer_velocity * self.config.dt
        for i in range(2):
            if new_xy[i] < -self.half_area:
                new_xy[i] = -2.0 * self.half_area - new_xy[i]
                self.jammer_velocity[i] *= -1.0
            elif new_xy[i] > self.half_area:
                new_xy[i] = 2.0 * self.half_area - new_xy[i]
                self.jammer_velocity[i] *= -1.0
        self.jammer_position[:2] = new_xy

    def _update_vehicles(self) -> None:
        for v in self.vehicles:
            v.update(dt=self.config.dt, half_area=self.half_area)

    def compute_vehicle_reflection_gains(self) -> dict:
        """Compute reflection channel gains for each vehicle.

        Returns dict with:
          - 'vehicle_to_user': list of gains |h_VU|^2 * rcs per vehicle
          - 'vehicle_to_eve': list of worst-case gains per vehicle
          - 'vehicle_info': list of vehicle state dicts
        """
        gains_uv = []
        gains_ve = []
        v_info = []

        for v in self.vehicles:
            d_RV = self._distance_2d(self.ris_position, v.position)
            d_VU = self._distance_2d(v.position, self.user_position)
            g_UV = compute_reflection_channel_gain(
                d_RV, d_VU, v.rcs, self.config.rician_k,
                self.config.alpha, self.config.beta0,
            )
            gains_uv.append(g_UV)

            if self.num_eves > 0:
                g_eves = []
                for ep in self.eve_positions:
                    d_VE = self._distance_2d(v.position, ep)
                    g_ve = compute_reflection_channel_gain(
                        d_RV, d_VE, v.rcs, self.config.rician_k,
                        self.config.alpha, self.config.beta0,
                    )
                    g_eves.append(g_ve)
                gains_ve.append(float(np.max(g_eves)))
            else:
                gains_ve.append(0.0)

            v_info.append({
                "id": v.vehicle_id,
                "position": v.position.copy(),
                "type": v.vehicle_type,
                "rcs": v.rcs,
                "speed": float(np.linalg.norm(v.velocity)),
            })

        return {
            "vehicle_to_user": gains_uv,
            "vehicle_to_eve": gains_ve,
            "vehicle_info": v_info,
        }

    def compute_rates(self) -> dict:
        noise_power = self.config.noise_psd * self.config.bandwidth
        g_user = float(
            np.abs(
                compute_effective_channel(self.h_RU, self.Phi, self.h_BR)
            )
            ** 2
        )
        jammer_interference_user = (
            self.config.jammer_power * compute_jammer_gain(self.h_JU, self.w)
        )
        sinr_user = (self.config.bs_power * g_user) / (
            jammer_interference_user + noise_power
        )
        R_legit = self.config.bandwidth * np.log2(1.0 + sinr_user)

        if self.num_eves > 0:
            g_eve_arr = np.array([
                float(
                    np.abs(
                        compute_effective_channel(self.h_RE[i], self.Phi, self.h_BR)
                    )
                    ** 2
                )
                for i in range(self.num_eves)
            ])
            jammer_gain_eve_arr = np.array([
                compute_jammer_gain(self.h_JE[i], self.w)
                for i in range(self.num_eves)
            ])
            jammer_interference_eve_arr = (
                self.config.jammer_power * jammer_gain_eve_arr
            )
            sinr_eve_arr = (self.config.bs_power * g_eve_arr) / (
                jammer_interference_eve_arr + noise_power
            )
            R_eve_arr = self.config.bandwidth * np.log2(1.0 + sinr_eve_arr)
            max_eve_idx = int(np.argmax(R_eve_arr))
            R_eve = float(R_eve_arr[max_eve_idx])
        else:
            R_eve = 0.0

        R_sec = max(R_legit - R_eve, 0.0)
        vg = self.compute_vehicle_reflection_gains()

        return {
            "g_user": float(g_user),
            "sinr_user": float(sinr_user),
            "sinr_eve_max": float(np.max(sinr_eve_arr)) if self.num_eves > 0 else 0.0,
            "R_legit": float(R_legit),
            "R_eve": float(R_eve),
            "R_sec": float(R_sec),
            "num_eves": self.num_eves,
            "noise_power": float(noise_power),
            "jammer_power": float(self.config.jammer_power),
            "num_vehicles": len(self.vehicles),
            "vehicle_reflection_gains_user": vg["vehicle_to_user"],
            "vehicle_reflection_gains_eve": vg["vehicle_to_eve"],
            "vehicle_info": vg["vehicle_info"],
        }

    def reset(self) -> dict:
        self._step_counter = 0
        self._reset_positions()
        self._generate_channels()
        self.phases = np.random.uniform(0.0, 2.0 * np.pi, size=self.config.N_ris)
        self.Phi = compute_ris_reflection_matrix(self.phases)
        self._compute_beamforming()
        return self.get_state()

    def set_phases(self, phases: np.ndarray) -> None:
        self.phases = phases.copy()
        self.Phi = compute_ris_reflection_matrix(self.phases)

    def set_jammer_power(self, power: float) -> None:
        self.config.jammer_power = float(
            np.clip(power, self.config.jammer_power_min, self.config.jammer_power_max)
        )

    def set_beamforming_mode(self, mode: str) -> None:
        self.config.beamforming_mode = mode
        self._compute_beamforming()

    def get_state(self) -> dict:
        rates = self.compute_rates()
        return {
            "positions": {
                "bs": self.bs_position.copy(),
                "ris": self.ris_position.copy(),
                "user": self.user_position.copy(),
                "jammer": self.jammer_position.copy(),
                "eves": self.eve_positions.copy(),
            },
            "phases": self.phases.copy(),
            "vehicles": rates.get("vehicle_info", []),
            "rates": rates,
        }

    def step(
        self,
        action_phases: np.ndarray | None = None,
        action_jammer_velocity: np.ndarray | None = None,
    ) -> Tuple[dict, float, bool, dict]:
        self._step_counter += 1
        if action_phases is not None:
            self.set_phases(np.clip(action_phases, 0.0, 2.0 * np.pi))
        else:
            self.phases = np.random.uniform(
                0.0, 2.0 * np.pi, size=self.config.N_ris
            )
            self.Phi = compute_ris_reflection_matrix(self.phases)

        if action_jammer_velocity is not None and not np.all(
            action_jammer_velocity == 0
        ):
            self.jammer_velocity = np.clip(action_jammer_velocity, -1.0, 1.0) * 10.0
        else:
            self._update_jammer_position()

        self._update_vehicles()
        self._generate_channels()
        self._compute_beamforming()
        rates = self.compute_rates()
        reward = rates["R_sec"]
        done = self._step_counter >= self.config.max_steps
        return self.get_state(), reward, done, rates
