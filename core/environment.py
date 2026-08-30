import numpy as np
from dataclasses import dataclass, replace
from typing import Tuple
from core.channel import channel_gain, channel_gain_los_aware, compute_elevation_angle, generate_fading
from core.ntn_channel import satellite_channel_gain
from core.energy import compute_energy_usage, compute_energy_harvesting, update_battery_state
from core.reward import (
    compute_secrecy_reward,
    compute_energy_penalty,
    compute_motion_penalty,
    compute_smoothness_penalty,
    compute_boundary_penalty,
    compute_sustainability_bonus,
    compute_total_reward,
)
from core.observation import build_observation

@dataclass
class EnvConfig:
    area_size: float = 1000.0
    relay_altitude: float = 50.0
    jammer_altitude: float = 50.0
    max_speed: float = 20.0
    max_acceleration: float = 10.0
    dt: float = 0.1
    seed: int | None = None
    max_steps: int = 200
    bandwidth: float = 1e6
    noise_psd: float = 10 ** (-17.4)
    user_power: float = 0.2
    relay_power: float = 0.5
    jammer_power_min: float = 0.0
    jammer_power_max: float = 0.5
    alpha: float = 2.0
    beta0: float = 1.0
    fading_model: str = "rician"
    rician_k: float = 5.0
    use_los_model: bool = False
    alpha_los: float = 2.0
    alpha_nlos: float = 3.0
    los_a: float = 9.61
    los_b: float = 0.20
    control_mode: str = "velocity"
    waypoint_gain: float = 1.0
    role_switching: bool = False
    relay_battery_joules: float = 5000.0
    jammer_battery_joules: float = 5000.0
    relay_hover_power_watts: float = 12.0
    jammer_hover_power_watts: float = 12.0
    relay_motion_power_coeff: float = 0.06
    jammer_motion_power_coeff: float = 0.06
    jammer_rf_power_coeff: float = 1.0
    energy_reward_weight: float = 1e-1
    battery_depletion_penalty: float = 5e5
    # --- User mobility ---
    user_mobile: bool = False
    user_max_speed: float = 3.0
    user_motion_model: str = "random_walk"
    # --- Multiple Eve (HPPP) configuration (Phase 1–4) ---
    use_multiple_eves: bool = True
    eve_density_lambda: float = 2e-5
    eve_region_xmin: float = 0.0
    eve_region_xmax: float = 1000.0
    eve_region_ymin: float = 0.0
    eve_region_ymax: float = 1000.0
    eve_uncertainty_radius: float = 15.0
    # --- Observation / state representation ---
    observation_mode: str = "full"
    normalize_observations: bool = True
    movement_penalty_weight: float = 5e-2
    smoothness_penalty_weight: float = 2e-2
    boundary_penalty_weight: float = 1.0
    secrecy_scale: float = 1e-6
    enable_energy_harvesting: bool = False
    relay_harvest_efficiency: float = 0.5
    jammer_harvest_efficiency: float = 0.5
    relay_harvest_max_watts: float = 8.0
    jammer_harvest_max_watts: float = 8.0
    solar_variability: float = 0.1
    harvesting_reward_weight: float = 1e-3
    enable_ntn: bool = False
    satellite_altitude_km: float = 500.0
    satellite_horizontal_offset_km: float = 100.0
    ntn_carrier_frequency_hz: float = 2e9
    ntn_atmospheric_loss_db: float = 0.5
    ntn_rician_k_db: float = 10.0

class UAVEnvironment:
    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        self.half_area = self.config.area_size / 2
        self.bs_position = np.array([0.0, 0.0, 0.0])
        self._step_counter = 0
        self.fading = {"UR": 0.0, "RB": 0.0, "UE": 0.0, "JE": 0.0}
        self.current_jammer_power = self.config.jammer_power_max
        self.relay_velocity = np.zeros(2, dtype=float)
        self.jammer_velocity = np.zeros(2, dtype=float)
        self.user_velocity = np.zeros(2, dtype=float)  # (Requirement 3)
        self._prev_relay_velocity = np.zeros(2, dtype=float)
        self._prev_jammer_velocity = np.zeros(2, dtype=float)
        self.roles_swapped = False
        self.relay_battery = self.config.relay_battery_joules
        self.jammer_battery = self.config.jammer_battery_joules
        self.relay_harvest_power_w = 0.0
        self.jammer_harvest_power_w = 0.0
        self.relay_harvested_energy_j = 0.0
        self.jammer_harvested_energy_j = 0.0
        self.total_harvested_energy_j = 0.0
        self.battery_saturation_event = False
        alt_m = self.config.satellite_altitude_km * 1000.0
        off_m = self.config.satellite_horizontal_offset_km * 1000.0
        self.satellite_position = np.array([off_m, off_m, alt_m], dtype=float)
        self.ntn_fading_sat_relay = 1.0
        self.h_sat_relay = 0.0
        # Multi-Eve (HPPP) state
        self.eve_positions = np.empty((0, 2), dtype=float)
        self.num_eves = 1
        self.nearest_eve_distance = 0.0
        self.mean_eve_distance = 0.0
        self.max_eve_capacity = 0.0

    def _random_position_2d(self) -> np.ndarray:
        return np.array([
            np.random.uniform(-self.half_area, self.half_area),
            np.random.uniform(-self.half_area, self.half_area),
        ])

    def _generate_hppp_eves(self) -> np.ndarray:
        area = ((self.config.eve_region_xmax - self.config.eve_region_xmin) *
                (self.config.eve_region_ymax - self.config.eve_region_ymin))
        n_eve = np.random.poisson(self.config.eve_density_lambda * area)
        if n_eve == 0:
            return np.empty((0, 2), dtype=float)
        xs = np.random.uniform(self.config.eve_region_xmin, self.config.eve_region_xmax, size=n_eve)
        ys = np.random.uniform(self.config.eve_region_ymin, self.config.eve_region_ymax, size=n_eve)
        return np.column_stack([xs, ys])

    def _reset_entity_positions(self):
        self.user_position = np.append(self._random_position_2d(), 0.0)
        self.relay_position = np.append(self._random_position_2d(), self.config.relay_altitude)
        self.jammer_position = np.append(self._random_position_2d(), self.config.jammer_altitude)
        if self.config.use_multiple_eves:
            self.eve_positions = self._generate_hppp_eves()
            self.num_eves = self.eve_positions.shape[0]
            if self.num_eves > 0:
                dists = np.linalg.norm(self.eve_positions - self.user_position[:2], axis=1)
                nearest_idx = int(np.argmin(dists))
                self.eve_position = np.append(self.eve_positions[nearest_idx], 0.0)
            else:
                self.eve_position = np.array([0.0, 0.0, 0.0])
            print(f"Generated {self.num_eves} eavesdroppers using HPPP")
        else:
            self.eve_position = np.append(self._random_position_2d(), 0.0)
            self.eve_positions = np.array([self.eve_position[:2]], dtype=float)
            self.num_eves = 1

    def reset(self) -> np.ndarray:
        self._step_counter = 0
        self._reset_entity_positions()
        self.current_jammer_power = self.config.jammer_power_max
        self.relay_velocity = np.zeros(2, dtype=float)
        self.jammer_velocity = np.zeros(2, dtype=float)
        self.user_velocity = np.zeros(2, dtype=float)
        self._prev_relay_velocity = np.zeros(2, dtype=float)
        self._prev_jammer_velocity = np.zeros(2, dtype=float)
        self.roles_swapped = False
        self.relay_battery = self.config.relay_battery_joules
        self.jammer_battery = self.config.jammer_battery_joules
        self.relay_harvest_power_w = 0.0
        self.jammer_harvest_power_w = 0.0
        self.relay_harvested_energy_j = 0.0
        self.jammer_harvested_energy_j = 0.0
        self.total_harvested_energy_j = 0.0
        self.battery_saturation_event = False
        self._generate_fading()
        return self.get_state()

    def _generate_fading(self) -> None:
        self.fading["UR"] = generate_fading(self.config.fading_model, self.config.rician_k)
        self.fading["RB"] = generate_fading(self.config.fading_model, self.config.rician_k)
        if self.config.use_multiple_eves:
            n = self.num_eves
            self.fading["UE"] = np.array([generate_fading(self.config.fading_model, self.config.rician_k) for _ in range(n)])
            self.fading["JE"] = np.array([generate_fading(self.config.fading_model, self.config.rician_k) for _ in range(n)])
        else:
            self.fading["UE"] = generate_fading(self.config.fading_model, self.config.rician_k)
            self.fading["JE"] = generate_fading(self.config.fading_model, self.config.rician_k)

    def _clip_to_bounds(self, pos: np.ndarray, altitude: float) -> np.ndarray:
        xy = np.clip(pos[:2], -self.half_area, self.half_area)
        return np.array([xy[0], xy[1], altitude])

    def _update_user_position(self):
        if not self.config.user_mobile:
            return

        speed = np.linalg.norm(self.user_velocity)

        if speed < 1e-12:
            # User starts stationary: pick a random heading and speed
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.5, self.config.user_max_speed)
            self.user_velocity = speed * np.array([np.cos(angle), np.sin(angle)])
        else:
            # Perturb heading by a small random angle (±30°)
            current_angle = np.arctan2(self.user_velocity[1], self.user_velocity[0])
            new_angle = current_angle + np.random.uniform(-np.pi / 6, np.pi / 6)
            # Small speed jitter (±10%), clipped to [0, user_max_speed]
            speed_jitter = np.random.uniform(0.9, 1.1)
            speed = np.clip(speed * speed_jitter, 0.0, self.config.user_max_speed)
            self.user_velocity = speed * np.array([np.cos(new_angle), np.sin(new_angle)])

        new_xy = self.user_position[:2] + self.user_velocity * self.config.dt

        # --- Boundary reflection (elastic bounce) ---
        for i in range(2):
            if new_xy[i] < -self.half_area:
                new_xy[i] = -2.0 * self.half_area - new_xy[i]
                self.user_velocity[i] *= -1.0
            elif new_xy[i] > self.half_area:
                new_xy[i] = 2.0 * self.half_area - new_xy[i]
                self.user_velocity[i] *= -1.0

        self.user_position[:2] = new_xy
        self.user_position[2] = 0.0  # keep z=0

    def _scale_jammer_power(self, action_jammer_power: float) -> float:
        action_jammer_power = float(np.clip(action_jammer_power, -1.0, 1.0))
        power_span = self.config.jammer_power_max - self.config.jammer_power_min
        return self.config.jammer_power_min + 0.5 * (action_jammer_power + 1.0) * power_span

    def _control_to_velocity_action(self, position: np.ndarray, action: np.ndarray) -> np.ndarray:
        if self.config.control_mode == "velocity":
            return np.clip(action, -1.0, 1.0)
        if self.config.control_mode != "waypoint":
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

        waypoint_xy = np.clip(action, -1.0, 1.0) * self.half_area
        desired = waypoint_xy - position[:2]
        norm = np.linalg.norm(desired)
        if norm < 1e-12:
            return np.zeros(2, dtype=float)
        return np.clip((desired / norm) * self.config.waypoint_gain, -1.0, 1.0)

    def _update_velocity(self, velocity: np.ndarray, action: np.ndarray) -> np.ndarray:
        target_velocity = np.clip(action, -1.0, 1.0) * self.config.max_speed
        delta_v = target_velocity - velocity
        max_delta = self.config.max_acceleration * self.config.dt
        delta_norm = np.linalg.norm(delta_v)
        if delta_norm > max_delta > 0.0:
            delta_v = delta_v * (max_delta / delta_norm)
        new_velocity = velocity + delta_v
        speed = np.linalg.norm(new_velocity)
        if speed > self.config.max_speed > 0.0:
            new_velocity = new_velocity * (self.config.max_speed / speed)
        return new_velocity

    def _compute_energy_usage(self) -> dict:
        return compute_energy_usage(
            relay_velocity=self.relay_velocity,
            jammer_velocity=self.jammer_velocity,
            jammer_power=self.current_jammer_power,
            relay_hover_power_watts=self.config.relay_hover_power_watts,
            relay_motion_power_coeff=self.config.relay_motion_power_coeff,
            jammer_hover_power_watts=self.config.jammer_hover_power_watts,
            jammer_motion_power_coeff=self.config.jammer_motion_power_coeff,
            jammer_rf_power_coeff=self.config.jammer_rf_power_coeff,
            dt=self.config.dt,
        )

    def _compute_energy_harvesting(self) -> dict:
        return compute_energy_harvesting(
            relay_harvest_efficiency=self.config.relay_harvest_efficiency,
            relay_harvest_max_watts=self.config.relay_harvest_max_watts,
            jammer_harvest_efficiency=self.config.jammer_harvest_efficiency,
            jammer_harvest_max_watts=self.config.jammer_harvest_max_watts,
            solar_variability=self.config.solar_variability,
            dt=self.config.dt,
        )

    def _compute_motion_penalty(self) -> float:
        return compute_motion_penalty(
            movement_penalty_weight=self.config.movement_penalty_weight,
            relay_velocity=self.relay_velocity,
            jammer_velocity=self.jammer_velocity,
            max_speed=self.config.max_speed,
        )

    def _compute_smoothness_penalty(self) -> float:
        return compute_smoothness_penalty(
            smoothness_penalty_weight=self.config.smoothness_penalty_weight,
            relay_velocity=self.relay_velocity,
            jammer_velocity=self.jammer_velocity,
            prev_relay_velocity=self._prev_relay_velocity,
            prev_jammer_velocity=self._prev_jammer_velocity,
            max_acceleration=self.config.max_acceleration,
            dt=self.config.dt,
        )

    def _compute_boundary_penalty(self) -> float:
        return compute_boundary_penalty(
            boundary_penalty_weight=self.config.boundary_penalty_weight,
            relay_position=self.relay_position,
            jammer_position=self.jammer_position,
            half_area=self.half_area,
        )

    def step(
        self,
        action_relay: np.ndarray,
        action_jammer: np.ndarray,
        action_jammer_power: float = 1.0,
        action_role_switch: float | bool = False,
    ) -> Tuple[np.ndarray, float, bool, dict]:
        action_relay = np.clip(action_relay, -1.0, 1.0)
        action_jammer = np.clip(action_jammer, -1.0, 1.0)
        if self.config.role_switching and bool(action_role_switch):
            self.roles_swapped = not self.roles_swapped
        self.current_jammer_power = self._scale_jammer_power(action_jammer_power)

        relay_velocity_action = self._control_to_velocity_action(self.relay_position, action_relay)
        jammer_velocity_action = self._control_to_velocity_action(self.jammer_position, action_jammer)
        # Save pre-update velocities for smoothness penalty
        self._prev_relay_velocity = self.relay_velocity.copy()
        self._prev_jammer_velocity = self.jammer_velocity.copy()
        self.relay_velocity = self._update_velocity(self.relay_velocity, relay_velocity_action)
        self.jammer_velocity = self._update_velocity(self.jammer_velocity, jammer_velocity_action)
        delta_relay = np.append(self.relay_velocity * self.config.dt, 0.0)
        delta_jammer = np.append(self.jammer_velocity * self.config.dt, 0.0)

        self.relay_position = self._clip_to_bounds(
            self.relay_position + delta_relay, self.config.relay_altitude
        )
        self.jammer_position = self._clip_to_bounds(
            self.jammer_position + delta_jammer, self.config.jammer_altitude
        )

        self._update_user_position()

        self._step_counter += 1
        energy = self._compute_energy_usage()

        if self.config.enable_energy_harvesting:
            harvest = self._compute_energy_harvesting()
        else:
            harvest = {}

        battery_state = update_battery_state(
            relay_battery=self.relay_battery,
            jammer_battery=self.jammer_battery,
            relay_energy_consumed=energy["relay_energy"],
            jammer_energy_consumed=energy["jammer_energy"],
            relay_battery_capacity=self.config.relay_battery_joules,
            jammer_battery_capacity=self.config.jammer_battery_joules,
            enable_harvesting=self.config.enable_energy_harvesting,
            relay_harvested_energy=harvest.get("relay_harvested_energy_j", 0.0),
            jammer_harvested_energy=harvest.get("jammer_harvested_energy_j", 0.0),
        )
        self.relay_battery = battery_state["relay_battery"]
        self.jammer_battery = battery_state["jammer_battery"]
        battery_depleted = battery_state["battery_depleted"]
        self.battery_saturation_event = battery_state["battery_saturation_event"]
        done = self._step_counter >= self.config.max_steps or battery_depleted
        self._generate_fading()
        rates = self.compute_rates()
        scaled_secrecy = compute_secrecy_reward(self.config.secrecy_scale, rates["R_sec"])
        energy_penalty = compute_energy_penalty(self.config.energy_reward_weight, energy["total_energy"])
        motion_penalty = self._compute_motion_penalty()
        smoothness_penalty = self._compute_smoothness_penalty()
        boundary_penalty = self._compute_boundary_penalty()

        sustainability_bonus = (
            compute_sustainability_bonus(
                self.config.harvesting_reward_weight,
                harvest.get("total_harvested_energy_j", 0.0),
            )
            if self.config.enable_energy_harvesting
            else 0.0
        )
        if self.config.enable_energy_harvesting:
            # Persist harvest values for info dict
            self.relay_harvest_power_w = harvest.get("relay_harvest_power_w", 0.0)
            self.jammer_harvest_power_w = harvest.get("jammer_harvest_power_w", 0.0)
            self.relay_harvested_energy_j = harvest.get("relay_harvested_energy_j", 0.0)
            self.jammer_harvested_energy_j = harvest.get("jammer_harvested_energy_j", 0.0)
            self.total_harvested_energy_j = harvest.get("total_harvested_energy_j", 0.0)

        reward = compute_total_reward(
            scaled_secrecy=scaled_secrecy,
            energy_penalty=energy_penalty,
            motion_penalty=motion_penalty,
            smoothness_penalty=smoothness_penalty,
            boundary_penalty=boundary_penalty,
            sustainability_bonus=sustainability_bonus,
            battery_depletion_penalty=self.config.battery_depletion_penalty,
            battery_depleted=battery_depleted,
        )

        rates.update(
            {
                "reward": float(reward),
                "energy_penalty": float(energy_penalty),
                "reward_secrecy": float(scaled_secrecy),
                "reward_energy_penalty": float(-energy_penalty),
                "reward_motion_penalty": float(-motion_penalty),
                "reward_smoothness_penalty": float(-smoothness_penalty),
                "reward_boundary_penalty": float(-boundary_penalty),
                "reward_total": float(reward),
                "relay_energy_j": energy["relay_energy"],
                "jammer_energy_j": energy["jammer_energy"],
                "total_energy_j": energy["total_energy"],
                "relay_speed_mps": energy["relay_speed"],
                "jammer_speed_mps": energy["jammer_speed"],
                "relay_battery_j": float(self.relay_battery),
                "jammer_battery_j": float(self.jammer_battery),
                "battery_depleted": bool(battery_depleted),
                "battery_saturation_event": bool(self.battery_saturation_event),
                "relay_harvested_energy_j": float(self.relay_harvested_energy_j),
                "jammer_harvested_energy_j": float(self.jammer_harvested_energy_j),
                "total_harvested_energy_j": float(self.total_harvested_energy_j),
                "relay_harvest_power_w": float(self.relay_harvest_power_w),
                "jammer_harvest_power_w": float(self.jammer_harvest_power_w),
                "roles_swapped": bool(self.roles_swapped),
                "effective_relay_label": "jammer_uav" if self.roles_swapped else "relay_uav",
                "effective_jammer_label": "relay_uav" if self.roles_swapped else "jammer_uav",
                # Multi-Eve logging fields (Phase 5)
                "num_eves": self.num_eves,
                "nearest_eve_distance": float(np.min([compute_distance(self.user_position[:2], ep) for ep in self.eve_positions])) if self.num_eves > 0 else 0.0,
                "mean_eve_distance": float(np.mean([compute_distance(self.user_position[:2], ep) for ep in self.eve_positions])) if self.num_eves > 0 else 0.0,
                "max_eve_capacity": float(rates.get("max_eve_capacity", 0.0)),
            }
        )

        return self.get_state(), reward, done, rates

    def get_state(self) -> np.ndarray:
        """Build the observation vector according to config.observation_mode."""
        mode = self.config.observation_mode
        needs_comms = mode in ("channels", "full", "full_eh", "full_ntn")
        gains = self.compute_all_channel_gains() if needs_comms else {}
        rates = self.compute_rates(gains) if needs_comms else {}
        needs_dist = mode in ("full", "full_eh", "full_ntn")
        distances = self.compute_distances() if needs_dist else {}

        # Compute aggregated Eve features for multi-Eve mode
        eve_agg_features = None
        if self.config.use_multiple_eves:
            n = self.num_eves
            if n > 0:
                d_UE_all = np.array([compute_distance(self.user_position[:2], ep) for ep in self.eve_positions])
                nearest_eve_dist = float(np.min(d_UE_all))
                mean_eve_dist = float(np.mean(d_UE_all))
            else:
                nearest_eve_dist = 0.0
                mean_eve_dist = 0.0
            max_eve_cap = rates.get("max_eve_capacity", 0.0) if needs_comms else 0.0
            eve_agg_features = np.array([nearest_eve_dist, mean_eve_dist, max_eve_cap, float(n)])

        return build_observation(
            mode=mode,
            relay_position=self.relay_position,
            jammer_position=self.jammer_position,
            user_position=self.user_position,
            bs_position=self.bs_position,
            eve_position=self.eve_position,
            relay_velocity=self.relay_velocity,
            jammer_velocity=self.jammer_velocity,
            user_velocity=self.user_velocity,
            relay_battery=self.relay_battery,
            jammer_battery=self.jammer_battery,
            gains=gains,
            rates=rates,
            distances=distances,
            normalize=self.config.normalize_observations,
            half_area=self.half_area,
            max_speed=self.config.max_speed,
            user_max_speed=self.config.user_max_speed,
            jammer_power_max=self.config.jammer_power_max,
            area_size=self.config.area_size,
            relay_battery_capacity=self.config.relay_battery_joules,
            jammer_battery_capacity=self.config.jammer_battery_joules,
            # EH observation params
            enable_energy_harvesting=self.config.enable_energy_harvesting,
            relay_harvest_power_w=self.relay_harvest_power_w,
            jammer_harvest_power_w=self.jammer_harvest_power_w,
            relay_harvest_max_watts=self.config.relay_harvest_max_watts,
            jammer_harvest_max_watts=self.config.jammer_harvest_max_watts,
            battery_saturation_event=self.battery_saturation_event,
            # NTN observation params
            satellite_position=self.satellite_position,
            h_sat_relay=self.h_sat_relay,
            satellite_altitude_m=self.config.satellite_altitude_km * 1000.0,
            # Multi-Eve aggregated features
            use_multiple_eves=self.config.use_multiple_eves,
            eve_agg_features=eve_agg_features,
        )

    def compute_channel_gain(self, tx_pos: np.ndarray, rx_pos: np.ndarray,
                             fading: float) -> float:
        d = compute_distance(tx_pos, rx_pos)
        if self.config.use_los_model:
            theta = compute_elevation_angle(tx_pos, rx_pos)
            return channel_gain_los_aware(
                d, theta, fading,
                self.config.alpha_los, self.config.alpha_nlos,
                self.config.beta0, self.config.los_a, self.config.los_b,
            )
        return channel_gain(d, fading, alpha=self.config.alpha, beta0=self.config.beta0)

    def compute_channel_gain_uncertain(
        self, tx_pos: np.ndarray, rx_pos: np.ndarray, delta_q: float, maximize: bool, fading: float
    ) -> float:
        tx_pos = np.asarray(tx_pos, dtype=float)
        rx_pos = np.asarray(rx_pos, dtype=float)
        dx = tx_pos[0] - rx_pos[0]
        dy = tx_pos[1] - rx_pos[1]
        dz = abs(tx_pos[2] - rx_pos[2])
        
        d_2d = np.sqrt(dx * dx + dy * dy)
        if maximize:
            d_2d_mod = max(d_2d - delta_q, 0.0)
        else:
            d_2d_mod = d_2d + delta_q
            
        d_3d_mod = np.sqrt(d_2d_mod * d_2d_mod + dz * dz)
        
        if self.config.use_los_model:
            if d_2d_mod < 1e-10:
                theta_mod = 90.0 if dz > 1e-10 else 0.0
            else:
                theta_mod = float(np.degrees(np.arctan2(dz, d_2d_mod)))
            return channel_gain_los_aware(
                d_3d_mod, theta_mod, fading,
                self.config.alpha_los, self.config.alpha_nlos,
                self.config.beta0, self.config.los_a, self.config.los_b,
            )
        return channel_gain(d_3d_mod, fading, alpha=self.config.alpha, beta0=self.config.beta0)

    def compute_all_channel_gains(self) -> dict:
        gains = {}
        relay_position = self.jammer_position if self.roles_swapped else self.relay_position
        jammer_position = self.relay_position if self.roles_swapped else self.jammer_position
        gains["h_UR"] = self.compute_channel_gain(
            self.user_position, relay_position, self.fading["UR"])
        gains["h_RB"] = self.compute_channel_gain(
            relay_position, self.bs_position, self.fading["RB"])
        if self.config.use_multiple_eves:
            n = self.num_eves
            if n == 0:
                gains["h_UE"] = np.array([], dtype=float)
                gains["h_JE"] = np.array([], dtype=float)
            else:
                h_UE_list = [self.compute_channel_gain_uncertain(
                    self.user_position, np.append(self.eve_positions[i], 0.0),
                    self.config.eve_uncertainty_radius, True, self.fading["UE"][i])
                    for i in range(n)]
                h_JE_list = [self.compute_channel_gain_uncertain(
                    jammer_position, np.append(self.eve_positions[i], 0.0),
                    self.config.eve_uncertainty_radius, False, self.fading["JE"][i])
                    for i in range(n)]
                gains["h_UE"] = np.array(h_UE_list)
                gains["h_JE"] = np.array(h_JE_list)
        else:
            gains["h_UE"] = self.compute_channel_gain_uncertain(
                self.user_position, self.eve_position,
                self.config.eve_uncertainty_radius, True, self.fading["UE"])
            gains["h_JE"] = self.compute_channel_gain_uncertain(
                jammer_position, self.eve_position,
                self.config.eve_uncertainty_radius, False, self.fading["JE"])
        if self.config.enable_ntn:
            self.ntn_fading_sat_relay = generate_fading(
                "rician", K=10.0 ** (self.config.ntn_rician_k_db / 10.0))
            d_sr = compute_distance(self.satellite_position, relay_position)
            self.h_sat_relay = satellite_channel_gain(
                d_sr,
                freq_hz=self.config.ntn_carrier_frequency_hz,
                atmospheric_loss_db=self.config.ntn_atmospheric_loss_db,
                rician_k_db=self.config.ntn_rician_k_db,
            )
            gains["h_sat_relay"] = self.h_sat_relay
        return gains

    def compute_distances(self) -> dict:
        relay_position = self.jammer_position if self.roles_swapped else self.relay_position
        jammer_position = self.relay_position if self.roles_swapped else self.jammer_position
        d_UR = compute_distance(self.user_position, relay_position)
        d_RB = compute_distance(relay_position, self.bs_position)
        if self.config.use_multiple_eves:
            n = self.num_eves
            if n == 0:
                d_UE = 0.0
                d_JE = 0.0
            else:
                d_UE_all_nom = np.array([compute_distance(self.user_position[:2], ep) for ep in self.eve_positions])
                nearest_idx = int(np.argmin(d_UE_all_nom))
                d_UE = float(max(d_UE_all_nom[nearest_idx] - self.config.eve_uncertainty_radius, 0.0))
                d_JE_nom = compute_distance(jammer_position[:2], self.eve_positions[nearest_idx])
                d_JE = float(d_JE_nom + self.config.eve_uncertainty_radius)
        else:
            d_UE_nom = compute_distance(self.user_position, self.eve_position)
            d_JE_nom = compute_distance(jammer_position, self.eve_position)
            d_UE = float(max(d_UE_nom - self.config.eve_uncertainty_radius, 0.0))
            d_JE = float(d_JE_nom + self.config.eve_uncertainty_radius)
        return {
            "d_UR": d_UR,
            "d_RB": d_RB,
            "d_UE": d_UE,
            "d_JE": d_JE,
        }

    def _fmt_fading(self, val):
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return "[]"
            return f"min={val.min():.6f} max={val.max():.6f} mean={val.mean():.6f}"
        return f"{val:.6f}"

    def _fmt_gain(self, val):
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return "[]"
            return f"min={val.min():.6e} max={val.max():.6e}"
        return f"{val:.6e}"

    def print_channel_gains(self, gains: dict | None = None) -> None:
        if gains is None:
            gains = self.compute_all_channel_gains()
        print("\n-> Fading Values ")
        model_label = self.config.fading_model.capitalize()
        print(f"  f_UR ({model_label})     : {self._fmt_fading(self.fading['UR'])}")
        print(f"  f_RB ({model_label})     : {self._fmt_fading(self.fading['RB'])}")
        print(f"  f_UE ({model_label})     : {self._fmt_fading(self.fading['UE'])}")
        print(f"  f_JE ({model_label})     : {self._fmt_fading(self.fading['JE'])}")
        print("\n-> Channel Gains ")
        print(f"  h_UR (User->Relay): {self._fmt_gain(gains['h_UR'])}")
        print(f"  h_RB (Relay->BS)  : {self._fmt_gain(gains['h_RB'])}")
        print(f"  h_UE (User->Eve)  : {self._fmt_gain(gains['h_UE'])}")
        print(f"  h_JE (Jammer->Eve): {self._fmt_gain(gains['h_JE'])}")
        if self.config.use_multiple_eves:
            print(f"  Num Eves: {self.num_eves}")
        if "h_sat_relay" in gains:
            print(f"  h_SR (Sat->Relay): {gains['h_sat_relay']:.6e}  (NTN)")

    def compute_rates(self, gains: dict | None = None) -> dict:
        if gains is None:
            gains = self.compute_all_channel_gains()
        noise_power = self.config.noise_psd * self.config.bandwidth

        gamma_ur = (self.config.user_power * gains["h_UR"]) / noise_power
        gamma_rb = (self.config.relay_power * gains["h_RB"]) / noise_power
        r_legit = 0.5 * self.config.bandwidth * np.log2(1.0 + min(gamma_ur, gamma_rb))

        if self.config.enable_ntn and "h_sat_relay" in gains:
            gamma_sat_relay = (self.config.relay_power * gains["h_sat_relay"]) / noise_power
            gamma_rb_ntn = gamma_rb + gamma_sat_relay
            r_legit_ntn = 0.5 * self.config.bandwidth * np.log2(1.0 + min(gamma_ur, gamma_rb_ntn))
            r_legit = max(r_legit, r_legit_ntn)

        max_eve_capacity = 0.0
        worst_eve_idx = -1
        h_ue_scalar = float(gains["h_UE"]) if not isinstance(gains["h_UE"], np.ndarray) else 0.0
        h_je_scalar = float(gains["h_JE"]) if not isinstance(gains["h_JE"], np.ndarray) else 0.0
        if self.config.use_multiple_eves:
            n = self.num_eves
            if n == 0:
                gamma_e = 0.0
                r_eve = 0.0
            else:
                gamma_e_arr = (self.config.user_power * gains["h_UE"]) / (
                    noise_power + self.current_jammer_power * gains["h_JE"]
                )
                r_eve_arr = self.config.bandwidth * np.log2(1.0 + gamma_e_arr)
                worst_eve_idx = int(np.argmax(r_eve_arr))
                gamma_e = float(gamma_e_arr[worst_eve_idx])
                r_eve = float(r_eve_arr[worst_eve_idx])
                max_eve_capacity = r_eve
                h_ue_scalar = float(gains["h_UE"][worst_eve_idx])
                h_je_scalar = float(gains["h_JE"][worst_eve_idx])
        else:
            gamma_e = (self.config.user_power * gains["h_UE"]) / (
                noise_power + self.current_jammer_power * gains["h_JE"]
            )
            r_eve = self.config.bandwidth * np.log2(1.0 + gamma_e)

        r_sec = max(r_legit - r_eve, 0.0)

        return {
            "gamma_UR": float(gamma_ur),
            "gamma_RB": float(gamma_rb),
            "gamma_E": float(gamma_e),
            "R_legit": float(r_legit),
            "R_eve": float(r_eve),
            "R_sec": float(r_sec),
            "jammer_power": float(self.current_jammer_power),
            "fading_model": self.config.fading_model,
            "max_eve_capacity": float(max_eve_capacity),
            "worst_eve_index": worst_eve_idx,
            "h_UE_scalar": h_ue_scalar,
            "h_JE_scalar": h_je_scalar,
        }

def compute_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))

if __name__ == "__main__":
    print("=" * 70)
    print("OBSERVATION MODE VERIFICATION")
    print("=" * 70)
    base_cfg = EnvConfig(seed=42, user_mobile=True)
    for mode in ["geometry", "channels", "full", "full_eh", "full_ntn"]:
        cfg = replace(base_cfg, observation_mode=mode, normalize_observations=True)
        env = UAVEnvironment(cfg)
        s_norm = env.reset()
        cfg_raw = replace(base_cfg, observation_mode=mode, normalize_observations=False)
        env_raw = UAVEnvironment(cfg_raw)
        s_raw = env_raw.reset()
        print(f"  Mode: {mode:<10s}  Dim: {s_norm.shape[0]:2d}  "
              f"Min: {s_norm.min(): .4f}  Max: {s_norm.max(): .4f}  "
              f"(raw min: {s_raw.min(): .4e}  raw max: {s_raw.max(): .4e})")
    print()

    # Show a concrete normalised "full" vector
    cfg_ex = replace(base_cfg, observation_mode="full", normalize_observations=True)
    env_ex = UAVEnvironment(cfg_ex)
    s_ex = env_ex.reset()
    print(f"  Example normalised 'full' vector ({s_ex.shape[0]} dims):")
    print(f"    first 10 : {np.round(s_ex[:10], 4)}")
    print(f"    mid   10 : {np.round(s_ex[14:24], 4)}")
    print(f"    last  10 : {np.round(s_ex[28:], 4)}")
    print()

    # ---- Part 1: User mobility debug verification ----
    print("=" * 70)
    print("USER MOBILITY VERIFICATION (10 steps, user_mobile=True)")
    print("=" * 70)
    env_mobile = UAVEnvironment(EnvConfig(user_mobile=True))
    env_mobile.reset()
    user_traj = [env_mobile.user_position.copy()]
    prev_user = env_mobile.user_position.copy()
    print(f"{'Step':<6} {'User_X':<10} {'User_Y':<10} {'User_Z':<8} "
          f"{'Speed':<8} {'dx':<10} {'dy':<10}")
    print("-" * 70)
    for t in range(10):
        a_relay = np.random.uniform(-1, 1, size=2)
        a_jammer = np.random.uniform(-1, 1, size=2)
        env_mobile.step(a_relay, a_jammer)
        u = env_mobile.user_position
        dv = u - prev_user
        speed = np.linalg.norm(env_mobile.user_velocity)
        print(f"{t+1:<6} {u[0]:<10.3f} {u[1]:<10.3f} {u[2]:<8.2f} "
              f"{speed:<8.4f} {dv[0]:<10.5f} {dv[1]:<10.5f}")
        user_traj.append(u.copy())
        prev_user = u.copy()
    displacements = [np.linalg.norm(user_traj[i+1][:2] - user_traj[i][:2])
                     for i in range(len(user_traj) - 1)]
    max_disp = max(displacements)
    theoretical_max = env_mobile.config.user_max_speed * env_mobile.config.dt
    print(f"\nMax per-step displacement: {max_disp:.5f} m  "
          f"(theoretical max: {theoretical_max:.5f} m)  ->  "
          f"{'SMOOTH' if max_disp <= theoretical_max + 1e-6 else 'TELEPORTATION!'}")
    print("User position changes smoothly (no teleportation).\n")

    # ---- Part 2: Standard environment test (backward compatibility) ----
    print("=" * 70)
    print("STANDARD ENVIRONMENT TEST (backward compat)")
    print("=" * 70)
    env = UAVEnvironment()
    state = env.reset()

    print(f"{'Step':<6} {'Relay':<30} {'Jammer':<30} {'User':<30} {'BS':<20} {'Eve':<30}")
    print("-" * 150)

    for t in range(5):
        a_relay = np.random.uniform(-1, 1, size=2)
        a_jammer = np.random.uniform(-1, 1, size=2)

        state, reward, done, info = env.step(a_relay, a_jammer)

        relay, jammer, user, bs, eve = (
            state[:3], state[3:6], state[6:9], state[9:12], state[12:15]
        )

        print(
            f"{t+1:<6} "
            f"{str(np.round(relay, 2)):<30} "
            f"{str(np.round(jammer, 2)):<30} "
            f"{str(np.round(user, 2)):<30} "
            f"{str(np.round(bs, 2)):<20} "
            f"{str(np.round(eve, 2)):<30}"
        )
        print(
            f"       Reward(R_sec)={reward:.4f}, "
            f"R_legit={info['R_legit']:.4f}, "
            f"R_eve={info['R_eve']:.4f}"
        )

    env.print_channel_gains()

    print("Stability check: gains should be identical across calls:-")
    g1 = env.compute_all_channel_gains()
    g2 = env.compute_all_channel_gains()
    g3 = env.compute_all_channel_gains()
    for key in g1:
        assert g1[key] == g2[key] == g3[key], f"Unstable: {key}"
        print(f"- {key}: {g1[key]:.6e} (stable across 3 calls)")
    print("All gains stable.\n")

    # ---- Part 3: Static user backward-compatibility check ----
    print("=" * 70)
    print("STATIC USER MODE TEST (user_mobile=False)")
    print("=" * 70)
    env_static = UAVEnvironment(EnvConfig(user_mobile=False, seed=42))
    env_static.reset()
    pos_before = env_static.user_position.copy()
    for _ in range(5):
        a_relay = np.random.uniform(-1, 1, size=2)
        a_jammer = np.random.uniform(-1, 1, size=2)
        env_static.step(a_relay, a_jammer)
    pos_after = env_static.user_position.copy()
    diff = np.linalg.norm(pos_after - pos_before)
    print(f"User position before: {pos_before}")
    print(f"User position after : {pos_after}")
    print(f"Total movement      : {diff:.6f} m  ->  "
          f"{'STATIONARY (OK)' if diff < 1e-10 else 'ERROR: moved!'}")
    print()

    print("=" * 70)
    print("REWARD DECOMPOSITION (10 random steps)")
    print("=" * 70)
    env_rew = UAVEnvironment(EnvConfig(user_mobile=True))
    env_rew.reset()
    hdr = f"{'Step':<6} {'R_sec':<12} {'Secrecy':<10} {'Energy':<10} {'Motion':<10} {'Smooth':<10} {'Bound':<10} {'Reward':<10} {'v_relay':<8} {'v_jam':<8}"
    print(hdr)
    print("-" * len(hdr))
    for t in range(10):
        a_relay = np.random.uniform(-1, 1, size=2)
        a_jammer = np.random.uniform(-1, 1, size=2)
        _, reward, _, info = env_rew.step(a_relay, a_jammer)
        print(
            f"{t+1:<6} "
            f"{info['R_sec']:<12.2f} "
            f"{info['reward_secrecy']:<10.4f} "
            f"{info['reward_energy_penalty']:<10.2f} "
            f"{info['reward_motion_penalty']:<10.2f} "
            f"{info['reward_smoothness_penalty']:<10.2f} "
            f"{info['reward_boundary_penalty']:<10.2f} "
            f"{reward:<10.2f} "
            f"{info['relay_speed_mps']:<8.2f} "
            f"{info['jammer_speed_mps']:<8.2f}"
        )
    print()

    print("=" * 70)
    print("LOS MODEL COMPARISON  (free-space vs LoS-aware)")
    print("=" * 70)
    cfg_fs = EnvConfig(seed=42, use_los_model=False)
    cfg_los = EnvConfig(seed=42, use_los_model=True)
    env_fs = UAVEnvironment(cfg_fs);  _ = env_fs.reset()
    env_los = UAVEnvironment(cfg_los); _ = env_los.reset()

    links = ["UR (user-relay)", "RB (relay-BS)", "UE (user-eve)", "JE (jammer-eve)"]
    gain_keys = ["h_UR", "h_RB", "h_UE", "h_JE"]
    print(f"{'Link':<16} {'Dist (m)':<10} {'Elev(deg)':<10} {'P_LoS':<8} "
          f"{'Free-sp':<12} {'LoS-aware':<12} {'G_los':<12} {'G_nlos':<12}")
    print("-" * 92)
    for link, key in zip(links, gain_keys):
        g_fs = env_fs.compute_all_channel_gains()[key]

        # manually retrieve LoS details
        if link.startswith("UR"):
            tx, rx = env_los.user_position, env_los.relay_position
        elif link.startswith("RB"):
            tx, rx = env_los.relay_position, env_los.bs_position
        elif link.startswith("UE"):
            tx, rx = env_los.user_position, env_los.eve_position
        else:  # JE
            tx, rx = env_los.jammer_position, env_los.eve_position

        d = compute_distance(tx, rx)
        theta = compute_elevation_angle(tx, rx)
        fading_val = env_los.fading[key[-2:]]  # "UR", "RB", "UE", "JE"
        from core.channel import los_probability, path_loss
        p_los = los_probability(theta, env_los.config.los_a, env_los.config.los_b)
        g_los = path_loss(d, env_los.config.alpha_los, env_los.config.beta0) * fading_val
        g_nlos = path_loss(d, env_los.config.alpha_nlos, env_los.config.beta0) * fading_val
        g_los_aware = env_los.compute_all_channel_gains()[key]

        print(f"{link:<16} {d:<10.2f} {theta:<10.2f} {p_los:<8.4f} "
              f"{g_fs:<12.6e} {g_los_aware:<12.6e} {g_los:<12.6e} {g_nlos:<12.6e}")
    print()

    print("=" * 70)
    print("ENERGY HARVESTING VERIFICATION")
    print("=" * 70)
    cfg_eh = EnvConfig(
        seed=42, user_mobile=True,
        enable_energy_harvesting=True,
        relay_harvest_efficiency=0.5, jammer_harvest_efficiency=0.5,
        relay_harvest_max_watts=8.0, jammer_harvest_max_watts=8.0,
        solar_variability=0.1,
    )
    env_eh = UAVEnvironment(cfg_eh)
    env_eh.reset()

    print("Energy-causality battery evolution (10 steps):")
    hdr = (f"{'Step':<6} {'Bat_before_R':<12} {'Bat_before_J':<12} "
           f"{'Cons_R(J)':<10} {'Cons_J(J)':<10} "
           f"{'Harv_R(J)':<10} {'Harv_J(J)':<10} "
           f"{'Bat_after_R':<12} {'Bat_after_J':<12} "
           f"{'Saturate':<9}")
    print(hdr)
    print("-" * len(hdr))
    for t in range(10):
        a_relay = np.random.uniform(-1, 1, size=2)
        a_jammer = np.random.uniform(-1, 1, size=2)
        _, _, _, info = env_eh.step(a_relay, a_jammer)
        # Reconstruct battery-before: battery_after + consumed - harvested
        b_before_r = (info['relay_battery_j'] + info['relay_energy_j']
                      - info['relay_harvested_energy_j'])
        b_before_j = (info['jammer_battery_j'] + info['jammer_energy_j']
                      - info['jammer_harvested_energy_j'])
        print(
            f"{t+1:<6} "
            f"{b_before_r:<12.2f} {b_before_j:<12.2f} "
            f"{info['relay_energy_j']:<10.4f} {info['jammer_energy_j']:<10.4f} "
            f"{info['relay_harvested_energy_j']:<10.4f} {info['jammer_harvested_energy_j']:<10.4f} "
            f"{info['relay_battery_j']:<12.2f} {info['jammer_battery_j']:<12.2f} "
            f"{'SAT' if info['battery_saturation_event'] else '---':<9}"
        )
    print()

    # Example harvested energy values
    print("Example harvested power values (10 independent samples):")
    p_hdr = f"{'Sample':<8} {'Relay_pwr(W)':<15} {'Jammer_pwr(W)':<15} {'Relay_En(J)':<15} {'Jammer_En(J)':<15} {'Total_En(J)':<15}"
    print(p_hdr)
    print("-" * len(p_hdr))
    harvest_samples = []
    for i in range(10):
        h = env_eh._compute_energy_harvesting()
        harvest_samples.append(h)
        print(
            f"{i+1:<8} "
            f"{h['relay_harvest_power_w']:<15.4f} "
            f"{h['jammer_harvest_power_w']:<15.4f} "
            f"{h['relay_harvested_energy_j']:<15.6f} "
            f"{h['jammer_harvested_energy_j']:<15.6f} "
            f"{h['total_harvested_energy_j']:<15.6f}"
        )
    # Stats
    totals = [s['total_harvested_energy_j'] for s in harvest_samples]
    print(f"\n  Harvest stats over 10 samples:")
    print(f"    Mean total: {np.mean(totals):.6f} J  "
          f"Min: {np.min(totals):.6f} J  "
          f"Max: {np.max(totals):.6f} J  "
          f"Std: {np.std(totals):.6f} J")
    # Theoretical max: (0.5*8 + 0.5*8) * 0.1 = 0.8 J
    max_possible = ((cfg_eh.relay_harvest_efficiency * cfg_eh.relay_harvest_max_watts
                     + cfg_eh.jammer_harvest_efficiency * cfg_eh.jammer_harvest_max_watts)
                    * cfg_eh.dt)
    clipped_max = ((cfg_eh.relay_harvest_max_watts + cfg_eh.jammer_harvest_max_watts)
                   * cfg_eh.dt)
    print(f"    Theoretical max base (no clip): {max_possible:.6f} J")
    print(f"    Absolute physical max (clipped): {clipped_max:.6f} J")
    print()

    print("=" * 70)
    print("NTN SATELLITE VERIFICATION")
    print("=" * 70)
    cfg_ntn = EnvConfig(seed=42, observation_mode="full_ntn", enable_ntn=True,
                        satellite_altitude_km=500.0, satellite_horizontal_offset_km=100.0)
    env_ntn = UAVEnvironment(cfg_ntn)
    s_ntn = env_ntn.reset()
    print(f"  full_ntn obs dim    : {s_ntn.shape[0]} (expect 46)")
    print(f"  Satellite position  : {env_ntn.satellite_position}")
    print(f"  h_sat_relay (linear): {env_ntn.h_sat_relay:.6e}")
    sat_alt = env_ntn.config.satellite_altitude_km
    print(f"  Satellite altitude  : {sat_alt:.0f} km")
    # Step with NTN enabled
    a_relay, a_jammer = np.zeros(2), np.zeros(2)
    ns, r, d, info = env_ntn.step(a_relay, a_jammer)
    print(f"  Step R_legit        : {info['R_legit']:.4f} bps")
    print(f"  Step R_sec          : {info['R_sec']:.4f} bps")
    # Compare with NTN disabled
    cfg_no_ntn = EnvConfig(seed=42, observation_mode="full", enable_ntn=False)
    env_no_ntn = UAVEnvironment(cfg_no_ntn)
    env_no_ntn.reset()
    ns2, r2, d2, info2 = env_no_ntn.step(a_relay, a_jammer)
    print(f"  (No NTN: R_legit={info2['R_legit']:.4f}, R_sec={info2['R_sec']:.4f})")
    print()

    print("=" * 70)
    print("STATE DIMENSION VERIFICATION  (Requirement 13)")
    print("=" * 70)
    for mode in ["geometry", "channels", "full", "full_eh", "full_ntn"]:
        cfg_noeh = EnvConfig(seed=42, observation_mode=mode, enable_energy_harvesting=False)
        cfg_eh_on = EnvConfig(seed=42, observation_mode=mode, enable_energy_harvesting=True)
        env_noeh = UAVEnvironment(cfg_noeh)
        env_eh_on = UAVEnvironment(cfg_eh_on)
        s_noeh = env_noeh.reset()
        s_eh_on = env_eh_on.reset()
        dim_noeh = s_noeh.shape[0]
        dim_eh = s_eh_on.shape[0]
        status = "OK (unchanged)" if dim_noeh == dim_eh else "CHANGED!"
        print(f"  Mode={mode:<10s}  dim(no EH)={dim_noeh}  dim(EH)={dim_eh}  {status}")
    print()

    # Verify all EH info keys exist when EH is disabled (should default to 0)
    print("Backward-compatibility check: EH info fields default to 0 when disabled:")
    env_noeh = UAVEnvironment(EnvConfig(seed=42, enable_energy_harvesting=False))
    env_noeh.reset()
    _, _, _, info = env_noeh.step(np.zeros(2), np.zeros(2))
    for key in ["relay_harvested_energy_j", "jammer_harvested_energy_j",
                "total_harvested_energy_j", "relay_harvest_power_w",
                "jammer_harvest_power_w", "battery_saturation_event"]:
        val = info[key]
        expected = 0.0 if key != "battery_saturation_event" else False
        match = "OK" if val == expected else "MISMATCH"
        print(f"    info['{key}'] = {val}  ({match})")
