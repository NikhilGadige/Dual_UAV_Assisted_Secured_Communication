import numpy as np
from dataclasses import dataclass
from typing import Tuple
from channel import channel_gain, generate_fading

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
    energy_reward_weight: float = 1e4
    battery_depletion_penalty: float = 5e5

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
        self.roles_swapped = False
        self.relay_battery = self.config.relay_battery_joules
        self.jammer_battery = self.config.jammer_battery_joules

    def _random_position_2d(self) -> np.ndarray:
        return np.array([
            np.random.uniform(-self.half_area, self.half_area),
            np.random.uniform(-self.half_area, self.half_area),
        ])

    def _reset_entity_positions(self):
        self.user_position = np.append(self._random_position_2d(), 0.0)
        self.eve_position = np.append(self._random_position_2d(), 0.0)
        self.relay_position = np.append(self._random_position_2d(), self.config.relay_altitude)
        self.jammer_position = np.append(self._random_position_2d(), self.config.jammer_altitude)

    def reset(self) -> np.ndarray:
        self._step_counter = 0
        self._reset_entity_positions()
        self.current_jammer_power = self.config.jammer_power_max
        self.relay_velocity = np.zeros(2, dtype=float)
        self.jammer_velocity = np.zeros(2, dtype=float)
        self.roles_swapped = False
        self.relay_battery = self.config.relay_battery_joules
        self.jammer_battery = self.config.jammer_battery_joules
        self._generate_fading()
        return self.get_state()

    def _generate_fading(self) -> None:
        self.fading["UR"] = generate_fading(self.config.fading_model, self.config.rician_k)
        self.fading["RB"] = generate_fading(self.config.fading_model, self.config.rician_k)
        self.fading["UE"] = generate_fading(self.config.fading_model, self.config.rician_k)
        self.fading["JE"] = generate_fading(self.config.fading_model, self.config.rician_k)

    def _clip_to_bounds(self, pos: np.ndarray, altitude: float) -> np.ndarray:
        xy = np.clip(pos[:2], -self.half_area, self.half_area)
        return np.array([xy[0], xy[1], altitude])

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
        relay_speed = np.linalg.norm(self.relay_velocity)
        jammer_speed = np.linalg.norm(self.jammer_velocity)
        relay_power_draw = (
            self.config.relay_hover_power_watts
            + self.config.relay_motion_power_coeff * relay_speed ** 2
        )
        jammer_power_draw = (
            self.config.jammer_hover_power_watts
            + self.config.jammer_motion_power_coeff * jammer_speed ** 2
            + self.config.jammer_rf_power_coeff * self.current_jammer_power
        )
        relay_energy = relay_power_draw * self.config.dt
        jammer_energy = jammer_power_draw * self.config.dt
        return {
            "relay_speed": float(relay_speed),
            "jammer_speed": float(jammer_speed),
            "relay_energy": float(relay_energy),
            "jammer_energy": float(jammer_energy),
            "total_energy": float(relay_energy + jammer_energy),
        }

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

        self._step_counter += 1
        energy = self._compute_energy_usage()
        self.relay_battery = max(0.0, self.relay_battery - energy["relay_energy"])
        self.jammer_battery = max(0.0, self.jammer_battery - energy["jammer_energy"])
        battery_depleted = self.relay_battery <= 0.0 or self.jammer_battery <= 0.0
        done = self._step_counter >= self.config.max_steps or battery_depleted
        self._generate_fading()
        rates = self.compute_rates()
        energy_penalty = self.config.energy_reward_weight * energy["total_energy"]
        reward = rates["R_sec"] - energy_penalty
        if battery_depleted:
            reward -= self.config.battery_depletion_penalty
        rates.update(
            {
                "reward": float(reward),
                "energy_penalty": float(energy_penalty),
                "relay_energy_j": energy["relay_energy"],
                "jammer_energy_j": energy["jammer_energy"],
                "total_energy_j": energy["total_energy"],
                "relay_speed_mps": energy["relay_speed"],
                "jammer_speed_mps": energy["jammer_speed"],
                "relay_battery_j": float(self.relay_battery),
                "jammer_battery_j": float(self.jammer_battery),
                "battery_depleted": bool(battery_depleted),
                "roles_swapped": bool(self.roles_swapped),
                "effective_relay_label": "jammer_uav" if self.roles_swapped else "relay_uav",
                "effective_jammer_label": "relay_uav" if self.roles_swapped else "jammer_uav",
            }
        )

        return self.get_state(), reward, done, rates

    def get_state(self) -> np.ndarray:
        gains = self.compute_all_channel_gains()
        rates = self.compute_rates(gains)
        distances = self.compute_distances()

        return np.concatenate([
            self.relay_position,
            self.jammer_position,
            self.user_position,
            self.bs_position,
            self.eve_position,
            self.relay_velocity,
            self.jammer_velocity,
            np.array([
                distances["d_UR"],
                distances["d_RB"],
                distances["d_UE"],
                distances["d_JE"],
            ], dtype=float),
            np.array([
                gains["h_UR"],
                gains["h_RB"],
                gains["h_UE"],
                gains["h_JE"],
            ], dtype=float),
            np.array([
                rates["gamma_UR"],
                rates["gamma_RB"],
                rates["gamma_E"],
                self.current_jammer_power,
            ], dtype=float),
            np.array([
                self.relay_battery,
                self.jammer_battery,
            ], dtype=float),
        ])

    def compute_channel_gain(self, tx_pos: np.ndarray, rx_pos: np.ndarray,
                             fading: float) -> float:
        return channel_gain(
            compute_distance(tx_pos, rx_pos),
            fading,
            alpha=self.config.alpha,
            beta0=self.config.beta0,
        )

    def compute_all_channel_gains(self) -> dict:
        gains = {}
        relay_position = self.jammer_position if self.roles_swapped else self.relay_position
        jammer_position = self.relay_position if self.roles_swapped else self.jammer_position
        gains["h_UR"] = self.compute_channel_gain(
            self.user_position, relay_position, self.fading["UR"])
        gains["h_RB"] = self.compute_channel_gain(
            relay_position, self.bs_position, self.fading["RB"])
        gains["h_UE"] = self.compute_channel_gain(
            self.user_position, self.eve_position, self.fading["UE"])
        gains["h_JE"] = self.compute_channel_gain(
            jammer_position, self.eve_position, self.fading["JE"])
        return gains

    def compute_distances(self) -> dict:
        relay_position = self.jammer_position if self.roles_swapped else self.relay_position
        jammer_position = self.relay_position if self.roles_swapped else self.jammer_position
        return {
            "d_UR": compute_distance(self.user_position, relay_position),
            "d_RB": compute_distance(relay_position, self.bs_position),
            "d_UE": compute_distance(self.user_position, self.eve_position),
            "d_JE": compute_distance(jammer_position, self.eve_position),
        }

    def print_channel_gains(self, gains: dict | None = None) -> None:
        if gains is None:
            gains = self.compute_all_channel_gains()
        print("\n-> Fading Values ")
        model_label = self.config.fading_model.capitalize()
        print(f"  f_UR ({model_label})     : {self.fading['UR']:.6f}")
        print(f"  f_RB ({model_label})     : {self.fading['RB']:.6f}")
        print(f"  f_UE ({model_label})     : {self.fading['UE']:.6f}")
        print(f"  f_JE ({model_label})     : {self.fading['JE']:.6f}")
        print("\n-> Channel Gains ")
        print(f"  h_UR (User->Relay): {gains['h_UR']:.6e}")
        print(f"  h_RB (Relay->BS)  : {gains['h_RB']:.6e}")
        print(f"  h_UE (User->Eve)  : {gains['h_UE']:.6e}")
        print(f"  h_JE (Jammer->Eve): {gains['h_JE']:.6e}")

    def compute_rates(self, gains: dict | None = None) -> dict:
        if gains is None:
            gains = self.compute_all_channel_gains()
        noise_power = self.config.noise_psd * self.config.bandwidth

        gamma_ur = (self.config.user_power * gains["h_UR"]) / noise_power
        gamma_rb = (self.config.relay_power * gains["h_RB"]) / noise_power
        r_legit = 0.5 * self.config.bandwidth * np.log2(1.0 + min(gamma_ur, gamma_rb))

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
        }

def compute_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))

if __name__ == "__main__":
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
