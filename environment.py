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
    dt: float = 0.1
    seed: int | None = None
    max_steps: int = 200
    bandwidth: float = 1e6
    noise_psd: float = 10 ** (-17.4)
    user_power: float = 0.2
    relay_power: float = 0.5
    jammer_power: float = 0.5
    alpha: float = 2.0
    beta0: float = 1.0

class UAVEnvironment:
    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        self.half_area = self.config.area_size / 2
        self.bs_position = np.array([0.0, 0.0, 0.0])
        self._step_counter = 0
        self.fading = {"UR": 0.0, "RB": 0.0, "UE": 0.0, "JE": 0.0}

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
        self._generate_fading()
        return self.get_state()

    def _generate_fading(self, model: str = "rician", K: float = 5.0) -> None:
        self.fading["UR"] = generate_fading(model, K)
        self.fading["RB"] = generate_fading(model, K)
        self.fading["UE"] = generate_fading(model, K)
        self.fading["JE"] = generate_fading(model, K)

    def _clip_to_bounds(self, pos: np.ndarray, altitude: float) -> np.ndarray:
        xy = np.clip(pos[:2], -self.half_area, self.half_area)
        return np.array([xy[0], xy[1], altitude])

    def step(self, action_relay: np.ndarray, action_jammer: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action_relay = np.clip(action_relay, -1.0, 1.0)
        action_jammer = np.clip(action_jammer, -1.0, 1.0)

        delta_relay = np.append(action_relay * self.config.max_speed * self.config.dt, 0.0)
        delta_jammer = np.append(action_jammer * self.config.max_speed * self.config.dt, 0.0)

        self.relay_position = self._clip_to_bounds(
            self.relay_position + delta_relay, self.config.relay_altitude
        )
        self.jammer_position = self._clip_to_bounds(
            self.jammer_position + delta_jammer, self.config.jammer_altitude
        )

        self._step_counter += 1
        done = self._step_counter >= self.config.max_steps
        self._generate_fading()
        rates = self.compute_rates()
        reward = rates["R_sec"]

        return self.get_state(), reward, done, rates

    def get_state(self) -> np.ndarray:
        return np.concatenate([
            self.relay_position,
            self.jammer_position,
            self.user_position,
            self.bs_position,
            self.eve_position,
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
        gains["h_UR"] = self.compute_channel_gain(
            self.user_position, self.relay_position, self.fading["UR"])
        gains["h_RB"] = self.compute_channel_gain(
            self.relay_position, self.bs_position, self.fading["RB"])
        gains["h_UE"] = self.compute_channel_gain(
            self.user_position, self.eve_position, self.fading["UE"])
        gains["h_JE"] = self.compute_channel_gain(
            self.jammer_position, self.eve_position, self.fading["JE"])
        return gains

    def print_channel_gains(self, gains: dict | None = None) -> None:
        if gains is None:
            gains = self.compute_all_channel_gains()
        print("\n-> Fading Values ")
        print(f"  f_UR (Rician)     : {self.fading['UR']:.6f}")
        print(f"  f_RB (Rician)     : {self.fading['RB']:.6f}")
        print(f"  f_UE (Rician)     : {self.fading['UE']:.6f}")
        print(f"  f_JE (Rician)     : {self.fading['JE']:.6f}")
        print("\n-> Channel Gains ")
        print(f"  h_UR (User->Relay): {gains['h_UR']:.6e}")
        print(f"  h_RB (Relay->BS)  : {gains['h_RB']:.6e}")
        print(f"  h_UE (User->Eve)  : {gains['h_UE']:.6e}")
        print(f"  h_JE (Jammer->Eve): {gains['h_JE']:.6e}")

    def compute_rates(self) -> dict:
        gains = self.compute_all_channel_gains()
        noise_power = self.config.noise_psd * self.config.bandwidth

        gamma_ur = (self.config.user_power * gains["h_UR"]) / noise_power
        gamma_rb = (self.config.relay_power * gains["h_RB"]) / noise_power
        r_legit = 0.5 * self.config.bandwidth * np.log2(1.0 + min(gamma_ur, gamma_rb))

        gamma_e = (self.config.user_power * gains["h_UE"]) / (
            noise_power + self.config.jammer_power * gains["h_JE"]
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
            state[:3], state[3:6], state[6:9], state[9:12], state[12:]
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
