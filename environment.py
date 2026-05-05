import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class EnvConfig:
    area_size: float = 1000.0
    relay_altitude: float = 50.0
    jammer_altitude: float = 50.0
    max_speed: float = 20.0
    dt: float = 0.1
    seed: int | None = None
    max_steps: int = 200

class UAVEnvironment:
    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        self.half_area = self.config.area_size / 2
        self.bs_position = np.array([0.0, 0.0, 0.0])
        self._step_counter = 0

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
        return self.get_state()

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

        return self.get_state(), 0.0, done, {}

    def get_state(self) -> np.ndarray:
        return np.concatenate([
            self.relay_position,
            self.jammer_position,
            self.user_position,
            self.bs_position,
            self.eve_position,
        ])

def compute_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))

if __name__ == "__main__":
    env = UAVEnvironment()
    state = env.reset()

    print(f"{'Step':<6} {'Relay':<30} {'Jammer':<30} {'User':<30} {'BS':<20} {'Eve':<30}")

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