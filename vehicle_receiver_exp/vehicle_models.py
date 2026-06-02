import numpy as np
from core.environment import UAVEnvironment


class VehicleReceiver:
    """Mobile vehicle receiver with configurable mobility patterns.

    Replaces the stationary IoT user with a moving vehicle.
    The vehicle moves continuously within the area boundary.
    """

    def __init__(self, position: np.ndarray, mobility_mode: str = "straight_road", max_speed: float = 10.0):
        self.position = position.copy()
        self.velocity = np.zeros(2, dtype=float)
        self.heading = 0.0
        self.max_speed = max_speed
        self.mobility_mode = mobility_mode
        self._init_mobility()

    def _init_mobility(self) -> None:
        h = self.max_speed
        if self.mobility_mode == "straight_road":
            self.heading = np.random.uniform(-np.pi, np.pi)
            self.velocity = h * np.array([np.cos(self.heading), np.sin(self.heading)])
        elif self.mobility_mode == "lane_change":
            self.heading = 0.0
            self.velocity = np.array([h, 0.0])
            self._lc_timer = 0
            self._lc_interval = np.random.randint(30, 70)
        elif self.mobility_mode == "urban_grid":
            self.heading = 0.0
            self.velocity = np.array([h, 0.0])
            self._grid_timer = 0
            self._grid_target = np.random.randint(40, 120)
        else:
            raise ValueError(f"Unknown mobility_mode: {self.mobility_mode}")

    def _reflect(self, pos: np.ndarray, half_area: float) -> None:
        for i in range(2):
            if pos[i] < -half_area:
                pos[i] = -2.0 * half_area - pos[i]
                self.velocity[i] *= -1.0
                self.heading = np.arctan2(self.velocity[1], self.velocity[0])
            elif pos[i] > half_area:
                pos[i] = 2.0 * half_area - pos[i]
                self.velocity[i] *= -1.0
                self.heading = np.arctan2(self.velocity[1], self.velocity[0])

    def update(self, dt: float = 0.1, half_area: float = 500.0) -> None:
        if self.mobility_mode == "straight_road":
            self.position += self.velocity * dt
            self._reflect(self.position, half_area)
        elif self.mobility_mode == "lane_change":
            self.position += self.velocity * dt
            self._lc_timer += 1
            if self._lc_timer >= self._lc_interval:
                self._lc_timer = 0
                self._lc_interval = np.random.randint(30, 70)
                dh = np.random.choice([-np.pi / 4, 0.0, np.pi / 4])
                self.heading += dh
                self.velocity = self.max_speed * np.array([np.cos(self.heading), np.sin(self.heading)])
            self._reflect(self.position, half_area)
        elif self.mobility_mode == "urban_grid":
            self.position += self.velocity * dt
            self._grid_timer += 1
            if self._grid_timer >= self._grid_target:
                self._grid_timer = 0
                self._grid_target = np.random.randint(40, 120)
                turn = np.random.choice([-np.pi / 2, np.pi / 2])
                self.heading += turn
                self.velocity = self.max_speed * np.array([np.cos(self.heading), np.sin(self.heading)])
            self._reflect(self.position, half_area)


class VehicleUAVEnvironment(UAVEnvironment):
    """UAV environment where the user is a mobile vehicle instead of stationary IoT.

    Overrides _update_user_position() to use VehicleReceiver mobility.
    All secrecy equations, channel models, and reward functions remain unchanged.
    """

    def __init__(self, config=None, mobility_mode: str = "straight_road", vehicle_max_speed: float = 10.0):
        super().__init__(config)
        self.mobility_mode = mobility_mode
        self.vehicle_max_speed = vehicle_max_speed
        self.vehicle = None

    def reset(self):
        state = super().reset()
        self.vehicle = VehicleReceiver(
            position=self.user_position[:2].copy(),
            mobility_mode=self.mobility_mode,
            max_speed=self.vehicle_max_speed,
        )
        return self.get_state()

    def _update_user_position(self) -> None:
        if self.vehicle is None:
            return
        self.vehicle.update(dt=self.config.dt, half_area=self.half_area)
        self.user_position[:2] = self.vehicle.position
        self.user_velocity[:] = self.vehicle.velocity
