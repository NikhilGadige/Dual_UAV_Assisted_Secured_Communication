import numpy as np
from core.channel import path_loss


def compute_rcs(
    vehicle_type: str = "car",
    frequency_hz: float = 2e9,
    aspect_angle_deg: float = 0.0,
) -> float:
    """Radar Cross Section model (square-metres).

    Simple constant-RCS model per vehicle type, with a cosine
    aspect-angle taper.
    """
    rcs_db = {"car": 10.0, "truck": 20.0, "bus": 25.0, "motorcycle": 5.0}
    db = rcs_db.get(vehicle_type, 10.0)
    rcs_linear = 10.0 ** (db / 10.0)
    angular_factor = np.cos(np.radians(aspect_angle_deg)) ** 2
    return rcs_linear * max(angular_factor, 0.01)


class Vehicle:
    """Mobile reflecting vehicle with configurable mobility patterns."""

    def __init__(
        self,
        vehicle_id: int,
        position: np.ndarray,
        mobility_mode: str = "straight_road",
        max_speed: float = 10.0,
        vehicle_type: str = "car",
        rcs: float | None = None,
    ):
        self.vehicle_id = vehicle_id
        self.position = position.copy()
        self.velocity = np.zeros(2, dtype=float)
        self.heading = 0.0
        self.max_speed = max_speed
        self.mobility_mode = mobility_mode
        self.vehicle_type = vehicle_type
        self.rcs = rcs if rcs is not None else compute_rcs(vehicle_type)
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
                self.velocity = self.max_speed * np.array(
                    [np.cos(self.heading), np.sin(self.heading)]
                )
            self._reflect(self.position, half_area)
        elif self.mobility_mode == "urban_grid":
            self.position += self.velocity * dt
            self._grid_timer += 1
            if self._grid_timer >= self._grid_target:
                self._grid_timer = 0
                self._grid_target = np.random.randint(40, 120)
                turn = np.random.choice([-np.pi / 2, np.pi / 2])
                self.heading += turn
                self.velocity = self.max_speed * np.array(
                    [np.cos(self.heading), np.sin(self.heading)]
                )
            self._reflect(self.position, half_area)

    def distance_to(self, other_pos: np.ndarray) -> float:
        return float(np.linalg.norm(self.position[:2] - other_pos[:2]))


def _generate_scalar_rician(
    K: float = 5.0, path_loss_factor: float = 1.0
) -> complex:
    K_linear = 10.0 ** (K / 10.0) if K < 100 else K
    scale_los = np.sqrt(K_linear / (K_linear + 1.0))
    scale_nlos = np.sqrt(1.0 / (K_linear + 1.0))
    los_phase = np.exp(1j * np.random.uniform(0.0, 2.0 * np.pi))
    nlos = np.random.normal(0.0, 1.0 / np.sqrt(2.0)) + 1j * np.random.normal(
        0.0, 1.0 / np.sqrt(2.0)
    )
    h = np.sqrt(path_loss_factor) * (scale_los * los_phase + scale_nlos * nlos)
    return complex(h)


def compute_reflection_channel_gain(
    d_UV: float,
    d_VR: float,
    sigma_rcs: float,
    K: float = 5.0,
    alpha: float = 2.0,
    beta0: float = 1.0,
) -> float:
    """Compute |h_vehicle|^2 = |h_UV|^2 * sigma_rcs * |h_VR|^2.

    Each link is independent Rician fading with path loss.
    Returns the total reflection channel power gain (linear).
    """
    pl_UV = path_loss(d_UV, alpha, beta0)
    pl_VR = path_loss(d_VR, alpha, beta0)
    h_UV = _generate_scalar_rician(K, pl_UV)
    h_VR = _generate_scalar_rician(K, pl_VR)
    return float(np.abs(h_UV) ** 2) * sigma_rcs * float(np.abs(h_VR) ** 2)


def compute_cascaded_reflection_gain(
    d_UV: float,
    d_VR: float,
    sigma_rcs: float,
    K: float = 5.0,
    alpha: float = 2.0,
    beta0: float = 1.0,
) -> tuple:
    """Compute both the direct scalar channel and the reflected power gain.

    Returns (h_vehicle_complex, |h_vehicle|^2).
    """
    pl_UV = path_loss(d_UV, alpha, beta0)
    pl_VR = path_loss(d_VR, alpha, beta0)
    h_UV = _generate_scalar_rician(K, pl_UV)
    h_VR = _generate_scalar_rician(K, pl_VR)
    h_vehicle = h_UV * np.sqrt(sigma_rcs) * h_VR
    return h_vehicle, float(np.abs(h_vehicle) ** 2)
