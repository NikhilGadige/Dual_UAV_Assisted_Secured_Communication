from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SCABCDConfig:
    channel_model: str = "rician"
    seed: int = 42
    horizon: int = 12
    max_bcd_iters: int = 100
    min_bcd_iters: int = 20
    bcd_patience: int = 8
    max_sca_iters: int = 8
    bcd_abs_tolerance: float = 1e-3
    bcd_rel_tolerance: float = 5e-4
    sca_tolerance: float = 1e-4
    trust_region_weight: float = 1.0
    power_trust_region_radius: float = 0.35
    trajectory_trust_region_radius: float = 180.0
    area_size: float = 1000.0
    max_flight_radius: float = 350.0
    relay_altitude: float = 50.0
    jammer_altitude: float = 50.0
    slot_duration: float = 4.0
    max_speed: float = 20.0
    collision_distance: float = 30.0
    bandwidth: float = 1e6
    noise_psd: float = 10 ** (-17.4)
    beta0: float = 1.0
    alpha: float = 2.0
    use_los_model: bool = False
    alpha_los: float = 2.0
    alpha_nlos: float = 3.0
    los_a: float = 9.61
    los_b: float = 0.20
    rician_k: float = 5.0
    user_power_min: float = 1e-3
    user_power_max: float = 0.2
    relay_power_min: float = 1e-3
    relay_power_max: float = 0.5
    jammer_power_min: float = 0.0
    jammer_power_max: float = 0.5
    avg_user_power_budget: float = 0.15
    avg_relay_power_budget: float = 0.35
    avg_jammer_power_budget: float = 0.25
    min_average_secrecy_rate: float = 0.0
    alpha_min: float = 0.05
    alpha_max: float = 0.95
    alpha_trust_region_radius: float = 0.5
    eve_uncertainty_radius: float = 30.0
    use_vehicle_receiver: bool = True
    vehicle_mobility_mode: str = "straight_road"
    vehicle_max_speed: float = 8.0
    use_multiple_eves: bool = True
    eve_density_lambda: float = 2e-5
    generate_future_study_artifacts: bool = True
    future_long_run_iterations: int = 4000

    @property
    def candidate_step_sizes(self) -> tuple[float, ...]:
        return (0.25, 0.2, 0.15, 0.1, 0.05, 0.02)

    @property
    def half_area(self) -> float:
        return self.area_size / 2.0

    @property
    def noise_power(self) -> float:
        return self.bandwidth * self.noise_psd

    def output_root(self) -> Path:
        return Path("outputs") / "sca_bcd" / self.channel_model

    def ensure_output_dirs(self) -> dict[str, Path]:
        root = self.output_root()
        dirs = {
            "root": root,
            "convergence": root / "convergence",
            "checkpoints": root / "checkpoints",
            "plots": root / "plots",
            "reports": root / "reports",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs
