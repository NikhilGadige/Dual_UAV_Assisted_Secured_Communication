"""Experiment configurations for Vehicle Reflection model."""

from dataclasses import dataclass
from vehicle_reflection_exp.environments.vehicle_reflection_env import (
    VehicleReflectionConfig,
)


@dataclass
class VehicleReflectionExperimentConfig:
    seed: int = 42
    episodes: int = 50
    max_steps: int = 200
    area_size: float = 1000.0
    ris_altitude: float = 50.0
    jammer_altitude: float = 50.0
    N_ris: int = 16
    N_j: int = 4
    rician_k: float = 5.0
    alpha: float = 2.0
    beta0: float = 1.0
    bs_power: float = 0.5
    jammer_power: float = 0.2
    bandwidth: float = 1e6
    noise_psd: float = 10 ** (-17.4)
    eve_density_lambda: float = 2e-5
    beamforming_mode: str = "isotropic"
    output_root: str = "outputs/vehicle_reflection"
    num_vehicles: int = 3
    vehicle_max_speed: float = 10.0
    vehicle_mobility_mode: str = "straight_road"


def build_env_config(
    config: VehicleReflectionExperimentConfig | None = None,
) -> VehicleReflectionConfig:
    if config is None:
        config = VehicleReflectionExperimentConfig()
    return VehicleReflectionConfig(
        area_size=config.area_size,
        ris_altitude=config.ris_altitude,
        jammer_altitude=config.jammer_altitude,
        seed=config.seed,
        max_steps=config.max_steps,
        bandwidth=config.bandwidth,
        noise_psd=config.noise_psd,
        bs_power=config.bs_power,
        jammer_power=config.jammer_power,
        alpha=config.alpha,
        beta0=config.beta0,
        N_ris=config.N_ris,
        N_j=config.N_j,
        rician_k=config.rician_k,
        output_root=config.output_root,
        beamforming_mode=config.beamforming_mode,
        eve_density_lambda=config.eve_density_lambda,
        num_vehicles=config.num_vehicles,
        vehicle_max_speed=config.vehicle_max_speed,
        vehicle_mobility_mode=config.vehicle_mobility_mode,
    )


_EXAMPLE_CONFIGS = {
    "default": VehicleReflectionExperimentConfig(),
    "many_vehicles": VehicleReflectionExperimentConfig(num_vehicles=6),
    "single_vehicle": VehicleReflectionExperimentConfig(num_vehicles=1),
    "lane_change": VehicleReflectionExperimentConfig(
        vehicle_mobility_mode="lane_change"
    ),
    "urban_grid": VehicleReflectionExperimentConfig(
        vehicle_mobility_mode="urban_grid"
    ),
}
