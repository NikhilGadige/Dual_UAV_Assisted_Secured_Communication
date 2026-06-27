"""Experiment configurations for Bistatic Sensing model."""

from dataclasses import dataclass
from bistatic_sensing_exp.environments.bistatic_sensing_env import (
    BistaticSensingConfig,
)


@dataclass
class BistaticSensingExperimentConfig:
    seed: int = 42
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
    sensing_power: float = 1.0
    bandwidth: float = 1e6
    noise_psd: float = 10 ** (-17.4)
    eve_density_lambda: float = 2e-5
    output_root: str = "outputs/bistatic_sensing"
    num_vehicles: int = 3
    vehicle_max_speed: float = 10.0
    vehicle_mobility_mode: str = "straight_road"


def build_env_config(
    config: BistaticSensingExperimentConfig | None = None,
) -> BistaticSensingConfig:
    if config is None:
        config = BistaticSensingExperimentConfig()
    return BistaticSensingConfig(
        area_size=config.area_size,
        ris_altitude=config.ris_altitude,
        jammer_altitude=config.jammer_altitude,
        seed=config.seed,
        max_steps=config.max_steps,
        bandwidth=config.bandwidth,
        noise_psd=config.noise_psd,
        bs_power=config.bs_power,
        jammer_power=config.jammer_power,
        sensing_power=config.sensing_power,
        alpha=config.alpha,
        beta0=config.beta0,
        N_ris=config.N_ris,
        N_j=config.N_j,
        rician_k=config.rician_k,
        output_root=config.output_root,
        eve_density_lambda=config.eve_density_lambda,
        num_vehicles=config.num_vehicles,
        vehicle_max_speed=config.vehicle_max_speed,
        vehicle_mobility_mode=config.vehicle_mobility_mode,
    )


_EXAMPLE_CONFIGS = {
    "default": BistaticSensingExperimentConfig(),
    "high_power": BistaticSensingExperimentConfig(sensing_power=5.0),
    "low_power": BistaticSensingExperimentConfig(sensing_power=0.1),
    "many_targets": BistaticSensingExperimentConfig(num_vehicles=6),
}
