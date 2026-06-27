"""Experiment configurations for RIS-mounted UAV communication model."""

from dataclasses import dataclass
from ris_uav_exp.environments.ris_uav_env import RISUAVConfig


@dataclass
class RISExperimentConfig:
    seed: int = 42
    episodes: int = 50
    max_steps: int = 200
    area_size: float = 1000.0
    ris_altitude: float = 50.0
    N_ris: int = 16
    rician_k: float = 5.0
    alpha: float = 2.0
    beta0: float = 1.0
    bs_power: float = 0.5
    bandwidth: float = 1e6
    noise_psd: float = 10 ** (-17.4)
    eve_density_lambda: float = 2e-5
    output_root: str = "outputs/ris_uav"
    algorithm: str = "baseline"


def build_ris_env_config(config: RISExperimentConfig | None = None) -> RISUAVConfig:
    if config is None:
        config = RISExperimentConfig()
    return RISUAVConfig(
        area_size=config.area_size,
        ris_altitude=config.ris_altitude,
        seed=config.seed,
        max_steps=config.max_steps,
        bandwidth=config.bandwidth,
        noise_psd=config.noise_psd,
        bs_power=config.bs_power,
        alpha=config.alpha,
        beta0=config.beta0,
        N_ris=config.N_ris,
        rician_k=config.rician_k,
        output_root=config.output_root,
        eve_density_lambda=config.eve_density_lambda,
    )


_EXAMPLE_CONFIGS = {
    "default": RISExperimentConfig(),
    "high_altitude": RISExperimentConfig(ris_altitude=100.0),
    "low_altitude": RISExperimentConfig(ris_altitude=20.0),
    "dense_eve": RISExperimentConfig(eve_density_lambda=5e-5),
    "sparse_eve": RISExperimentConfig(eve_density_lambda=5e-6),
    "strong_los": RISExperimentConfig(rician_k=15.0),
    "weak_los": RISExperimentConfig(rician_k=0.0),
}
