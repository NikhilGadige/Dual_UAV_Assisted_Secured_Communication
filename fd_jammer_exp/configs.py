"""Experiment configurations for Full-Duplex UAV Jammer experiments."""

from dataclasses import dataclass
from fd_jammer_exp.environments.fd_jammer_env import FDJammerConfig


@dataclass
class FDJammerExperimentConfig:
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
    output_root: str = "outputs/fd_jammer"


def build_fd_jammer_env_config(
    config: FDJammerExperimentConfig | None = None,
) -> FDJammerConfig:
    if config is None:
        config = FDJammerExperimentConfig()
    return FDJammerConfig(
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
    )


_EXAMPLE_CONFIGS = {
    "default": FDJammerExperimentConfig(),
    "high_jammer": FDJammerExperimentConfig(jammer_power=0.5),
    "low_jammer": FDJammerExperimentConfig(jammer_power=0.05),
    "mrt_mode": FDJammerExperimentConfig(beamforming_mode="mrt"),
    "nullspace_mode": FDJammerExperimentConfig(beamforming_mode="nullspace"),
    "isotropic_mode": FDJammerExperimentConfig(beamforming_mode="isotropic"),
}
