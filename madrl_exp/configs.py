"""Configuration dataclasses for MADRL experiments — 3-agent version."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EnvConfig:
    """Environment configuration (mirrors OptimizationConfig)."""
    M_bs: int = 4
    N_ris: int = 16
    N_j: int = 4
    N_time: int = 5
    N_tx_sense: int = 16
    N_rx_sense: int = 16
    L_pilot: int = 32
    P_bs_max: float = 10.0
    P_j_max: float = 0.05
    sigma2: float = 1e-8
    noise_power_sense: float = 1e-8
    v_max: float = 50.0
    dt: float = 1.0
    d_ant: float = 0.5
    wavelength: float = 1.0
    eta_ris: float = 0.3
    seed: int = 42
    sensing_utility_mode: str = "log"
    alpha: float = 0.5
    include_direct_links: bool = False
    action_range: float = 1.0  # range for action spaces [-action_range, action_range]
    beamform_mode: str = "reim"  # "reim" (A), "direction" (B), "power_direction" (C)


@dataclass
class AgentConfig:
    """Single agent configuration."""
    name: str = ""
    obs_dim: int = 64
    act_dim: int = 16
    hidden_dim: int = 256
    n_layers: int = 2
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 1_000_000
    batch_size: int = 256
    update_freq: int = 1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1
    clip_epsilon: float = 0.2
    target_kl: float = 0.01
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    temperature: float = 1.0
    clip_logits: bool = False
    weight_decay: float = 0.0  # L2 penalty on actor output layer only


@dataclass
class TrainingConfig:
    """Training configuration."""
    n_episodes: int = 100
    max_steps_per_episode: int = 100
    algorithm: Literal["mappo", "matd3"] = "mappo"
    eval_interval: int = 20
    n_eval_episodes: int = 5
    log_interval: int = 10
    save_interval: int = 50
    output_root: str = "outputs/reinforcement_learning/training_runs"
    seed: int = 42


@dataclass
class RewardConfig:
    """Reward weighting."""
    reward_mode: str = "original"  # "original", "normalized", "normalized_penalty", "full"
    lambda_constraint: float = 0.1
    lambda_outage: float = 0.5
    lambda_power: float = 0.01
    secrecy_outage_threshold: float = 0.01
    # Secrecy floor penalty (Part 2)
    lambda_secret: float = 1.0
    R_target: float = 0.0   # will be set from random baseline secrecy
    # Action regularization (Part 3)
    lambda_action: float = 0.0
    classification_threshold: float = 0.5
    obs_clip: float = 0.0  # clip observations to [-obs_clip, obs_clip]; 0 = no clip


@dataclass
class MADRLConfig:
    """Top-level experiment configuration."""
    env: EnvConfig = field(default_factory=EnvConfig)
    agents: list[AgentConfig] = field(default_factory=list)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    output_root: str = "outputs/reinforcement_learning/training_runs"

    def __post_init__(self):
        if not self.agents:
            bs_act = 2 * self.env.N_time * self.env.M_bs
            if self.env.beamform_mode == "power_direction":
                bs_act += self.env.N_time
            self.agents = [
                AgentConfig(name="bs_beamformer", act_dim=bs_act),
                AgentConfig(name="uav_trajectory", act_dim=3 * self.env.N_time),
                AgentConfig(name="jammer_beamformer", act_dim=2 * self.env.N_time * self.env.N_j),
            ]
