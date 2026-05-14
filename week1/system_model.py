"""Week 1 secure dual-UAV secrecy system model.

This module makes the week 1 assumptions explicit:
- a static base station at the origin,
- one mobile user following a random walk,
- one eavesdropper,
- two UAVs: a relay drone and a jammer drone,
- two channel families: Rician and Rayleigh.

It also provides helper constructors for the DDPG and DQN training configs so
the week1 runner can keep the environment and RL settings aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.config_utils import build_env_config
from core.environment import EnvConfig
from rl.ddpg_train import DDPGConfig
from rl.dqn_train import DQNConfig

CHANNEL_MODELS: tuple[str, ...] = ("rician", "rayleigh")
ALGORITHMS: tuple[str, ...] = ("dqn", "ddpg")
DEFAULT_TRAIN_EPISODES = 600
DEFAULT_EVAL_EPISODES = 20
DEFAULT_EVAL_SEEDS: tuple[int, ...] = (7, 21, 42)


@dataclass(frozen=True)
class Week1Scenario:
    algorithm: str
    fading_model: str
    train_episodes: int = DEFAULT_TRAIN_EPISODES
    eval_episodes: int = DEFAULT_EVAL_EPISODES
    seed: int = 42
    hidden_dim: int = 128
    control_mode: str = "velocity"
    observation_mode: str = "full"
    normalize_observations: bool = True
    user_mobile: bool = True
    use_los_model: bool = False
    rician_k: float = 5.0
    device: str = "cpu"


def build_week1_env_config(
    seed: int,
    fading_model: str,
    *,
    control_mode: str = "velocity",
    observation_mode: str = "full",
    normalize_observations: bool = True,
    user_mobile: bool = True,
    use_los_model: bool = False,
    rician_k: float = 5.0,
) -> EnvConfig:
    return build_env_config(
        seed=seed,
        fading_model=fading_model,
        rician_k=rician_k,
        control_mode=control_mode,
        user_mobile=user_mobile,
        use_los_model=use_los_model,
        observation_mode=observation_mode,
        normalize_observations=normalize_observations,
    )


def build_week1_dqn_config(scenario: Week1Scenario) -> DQNConfig:
    return DQNConfig(
        episodes=scenario.train_episodes,
        hidden_dim=scenario.hidden_dim,
        seed=scenario.seed,
        device=scenario.device,
        fading_model=scenario.fading_model,
        rician_k=scenario.rician_k,
        evaluation_episodes=scenario.eval_episodes,
        control_mode=scenario.control_mode,
        user_mobile=scenario.user_mobile,
        use_los_model=scenario.use_los_model,
        observation_mode=scenario.observation_mode,
        normalize_observations=scenario.normalize_observations,
        epsilon_decay_steps=max(scenario.train_episodes * 180, 60000),
    )


def build_week1_ddpg_config(scenario: Week1Scenario) -> DDPGConfig:
    return DDPGConfig(
        episodes=scenario.train_episodes,
        hidden_dim=scenario.hidden_dim,
        seed=scenario.seed,
        device=scenario.device,
        fading_model=scenario.fading_model,
        rician_k=scenario.rician_k,
        evaluation_episodes=scenario.eval_episodes,
        control_mode=scenario.control_mode,
        user_mobile=scenario.user_mobile,
        use_los_model=scenario.use_los_model,
        observation_mode=scenario.observation_mode,
        normalize_observations=scenario.normalize_observations,
        noise_decay_steps=max(scenario.train_episodes * 180, 60000),
    )


def default_week1_scenarios(
    *,
    train_episodes: int = DEFAULT_TRAIN_EPISODES,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 42,
    hidden_dim: int = 128,
    device: str = "cpu",
) -> list[Week1Scenario]:
    scenarios: list[Week1Scenario] = []
    for algorithm in ALGORITHMS:
        for fading_model in CHANNEL_MODELS:
            scenarios.append(
                Week1Scenario(
                    algorithm=algorithm,
                    fading_model=fading_model,
                    train_episodes=train_episodes,
                    eval_episodes=eval_episodes,
                    seed=seed,
                    hidden_dim=hidden_dim,
                    device=device,
                )
            )
    return scenarios


def scenario_output_dir(root: Path, algorithm: str, fading_model: str) -> Path:
    return root / "outputs" / "week1" / "training" / algorithm / fading_model

