"""Week 1 secure dual-UAV system model and experiment pipeline."""

from .system_model import (
    CHANNEL_MODELS,
    ALGORITHMS,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_TRAIN_EPISODES,
    DEFAULT_EVAL_SEEDS,
    Week1Scenario,
    build_week1_ddpg_config,
    build_week1_dqn_config,
    build_week1_env_config,
    default_week1_scenarios,
)
