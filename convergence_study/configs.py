from dataclasses import dataclass

from core.environment import EnvConfig
from core.config_utils import build_env_config as build_base_env_config
from rl.ddpg_train import DDPGConfig
from rl.dqn_train import DQNConfig


@dataclass
class ConvergenceExperimentConfig:
    algorithm: str
    fading_model: str
    episodes: int = 3000
    hidden_dim: int = 32
    learning_rate: float = 1e-2
    seed: int = 42
    device: str = "cpu"


def build_env_config(experiment: ConvergenceExperimentConfig) -> EnvConfig:
    return build_base_env_config(
        seed=experiment.seed,
        fading_model=experiment.fading_model,
        control_mode="velocity",
        role_switching=False,
        user_mobile=True,
        use_los_model=False,
        observation_mode="full",
        normalize_observations=True,
    )


def build_dqn_convergence_config(
    fading_model: str,
    episodes: int = 3000,
    hidden_dim: int = 32,
    seed: int = 42,
    device: str = "cpu",
) -> DQNConfig:
    experiment = ConvergenceExperimentConfig(
        algorithm="dqn",
        fading_model=fading_model,
        episodes=episodes,
        hidden_dim=hidden_dim,
        seed=seed,
        device=device,
    )
    env_cfg = build_env_config(experiment)
    return DQNConfig(
        episodes=experiment.episodes,
        lr=experiment.learning_rate,
        hidden_dim=experiment.hidden_dim,
        seed=experiment.seed,
        device=experiment.device,
        fading_model=experiment.fading_model,
        evaluation_episodes=20,
        control_mode=env_cfg.control_mode,
        user_mobile=env_cfg.user_mobile,
        use_los_model=env_cfg.use_los_model,
        observation_mode=env_cfg.observation_mode,
        normalize_observations=env_cfg.normalize_observations,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=experiment.episodes * 1200,
        target_update_tau=0.005,
        grad_clip_norm=5.0,
        td_target_clip=20.0,
        batch_size=64,
        replay_size=50000,
        min_replay_size=2000,
        eval_interval=50,
        train_eval_episodes=5,
    )


def build_ddpg_convergence_config(
    fading_model: str,
    episodes: int = 3000,
    hidden_dim: int = 32,
    seed: int = 42,
    device: str = "cpu",
) -> DDPGConfig:
    experiment = ConvergenceExperimentConfig(
        algorithm="ddpg",
        fading_model=fading_model,
        episodes=episodes,
        hidden_dim=hidden_dim,
        seed=seed,
        device=device,
    )
    env_cfg = build_env_config(experiment)
    return DDPGConfig(
        episodes=experiment.episodes,
        actor_lr=experiment.learning_rate,
        critic_lr=experiment.learning_rate,
        hidden_dim=experiment.hidden_dim,
        seed=experiment.seed,
        device=experiment.device,
        fading_model=experiment.fading_model,
        evaluation_episodes=20,
        control_mode=env_cfg.control_mode,
        role_switching=False,
        user_mobile=env_cfg.user_mobile,
        use_los_model=env_cfg.use_los_model,
        observation_mode=env_cfg.observation_mode,
        normalize_observations=env_cfg.normalize_observations,
        tau=0.005,
        batch_size=64,
        replay_size=50000,
        min_replay_size=512,
        noise_decay_steps=experiment.episodes * 400,
        eval_interval=50,
        train_eval_episodes=5,
    )


def build_run_name(algorithm: str, fading_model: str, hidden_dim: int) -> str:
    return f"{algorithm}_{fading_model}_h{hidden_dim}"


def build_output_dir(algorithm: str, fading_model: str, hidden_dim: int) -> str:
    return f"outputs/convergence/{build_run_name(algorithm, fading_model, hidden_dim)}"
