from __future__ import annotations

from dataclasses import dataclass

from core.config_utils import build_env_config
from core.environment import EnvConfig


@dataclass
class TD3PGStudyConfig:
    episodes: int = 4000
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 100000
    min_replay_size: int = 1024
    hidden_dim: int = 64
    seed: int = 42
    device: str = "cpu"
    fading_model: str = "rician"
    rician_k: float = 5.0
    exploration_noise_start: float = 0.30
    exploration_noise_end: float = 0.05
    exploration_noise_decay_steps: int = 800000
    target_policy_noise: float = 0.20
    target_noise_clip: float = 0.50
    policy_delay: int = 2
    grad_clip_norm: float = 5.0
    eval_interval: int = 50
    train_eval_episodes: int = 5
    final_eval_episodes: int = 20
    control_mode: str = "velocity"
    role_switching: bool = False
    user_mobile: bool = True
    use_los_model: bool = False
    observation_mode: str = "full"
    normalize_observations: bool = True
    output_root: str = "td3pg_study/output"


def make_env_config(seed: int, cfg: TD3PGStudyConfig) -> EnvConfig:
    return build_env_config(
        seed=seed,
        fading_model=cfg.fading_model,
        rician_k=cfg.rician_k,
        control_mode=cfg.control_mode,
        role_switching=cfg.role_switching,
        user_mobile=cfg.user_mobile,
        use_los_model=cfg.use_los_model,
        observation_mode=cfg.observation_mode,
        normalize_observations=cfg.normalize_observations,
    )


def build_run_name(fading_model: str, hidden_dim: int) -> str:
    return f"td3pg_{fading_model}_h{hidden_dim}"


def build_output_dir(cfg: TD3PGStudyConfig) -> str:
    return f"{cfg.output_root}/{build_run_name(cfg.fading_model, cfg.hidden_dim)}"
