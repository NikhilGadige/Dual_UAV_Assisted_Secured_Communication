from __future__ import annotations

from dataclasses import dataclass

from convergence_study.configs import (
    build_ddpg_convergence_config,
    build_dqn_convergence_config,
)


@dataclass
class BasicExperimentConfig:
    episodes: int = 4000
    hidden_dim: int = 32
    learning_rate: float = 5e-4
    seed: int = 42
    device: str = "cpu"
    output_root: str = "outputs/basic_outputs"
    plots_subdir: str = "plots_update"
    runs_subdir: str = "runs"


def build_basic_dqn_config(
    fading_model: str,
    episodes: int = 4000,
    hidden_dim: int = 32,
    learning_rate: float = 5e-4,
    seed: int = 42,
    device: str = "cpu",
):
    cfg = build_dqn_convergence_config(
        fading_model=fading_model,
        episodes=episodes,
        hidden_dim=hidden_dim,
        seed=seed,
        device=device,
    )
    cfg.lr = learning_rate
    cfg.hidden_dim = hidden_dim
    return cfg


def build_basic_ddpg_config(
    fading_model: str,
    episodes: int = 4000,
    hidden_dim: int = 32,
    learning_rate: float = 5e-4,
    seed: int = 42,
    device: str = "cpu",
):
    cfg = build_ddpg_convergence_config(
        fading_model=fading_model,
        episodes=episodes,
        hidden_dim=hidden_dim,
        seed=seed,
        device=device,
    )
    cfg.actor_lr = learning_rate
    cfg.critic_lr = learning_rate
    cfg.hidden_dim = hidden_dim
    return cfg


def build_run_key(algorithm: str, fading_model: str, hidden_dim: int = 32) -> str:
    return f"{algorithm}_{fading_model}_h{hidden_dim}"


def build_run_dir(
    algorithm: str,
    fading_model: str,
    hidden_dim: int = 32,
    output_root: str = "outputs/basic_outputs",
) -> str:
    return f"{output_root}/runs/{build_run_key(algorithm, fading_model, hidden_dim)}"


def build_plot_dir(
    algorithm: str,
    fading_model: str,
    hidden_dim: int = 32,
    output_root: str = "outputs/basic_outputs",
) -> str:
    return f"{output_root}/plots_update/{build_run_key(algorithm, fading_model, hidden_dim)}"
