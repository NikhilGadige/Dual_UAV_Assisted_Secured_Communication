from __future__ import annotations

from basic_implementation.configs import (
    BasicExperimentConfig,
    build_basic_ddpg_config,
    build_basic_dqn_config,
    build_plot_dir,
    build_run_dir,
    build_run_key,
)
from basic_implementation.plotting import generate_per_run_plots
from rl.ddpg_train import train_ddpg
from rl.dqn_train import train_dqn


def run_basic_experiment(
    algorithm: str,
    fading_model: str,
    episodes: int,
    seed: int,
    device: str,
    hidden_dim: int = 32,
    learning_rate: float = 5e-4,
    output_root: str = "outputs/basic_outputs",
) -> dict:
    basic_cfg = BasicExperimentConfig(
        episodes=episodes,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
        output_root=output_root,
    )
    run_key = build_run_key(algorithm, fading_model, hidden_dim)
    run_dir = build_run_dir(
        algorithm,
        fading_model,
        hidden_dim=hidden_dim,
        output_root=output_root,
    )
    plot_dir = build_plot_dir(
        algorithm,
        fading_model,
        hidden_dim=hidden_dim,
        output_root=output_root,
    )

    if algorithm == "dqn":
        summary = train_dqn(
            build_basic_dqn_config(
                fading_model=fading_model,
                episodes=episodes,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                seed=seed,
                device=device,
            ),
            output_dir=run_dir,
        )
        model_path = summary["model_path"]
    elif algorithm == "ddpg":
        summary = train_ddpg(
            build_basic_ddpg_config(
                fading_model=fading_model,
                episodes=episodes,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                seed=seed,
                device=device,
            ),
            output_dir=run_dir,
        )
        model_path = summary["actor_path"]
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    plot_paths = generate_per_run_plots(summary["training_log_csv"], plot_dir, run_key)
    print(f"Run key: {run_key}")
    print(f"Training outputs: {run_dir}")
    print(f"Plots: {plot_dir}")
    print(f"Training CSV: {summary['training_log_csv']}")
    print(f"Model: {model_path}")
    return {
        "run_key": run_key,
        "run_dir": run_dir,
        "plot_dir": plot_dir,
        "training_log_csv": summary["training_log_csv"],
        "model_path": model_path,
        "plot_paths": plot_paths,
    }
