import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convergence_study.configs import (  # noqa: E402
    build_ddpg_convergence_config,
    build_dqn_convergence_config,
    build_output_dir,
    build_run_name,
)
from convergence_study.plot_convergence import generate_comparison_plots, generate_single_run_plots  # noqa: E402
from rl.ddpg_train import train_ddpg  # noqa: E402
from rl.dqn_train import train_dqn  # noqa: E402


def run_convergence_experiment(
    algorithm: str,
    fading_model: str,
    hidden_dim: int,
    episodes: int,
    seed: int,
    device: str,
) -> dict:
    run_name = build_run_name(algorithm, fading_model, hidden_dim)
    output_dir = build_output_dir(algorithm, fading_model, hidden_dim)
    if algorithm == "dqn":
        summary = train_dqn(
            build_dqn_convergence_config(
                fading_model=fading_model,
                episodes=episodes,
                hidden_dim=hidden_dim,
                seed=seed,
                device=device,
            ),
            output_dir=output_dir,
        )
    elif algorithm == "ddpg":
        summary = train_ddpg(
            build_ddpg_convergence_config(
                fading_model=fading_model,
                episodes=episodes,
                hidden_dim=hidden_dim,
                seed=seed,
                device=device,
            ),
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    csv_path = summary["training_log_csv"]
    generate_single_run_plots(csv_path, output_dir, run_name)
    generate_comparison_plots("outputs/convergence")
    print(f"Saved outputs to {output_dir} using {csv_path}")
    return summary
