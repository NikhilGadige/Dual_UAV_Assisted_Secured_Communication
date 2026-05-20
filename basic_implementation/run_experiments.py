from __future__ import annotations

import argparse
from pathlib import Path

from basic_implementation.configs import BasicExperimentConfig
from basic_implementation.plotting import (
    generate_four_way_plots,
    write_summary_csv,
)
from basic_implementation.run_common import run_basic_experiment


def run_all_experiments(
    episodes: int = 4000,
    hidden_dim: int = 32,
    learning_rate: float = 1e-2,
    seed: int = 42,
    device: str = "cpu",
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
    runs = [
        ("dqn", "rayleigh"),
        ("dqn", "rician"),
        ("ddpg", "rayleigh"),
        ("ddpg", "rician"),
    ]
    raw_results = [run_basic_experiment(algorithm, fading_model, basic_cfg.episodes, basic_cfg.seed, basic_cfg.device, basic_cfg.hidden_dim, basic_cfg.learning_rate, basic_cfg.output_root) for algorithm, fading_model in runs]
    results = [
        {
            "algorithm": algorithm,
            "channel_model": fading_model,
            "episodes": basic_cfg.episodes,
            "hidden_dim": basic_cfg.hidden_dim,
            "learning_rate": basic_cfg.learning_rate,
            "seed": basic_cfg.seed,
            "device": basic_cfg.device,
            **result,
        }
        for (algorithm, fading_model), result in zip(runs, raw_results)
    ]
    csv_map = {row["run_key"]: row["training_log_csv"] for row in results}
    plot_paths = generate_four_way_plots(csv_map, output_dir=f"{output_root}/plots_update")
    summary_csv = write_summary_csv(results, output_root=output_root)
    return {
        "results": results,
        "plot_paths": plot_paths,
        "summary_csv": summary_csv,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DQN/DDPG under Rayleigh/Rician fading using the refined basic implementation wrapper."
    )
    parser.add_argument("--episodes", type=int, default=4000, help="Training episodes per run. Use 4000 or 4500.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension for DQN/DDPG.")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Learning rate for DQN/DDPG.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/basic_outputs",
        help="Root directory for this refined convergence update.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    result = run_all_experiments(
        episodes=args.episodes,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        output_root=args.output_root,
    )
    print(f"Summary CSV: {result['summary_csv']}")
    print("Generated plots:")
    for plot_path in result["plot_paths"]:
        print(f"  {plot_path}")
