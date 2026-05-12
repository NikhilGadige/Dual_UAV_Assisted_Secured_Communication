import argparse
import csv
from pathlib import Path

from ddpg_train import DDPGConfig, train_ddpg
from dqn_train import DQNConfig, train_dqn


def run_rl_channel_matrix(
    episodes: int = 60,
    eval_episodes: int = 20,
    seed: int = 42,
    output_dir: str = "outputs/rl_channel_matrix",
    control_mode: str = "velocity",
    user_mobile: bool = False,
    use_los_model: bool = False,
    observation_mode: str = "full",
    normalize_observations: bool = True,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("dqn", "rician"),
        ("dqn", "rayleigh"),
        ("ddpg", "rician"),
        ("ddpg", "rayleigh"),
    ]

    rows: list[dict] = []
    for algorithm, fading_model in runs:
        run_dir = out_dir / f"{algorithm}_{fading_model}"
        replay_warmup = min(1000, max(4, episodes * 2))
        batch_size = min(32, replay_warmup)
        print(f"\nRunning {algorithm.upper()} with {fading_model} fading...")

        if algorithm == "dqn":
            summary = train_dqn(
                DQNConfig(
                    episodes=episodes,
                    min_replay_size=replay_warmup,
                    batch_size=batch_size,
                    seed=seed,
                    fading_model=fading_model,
                    evaluation_episodes=eval_episodes,
                    control_mode=control_mode,
                    user_mobile=user_mobile,
                    use_los_model=use_los_model,
                    observation_mode=observation_mode,
                    normalize_observations=normalize_observations,
                ),
                output_dir=str(run_dir),
            )
            rows.append(
                {
                    "algorithm": algorithm,
                    "fading_model": fading_model,
                    "user_mobile": user_mobile,
                    "use_los_model": use_los_model,
                    "observation_mode": observation_mode,
                    "normalize_observations": normalize_observations,
                    "enable_energy_harvesting": False,
                    "observation_has_eh": observation_mode == "full_eh",
                    "mean_avg_rsec_mbps": summary["dqn_mean_avg_rsec_mbps"],
                    "mean_episode_secrecy_mbits": summary["dqn_mean_episode_secrecy_mbits"],
                    "baseline_random_avg_rsec_mbps": summary["random_mean_avg_rsec_mbps"],
                    "baseline_greedy_avg_rsec_mbps": summary["greedy_mean_avg_rsec_mbps"],
                    "training_log_csv": summary["training_log_csv"],
                    "model_path": summary["model_path"],
                }
            )
        else:
            summary = train_ddpg(
                DDPGConfig(
                    episodes=episodes,
                    min_replay_size=replay_warmup,
                    batch_size=batch_size,
                    seed=seed,
                    fading_model=fading_model,
                    evaluation_episodes=eval_episodes,
                    control_mode=control_mode,
                    user_mobile=user_mobile,
                    use_los_model=use_los_model,
                    observation_mode=observation_mode,
                    normalize_observations=normalize_observations,
                ),
                output_dir=str(run_dir),
            )
            rows.append(
                {
                    "algorithm": algorithm,
                    "fading_model": fading_model,
                    "user_mobile": user_mobile,
                    "use_los_model": use_los_model,
                    "observation_mode": observation_mode,
                    "normalize_observations": normalize_observations,
                    "enable_energy_harvesting": False,
                    "observation_has_eh": observation_mode == "full_eh",
                    "mean_avg_rsec_mbps": summary["ddpg_mean_avg_rsec_mbps"],
                    "mean_episode_secrecy_mbits": summary["ddpg_mean_episode_secrecy_mbits"],
                    "baseline_random_avg_rsec_mbps": summary["random_mean_avg_rsec_mbps"],
                    "baseline_greedy_avg_rsec_mbps": summary["greedy_mean_avg_rsec_mbps"],
                    "training_log_csv": summary["training_log_csv"],
                    "model_path": summary["actor_path"],
                }
            )

    summary_csv = out_dir / "rl_channel_matrix_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nRL channel matrix complete.")
    print(f"  Summary CSV: {summary_csv.resolve()}")
    for row in rows:
        print(
            f"  - {row['algorithm'].upper()} | {row['fading_model']:<8} | "
            f"avg_R_sec={row['mean_avg_rsec_mbps']:.4f} Mbps"
        )

    return {"summary_csv": str(summary_csv.resolve()), "rows": rows}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run DQN and DDPG under both Rician and Rayleigh fading."
    )
    parser.add_argument("--episodes", type=int, default=60, help="Episodes per run")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Evaluation episodes after each run")
    parser.add_argument("--seed", type=int, default=42, help="Shared random seed")
    parser.add_argument(
        "--control-mode",
        type=str,
        default="velocity",
        choices=["velocity", "waypoint"],
        help="Velocity-vector or normalized waypoint control",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/rl_channel_matrix",
        help="Directory to store all run outputs",
    )
    parser.add_argument("--user-mobile", action="store_true", help="Enable mobile user")
    parser.add_argument("--use-los-model", action="store_true", help="Use LoS path-loss model")
    parser.add_argument(
        "--observation-mode",
        type=str,
        default="full",
        choices=["full", "geometry", "channels"],
        help="Observation space mode",
    )
    parser.add_argument("--no-normalize", action="store_true", help="Disable observation normalization")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_rl_channel_matrix(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        control_mode=args.control_mode,
        user_mobile=args.user_mobile,
        use_los_model=args.use_los_model,
        observation_mode=args.observation_mode,
        normalize_observations=not args.no_normalize,
    )
