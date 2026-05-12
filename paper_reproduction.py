import argparse
import csv
from pathlib import Path

from advanced_rl_train import AdvancedRLConfig, train_advanced
from baselines import distance_greedy_policy, evaluate_policy, random_policy
from config_utils import build_env_config
from ddpg_analysis import plot_ddpg_training_curves
from dqn_analysis import plot_dqn_training_curves
from environment import EnvConfig
from final_comparison import run_final_comparison
from paper_plots import plot_final_paper_comparisons, plot_training_comparison
from rl_channel_experiments import run_rl_channel_matrix
from trajectory_plots import generate_trajectory_suite


def _parse_seed_list(raw: str) -> list[int]:
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def _write_baseline_channel_table(
    output_dir: Path,
    episodes_per_seed: int,
    seeds: list[int],
    user_mobile: bool,
    use_los_model: bool,
    observation_mode: str,
    normalize_observations: bool,
) -> str:
    rows = []
    policies = [
        ("Random Walk", random_policy),
        ("Distance-Greedy", distance_greedy_policy),
    ]
    for fading_model in ["rician", "rayleigh"]:
        for policy_name, policy_fn in policies:
            for seed in seeds:
                summary = evaluate_policy(
                    policy_name,
                    policy_fn,
                    episodes=episodes_per_seed,
                    seed=seed,
                    env_config=build_env_config(seed=seed, fading_model=fading_model, user_mobile=user_mobile, use_los_model=use_los_model, observation_mode=observation_mode, normalize_observations=normalize_observations),
                )
                rows.append(
                    {
                        "policy": policy_name,
                        "fading_model": fading_model,
                        "seed": seed,
                        "episodes": episodes_per_seed,
                        "enable_energy_harvesting": False,
                        "observation_has_eh": observation_mode == "full_eh",
                        "mean_avg_R_sec_mbps": summary["mean_avg_R_sec_mbps"],
                        "mean_episode_secrecy_mbits": summary["mean_episode_secrecy_mbits"],
                        "mean_avg_R_legit_mbps": summary["mean_avg_R_legit_mbps"],
                        "mean_avg_R_eve_mbps": summary["mean_avg_R_eve_mbps"],
                    }
                )

    out_csv = output_dir / "baseline_channel_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return str(out_csv.resolve())


def _model_paths(matrix_rows: list[dict]) -> dict[tuple[str, str], str]:
    return {
        (row["algorithm"], row["fading_model"]): row["model_path"]
        for row in matrix_rows
    }


def run_paper_reproduction(
    train_episodes: int = 60,
    eval_episodes: int = 10,
    seeds: list[int] | None = None,
    train_seed: int = 42,
    output_dir: str = "outputs/paper_reproduction",
    make_trajectories: bool = True,
    control_mode: str = "velocity",
    include_advanced: bool = False,
    advanced_methods: list[str] | None = None,
    user_mobile: bool = False,
    use_los_model: bool = False,
    observation_mode: str = "full",
    normalize_observations: bool = True,
) -> dict:
    if seeds is None:
        seeds = [7, 21, 42]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Running baselines for Rician and Rayleigh...")
    baseline_csv = _write_baseline_channel_table(
        out_dir,
        eval_episodes,
        seeds,
        user_mobile,
        use_los_model,
        observation_mode,
        normalize_observations,
    )

    print("\n[2/6] Training DQN/DDPG channel matrix...")
    matrix = run_rl_channel_matrix(
        episodes=train_episodes,
        eval_episodes=eval_episodes,
        seed=train_seed,
        output_dir=str(out_dir / "rl_channel_matrix"),
        control_mode=control_mode,
        user_mobile=user_mobile,
        use_los_model=use_los_model,
        observation_mode=observation_mode,
        normalize_observations=normalize_observations,
    )
    paths = _model_paths(matrix["rows"])

    print("\n[3/6] Plotting per-agent learning curves...")
    analysis_outputs = {}
    log_specs = []
    for row in matrix["rows"]:
        algorithm = row["algorithm"]
        fading_model = row["fading_model"]
        csv_path = row["training_log_csv"]
        analysis_dir = out_dir / "analysis" / f"{algorithm}_{fading_model}"
        if algorithm == "dqn":
            analysis_outputs[f"{algorithm}_{fading_model}"] = plot_dqn_training_curves(
                csv_path,
                output_dir=str(analysis_dir),
            )
        else:
            analysis_outputs[f"{algorithm}_{fading_model}"] = plot_ddpg_training_curves(
                csv_path,
                output_dir=str(analysis_dir),
            )
        log_specs.append(
            {
                "algorithm": algorithm,
                "fading_model": fading_model,
                "csv_path": csv_path,
            }
        )

    print("\n[4/6] Building final tables and method/channel comparisons...")
    final = run_final_comparison(
        dqn_rician_model=paths[("dqn", "rician")],
        dqn_rayleigh_model=paths[("dqn", "rayleigh")],
        ddpg_rician_actor=paths[("ddpg", "rician")],
        ddpg_rayleigh_actor=paths[("ddpg", "rayleigh")],
        episodes_per_seed=eval_episodes,
        seeds=seeds,
        output_dir=str(out_dir / "final_comparison"),
        control_mode=control_mode,
        user_mobile=user_mobile,
        use_los_model=use_los_model,
        observation_mode=observation_mode,
        normalize_observations=normalize_observations,
    )

    print("\n[5/6] Creating paper-style aggregate plots...")
    paper_training_plots = plot_training_comparison(log_specs, str(out_dir / "paper_plots"))
    paper_final_plots = plot_final_paper_comparisons(final["csv_path"], str(out_dir / "paper_plots"))

    advanced_outputs = {}
    advanced_summary_csv = ""
    if include_advanced:
        print("\n[6/7] Running advanced RL methods...")
        if advanced_methods is None:
            advanced_methods = ["td3", "sac", "ppo"]
        advanced_rows = []
        for method in advanced_methods:
            for fading_model in ["rician", "rayleigh"]:
                result = train_advanced(
                    method,
                    AdvancedRLConfig(
                        episodes=train_episodes,
                        evaluation_episodes=eval_episodes,
                        seed=train_seed,
                        fading_model=fading_model,
                        control_mode=control_mode,
                        user_mobile=user_mobile,
                        use_los_model=use_los_model,
                        observation_mode=observation_mode,
                        normalize_observations=normalize_observations,
                    ),
                    str(out_dir / "advanced_rl" / f"{method}_{fading_model}"),
                )
                advanced_outputs[f"{method}_{fading_model}"] = result
                advanced_rows.append(
                    {
                        "algorithm": method,
                        "fading_model": fading_model,
                        "user_mobile": user_mobile,
                        "use_los_model": use_los_model,
                        "observation_mode": observation_mode,
                        "normalize_observations": normalize_observations,
                        "mean_avg_rsec_mbps": result["mean_avg_rsec_mbps"],
                        "mean_episode_secrecy_mbits": result["mean_episode_secrecy_mbits"],
                        "training_log_csv": result["training_log_csv"],
                        "model_path": result["model_path"],
                    }
                )
        advanced_summary = out_dir / "advanced_rl_summary.csv"
        with advanced_summary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(advanced_rows[0].keys()))
            writer.writeheader()
            writer.writerows(advanced_rows)
        advanced_summary_csv = str(advanced_summary.resolve())

    trajectory_outputs = {}
    if make_trajectories:
        print("\n[7/7] Generating trajectory plots..." if include_advanced else "\n[6/6] Generating trajectory plots...")
        for fading_model in ["rician", "rayleigh"]:
            trajectory_outputs[fading_model] = generate_trajectory_suite(
                dqn_model=paths[("dqn", fading_model)],
                ddpg_actor=paths[("ddpg", fading_model)],
                fading_model=fading_model,
                seed=train_seed,
                output_dir=str(out_dir / "trajectories" / fading_model),
            )
    else:
        print("\n[7/7] Skipping trajectory plots by request." if include_advanced else "\n[6/6] Skipping trajectory plots by request.")

    manifest = {
        "baseline_csv": baseline_csv,
        "rl_matrix_csv": matrix["summary_csv"],
        "final_csv": final["csv_path"],
        "final_plots": final["plot_paths"],
        "paper_training_plots": paper_training_plots,
        "paper_final_plots": paper_final_plots,
        "trajectory_plots": trajectory_outputs,
        "analysis_outputs": analysis_outputs,
        "advanced_outputs": advanced_outputs,
        "advanced_summary_csv": advanced_summary_csv,
    }

    manifest_csv = out_dir / "paper_reproduction_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["artifact", "path"])
        writer.writeheader()
        writer.writerow({"artifact": "baseline_csv", "path": baseline_csv})
        writer.writerow({"artifact": "rl_matrix_csv", "path": matrix["summary_csv"]})
        writer.writerow({"artifact": "final_csv", "path": final["csv_path"]})
        if advanced_summary_csv:
            writer.writerow({"artifact": "advanced_summary_csv", "path": advanced_summary_csv})
        for group in ["final_plots", "paper_training_plots", "paper_final_plots"]:
            for name, path in manifest[group].items():
                writer.writerow({"artifact": f"{group}.{name}", "path": path})
        for fading_model, outputs in trajectory_outputs.items():
            for method, path in outputs.items():
                writer.writerow({"artifact": f"trajectory.{fading_model}.{method}", "path": path})

    manifest["manifest_csv"] = str(manifest_csv.resolve())
    print("\nPaper reproduction pipeline complete.")
    print(f"  Manifest: {manifest['manifest_csv']}")
    return manifest


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full paper reproduction pipeline: baselines, DQN, DDPG, channels, tables, plots."
    )
    parser.add_argument("--train-episodes", type=int, default=60, help="Training episodes per RL run")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes per seed")
    parser.add_argument("--seeds", type=str, default="7,21,42", help="Comma-separated evaluation seeds")
    parser.add_argument("--train-seed", type=int, default=42, help="Shared training seed")
    parser.add_argument("--output-dir", type=str, default="outputs/paper_reproduction", help="Output directory")
    parser.add_argument("--skip-trajectories", action="store_true", help="Skip rollout trajectory plots")
    parser.add_argument(
        "--control-mode",
        type=str,
        default="velocity",
        choices=["velocity", "waypoint"],
        help="Velocity-vector or normalized waypoint control",
    )
    parser.add_argument("--include-advanced", action="store_true", help="Also run TD3, SAC, and PPO")
    parser.add_argument("--advanced-methods", type=str, default="td3,sac,ppo", help="Comma-separated advanced methods")
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
    run_paper_reproduction(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        seeds=_parse_seed_list(args.seeds),
        train_seed=args.train_seed,
        output_dir=args.output_dir,
        make_trajectories=not args.skip_trajectories,
        control_mode=args.control_mode,
        include_advanced=args.include_advanced,
        advanced_methods=[m.strip() for m in args.advanced_methods.split(",") if m.strip()],
        user_mobile=args.user_mobile,
        use_los_model=args.use_los_model,
        observation_mode=args.observation_mode,
        normalize_observations=not args.no_normalize,
    )
