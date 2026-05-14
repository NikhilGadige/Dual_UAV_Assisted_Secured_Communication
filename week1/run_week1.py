"""Run the week 1 dual-UAV secrecy study.

This script trains DQN and DDPG on both Rician and Rayleigh channels, then
produces convergence plots, final comparison plots, and representative
trajectories.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from analysis.baselines import distance_greedy_policy, evaluate_policy, random_policy
from analysis.ddpg_analysis import plot_ddpg_training_curves
from analysis.dqn_analysis import plot_dqn_training_curves
from analysis.paper_plots import plot_final_paper_comparisons, plot_training_comparison
from analysis.trajectory_plots import generate_trajectory_suite
from core.environment import UAVEnvironment
from rl.ddpg_train import Actor, evaluate_ddpg, train_ddpg
from rl.dqn_train import QNetwork, evaluate_dqn, make_action_table, train_dqn

from week1.system_model import (
    CHANNEL_MODELS,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_EVAL_SEEDS,
    DEFAULT_TRAIN_EPISODES,
    Week1Scenario,
    build_week1_ddpg_config,
    build_week1_dqn_config,
    build_week1_env_config,
    default_week1_scenarios,
    scenario_output_dir,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
WEEK1_OUTPUT_DIR = ROOT_DIR / "outputs" / "week1"


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _load_dqn_model(model_path: str, state_dim: int, hidden_dim: int, device: torch.device):
    action_table = make_action_table()
    q_net = QNetwork(state_dim, len(action_table), hidden_dim=hidden_dim).to(device)
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()
    return q_net, action_table


def _load_ddpg_model(model_path: str, state_dim: int, hidden_dim: int, device: torch.device):
    actor = Actor(state_dim, 5, hidden_dim=hidden_dim).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()
    return actor


def _evaluate_trained_model(
    algorithm: str,
    model_path: str,
    scenario: Week1Scenario,
    seeds: list[int],
) -> dict:
    device = torch.device(scenario.device if scenario.device == "cuda" and torch.cuda.is_available() else "cpu")
    per_seed: list[dict] = []

    for seed in seeds:
        env_cfg = build_week1_env_config(
            seed=seed,
            fading_model=scenario.fading_model,
            control_mode=scenario.control_mode,
            observation_mode=scenario.observation_mode,
            normalize_observations=scenario.normalize_observations,
            user_mobile=scenario.user_mobile,
            use_los_model=scenario.use_los_model,
            rician_k=scenario.rician_k,
        )
        env = UAVEnvironment(env_cfg)

        if algorithm == "dqn":
            state_dim = env.reset().shape[0]
            q_net, action_table = _load_dqn_model(model_path, state_dim, scenario.hidden_dim, device)
            metrics = evaluate_dqn(env, q_net, action_table, device=device, episodes=scenario.eval_episodes)
        elif algorithm == "ddpg":
            state_dim = env.reset().shape[0]
            actor = _load_ddpg_model(model_path, state_dim, scenario.hidden_dim, device)
            metrics = evaluate_ddpg(env, actor, device=device, episodes=scenario.eval_episodes)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        per_seed.append(metrics)

    return {
        "mean_avg_rsec_mbps": float(np.mean([item["mean_avg_rsec_mbps"] for item in per_seed])),
        "mean_episode_secrecy_mbits": float(np.mean([item["mean_episode_secrecy_mbits"] for item in per_seed])),
    }


def _evaluate_baseline(
    policy_name: str,
    policy_fn,
    scenario: Week1Scenario,
    seeds: list[int],
) -> dict:
    summaries = []
    for seed in seeds:
        env_cfg = build_week1_env_config(
            seed=seed,
            fading_model=scenario.fading_model,
            control_mode=scenario.control_mode,
            observation_mode=scenario.observation_mode,
            normalize_observations=scenario.normalize_observations,
            user_mobile=scenario.user_mobile,
            use_los_model=scenario.use_los_model,
            rician_k=scenario.rician_k,
        )
        summaries.append(
            evaluate_policy(
                policy_name,
                policy_fn,
                episodes=scenario.eval_episodes,
                seed=seed,
                env_config=env_cfg,
            )
        )
    return {
        "mean_avg_R_sec_mbps": float(np.mean([item["mean_avg_R_sec_mbps"] for item in summaries])),
        "mean_episode_secrecy_mbits": float(np.mean([item["mean_episode_secrecy_mbits"] for item in summaries])),
    }


def _train_one_scenario(scenario: Week1Scenario) -> dict:
    scenario_dir = scenario_output_dir(ROOT_DIR, scenario.algorithm, scenario.fading_model)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    if scenario.algorithm == "dqn":
        train_summary = train_dqn(build_week1_dqn_config(scenario), output_dir=str(scenario_dir))
        analysis = plot_dqn_training_curves(train_summary["training_log_csv"], output_dir=str(scenario_dir / "analysis"))
        model_path = train_summary["model_path"]
    elif scenario.algorithm == "ddpg":
        train_summary = train_ddpg(build_week1_ddpg_config(scenario), output_dir=str(scenario_dir))
        analysis = plot_ddpg_training_curves(train_summary["training_log_csv"], output_dir=str(scenario_dir / "analysis"))
        model_path = train_summary["actor_path"]
    else:
        raise ValueError(f"Unsupported algorithm: {scenario.algorithm}")

    return {
        "algorithm": scenario.algorithm,
        "fading_model": scenario.fading_model,
        "training_log_csv": train_summary["training_log_csv"],
        "model_path": model_path,
        "training_summary": train_summary,
        "analysis_outputs": analysis,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_week1_suite(
    train_episodes: int = DEFAULT_TRAIN_EPISODES,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    seeds: list[int] | None = None,
    seed: int = 42,
    hidden_dim: int = 128,
    device: str | None = None,
    output_dir: str | None = None,
    make_trajectories: bool = True,
) -> dict:
    if seeds is None:
        seeds = list(DEFAULT_EVAL_SEEDS)

    resolved_output = Path(output_dir) if output_dir is not None else WEEK1_OUTPUT_DIR
    resolved_output.mkdir(parents=True, exist_ok=True)

    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    scenarios = default_week1_scenarios(
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        seed=seed,
        hidden_dim=hidden_dim,
        device=device_name,
    )

    manifest_rows = []
    training_specs = []
    trained: dict[tuple[str, str], dict] = {}

    print("\n[1/4] Training DQN and DDPG on Rician and Rayleigh channels...")
    for scenario in scenarios:
        result = _train_one_scenario(scenario)
        trained[(scenario.algorithm, scenario.fading_model)] = result
        training_specs.append(
            {
                "algorithm": scenario.algorithm,
                "fading_model": scenario.fading_model,
                "csv_path": result["training_log_csv"],
            }
        )
        manifest_rows.append(
            {
                "algorithm": scenario.algorithm,
                "fading_model": scenario.fading_model,
                "train_episodes": scenario.train_episodes,
                "eval_episodes": scenario.eval_episodes,
                "seed": scenario.seed,
                "hidden_dim": scenario.hidden_dim,
                "user_mobile": scenario.user_mobile,
                "control_mode": scenario.control_mode,
                "observation_mode": scenario.observation_mode,
                "normalize_observations": scenario.normalize_observations,
                "use_los_model": scenario.use_los_model,
                "training_log_csv": result["training_log_csv"],
                "model_path": result["model_path"],
            }
        )

    manifest_csv = resolved_output / "manifests" / "week1_system_model_manifest.csv"
    _write_csv(manifest_csv, manifest_rows)

    print("\n[2/4] Building convergence plots and training summaries...")
    aggregate_plots = plot_training_comparison(training_specs, str(resolved_output / "paper_plots"))

    print("\n[3/4] Evaluating checkpoints and generating final comparisons...")
    comparison_rows: list[dict] = []
    for fading_model in CHANNEL_MODELS:
        template_scenario = Week1Scenario(
            algorithm="dqn",
            fading_model=fading_model,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            seed=seed,
            hidden_dim=hidden_dim,
            device=device_name,
        )

        random_eval = _evaluate_baseline("Random Walk", random_policy, template_scenario, seeds)
        greedy_eval = _evaluate_baseline("Distance-Greedy", distance_greedy_policy, template_scenario, seeds)

        dqn_eval = _evaluate_trained_model(
            "dqn",
            trained[("dqn", fading_model)]["model_path"],
            template_scenario,
            seeds,
        )
        ddpg_eval = _evaluate_trained_model(
            "ddpg",
            trained[("ddpg", fading_model)]["model_path"],
            template_scenario,
            seeds,
        )

        comparison_rows.extend(
            [
                {
                    "method": "Random Walk",
                    "fading_model": fading_model,
                    "avg_rsec_mbps": random_eval["mean_avg_R_sec_mbps"],
                    "episode_secrecy_mbits": random_eval["mean_episode_secrecy_mbits"],
                },
                {
                    "method": "Distance-Greedy",
                    "fading_model": fading_model,
                    "avg_rsec_mbps": greedy_eval["mean_avg_R_sec_mbps"],
                    "episode_secrecy_mbits": greedy_eval["mean_episode_secrecy_mbits"],
                },
                {
                    "method": "DQN",
                    "fading_model": fading_model,
                    "avg_rsec_mbps": dqn_eval["mean_avg_rsec_mbps"],
                    "episode_secrecy_mbits": dqn_eval["mean_episode_secrecy_mbits"],
                },
                {
                    "method": "DDPG",
                    "fading_model": fading_model,
                    "avg_rsec_mbps": ddpg_eval["mean_avg_rsec_mbps"],
                    "episode_secrecy_mbits": ddpg_eval["mean_episode_secrecy_mbits"],
                },
            ]
        )

    comparison_csv = resolved_output / "comparisons" / "week1_comparison_summary.csv"
    _write_csv(comparison_csv, comparison_rows)
    final_plots = plot_final_paper_comparisons(str(comparison_csv), str(resolved_output / "paper_plots"))

    trajectory_outputs = {}
    if make_trajectories:
        print("\n[4/4] Rendering trajectory plots...")
        for fading_model in CHANNEL_MODELS:
            dqn_model_path = trained[("dqn", fading_model)]["model_path"]
            ddpg_actor_path = trained[("ddpg", fading_model)]["model_path"]
            trajectory_dir = resolved_output / "trajectories" / fading_model
            trajectory_outputs[fading_model] = {
                "random": generate_trajectory_suite(
                    method="random",
                    fading_model=fading_model,
                    seed=seed,
                    output_dir=str(trajectory_dir / "random"),
                    control_mode="velocity",
                    user_mobile=True,
                )["random"],
                "greedy": generate_trajectory_suite(
                    method="greedy",
                    fading_model=fading_model,
                    seed=seed,
                    output_dir=str(trajectory_dir / "greedy"),
                    control_mode="velocity",
                    user_mobile=True,
                )["greedy"],
                "dqn": generate_trajectory_suite(
                    method="dqn",
                    dqn_model=dqn_model_path,
                    fading_model=fading_model,
                    seed=seed,
                    output_dir=str(trajectory_dir / "dqn"),
                    control_mode="velocity",
                    user_mobile=True,
                )["dqn"],
                "ddpg": generate_trajectory_suite(
                    method="ddpg",
                    ddpg_actor=ddpg_actor_path,
                    fading_model=fading_model,
                    seed=seed,
                    output_dir=str(trajectory_dir / "ddpg"),
                    control_mode="velocity",
                    user_mobile=True,
                )["ddpg"],
            }

    return {
        "manifest_csv": str(manifest_csv.resolve()),
        "comparison_csv": str(comparison_csv.resolve()),
        "training_specs": training_specs,
        "training_plots": aggregate_plots,
        "final_plots": final_plots,
        "trajectory_plots": trajectory_outputs,
        "trained": trained,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Train and compare the week1 dual-UAV secrecy system.")
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES, help="Training episodes per run")
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES, help="Evaluation episodes per seed")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_EVAL_SEEDS), help="Comma-separated evaluation seeds")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer width for DQN/DDPG")
    parser.add_argument("--device", type=str, default=None, help="Torch device override, e.g. cpu or cuda")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for week1 artifacts")
    parser.add_argument("--skip-trajectories", action="store_true", help="Skip trajectory rendering")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_week1_suite(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        seeds=_parse_int_list(args.seeds),
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        device=args.device,
        output_dir=args.output_dir,
        make_trajectories=not args.skip_trajectories,
    )
