import argparse
import csv
from pathlib import Path

from baselines import distance_greedy_policy, evaluate_policy, random_policy
from config_utils import build_env_config
from ddpg_evaluate import evaluate_ddpg_multi_seed
from dqn_evaluate import evaluate_dqn_multi_seed
from environment import EnvConfig


def _safe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _aggregate_baseline(
    policy_name: str,
    policy_fn,
    episodes_per_seed: int,
    seeds: list[int],
    env_cfg: EnvConfig,
) -> dict:
    summaries = [
        evaluate_policy(
            policy_name,
            policy_fn,
            episodes=episodes_per_seed,
            seed=seed,
            env_config=env_cfg,
        )
        for seed in seeds
    ]
    return {
        "mean_avg_R_sec_mbps": sum(s["mean_avg_R_sec_mbps"] for s in summaries) / len(summaries),
        "mean_episode_secrecy_mbits": sum(s["mean_episode_secrecy_mbits"] for s in summaries) / len(summaries),
    }


def run_final_comparison(
    dqn_rician_model: str,
    dqn_rayleigh_model: str,
    ddpg_rician_actor: str,
    ddpg_rayleigh_actor: str,
    episodes_per_seed: int = 20,
    seeds: list[int] | None = None,
    output_dir: str = "outputs/final_comparison",
    control_mode: str = "velocity",
    user_mobile: bool = False,
    use_los_model: bool = False,
    observation_mode: str = "full",
    normalize_observations: bool = True,
) -> dict:
    if seeds is None:
        seeds = [7, 21, 42, 84, 168]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for fading_model in ["rician", "rayleigh"]:
        env_cfg = build_env_config(fading_model=fading_model, control_mode=control_mode, user_mobile=user_mobile, use_los_model=use_los_model, observation_mode=observation_mode, normalize_observations=normalize_observations)

        random_summary = _aggregate_baseline(
            "Random Walk",
            random_policy,
            episodes_per_seed,
            seeds,
            env_cfg,
        )
        greedy_summary = _aggregate_baseline(
            "Distance-Greedy",
            distance_greedy_policy,
            episodes_per_seed,
            seeds,
            env_cfg,
        )

        dqn_result = evaluate_dqn_multi_seed(
            model_path=dqn_rician_model if fading_model == "rician" else dqn_rayleigh_model,
            episodes_per_seed=episodes_per_seed,
            seeds=seeds,
            output_dir=str(out_dir / f"dqn_{fading_model}_eval"),
            channel_model=fading_model,
            control_mode=control_mode,
            user_mobile=user_mobile,
            use_los_model=use_los_model,
            observation_mode=observation_mode,
            normalize_observations=normalize_observations,
        )["aggregate"]

        ddpg_result = evaluate_ddpg_multi_seed(
            actor_path=ddpg_rician_actor if fading_model == "rician" else ddpg_rayleigh_actor,
            episodes_per_seed=episodes_per_seed,
            seeds=seeds,
            output_dir=str(out_dir / f"ddpg_{fading_model}_eval"),
            channel_model=fading_model,
            control_mode=control_mode,
            user_mobile=user_mobile,
            use_los_model=use_los_model,
            observation_mode=observation_mode,
            normalize_observations=normalize_observations,
        )["aggregate"]

        eh_meta = {"enable_energy_harvesting": False, "observation_has_eh": observation_mode == "full_eh"}
        rows.extend(
            [
                {
                    "method": "Random Walk",
                    "fading_model": fading_model,
                    **eh_meta,
                    "avg_rsec_mbps": random_summary["mean_avg_R_sec_mbps"],
                    "episode_secrecy_mbits": random_summary["mean_episode_secrecy_mbits"],
                },
                {
                    "method": "Distance-Greedy",
                    "fading_model": fading_model,
                    **eh_meta,
                    "avg_rsec_mbps": greedy_summary["mean_avg_R_sec_mbps"],
                    "episode_secrecy_mbits": greedy_summary["mean_episode_secrecy_mbits"],
                },
                {
                    "method": "DQN",
                    "fading_model": fading_model,
                    **eh_meta,
                    "avg_rsec_mbps": dqn_result["dqn_avg_rsec_mbps"],
                    "episode_secrecy_mbits": dqn_result["dqn_episode_secrecy_mbits"],
                },
                {
                    "method": "DDPG",
                    "fading_model": fading_model,
                    **eh_meta,
                    "avg_rsec_mbps": ddpg_result["ddpg_avg_rsec_mbps"],
                    "episode_secrecy_mbits": ddpg_result["ddpg_episode_secrecy_mbits"],
                },
            ]
        )

    csv_path = out_dir / "final_method_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_paths = _plot_final_comparison(out_dir, rows)
    return {"csv_path": str(csv_path.resolve()), "plot_paths": plot_paths, "rows": rows}


def _plot_final_comparison(out_dir: Path, rows: list[dict]) -> dict:
    plt = _safe_import_matplotlib()
    if plt is None:
        return {}

    methods = ["Random Walk", "Distance-Greedy", "DQN", "DDPG"]
    fading_models = ["rician", "rayleigh"]
    x = list(range(len(methods)))
    width = 0.35

    rate_plot = out_dir / "final_compare_avg_rsec_mbps.png"
    fig1 = plt.figure(figsize=(9, 4.5))
    ax1 = fig1.add_subplot(111)
    for idx, fading_model in enumerate(fading_models):
        values = [
            next(r["avg_rsec_mbps"] for r in rows if r["method"] == method and r["fading_model"] == fading_model)
            for method in methods
        ]
        offsets = [i + (idx - 0.5) * width for i in x]
        ax1.bar(offsets, values, width=width, label=fading_model.capitalize())
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15)
    ax1.set_ylabel("Avg Secrecy Rate (Mbps)")
    ax1.set_title("Final Secrecy Comparison Across Methods")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.25)
    fig1.tight_layout()
    fig1.savefig(rate_plot, dpi=150)
    plt.close(fig1)

    payload_plot = out_dir / "final_compare_episode_mbits.png"
    fig2 = plt.figure(figsize=(9, 4.5))
    ax2 = fig2.add_subplot(111)
    for idx, fading_model in enumerate(fading_models):
        values = [
            next(
                r["episode_secrecy_mbits"]
                for r in rows
                if r["method"] == method and r["fading_model"] == fading_model
            )
            for method in methods
        ]
        offsets = [i + (idx - 0.5) * width for i in x]
        ax2.bar(offsets, values, width=width, label=fading_model.capitalize())
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15)
    ax2.set_ylabel("Episode Secrecy Payload (Mbits)")
    ax2.set_title("Final Payload Comparison Across Methods")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(payload_plot, dpi=150)
    plt.close(fig2)

    return {
        "avg_rsec_plot": str(rate_plot.resolve()),
        "episode_mbits_plot": str(payload_plot.resolve()),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Final comparison across baselines, DQN, and DDPG.")
    parser.add_argument("--dqn-rician-model", type=str, required=True, help="Path to Rician DQN model")
    parser.add_argument("--dqn-rayleigh-model", type=str, required=True, help="Path to Rayleigh DQN model")
    parser.add_argument("--ddpg-rician-actor", type=str, required=True, help="Path to Rician DDPG actor")
    parser.add_argument("--ddpg-rayleigh-actor", type=str, required=True, help="Path to Rayleigh DDPG actor")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per seed")
    parser.add_argument("--seeds", type=str, default="7,21,42,84,168", help="Comma-separated seeds")
    parser.add_argument("--output-dir", type=str, default="outputs/final_comparison", help="Output directory")
    parser.add_argument(
        "--control-mode",
        type=str,
        default="velocity",
        choices=["velocity", "waypoint"],
        help="Velocity-vector or normalized waypoint control",
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
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    result = run_final_comparison(
        dqn_rician_model=args.dqn_rician_model,
        dqn_rayleigh_model=args.dqn_rayleigh_model,
        ddpg_rician_actor=args.ddpg_rician_actor,
        ddpg_rayleigh_actor=args.ddpg_rayleigh_actor,
        episodes_per_seed=args.episodes,
        seeds=seeds,
        output_dir=args.output_dir,
        control_mode=args.control_mode,
        user_mobile=args.user_mobile,
        use_los_model=args.use_los_model,
        observation_mode=args.observation_mode,
        normalize_observations=not args.no_normalize,
    )

    print("Final comparison complete:")
    print(f"  CSV: {result['csv_path']}")
    for key, value in result["plot_paths"].items():
        print(f"  {key}: {value}")
