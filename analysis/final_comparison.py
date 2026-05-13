import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from analysis.baselines import distance_greedy_policy, evaluate_policy, random_policy
from core.config_utils import build_env_config
from rl.advanced_rl_train import GaussianActor, rollout_episode
from rl.ddpg_train import Actor, evaluate_ddpg
from rl.dqn_train import QNetwork, evaluate_dqn, make_action_table
from rl.marl_dqn_train import evaluate_marl_dqn
from rl.marl_utils import (
    jammer_observation,
    make_jammer_action_table,
    make_relay_action_table,
    relay_observation,
)
from core.environment import UAVEnvironment

_DISPLAY_NAMES = {
    "dqn": "DQN",
    "ddpg": "DDPG",
    "td3": "TD3",
    "sac": "SAC",
    "greedy": "Distance-Greedy",
    "marl_shared": "MARL Shared",
    "marl_split": "MARL Split",
    "random": "Random Walk",
}

_METHOD_ORDER = ["random", "greedy", "dqn", "ddpg", "td3", "sac", "marl_shared", "marl_split"]


def _safe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _infer_action_dim(checkpoint_path: str) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "mean.weight" in checkpoint:
        return int(checkpoint["mean.weight"].shape[0])
    output_key = "net.4.weight"
    return int(checkpoint[output_key].shape[0]) if output_key in checkpoint else 5


def _discover_available_methods() -> dict:
    training_dir = Path("outputs/training")
    methods = {}

    dqn_path = training_dir / "dqn" / "dqn_qnet.pt"
    if dqn_path.exists():
        methods["dqn"] = str(dqn_path)

    ddpg_path = training_dir / "ddpg" / "ddpg_actor.pt"
    if ddpg_path.exists():
        methods["ddpg"] = str(ddpg_path)

    td3_path = training_dir / "td3" / "td3_actor.pt"
    if td3_path.exists():
        methods["td3"] = str(td3_path)

    sac_path = training_dir / "sac" / "sac_actor.pt"
    if sac_path.exists():
        methods["sac"] = str(sac_path)

    marl_shared_relay = training_dir / "marl_shared" / "marl_relay_qnet.pt"
    marl_shared_jammer = training_dir / "marl_shared" / "marl_jammer_qnet.pt"
    if marl_shared_relay.exists() and marl_shared_jammer.exists():
        methods["marl_shared"] = {
            "relay": str(marl_shared_relay),
            "jammer": str(marl_shared_jammer),
        }

    marl_split_relay = training_dir / "marl_split" / "marl_relay_qnet.pt"
    marl_split_jammer = training_dir / "marl_split" / "marl_jammer_qnet.pt"
    if marl_split_relay.exists() and marl_split_jammer.exists():
        methods["marl_split"] = {
            "relay": str(marl_split_relay),
            "jammer": str(marl_split_jammer),
        }

    methods["greedy"] = None
    methods["random"] = None
    return methods


def _build_env(env_params: dict, seed: int):
    return build_env_config(seed=seed, **env_params)


def _evaluate_method(
    method: str,
    checkpoint,
    env_params: dict,
    episodes_per_seed: int,
    seeds: list[int],
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    template_env = UAVEnvironment(_build_env(env_params, 0))

    if method == "dqn":
        action_table = make_action_table()
        state_dim = template_env.reset().shape[0]
        q_net = QNetwork(state_dim, len(action_table), hidden_dim=128).to(device)
        q_net.load_state_dict(torch.load(checkpoint, map_location=device))
        q_net.eval()
        all_m = [evaluate_dqn(UAVEnvironment(_build_env(env_params, s)), q_net, action_table, device=device, episodes=episodes_per_seed) for s in seeds]
    elif method in ("ddpg", "td3"):
        state_dim = template_env.reset().shape[0]
        action_dim = _infer_action_dim(checkpoint)
        actor = Actor(state_dim, action_dim, hidden_dim=128).to(device)
        actor.load_state_dict(torch.load(checkpoint, map_location=device))
        actor.eval()
        all_m = [evaluate_ddpg(UAVEnvironment(_build_env(env_params, s)), actor, device=device, episodes=episodes_per_seed) for s in seeds]
    elif method == "sac":
        state_dim = template_env.reset().shape[0]
        action_dim = _infer_action_dim(checkpoint)
        actor = GaussianActor(state_dim, action_dim, hidden_dim=128).to(device)
        actor.load_state_dict(torch.load(checkpoint, map_location=device))
        actor.eval()

        def _run(env):
            rs = [rollout_episode(env, lambda s: actor.deterministic(s), device) for _ in range(episodes_per_seed)]
            return {"mean_avg_rsec_mbps": float(np.mean([r["avg_rsec_mbps"] for r in rs])), "mean_episode_secrecy_mbits": float(np.mean([r["episode_secrecy_mbits"] for r in rs]))}

        all_m = [_run(UAVEnvironment(_build_env(env_params, s))) for s in seeds]
    elif method in ("marl_shared", "marl_split"):
        agent_obs_mode = "shared" if method == "marl_shared" else "split"
        relay_at = make_relay_action_table()
        jammer_at = make_jammer_action_table()
        full_obs = template_env.reset()
        full_dim = full_obs.shape[0]
        if agent_obs_mode == "shared":
            relay_dim = jammer_dim = full_dim
        else:
            relay_dim = relay_observation(full_obs).shape[0]
            jammer_dim = jammer_observation(full_obs).shape[0]
        relay_qnet = QNetwork(relay_dim, len(relay_at), hidden_dim=128).to(device)
        jammer_qnet = QNetwork(jammer_dim, len(jammer_at), hidden_dim=128).to(device)
        relay_qnet.load_state_dict(torch.load(checkpoint["relay"], map_location=device))
        jammer_qnet.load_state_dict(torch.load(checkpoint["jammer"], map_location=device))
        relay_qnet.eval()
        jammer_qnet.eval()
        all_m = [evaluate_marl_dqn(UAVEnvironment(_build_env(env_params, s)), relay_qnet, jammer_qnet, relay_at, jammer_at, device, agent_obs_mode, episodes=episodes_per_seed) for s in seeds]
    else:
        raise ValueError(f"Unknown method: {method}")

    key = method.replace("marl_", "marl")
    return {
        f"{key}_avg_rsec_mbps": float(np.mean([m["mean_avg_rsec_mbps"] for m in all_m])),
        f"{key}_episode_secrecy_mbits": float(np.mean([m["mean_episode_secrecy_mbits"] for m in all_m])),
    }


def _aggregate_baseline(
    policy_name: str,
    policy_fn,
    episodes_per_seed: int,
    seeds: list[int],
    env_cfg,
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
    dqn_rician_model: str | None = None,
    dqn_rayleigh_model: str | None = None,
    ddpg_rician_actor: str | None = None,
    ddpg_rayleigh_actor: str | None = None,
    episodes_per_seed: int = 20,
    seeds: list[int] | None = None,
    output_dir: str = "outputs/comparisons",
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

    available = _discover_available_methods()
    method_keys = [m for m in _METHOD_ORDER if m in available]
    env_params = dict(
        control_mode=control_mode,
        user_mobile=user_mobile,
        use_los_model=use_los_model,
        observation_mode=observation_mode,
        normalize_observations=normalize_observations,
    )

    rows: list[dict] = []
    for fading_model in ["rician", "rayleigh"]:
        fm_env_params = {**env_params, "fading_model": fading_model}
        base_cfg = build_env_config(**fm_env_params)

        for method in method_keys:
            if method in ("random", "greedy"):
                fn = random_policy if method == "random" else distance_greedy_policy
                s = _aggregate_baseline(_DISPLAY_NAMES[method], fn, episodes_per_seed, seeds, base_cfg)
                rows.append({
                    "method": _DISPLAY_NAMES[method],
                    "fading_model": fading_model,
                    "enable_energy_harvesting": False,
                    "observation_has_eh": observation_mode == "full_eh",
                    "avg_rsec_mbps": s["mean_avg_R_sec_mbps"],
                    "episode_secrecy_mbits": s["mean_episode_secrecy_mbits"],
                })
            else:
                ckpt = available[method]
                result = _evaluate_method(method, ckpt, fm_env_params, episodes_per_seed, seeds)
                key = method.replace("marl_", "marl")
                rows.append({
                    "method": _DISPLAY_NAMES[method],
                    "fading_model": fading_model,
                    "enable_energy_harvesting": False,
                    "observation_has_eh": observation_mode == "full_eh",
                    "avg_rsec_mbps": result[f"{key}_avg_rsec_mbps"],
                    "episode_secrecy_mbits": result[f"{key}_episode_secrecy_mbits"],
                })

    csv_path = out_dir / "comparison_summary.csv"
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

    fading_models = ["rician", "rayleigh"]
    methods_in_order = list(dict.fromkeys(r["method"] for r in rows if r["fading_model"] == "rician"))
    x = list(range(len(methods_in_order)))
    width = 0.35

    rate_plot = out_dir / "secrecy_rate_barplot.png"
    fig1 = plt.figure(figsize=(10, 5))
    ax1 = fig1.add_subplot(111)
    for idx, fm in enumerate(fading_models):
        values = [next(r["avg_rsec_mbps"] for r in rows if r["method"] == m and r["fading_model"] == fm) for m in methods_in_order]
        offsets = [i + (idx - 0.5) * width for i in x]
        ax1.bar(offsets, values, width=width, label=fm.capitalize())
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_in_order, rotation=15)
    ax1.set_ylabel("Avg Secrecy Rate (Mbps)")
    ax1.set_title("Secrecy Rate Comparison Across Methods")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.25)
    fig1.tight_layout()
    fig1.savefig(rate_plot, dpi=150)
    plt.close(fig1)

    payload_plot = out_dir / "secrecy_payload_barplot.png"
    fig2 = plt.figure(figsize=(10, 5))
    ax2 = fig2.add_subplot(111)
    for idx, fm in enumerate(fading_models):
        values = [next(r["episode_secrecy_mbits"] for r in rows if r["method"] == m and r["fading_model"] == fm) for m in methods_in_order]
        offsets = [i + (idx - 0.5) * width for i in x]
        ax2.bar(offsets, values, width=width, label=fm.capitalize())
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods_in_order, rotation=15)
    ax2.set_ylabel("Episode Secrecy Payload (Mbits)")
    ax2.set_title("Secrecy Payload Comparison Across Methods")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(payload_plot, dpi=150)
    plt.close(fig2)

    rank_plot = out_dir / "method_ranking.png"
    fig3 = plt.figure(figsize=(10, 5))
    ax3 = fig3.add_subplot(111)
    for idx, fm in enumerate(fading_models):
        values = [next(r["avg_rsec_mbps"] for r in rows if r["method"] == m and r["fading_model"] == fm) for m in methods_in_order]
        order = np.argsort(values)[::-1]
        ranked_methods = [methods_in_order[i] for i in order]
        ranked_vals = [values[i] for i in order]
        offsets = [i + (idx - 0.5) * width for i in range(len(ranked_methods))]
        ax3.bar(offsets, ranked_vals, width=width, label=fm.capitalize())
        for i, (v, m) in enumerate(zip(ranked_vals, ranked_methods)):
            ax3.text(i + (idx - 0.5) * width, v + 0.01, f"{i+1}", ha="center", va="bottom", fontsize=8)
    ax3.set_xticks(list(range(len(methods_in_order))))
    ax3.set_xticklabels([f"#{i+1}" for i in range(len(methods_in_order))], rotation=0)
    ax3.set_ylabel("Avg Secrecy Rate (Mbps)")
    ax3.set_title("Method Ranking by Secrecy Rate")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.25)
    fig3.tight_layout()
    fig3.savefig(rank_plot, dpi=150)
    plt.close(fig3)

    return {
        "avg_rsec_plot": str(rate_plot.resolve()),
        "episode_mbits_plot": str(payload_plot.resolve()),
        "method_ranking_plot": str(rank_plot.resolve()),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Final comparison across all available RL methods.")
    parser.add_argument(
        "--dqn-rician-model",
        type=str,
        default=None,
        help="Path to Rician DQN model (optional, auto-resolved if omitted)",
    )
    parser.add_argument(
        "--dqn-rayleigh-model",
        type=str,
        default=None,
        help="Path to Rayleigh DQN model (optional, auto-resolved if omitted)",
    )
    parser.add_argument(
        "--ddpg-rician-actor",
        type=str,
        default=None,
        help="Path to Rician DDPG actor (optional, auto-resolved if omitted)",
    )
    parser.add_argument(
        "--ddpg-rayleigh-actor",
        type=str,
        default=None,
        help="Path to Rayleigh DDPG actor (optional, auto-resolved if omitted)",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per seed")
    parser.add_argument("--seeds", type=str, default="7,21,42,84,168", help="Comma-separated seeds")
    parser.add_argument("--output-dir", type=str, default="outputs/comparisons", help="Output directory")
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
