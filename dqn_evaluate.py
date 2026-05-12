import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from Summer_Internship_2026.environment import EnvConfig, UAVEnvironment
from baselines import distance_greedy_policy, evaluate_policy, random_policy
from dqn_train import QNetwork, evaluate_dqn, make_action_table


def evaluate_dqn_multi_seed(
    model_path: str,
    episodes_per_seed: int = 20,
    seeds: list[int] | None = None,
    output_dir: str = "outputs/dqn_eval",
    channel_model: str = "rician",
    rician_k: float = 5.0,
    control_mode: str = "velocity",
) -> dict:
    if seeds is None:
        seeds = [7, 21, 42, 84, 168]

    action_table = make_action_table()
    env_cfg = EnvConfig(seed=0, fading_model=channel_model, rician_k=rician_k, control_mode=control_mode)
    state_dim = UAVEnvironment(env_cfg).reset().shape[0]
    action_dim = len(action_table)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QNetwork(state_dim, action_dim, hidden_dim=128).to(device)
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()

    rows = []
    for seed in seeds:
        eval_cfg = EnvConfig(seed=seed, fading_model=channel_model, rician_k=rician_k, control_mode=control_mode)
        env = UAVEnvironment(eval_cfg)
        dqn = evaluate_dqn(env, q_net, action_table, device=device, episodes=episodes_per_seed)
        rnd = evaluate_policy(
            "Random Walk",
            random_policy,
            episodes=episodes_per_seed,
            seed=seed,
            env_config=eval_cfg,
        )
        grd = evaluate_policy(
            "Distance-Greedy",
            distance_greedy_policy,
            episodes=episodes_per_seed,
            seed=seed,
            env_config=eval_cfg,
        )

        rows.append(
            {
                "seed": seed,
                "fading_model": channel_model,
                "dqn_avg_rsec_mbps": dqn["mean_avg_rsec_mbps"],
                "dqn_episode_secrecy_mbits": dqn["mean_episode_secrecy_mbits"],
                "random_avg_rsec_mbps": rnd["mean_avg_R_sec_mbps"],
                "greedy_avg_rsec_mbps": grd["mean_avg_R_sec_mbps"],
            }
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "dqn_vs_baselines.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    agg = {
        "fading_model": channel_model,
        "dqn_avg_rsec_mbps": float(np.mean([r["dqn_avg_rsec_mbps"] for r in rows])),
        "random_avg_rsec_mbps": float(np.mean([r["random_avg_rsec_mbps"] for r in rows])),
        "greedy_avg_rsec_mbps": float(np.mean([r["greedy_avg_rsec_mbps"] for r in rows])),
        "dqn_episode_secrecy_mbits": float(np.mean([r["dqn_episode_secrecy_mbits"] for r in rows])),
    }
    return {"csv_path": str(out_csv.resolve()), "aggregate": agg}


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN against baselines.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/dqn_smoke/dqn_qnet.pt",
        help="Path to trained DQN model (.pt)",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per seed")
    parser.add_argument("--seeds", type=str, default="7,21,42,84,168", help="Comma-separated seeds")
    parser.add_argument("--output-dir", type=str, default="outputs/dqn_eval", help="Output directory")
    parser.add_argument(
        "--channel-model",
        type=str,
        default="rician",
        choices=["rician", "rayleigh"],
        help="Fading model used by the trained environment",
    )
    parser.add_argument("--rician-k", type=float, default=5.0, help="Rician K-factor")
    parser.add_argument(
        "--control-mode",
        type=str,
        default="velocity",
        choices=["velocity", "waypoint"],
        help="Velocity-vector or normalized waypoint control",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    result = evaluate_dqn_multi_seed(
        model_path=args.model_path,
        episodes_per_seed=args.episodes,
        seeds=seeds,
        output_dir=args.output_dir,
        channel_model=args.channel_model,
        rician_k=args.rician_k,
        control_mode=args.control_mode,
    )

    agg = result["aggregate"]
    print("Evaluation complete:")
    print(f"  Channel model          : {agg['fading_model']}")
    print(f"  DQN avg secrecy rate    : {agg['dqn_avg_rsec_mbps']:.4f} Mbps")
    print(f"  Random avg secrecy rate : {agg['random_avg_rsec_mbps']:.4f} Mbps")
    print(f"  Greedy avg secrecy rate : {agg['greedy_avg_rsec_mbps']:.4f} Mbps")
    print(f"  DQN secrecy payload/ep  : {agg['dqn_episode_secrecy_mbits']:.4f} Mbits")
    print(f"  CSV saved at            : {result['csv_path']}")
