import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from baselines import distance_greedy_policy, evaluate_policy, random_policy
from Summer_Internship_2026.ddpg_train import Actor, evaluate_ddpg
from Summer_Internship_2026.environment import EnvConfig, UAVEnvironment


def evaluate_ddpg_multi_seed(
    actor_path: str,
    episodes_per_seed: int = 20,
    seeds: list[int] | None = None,
    output_dir: str = "outputs/ddpg_eval",
    channel_model: str = "rician",
    rician_k: float = 5.0,
    control_mode: str = "velocity",
    role_switching: bool = False,
) -> dict:
    if seeds is None:
        seeds = [7, 21, 42, 84, 168]

    env_cfg = EnvConfig(
        seed=0,
        fading_model=channel_model,
        rician_k=rician_k,
        control_mode=control_mode,
        role_switching=role_switching,
    )
    state_dim = UAVEnvironment(env_cfg).reset().shape[0]
    checkpoint = torch.load(actor_path, map_location="cpu")
    output_key = "net.4.weight"
    action_dim = checkpoint[output_key].shape[0] if output_key in checkpoint else (6 if role_switching else 5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(state_dim, action_dim, hidden_dim=128).to(device)
    actor.load_state_dict(checkpoint)
    actor.eval()

    rows = []
    for seed in seeds:
        eval_cfg = EnvConfig(
            seed=seed,
            fading_model=channel_model,
            rician_k=rician_k,
            control_mode=control_mode,
            role_switching=role_switching,
        )
        env = UAVEnvironment(eval_cfg)
        ddpg = evaluate_ddpg(env, actor, device=device, episodes=episodes_per_seed)
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
                "ddpg_avg_rsec_mbps": ddpg["mean_avg_rsec_mbps"],
                "ddpg_episode_secrecy_mbits": ddpg["mean_episode_secrecy_mbits"],
                "random_avg_rsec_mbps": rnd["mean_avg_R_sec_mbps"],
                "greedy_avg_rsec_mbps": grd["mean_avg_R_sec_mbps"],
            }
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ddpg_vs_baselines.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    agg = {
        "fading_model": channel_model,
        "ddpg_avg_rsec_mbps": float(np.mean([r["ddpg_avg_rsec_mbps"] for r in rows])),
        "random_avg_rsec_mbps": float(np.mean([r["random_avg_rsec_mbps"] for r in rows])),
        "greedy_avg_rsec_mbps": float(np.mean([r["greedy_avg_rsec_mbps"] for r in rows])),
        "ddpg_episode_secrecy_mbits": float(np.mean([r["ddpg_episode_secrecy_mbits"] for r in rows])),
    }
    return {"csv_path": str(out_csv.resolve()), "aggregate": agg}


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained DDPG against baselines.")
    parser.add_argument(
        "--actor-path",
        type=str,
        default="outputs/ddpg/ddpg_actor.pt",
        help="Path to trained DDPG actor (.pt)",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per seed")
    parser.add_argument("--seeds", type=str, default="7,21,42,84,168", help="Comma-separated seeds")
    parser.add_argument("--output-dir", type=str, default="outputs/ddpg_eval", help="Output directory")
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
    parser.add_argument("--role-switching", action="store_true", help="Enable relay/jammer role switching")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    result = evaluate_ddpg_multi_seed(
        actor_path=args.actor_path,
        episodes_per_seed=args.episodes,
        seeds=seeds,
        output_dir=args.output_dir,
        channel_model=args.channel_model,
        rician_k=args.rician_k,
        control_mode=args.control_mode,
        role_switching=args.role_switching,
    )

    agg = result["aggregate"]
    print("Evaluation complete:")
    print(f"  Channel model          : {agg['fading_model']}")
    print(f"  DDPG avg secrecy rate   : {agg['ddpg_avg_rsec_mbps']:.4f} Mbps")
    print(f"  Random avg secrecy rate : {agg['random_avg_rsec_mbps']:.4f} Mbps")
    print(f"  Greedy avg secrecy rate : {agg['greedy_avg_rsec_mbps']:.4f} Mbps")
    print(f"  DDPG secrecy payload/ep : {agg['ddpg_episode_secrecy_mbits']:.4f} Mbits")
    print(f"  CSV saved at            : {result['csv_path']}")
