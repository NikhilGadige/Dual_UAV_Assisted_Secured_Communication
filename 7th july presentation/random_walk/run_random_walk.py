"""Random-Walk baseline for the multi-agent sensing study (CRB + Pd).

No learning: each sensing UAV moves according to
mobility_types_exp.mobility_models.RandomWalkMobility (the already-built
correlated-random-walk model used elsewhere in this repo for mobile-user
mobility), reused here as-is to drive each agent's trajectory. This gives
a non-learning floor to compare MAPPO / MATD3PG / MADDPG against in the
convergence plots — a flat/noisy line, since there's no policy to
improve, but a genuinely useful reference for "how much does learning
actually help sensing geometry."
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_PRESENTATION_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PRESENTATION_DIR.parent
sys.path.insert(0, str(_PRESENTATION_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from mobility_types_exp.mobility_models import RandomWalkMobility  # noqa: E402
from common.sensing_env import SensingEnvConfig, MultiAgentSensingEnv  # noqa: E402
from common.plotting import generate_convergence_plots  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Random Walk baseline — multi-agent sensing (CRB + Pd)")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--n-agents", type=int, default=2)
    p.add_argument("--alpha-pd", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--n-eval-episodes", type=int, default=5)
    p.add_argument("--output-root", type=str, default=str(Path(__file__).resolve().parent / "output"))
    return p.parse_args()


def run_episode(env: MultiAgentSensingEnv, mobility: dict, velocity: dict) -> dict:
    obs, _ = env.reset()
    for name in env.agent_names:
        mobility[name].reset()
        velocity[name] = np.zeros(2)

    crbs, pds, rewards = [], [], []
    for _ in range(env.cfg.steps_per_episode):
        actions = {}
        for i, name in enumerate(env.agent_names):
            xy = env.q_agents[i].copy()
            new_xy, velocity[name] = mobility[name].step(
                xy, velocity[name], dt=env.cfg.dt,
                half_area=1.0e6, user_max_speed=env.cfg.v_max,
            )
            delta = new_xy - xy
            actions[name] = np.clip(delta / max(env.cfg.v_max * env.cfg.dt, 1e-9), -1.0, 1.0)

        obs, r, terminated, truncated, info = env.step(actions)
        crbs.append(info["crb_mean"])
        pds.append(info["pd_mean"])
        rewards.append(info["reward"])
        if truncated.get("__all__", False):
            break

    return {"avg_crb": float(np.mean(crbs)), "avg_pd": float(np.mean(pds)), "avg_reward": float(np.mean(rewards))}


def evaluate_random_walk(env_cfg: SensingEnvConfig, n_eval_episodes: int, base_seed: int) -> dict:
    rng_state = np.random.get_state()
    rewards, crbs, pds = [], [], []
    eval_env = MultiAgentSensingEnv(env_cfg)
    try:
        for i in range(n_eval_episodes):
            seed = base_seed + 10_000 + i
            np.random.seed(seed)
            mobility = {name: RandomWalkMobility() for name in eval_env.agent_names}
            velocity = {name: np.zeros(2) for name in eval_env.agent_names}
            eval_env.reset(seed=seed, num_mc=env_cfg.num_mc_eval)
            for name in eval_env.agent_names:
                mobility[name].reset()
            ep_rewards, ep_crbs, ep_pds = [], [], []
            for _ in range(eval_env.cfg.steps_per_episode):
                actions = {}
                for j, name in enumerate(eval_env.agent_names):
                    xy = eval_env.q_agents[j].copy()
                    new_xy, velocity[name] = mobility[name].step(
                        xy, velocity[name], dt=eval_env.cfg.dt,
                        half_area=1.0e6, user_max_speed=eval_env.cfg.v_max,
                    )
                    delta = new_xy - xy
                    actions[name] = np.clip(delta / max(eval_env.cfg.v_max * eval_env.cfg.dt, 1e-9), -1.0, 1.0)
                _, rewards_dict, terminated, truncated, info = eval_env.step(actions, num_mc=env_cfg.num_mc_eval)
                ep_rewards.append(float(rewards_dict[eval_env.agent_names[0]]))
                ep_crbs.append(float(info["crb_mean"]))
                ep_pds.append(float(info["pd_mean"]))
                if terminated.get("__all__", False) or truncated.get("__all__", False):
                    break
            rewards.append(float(np.mean(ep_rewards)))
            crbs.append(float(np.mean(ep_crbs)))
            pds.append(float(np.mean(ep_pds)))
    finally:
        np.random.set_state(rng_state)
    return {
        "avg_crb": float(np.mean(crbs)),
        "avg_pd": float(np.mean(pds)),
        "avg_reward": float(np.mean(rewards)),
        "std_crb": float(np.std(crbs)),
        "std_pd": float(np.std(pds)),
        "std_reward": float(np.std(rewards)),
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    env_cfg = SensingEnvConfig(
        n_agents=args.n_agents, steps_per_episode=args.steps,
        alpha_pd=args.alpha_pd, seed=args.seed,
    )
    env = MultiAgentSensingEnv(env_cfg)
    mobility = {name: RandomWalkMobility() for name in env.agent_names}
    velocity = {name: np.zeros(2) for name in env.agent_names}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"random_walk_{ts}"
    csv_dir = run_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting Random-Walk baseline: {args.episodes} episodes x {args.steps} steps, "
          f"{args.n_agents} sensing agents, alpha_pd={args.alpha_pd}")

    rows = []
    for ep in range(1, args.episodes + 1):
        result = run_episode(env, mobility, velocity)
        row = {"episode": ep, **result}
        if ep % args.eval_interval == 0 or ep == 1 or ep == args.episodes:
            eval_stats = evaluate_random_walk(env_cfg, args.n_eval_episodes, args.seed)
            for k, v in eval_stats.items():
                row[f"eval_{k}"] = v
        rows.append(row)
        if ep % 10 == 0 or ep == 1 or ep == args.episodes:
            print(f"Ep {ep:4d}/{args.episodes} | avg_CRB={result['avg_crb']:.5f} | "
                  f"avg_Pd={result['avg_pd']:.3f} | reward={result['avg_reward']:.3f}")
        if "eval_avg_crb" in row:
            print(f"          eval_CRB={row['eval_avg_crb']:.5f} | eval_Pd={row['eval_avg_pd']:.3f} | eval_reward={row['eval_avg_reward']:.3f}")

    csv_path = csv_dir / "training_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for r in rows for k in r.keys()}, key=lambda k: (k != "episode", k))
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    with open(run_dir / "history.json", "w") as f:
        json.dump({
            "crb": [r["avg_crb"] for r in rows],
            "pd": [r["avg_pd"] for r in rows],
            "reward": [r["avg_reward"] for r in rows],
        }, f, indent=2)

    plots_dir = run_dir / "plots"
    paths = generate_convergence_plots(str(csv_path), str(plots_dir), title="Random Walk", color="#7f7f7f")
    print(f"Random-Walk baseline complete. Results in {run_dir}")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
