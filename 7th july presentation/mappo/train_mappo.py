"""MAPPO for the multi-agent sensing study (CRB + Pd).

Reuses madrl_updated_exp.agents.mappo.MAPPOAgent as-is (the already-built
multi-agent-ready PPO implementation in this repo — same actor-critic /
tanh-squashed-Gaussian / clipped-surrogate design as PPO_study/train_ppo.py,
just packaged per-agent). Nothing in the agent code changes here; only the
environment (common/sensing_env.py) and the training loop are new.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PRESENTATION_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PRESENTATION_DIR.parent
sys.path.insert(0, str(_PRESENTATION_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from madrl_updated_exp.agents.mappo import MAPPOAgent  # noqa: E402
from common.sensing_env import SensingEnvConfig, MultiAgentSensingEnv  # noqa: E402
from common.trainer import SensingTrainer  # noqa: E402
from common.plotting import generate_convergence_plots  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="MAPPO — multi-agent sensing (CRB + Pd)")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--n-agents", type=int, default=2)
    p.add_argument("--alpha-pd", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", type=str, default=str(Path(__file__).resolve().parent / "output"))
    return p.parse_args()


def main():
    args = parse_args()
    env_cfg = SensingEnvConfig(
        n_agents=args.n_agents, steps_per_episode=args.steps,
        alpha_pd=args.alpha_pd, seed=args.seed,
    )
    probe_env = MultiAgentSensingEnv(env_cfg)
    agents = {
        name: MAPPOAgent(obs_dim=probe_env.obs_dim, act_dim=probe_env.act_dim, name=name,
                         hidden_dim=64, lr=1e-4, entropy_coef=0.005)
        for name in probe_env.agent_names
    }

    trainer = SensingTrainer(
        env_cfg=env_cfg, agents=agents, n_episodes=args.episodes,
        output_root=args.output_root, run_name="mappo",
    )
    print(f"Starting MAPPO training: {args.episodes} episodes x {args.steps} steps, "
          f"{args.n_agents} sensing agents, alpha_pd={args.alpha_pd}")
    run_dir = trainer.train()

    csv_path = str(Path(trainer.csv_dir) / "training_log.csv")
    plots_dir = str(Path(run_dir) / "plots")
    paths = generate_convergence_plots(csv_path, plots_dir, title="MAPPO", color="#ff7f0e")
    print(f"Training complete. Results in {run_dir}")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
