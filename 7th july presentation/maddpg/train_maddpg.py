"""MADDPG for the multi-agent sensing study (CRB + Pd).

Uses common/ddpg_agent.py's DDPGAgent — the Actor/Critic/OU-noise design
from rl/ddpg_train.py (the already-built single-agent DDPG in this repo),
repackaged per-agent behind the same interface as MAPPOAgent/MATD3Agent.
Only the environment (common/sensing_env.py) and training loop are new.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PRESENTATION_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PRESENTATION_DIR.parent
sys.path.insert(0, str(_PRESENTATION_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from common.ddpg_agent import DDPGAgent  # noqa: E402
from common.sensing_env import SensingEnvConfig, MultiAgentSensingEnv  # noqa: E402
from common.trainer import SensingTrainer  # noqa: E402
from common.plotting import generate_convergence_plots  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="MADDPG — multi-agent sensing (CRB + Pd)")
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
        name: DDPGAgent(obs_dim=probe_env.obs_dim, act_dim=probe_env.act_dim, name=name,
                         hidden_dim=64, batch_size=64)
        for name in probe_env.agent_names
    }

    trainer = SensingTrainer(
        env_cfg=env_cfg, agents=agents, n_episodes=args.episodes,
        output_root=args.output_root, run_name="maddpg",
    )
    print(f"Starting MADDPG training: {args.episodes} episodes x {args.steps} steps, "
          f"{args.n_agents} sensing agents, alpha_pd={args.alpha_pd}")
    run_dir = trainer.train()

    csv_path = str(Path(trainer.csv_dir) / "training_log.csv")
    plots_dir = str(Path(run_dir) / "plots")
    paths = generate_convergence_plots(csv_path, plots_dir, title="MADDPG", color="#1f77b4")
    print(f"Training complete. Results in {run_dir}")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
