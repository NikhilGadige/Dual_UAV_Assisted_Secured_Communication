"""Run MADRL experiments: training, evaluation, comparison — updated 4-agent version."""

from __future__ import annotations

import argparse
import os

from madrl_updated_exp.configs import MADRLConfig, EnvConfig, AgentConfig, TrainingConfig, RewardConfig
from madrl_updated_exp.training.trainer import MARLTrainer
from madrl_updated_exp.evaluation import compare


def parse_args():
    parser = argparse.ArgumentParser(description="MADRL for ISAC — 4 agents (+ RIS phase, HPPP eves, Random-Walk user)")
    parser.add_argument("--algorithm", choices=["mappo", "matd3"], default="mappo")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="train")
    return parser.parse_args()


def make_default_cfg(args) -> MADRLConfig:
    env_cfg = EnvConfig(alpha=args.alpha, seed=args.seed)
    train_cfg = TrainingConfig(
        algorithm=args.algorithm,
        n_episodes=args.episodes,
        max_steps_per_episode=args.steps,
        seed=args.seed,
    )
    agents = [
        AgentConfig(name="bs_beamformer",
                    act_dim=2 * env_cfg.N_time * env_cfg.M_bs),
        AgentConfig(name="uav_trajectory",
                    act_dim=3 * env_cfg.N_time),
        AgentConfig(name="jammer_beamformer",
                    act_dim=2 * env_cfg.N_time * env_cfg.N_j),
        AgentConfig(name="ris_phase",
                    act_dim=env_cfg.N_time * env_cfg.N_ris),
    ]
    reward_cfg = RewardConfig()
    return MADRLConfig(
        env=env_cfg,
        agents=agents,
        training=train_cfg,
        reward=reward_cfg,
        output_root="outputs/reinforcement_learning/training_runs",
    )


def main():
    args = parse_args()
    cfg = make_default_cfg(args)
    print(f"Starting {args.algorithm} training for {args.episodes} episodes (alpha={args.alpha})")
    print(f"Agents: {[a.name for a in cfg.agents]}")

    trainer = MARLTrainer(cfg)

    if args.mode in ("train", "all"):
        trainer.train()

    if args.mode in ("eval", "all"):
        print("\nRunning comparison vs baselines...")
        results = compare(trainer.run_dir, cfg, n_episodes=10)
        print("\n=== Comparison Results ===")
        for label, metrics in results.items():
            print(f"{label}: Secrecy={metrics['avg_secrecy']:.4f}±{metrics['std_secrecy']:.4f}, "
                  f"Sensing={metrics['avg_sensing']:.4f}±{metrics['std_sensing']:.4f}, "
                  f"Violation={metrics['avg_violation']:.4f}")


if __name__ == "__main__":
    main()
