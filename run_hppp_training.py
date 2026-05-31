import argparse
import sys
from pathlib import Path

import numpy as np

from hppp_training_utils import (
    _read_csv,
    _float,
    estimate_convergence_episode,
    generate_convergence_summary,
    plot_training_curves,
)

OUTPUT_ROOT = Path("outputs/hppp_training")
SMOKE = True


def _smoke_cfg(episodes: int | None, default: int) -> int:
    return episodes if episodes is not None else (50 if SMOKE else default)


def train_dqn(episodes: int | None, device: str, channel_model: str = "rician") -> dict:
    from basic_implementation.configs import build_basic_dqn_config
    from basic_implementation.run_common import train_dqn as basic_train_dqn

    n_ep = _smoke_cfg(episodes, 4000)
    cfg = build_basic_dqn_config(
        fading_model=channel_model, episodes=n_ep, hidden_dim=32,
        learning_rate=5e-4, seed=42, device=device,
    )
    out = str(OUTPUT_ROOT / "dqn" / channel_model)
    print(f"\n{'='*60}\nTraining DQN (HPPP) via basic_implementation for {n_ep} episodes [{channel_model}]\n{'='*60}")
    return basic_train_dqn(cfg, output_dir=out)


def train_ddpg(episodes: int | None, device: str, channel_model: str = "rician") -> dict:
    from basic_implementation.configs import build_basic_ddpg_config
    from basic_implementation.run_common import train_ddpg as basic_train_ddpg

    n_ep = _smoke_cfg(episodes, 4000)
    cfg = build_basic_ddpg_config(
        fading_model=channel_model, episodes=n_ep, hidden_dim=32,
        learning_rate=5e-4, seed=42, device=device,
    )
    out = str(OUTPUT_ROOT / "ddpg" / channel_model)
    print(f"\n{'='*60}\nTraining DDPG (HPPP) via basic_implementation for {n_ep} episodes [{channel_model}]\n{'='*60}")
    return basic_train_ddpg(cfg, output_dir=out)


def train_d3qn(episodes: int | None, device: str, channel_model: str = "rician") -> dict:
    from d3qn_study.train_d3qn import D3QNConfig, train_d3qn

    n_ep = _smoke_cfg(episodes, 4000)
    cfg = D3QNConfig(episodes=n_ep, device=device, eval_interval=0,
                     epsilon_decay_steps=n_ep * 120)
    cfg.seed = 42
    cfg.fading_model = channel_model
    out = str(OUTPUT_ROOT / "d3qn" / channel_model)
    print(f"\n{'='*60}\nTraining D3QN (HPPP) for {n_ep} episodes [{channel_model}]\n{'='*60}")
    return train_d3qn(cfg, output_dir=out)


def train_ppo(episodes: int | None, device: str, channel_model: str = "rician") -> dict:
    from PPO_study.train_ppo import PPOConfig, train_ppo

    n_ep = _smoke_cfg(episodes, 4000)
    cfg = PPOConfig(episodes=n_ep, device=device, eval_interval=0)
    cfg.seed = 42
    cfg.fading_model = channel_model
    out = str(OUTPUT_ROOT / "ppo" / channel_model)
    print(f"\n{'='*60}\nTraining PPO (HPPP) via PPO_study for {n_ep} episodes [{channel_model}]\n{'='*60}")
    return train_ppo(cfg, output_dir=out)


def train_sac(episodes: int | None, device: str, channel_model: str = "rician") -> dict:
    from sac_study.sac_train import train_sac
    from sac_study.configs import SACStudyConfig

    n_ep = _smoke_cfg(episodes, 4000)
    cfg = SACStudyConfig(episodes=n_ep, device=device, eval_interval=0,
                         output_root=str(OUTPUT_ROOT / "sac" / channel_model))
    cfg.seed = 42
    cfg.fading_model = channel_model
    out = str(OUTPUT_ROOT / "sac" / channel_model)
    print(f"\n{'='*60}\nTraining SAC (HPPP) via sac_study for {n_ep} episodes [{channel_model}]\n{'='*60}")
    return train_sac(cfg, output_dir=out)


def train_td3pg(episodes: int | None, device: str, channel_model: str = "rician") -> dict:
    from td3pg_study.td3pg_train import train_td3pg
    from td3pg_study.configs import TD3PGStudyConfig

    n_ep = _smoke_cfg(episodes, 4000)
    cfg = TD3PGStudyConfig(episodes=n_ep, device=device, eval_interval=0,
                           output_root=str(OUTPUT_ROOT / "td3pg" / channel_model))
    cfg.seed = 42
    cfg.fading_model = channel_model
    out = str(OUTPUT_ROOT / "td3pg" / channel_model)
    print(f"\n{'='*60}\nTraining TD3PG (HPPP) via td3pg_study for {n_ep} episodes [{channel_model}]\n{'='*60}")
    return train_td3pg(cfg, output_dir=out)


ALGORITHMS = {
    "dqn": train_dqn,
    "ddpg": train_ddpg,
    "d3qn": train_d3qn,
    "ppo": train_ppo,
    "sac": train_sac,
    "td3pg": train_td3pg,
}


def _find_csv(output_root: Path, algo: str, channel: str) -> Path | None:
    base = output_root / algo / channel
    p = base / "training_log.csv"
    if p.exists():
        return p
    return None


def _parse_args():
    p = argparse.ArgumentParser(description="HPPP multi-eavesdropper training orchestrator")
    algo_choices = list(ALGORITHMS) + ["all"]
    p.add_argument("--algos", type=str, nargs="+", choices=algo_choices, default=["all"])
    p.add_argument("--channel-model", choices=["rician", "rayleigh"], default="rician",
                   help="Fading channel model")
    p.add_argument("--episodes", type=int, default=None, help="Override episode count")
    p.add_argument("--smoke", action="store_true", default=True, help="50-episode smoke test")
    p.add_argument("--no-smoke", dest="smoke", action="store_false", help="Full training (3000+ episodes)")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--only-analysis", type=str, nargs="+", default=None,
                   help="Skip training, only generate plots/summaries from existing CSVs")
    return p.parse_args()


def main():
    global SMOKE
    args = _parse_args()
    SMOKE = args.smoke
    channel = args.channel_model

    if args.only_analysis is not None:
        algos_to_run = [a for a in args.only_analysis if a in ALGORITHMS]
    elif "all" in args.algos:
        algos_to_run = list(ALGORITHMS)
    else:
        algos_to_run = args.algos

    training_results = {}
    if not args.only_analysis:
        for name in algos_to_run:
            fn = ALGORITHMS[name]
            try:
                result = fn(args.episodes, args.device, channel)
                training_results[name] = result
            except Exception as e:
                print(f"\n!!! {name} FAILED: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

    eval_results = {}
    convergence_data = {}

    for name in algos_to_run:
        if name in training_results:
            log_csv = training_results[name].get("training_log_csv", "")
            if log_csv and Path(log_csv).exists():
                log_csv = Path(log_csv)
            else:
                log_csv = None
        else:
            log_csv = _find_csv(OUTPUT_ROOT, name, channel)
        if log_csv:
            print(f"\n--- Processing {name} (CSV: {log_csv}) ---")
            plots_out = OUTPUT_ROOT / name / channel / "plots"
            plots_out.mkdir(parents=True, exist_ok=True)
            plot_training_curves(str(log_csv), str(plots_out))
            rows = _read_csv(str(log_csv))
            if len(rows) >= 100:
                rsecs = np.array([_float(r, "avg_R_sec_mbps") for r in rows])
                stable_ep = estimate_convergence_episode(rsecs)
                final_roll_rew = float(np.mean([_float(r, "avg_shaped_reward") for r in rows[-100:]]))
                final_roll_sec = float(np.mean(rsecs[-100:]))
                convergence_data[name] = {
                    "stable_episode": stable_ep,
                    "final_rolling_reward": final_roll_rew,
                    "final_rolling_secrecy": final_roll_sec,
                }
        else:
            print(f"  WARNING: no CSV found for {name}")

    if convergence_data:
        reports_dir = OUTPUT_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        conv_path = reports_dir / f"convergence_summary_{channel}.md"
        generate_convergence_summary(convergence_data, str(conv_path))
        print(f"\nConvergence summary saved to {conv_path}")

    print("\n=== HPPP Training Complete ===")


if __name__ == "__main__":
    main()
