import argparse
from pathlib import Path

from d3qn_study.plot_d3qn import generate_channel_comparison, generate_single_run_plots
from d3qn_study.train_d3qn import D3QNConfig, train_d3qn


def run_one_channel(channel: str, episodes: int, hidden_dim: int, seed: int, device: str, out_root: Path) -> dict:
    run_dir = out_root / f"d3qn_{channel}_h{hidden_dim}"
    cfg = D3QNConfig(
        episodes=episodes,
        hidden_dim=hidden_dim,
        seed=seed,
        device=device,
        fading_model=channel,
        epsilon_decay_steps=episodes * 120,
    )
    summary = train_d3qn(cfg, output_dir=str(run_dir))
    title = f"D3QN + {channel.title()} (h{hidden_dim})"
    color = "#1f77b4" if channel == "rician" else "#d62728"
    generate_single_run_plots(summary["training_log_csv"], str(run_dir), title, color)
    return summary


def run_study(episodes: int, hidden_dim: int, seed: int, device: str, out_root: str = "d3qn_study/output") -> dict:
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

    ray = run_one_channel("rayleigh", episodes, hidden_dim, seed, device, root)
    ric = run_one_channel("rician", episodes, hidden_dim, seed + 11, device, root)

    comparison_plot = generate_channel_comparison(
        rayleigh_csv=ray["training_log_csv"],
        rician_csv=ric["training_log_csv"],
        output_dir=str(root),
    )

    return {
        "rayleigh_log": ray["training_log_csv"],
        "rician_log": ric["training_log_csv"],
        "rayleigh_model": ray["model_path"],
        "rician_model": ric["model_path"],
        "comparison_plot": comparison_plot,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D3QN convergence study for Rayleigh and Rician channels")
    parser.add_argument("--episodes", type=int, default=2500)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="d3qn_study/output")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_study(args.episodes, args.hidden_dim, args.seed, args.device, args.output_dir)
    print("D3QN study complete:")
    for key, value in result.items():
        print(f"  {key}: {value}")
