import argparse

from basic_implementation.run_common import run_basic_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run refined basic DQN with Rician fading and h32.")
    parser.add_argument("--episodes", type=int, default=4000, help="Training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device preference")
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/basic_outputs",
        help="Root directory for this basic implementation update",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_basic_experiment("dqn", "rician", args.episodes, args.seed, args.device, 32, 5e-4, args.output_root)
