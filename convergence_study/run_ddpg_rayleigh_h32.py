import argparse

from run_common import run_convergence_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DDPG convergence study with Rayleigh fading and h32.")
    parser.add_argument("--episodes", type=int, default=3000, help="Training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device preference")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_convergence_experiment("ddpg", "rayleigh", 32, args.episodes, args.seed, args.device)
