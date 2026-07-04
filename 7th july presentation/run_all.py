"""Run MAPPO, MATD3PG, MADDPG, and the Random-Walk baseline back-to-back
with matched settings, then build cross-algorithm CRB/Pd/reward
comparison plots for the 2026-07-07 presentation.

Usage:
    python "7th july presentation/run_all.py" --episodes 300 --steps 40
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from common.comparison import generate_comparison_plots  # noqa: E402

SCRIPTS = {
    "MAPPO": _THIS_DIR / "mappo" / "train_mappo.py",
    "MATD3PG": _THIS_DIR / "matd3pg" / "train_matd3pg.py",
    "MADDPG": _THIS_DIR / "maddpg" / "train_maddpg.py",
    "Random Walk": _THIS_DIR / "random_walk" / "run_random_walk.py",
}


def parse_args():
    p = argparse.ArgumentParser(description="Run all 4 sensing-study methods and compare")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--n-agents", type=int, default=2)
    p.add_argument("--alpha-pd", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def latest_run_dir(output_root: Path) -> Path:
    candidates = [p for p in output_root.iterdir() if p.is_dir()]
    return max(candidates, key=lambda p: p.name)


def main():
    args = parse_args()
    common_args = [
        "--episodes", str(args.episodes),
        "--steps", str(args.steps),
        "--n-agents", str(args.n_agents),
        "--alpha-pd", str(args.alpha_pd),
        "--seed", str(args.seed),
    ]

    csv_paths = {}
    for label, script in SCRIPTS.items():
        print(f"\n{'=' * 60}\nRunning {label}\n{'=' * 60}")
        output_root = script.parent / "output"
        subprocess.run([sys.executable, str(script), *common_args], check=True)
        run_dir = latest_run_dir(output_root)
        csv_paths[label] = str(run_dir / "csv" / "training_log.csv")

    comparison_dir = _THIS_DIR / "comparison_plots"
    paths = generate_comparison_plots(csv_paths, str(comparison_dir))

    print(f"\n{'=' * 60}\nAll runs complete. Comparison plots:\n{'=' * 60}")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
