import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_training_csv(csv_path: Path) -> list[dict[str, float | None]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        eval_val = row.get("eval_R_sec_mbps", "")
        parsed.append(
            {
                "episode": float(row["episode"]),
                "avg_R_sec_mbps": float(row["avg_R_sec_mbps"]),
                "rolling20": float(row["rolling20_avg_R_sec_mbps"]),
                "rolling100": float(row["rolling100_avg_R_sec_mbps"]),
                "avg_shaped_reward": float(row["avg_shaped_reward"]),
                "convergence_gap": float(row["convergence_gap"]),
                "eval_R_sec_mbps": float(eval_val) if str(eval_val).strip() else None,
            }
        )
    return parsed


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300})


def _save(x, y, title, ylabel, out, color):
    _style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(x, y, color=color, linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def generate_single_run_plots(csv_path: str, output_dir: str, title: str, color: str) -> list[str]:
    rows = read_training_csv(Path(csv_path))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ep = [r["episode"] for r in rows]
    avg = [r["avg_R_sec_mbps"] for r in rows]
    r20 = [r["rolling20"] for r in rows]
    r100 = [r["rolling100"] for r in rows]
    rew = [r["avg_shaped_reward"] for r in rows]
    gap = [r["convergence_gap"] for r in rows]

    paths = []
    p = out_dir / "secrecy_vs_episode.png"
    _save(ep, avg, title, "Average Secrecy Rate (Mbps)", p, color)
    paths.append(str(p.resolve()))

    p = out_dir / "rolling20_vs_episode.png"
    _save(ep, r20, title, "Rolling 20-Episode Secrecy (Mbps)", p, color)
    paths.append(str(p.resolve()))

    p = out_dir / "rolling100_vs_episode.png"
    _save(ep, r100, title, "Rolling 100-Episode Secrecy (Mbps)", p, color)
    paths.append(str(p.resolve()))

    p = out_dir / "shaped_reward_vs_episode.png"
    _save(ep, rew, title, "Average Shaped Reward", p, color)
    paths.append(str(p.resolve()))

    p = out_dir / "convergence_gap.png"
    _save(ep, gap, title, "|Rolling20-Rolling100| (Mbps)", p, color)
    paths.append(str(p.resolve()))

    _style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(ep, r100, color=color, label="Training rolling100")
    eval_pairs = [(r["episode"], r["eval_R_sec_mbps"]) for r in rows if r["eval_R_sec_mbps"] is not None]
    if eval_pairs:
        ax.plot([x for x, _ in eval_pairs], [y for _, y in eval_pairs], "o--", color="#111111", label="Evaluation")
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Secrecy Rate (Mbps)")
    ax.legend(frameon=True)
    fig.tight_layout()
    p = out_dir / "evaluation_vs_training.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p.resolve()))

    return paths


def generate_channel_comparison(rayleigh_csv: str, rician_csv: str, output_dir: str) -> str:
    ray_rows = read_training_csv(Path(rayleigh_csv))
    ric_rows = read_training_csv(Path(rician_csv))

    _style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot([r["episode"] for r in ray_rows], [r["rolling100"] for r in ray_rows], label="PPO + Rayleigh", color="#d62728")
    ax.plot([r["episode"] for r in ric_rows], [r["rolling100"] for r in ric_rows], label="PPO + Rician", color="#1f77b4")
    ax.set_title("PPO Convergence: Rayleigh vs Rician")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling 100-Episode Secrecy (Mbps)")
    ax.legend(frameon=True)
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "ppo_channel_comparison_rolling100.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return str(p.resolve())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PPO convergence plots")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--title", type=str, default="PPO")
    parser.add_argument("--color", type=str, default="#1f77b4")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_single_run_plots(args.csv, args.output_dir, args.title, args.color)
