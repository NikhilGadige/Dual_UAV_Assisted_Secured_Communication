import argparse
import csv
from pathlib import Path

import numpy as np


def _safe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def load_training_log(csv_path: str) -> list[dict]:
    rows = []
    with Path(csv_path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "episode": int(r["episode"]),
                    "noise_std": float(r["noise_std"]),
                    "avg_R_sec_mbps": float(r["avg_R_sec_mbps"]),
                    "rolling20_avg_R_sec_mbps": float(
                        r.get("rolling20_avg_R_sec_mbps", r["rolling100_avg_R_sec_mbps"])
                    ),
                    "rolling100_avg_R_sec_mbps": float(r["rolling100_avg_R_sec_mbps"]),
                    "convergence_gap20_100_mbps": float(r.get("convergence_gap20_100_mbps", 0.0)),
                    "episode_secrecy_mbits": float(r["episode_secrecy_mbits"]),
                    "avg_shaped_reward": float(r.get("avg_shaped_reward", 0.0)),
                    "relay_path_m": float(r.get("relay_path_m", 0.0)),
                    "jammer_path_m": float(r.get("jammer_path_m", 0.0)),
                }
            )
    return rows


def plot_ddpg_training_curves(csv_path: str, output_dir: str | None = None) -> dict:
    rows = load_training_log(csv_path)
    if not rows:
        raise ValueError("Training log is empty.")

    out_dir = Path(output_dir) if output_dir is not None else Path(csv_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plt = _safe_import_matplotlib()
    if plt is None:
        raise RuntimeError("matplotlib is not installed.")

    episodes = [r["episode"] for r in rows]
    avg_r_sec = [r["avg_R_sec_mbps"] for r in rows]
    roll20 = [r["rolling20_avg_R_sec_mbps"] for r in rows]
    roll100 = [r["rolling100_avg_R_sec_mbps"] for r in rows]
    noise = [r["noise_std"] for r in rows]
    ep_mbits = [r["episode_secrecy_mbits"] for r in rows]
    rewards = [r["avg_shaped_reward"] for r in rows]
    convergence_gap = [r["convergence_gap20_100_mbps"] for r in rows]
    relay_path = [r["relay_path_m"] for r in rows]
    jammer_path = [r["jammer_path_m"] for r in rows]

    curve_path = out_dir / "ddpg_curve_rsec_mbps.png"
    fig1 = plt.figure(figsize=(8, 4.5))
    ax1 = fig1.add_subplot(111)
    ax1.plot(episodes, avg_r_sec, label="Episode avg R_sec (Mbps)", alpha=0.5)
    ax1.plot(episodes, roll20, label="Rolling-20 avg R_sec (Mbps)", linewidth=1.5)
    ax1.plot(episodes, roll100, label="Rolling-100 avg R_sec (Mbps)", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Secrecy Rate (Mbps)")
    ax1.set_title("DDPG Training Curve")
    ax1.grid(alpha=0.25)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(curve_path, dpi=150)
    plt.close(fig1)

    noise_path = out_dir / "ddpg_curve_noise.png"
    fig2 = plt.figure(figsize=(8, 4.5))
    ax2 = fig2.add_subplot(111)
    ax2.plot(episodes, noise)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Action Noise Std")
    ax2.set_title("DDPG Exploration Schedule")
    ax2.grid(alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(noise_path, dpi=150)
    plt.close(fig2)

    mbits_path = out_dir / "ddpg_curve_episode_mbits.png"
    fig3 = plt.figure(figsize=(8, 4.5))
    ax3 = fig3.add_subplot(111)
    ax3.plot(episodes, ep_mbits, alpha=0.6)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Episode Secrecy Payload (Mbits)")
    ax3.set_title("DDPG Episode Throughput")
    ax3.grid(alpha=0.25)
    fig3.tight_layout()
    fig3.savefig(mbits_path, dpi=150)
    plt.close(fig3)

    reward_path = out_dir / "ddpg_curve_shaped_reward.png"
    fig4 = plt.figure(figsize=(8, 4.5))
    ax4 = fig4.add_subplot(111)
    ax4.plot(episodes, rewards, alpha=0.7)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Shaped Reward")
    ax4.set_title("DDPG Shaped Reward Over Episodes")
    ax4.grid(alpha=0.25)
    fig4.tight_layout()
    fig4.savefig(reward_path, dpi=150)
    plt.close(fig4)

    convergence_path = out_dir / "ddpg_curve_convergence_gap.png"
    fig5 = plt.figure(figsize=(8, 4.5))
    ax5 = fig5.add_subplot(111)
    ax5.plot(episodes, convergence_gap)
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("|Rolling-20 - Rolling-100| (Mbps)")
    ax5.set_title("DDPG Convergence Behavior")
    ax5.grid(alpha=0.25)
    fig5.tight_layout()
    fig5.savefig(convergence_path, dpi=150)
    plt.close(fig5)

    movement_path = out_dir / "ddpg_curve_episode_path_length.png"
    fig6 = plt.figure(figsize=(8, 4.5))
    ax6 = fig6.add_subplot(111)
    ax6.plot(episodes, relay_path, label="Relay path length")
    ax6.plot(episodes, jammer_path, label="Jammer path length")
    ax6.set_xlabel("Episode")
    ax6.set_ylabel("Path Length (m)")
    ax6.set_title("DDPG Episode Trajectory Behavior")
    ax6.legend()
    ax6.grid(alpha=0.25)
    fig6.tight_layout()
    fig6.savefig(movement_path, dpi=150)
    plt.close(fig6)

    return {
        "curve_rsec_mbps": str(curve_path.resolve()),
        "curve_noise": str(noise_path.resolve()),
        "curve_episode_mbits": str(mbits_path.resolve()),
        "curve_shaped_reward": str(reward_path.resolve()),
        "curve_convergence_gap": str(convergence_path.resolve()),
        "curve_episode_path_length": str(movement_path.resolve()),
        "final_rolling100_mbps": float(roll100[-1]),
        "best_episode_mbps": float(np.max(avg_r_sec)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot DDPG training curves from CSV log.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "outputs" / "training" / "ddpg" / "ddpg_training_log.csv"),
        help="Path to ddpg_training_log.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/plots",
        help="Directory to save plots (default: same as CSV directory)",
    )
    args = parser.parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training log not found: {csv_path.resolve()}")

    out = plot_ddpg_training_curves(str(csv_path), output_dir=args.output_dir)
    print("Saved DDPG plots:")
    for key in [
        "curve_rsec_mbps",
        "curve_noise",
        "curve_episode_mbits",
        "curve_shaped_reward",
        "curve_convergence_gap",
        "curve_episode_path_length",
    ]:
        print(f"  - {out[key]}")
    print(f"Final rolling-100 avg secrecy: {out['final_rolling100_mbps']:.4f} Mbps")
    print(f"Best episode secrecy rate   : {out['best_episode_mbps']:.4f} Mbps")
