import csv
import argparse
from pathlib import Path

import numpy as np

_KNOWN_ALGOS = {"dqn"}


def _detect_algorithm(csv_path: str, algorithm: str | None = None) -> tuple[str, str]:
    if algorithm is not None:
        raw = algorithm.lower()
    else:
        path = Path(csv_path)
        raw = path.parent.name.lower()
        if raw not in _KNOWN_ALGOS and not raw.startswith("marl_"):
            stem = path.stem
            raw = next((p for p in _KNOWN_ALGOS | {"marl_shared", "marl_split"} if stem.startswith(p)), "dqn")
    if raw in _KNOWN_ALGOS:
        display = raw.upper()
    elif raw == "marl_shared":
        display = "MARL Shared"
    elif raw == "marl_split":
        display = "MARL Split"
    else:
        display = raw.replace("_", " ").title()
    return raw, display


def _schema_is_marl(first_row_keys: set) -> bool:
    return "relay_epsilon" in first_row_keys


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
                    "epsilon": float(r["epsilon"]) if "epsilon" in r else None,
                    "relay_epsilon": float(r["relay_epsilon"]) if "relay_epsilon" in r else None,
                    "jammer_epsilon": float(r["jammer_epsilon"]) if "jammer_epsilon" in r else None,
                    "avg_R_sec_mbps": float(r["avg_R_sec_mbps"]),
                    "rolling100_avg_R_sec_mbps": float(r["rolling100_avg_R_sec_mbps"]),
                    "rolling20_avg_R_sec_mbps": float(
                        r.get("rolling20_avg_R_sec_mbps", r["rolling100_avg_R_sec_mbps"])
                    ),
                    "convergence_gap20_100_mbps": float(r.get("convergence_gap20_100_mbps", 0.0)),
                    "episode_secrecy_mbits": float(r["episode_secrecy_mbits"]),
                    "avg_shaped_reward": float(r.get("avg_shaped_reward", 0.0)),
                    "relay_path_m": float(r.get("relay_path_m", 0.0)),
                    "jammer_path_m": float(r.get("jammer_path_m", 0.0)),
                    "relay_loss": float(r["relay_loss"]) if "relay_loss" in r else None,
                    "jammer_loss": float(r["jammer_loss"]) if "jammer_loss" in r else None,
                    "relay_q_entropy": float(r["relay_q_entropy"]) if "relay_q_entropy" in r else None,
                    "jammer_q_entropy": float(r["jammer_q_entropy"]) if "jammer_q_entropy" in r else None,
                }
            )
    return rows


def _maybe_plot(plt, has_data, episodes, values, path, xlabel, ylabel, title):
    if not has_data:
        return None
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(episodes, values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.resolve())


def plot_dqn_training_curves(
    csv_path: str,
    output_dir: str | None = None,
    algorithm: str | None = None,
) -> dict:
    rows = load_training_log(csv_path)
    if not rows:
        raise ValueError("Training log is empty.")

    algo, algo_display = _detect_algorithm(csv_path, algorithm)
    is_marl = _schema_is_marl(set(rows[0].keys()))

    out_dir = Path(output_dir) if output_dir is not None else Path(csv_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plt = _safe_import_matplotlib()
    if plt is None:
        raise RuntimeError("matplotlib is not installed.")

    episodes = [r["episode"] for r in rows]
    avg_r_sec = [r["avg_R_sec_mbps"] for r in rows]
    roll20 = [r["rolling20_avg_R_sec_mbps"] for r in rows]
    roll100 = [r["rolling100_avg_R_sec_mbps"] for r in rows]
    ep_mbits = [r["episode_secrecy_mbits"] for r in rows]
    rewards = [r["avg_shaped_reward"] for r in rows]
    convergence_gap = [r["convergence_gap20_100_mbps"] for r in rows]
    relay_path = [r["relay_path_m"] for r in rows]
    jammer_path = [r["jammer_path_m"] for r in rows]
    has_movement = any(v > 0 for v in relay_path) or any(v > 0 for v in jammer_path)

    eps = [r["epsilon"] for r in rows]
    has_eps = any(v is not None for v in eps)

    relay_eps = [r["relay_epsilon"] for r in rows]
    jammer_eps = [r["jammer_epsilon"] for r in rows]
    has_relay_eps = any(v is not None for v in relay_eps)
    has_jammer_eps = any(v is not None for v in jammer_eps)

    relay_loss = [r["relay_loss"] for r in rows]
    jammer_loss = [r["jammer_loss"] for r in rows]
    has_relay_loss = any(v is not None for v in relay_loss)
    has_jammer_loss = any(v is not None for v in jammer_loss)

    relay_q_entropy = [r["relay_q_entropy"] for r in rows]
    jammer_q_entropy = [r["jammer_q_entropy"] for r in rows]
    has_relay_q_entropy = any(v is not None for v in relay_q_entropy)
    has_jammer_q_entropy = any(v is not None for v in jammer_q_entropy)

    result = {}

    # --- Common plots ---

    curve_path = out_dir / f"{algo}_curve_rsec_mbps.png"
    fig1 = plt.figure(figsize=(8, 4.5))
    ax1 = fig1.add_subplot(111)
    ax1.plot(episodes, avg_r_sec, label="Episode avg R_sec (Mbps)", alpha=0.5)
    ax1.plot(episodes, roll20, label="Rolling-20 avg R_sec (Mbps)", linewidth=1.5)
    ax1.plot(episodes, roll100, label="Rolling-100 avg R_sec (Mbps)", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Secrecy Rate (Mbps)")
    ax1.set_title(f"{algo_display} Training Curve")
    ax1.grid(alpha=0.25)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(curve_path, dpi=150)
    plt.close(fig1)
    result["curve_rsec_mbps"] = str(curve_path.resolve())

    mbits_path = out_dir / f"{algo}_curve_episode_mbits.png"
    fig3 = plt.figure(figsize=(8, 4.5))
    ax3 = fig3.add_subplot(111)
    ax3.plot(episodes, ep_mbits, alpha=0.6)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Episode Secrecy Payload (Mbits)")
    ax3.set_title(f"{algo_display} Episode Throughput")
    ax3.grid(alpha=0.25)
    fig3.tight_layout()
    fig3.savefig(mbits_path, dpi=150)
    plt.close(fig3)
    result["curve_episode_mbits"] = str(mbits_path.resolve())

    reward_path = out_dir / f"{algo}_curve_shaped_reward.png"
    fig4 = plt.figure(figsize=(8, 4.5))
    ax4 = fig4.add_subplot(111)
    ax4.plot(episodes, rewards, alpha=0.7)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Shaped Reward")
    ax4.set_title(f"{algo_display} Shaped Reward Over Episodes")
    ax4.grid(alpha=0.25)
    fig4.tight_layout()
    fig4.savefig(reward_path, dpi=150)
    plt.close(fig4)
    result["curve_shaped_reward"] = str(reward_path.resolve())

    convergence_path = out_dir / f"{algo}_curve_convergence_gap.png"
    fig5 = plt.figure(figsize=(8, 4.5))
    ax5 = fig5.add_subplot(111)
    ax5.plot(episodes, convergence_gap)
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("|Rolling-20 - Rolling-100| (Mbps)")
    ax5.set_title(f"{algo_display} Convergence Behavior")
    ax5.grid(alpha=0.25)
    fig5.tight_layout()
    fig5.savefig(convergence_path, dpi=150)
    plt.close(fig5)
    result["curve_convergence_gap"] = str(convergence_path.resolve())

    if has_movement:
        movement_path = out_dir / f"{algo}_curve_episode_path_length.png"
        fig6 = plt.figure(figsize=(8, 4.5))
        ax6 = fig6.add_subplot(111)
        ax6.plot(episodes, relay_path, label="Relay path length")
        ax6.plot(episodes, jammer_path, label="Jammer path length")
        ax6.set_xlabel("Episode")
        ax6.set_ylabel("Path Length (m)")
        ax6.set_title(f"{algo_display} Episode Trajectory Behavior")
        ax6.legend()
        ax6.grid(alpha=0.25)
        fig6.tight_layout()
        fig6.savefig(movement_path, dpi=150)
        plt.close(fig6)
        result["curve_episode_path_length"] = str(movement_path.resolve())

    # --- Single-agent epsilon plot ---
    if has_eps:
        eps_path = out_dir / f"{algo}_curve_epsilon.png"
        p = _maybe_plot(plt, True, episodes, eps, eps_path,
                        "Episode", "Epsilon", f"{algo_display} Exploration Schedule")
        if p:
            result["curve_epsilon"] = p

    # --- MARL-specific plots ---
    if is_marl:
        if has_relay_eps:
            relay_eps_path = out_dir / f"{algo}_curve_relay_epsilon.png"
            p = _maybe_plot(plt, True, episodes, relay_eps, relay_eps_path,
                            "Episode", "Relay Epsilon", f"{algo_display} Relay Exploration Schedule")
            if p:
                result["curve_relay_epsilon"] = p

        if has_jammer_eps:
            jammer_eps_path = out_dir / f"{algo}_curve_jammer_epsilon.png"
            p = _maybe_plot(plt, True, episodes, jammer_eps, jammer_eps_path,
                            "Episode", "Jammer Epsilon", f"{algo_display} Jammer Exploration Schedule")
            if p:
                result["curve_jammer_epsilon"] = p

        if has_relay_loss:
            relay_loss_path = out_dir / f"{algo}_curve_relay_loss.png"
            p = _maybe_plot(plt, True, episodes, relay_loss, relay_loss_path,
                            "Episode", "Relay Loss", f"{algo_display} Relay Loss")
            if p:
                result["curve_relay_loss"] = p

        if has_jammer_loss:
            jammer_loss_path = out_dir / f"{algo}_curve_jammer_loss.png"
            p = _maybe_plot(plt, True, episodes, jammer_loss, jammer_loss_path,
                            "Episode", "Jammer Loss", f"{algo_display} Jammer Loss")
            if p:
                result["curve_jammer_loss"] = p

        if has_relay_q_entropy:
            relay_ent_path = out_dir / f"{algo}_curve_relay_q_entropy.png"
            p = _maybe_plot(plt, True, episodes, relay_q_entropy, relay_ent_path,
                            "Episode", "Relay Q-Entropy", f"{algo_display} Relay Q-Entropy")
            if p:
                result["curve_relay_q_entropy"] = p

        if has_jammer_q_entropy:
            jammer_ent_path = out_dir / f"{algo}_curve_jammer_q_entropy.png"
            p = _maybe_plot(plt, True, episodes, jammer_q_entropy, jammer_ent_path,
                            "Episode", "Jammer Q-Entropy", f"{algo_display} Jammer Q-Entropy")
            if p:
                result["curve_jammer_q_entropy"] = p

    result["final_rolling100_mbps"] = float(roll100[-1])
    result["best_episode_mbps"] = float(np.max(avg_r_sec))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot DQN / MARL training curves from CSV log.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "outputs" / "training" / "dqn" / "dqn_training_log.csv"),
        help="Path to training_log.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as CSV directory)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["dqn", "marl_shared", "marl_split"],
        help="Override auto-detected algorithm name",
    )
    args = parser.parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training log not found: {csv_path.resolve()}")

    out = plot_dqn_training_curves(
        str(csv_path),
        output_dir=args.output_dir,
        algorithm=args.algorithm,
    )
    _, algo_display = _detect_algorithm(str(csv_path), args.algorithm)
    print(f"Saved {algo_display} plots:")
    for key, val in out.items():
        if key in ("final_rolling100_mbps", "best_episode_mbps"):
            continue
        print(f"  - {val}")
    print(f"Final rolling-100 avg secrecy: {out['final_rolling100_mbps']:.4f} Mbps")
    print(f"Best episode secrecy rate   : {out['best_episode_mbps']:.4f} Mbps")
