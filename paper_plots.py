import csv
from pathlib import Path


def _safe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def plot_training_comparison(log_specs: list[dict], output_dir: str) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = _safe_import_matplotlib()
    if plt is None:
        return {}

    series = []
    for spec in log_specs:
        rows = _read_csv(spec["csv_path"])
        if not rows:
            continue
        label = f"{spec['algorithm'].upper()}-{spec['fading_model'].capitalize()}"
        series.append((label, rows))

    if not series:
        return {}

    reward_path = out_dir / "paper_secrecy_reward_vs_episodes.png"
    fig1 = plt.figure(figsize=(9, 4.8))
    ax1 = fig1.add_subplot(111)
    for label, rows in series:
        episodes = [int(r["episode"]) for r in rows]
        rewards = [_float(r, "avg_shaped_reward") for r in rows]
        ax1.plot(episodes, rewards, label=label)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Shaped Reward")
    ax1.set_title("Shaped Reward Over Episodes")
    ax1.grid(alpha=0.25)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(reward_path, dpi=150)
    plt.close(fig1)

    rate_path = out_dir / "paper_secrecy_rate_vs_episodes.png"
    fig2 = plt.figure(figsize=(9, 4.8))
    ax2 = fig2.add_subplot(111)
    for label, rows in series:
        episodes = [int(r["episode"]) for r in rows]
        values = [
            _float(r, "rolling20_avg_R_sec_mbps", _float(r, "rolling100_avg_R_sec_mbps"))
            for r in rows
        ]
        ax2.plot(episodes, values, label=label)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Rolling Avg Secrecy Rate (Mbps)")
    ax2.set_title("Secrecy Rate Convergence")
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(rate_path, dpi=150)
    plt.close(fig2)

    convergence_path = out_dir / "paper_convergence_gap_vs_episodes.png"
    fig3 = plt.figure(figsize=(9, 4.8))
    ax3 = fig3.add_subplot(111)
    for label, rows in series:
        episodes = [int(r["episode"]) for r in rows]
        values = [_float(r, "convergence_gap20_100_mbps") for r in rows]
        ax3.plot(episodes, values, label=label)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("|Rolling-20 - Rolling-100| (Mbps)")
    ax3.set_title("Convergence Behavior")
    ax3.grid(alpha=0.25)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(convergence_path, dpi=150)
    plt.close(fig3)

    movement_path = out_dir / "paper_trajectory_path_length_vs_episodes.png"
    fig4 = plt.figure(figsize=(9, 4.8))
    ax4 = fig4.add_subplot(111)
    for label, rows in series:
        episodes = [int(r["episode"]) for r in rows]
        values = [_float(r, "relay_path_m") + _float(r, "jammer_path_m") for r in rows]
        ax4.plot(episodes, values, label=label)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Relay + Jammer Path Length (m)")
    ax4.set_title("Trajectory Behavior During Training")
    ax4.grid(alpha=0.25)
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(movement_path, dpi=150)
    plt.close(fig4)

    return {
        "secrecy_reward_vs_episodes": str(reward_path.resolve()),
        "secrecy_rate_vs_episodes": str(rate_path.resolve()),
        "convergence_gap_vs_episodes": str(convergence_path.resolve()),
        "trajectory_path_length_vs_episodes": str(movement_path.resolve()),
    }


def plot_final_paper_comparisons(final_csv: str, output_dir: str) -> dict:
    rows = _read_csv(final_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = _safe_import_matplotlib()
    if plt is None or not rows:
        return {}

    methods = ["Random Walk", "Distance-Greedy", "DQN", "DDPG"]
    fading_models = ["rician", "rayleigh"]

    method_path = out_dir / "paper_dqn_ddpg_baselines_avg_rsec.png"
    fig1 = plt.figure(figsize=(9, 4.8))
    ax1 = fig1.add_subplot(111)
    x = list(range(len(methods)))
    width = 0.35
    for idx, fading_model in enumerate(fading_models):
        values = [
            _float(
                next(r for r in rows if r["method"] == method and r["fading_model"] == fading_model),
                "avg_rsec_mbps",
            )
            for method in methods
        ]
        ax1.bar([i + (idx - 0.5) * width for i in x], values, width=width, label=fading_model.capitalize())
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15)
    ax1.set_ylabel("Avg Secrecy Rate (Mbps)")
    ax1.set_title("DQN vs DDPG vs Baselines")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(method_path, dpi=150)
    plt.close(fig1)

    channel_path = out_dir / "paper_rician_vs_rayleigh_avg_rsec.png"
    fig2 = plt.figure(figsize=(9, 4.8))
    ax2 = fig2.add_subplot(111)
    for idx, method in enumerate(methods):
        values = [
            _float(
                next(r for r in rows if r["method"] == method and r["fading_model"] == fading_model),
                "avg_rsec_mbps",
            )
            for fading_model in fading_models
        ]
        ax2.plot([m.capitalize() for m in fading_models], values, marker="o", label=method)
    ax2.set_ylabel("Avg Secrecy Rate (Mbps)")
    ax2.set_title("Rician vs Rayleigh")
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(channel_path, dpi=150)
    plt.close(fig2)

    payload_path = out_dir / "paper_final_secrecy_payload.png"
    fig3 = plt.figure(figsize=(9, 4.8))
    ax3 = fig3.add_subplot(111)
    for idx, fading_model in enumerate(fading_models):
        values = [
            _float(
                next(r for r in rows if r["method"] == method and r["fading_model"] == fading_model),
                "episode_secrecy_mbits",
            )
            for method in methods
        ]
        ax3.bar([i + (idx - 0.5) * width for i in x], values, width=width, label=fading_model.capitalize())
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=15)
    ax3.set_ylabel("Episode Secrecy Payload (Mbits)")
    ax3.set_title("Final Secrecy Payload Comparison")
    ax3.grid(axis="y", alpha=0.25)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(payload_path, dpi=150)
    plt.close(fig3)

    return {
        "dqn_ddpg_baselines_avg_rsec": str(method_path.resolve()),
        "rician_vs_rayleigh_avg_rsec": str(channel_path.resolve()),
        "final_secrecy_payload": str(payload_path.resolve()),
    }
