import csv
import numpy as np
from pathlib import Path
from core.environment import EnvConfig, UAVEnvironment
from core.config_utils import build_env_config

HPPP_CONFIG = {
    "use_multiple_eves": True,
    "eve_density_lambda": 2e-5,
}

ALGORITHM_OUTPUT_DIRS = {
    "sca": "SCA",
    "bcd": "BCD",
}


def resolve_hppp_algorithm_dir(algo: str) -> str:
    return ALGORITHM_OUTPUT_DIRS.get(algo.lower(), algo)


def build_hppp_env_config(seed: int = 42, **overrides) -> EnvConfig:
    cfg = build_env_config(seed=seed, **overrides)
    cfg.use_multiple_eves = HPPP_CONFIG["use_multiple_eves"]
    cfg.eve_density_lambda = HPPP_CONFIG["eve_density_lambda"]
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def plot_training_curves(csv_path: str, output_dir: str) -> dict:
    plt = _safe_import_matplotlib()
    if plt is None:
        return {}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(csv_path)
    if not rows:
        return {}

    episodes = np.array([int(r["episode"]) for r in rows])
    reward = np.array([_float(r, "avg_shaped_reward") for r in rows])
    r_sec = np.array([_float(r, "avg_R_sec_mbps") for r in rows])

    paths = {}

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(episodes, reward)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Shaped Reward")
    ax.set_title("Reward Curve")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = str(out / "reward_curve.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["reward_curve"] = p

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(episodes, r_sec)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Secrecy Rate (Mbps)")
    ax.set_title("Secrecy Rate Curve")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = str(out / "secrecy_curve.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["secrecy_curve"] = p

    window = 100
    if len(reward) >= window:
        roll_reward = np.convolve(reward, np.ones(window) / window, mode="valid")
        roll_secrecy = np.convolve(r_sec, np.ones(window) / window, mode="valid")
        roll_ep = episodes[window - 1:]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(roll_ep, roll_reward)
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Rolling-{window} Reward")
        ax.set_title(f"Rolling-{window} Reward Curve")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        p = str(out / "rolling_reward_100.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths["rolling_reward_100"] = p

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(roll_ep, roll_secrecy)
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Rolling-{window} Secrecy (Mbps)")
        ax.set_title(f"Rolling-{window} Secrecy Rate")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        p = str(out / "rolling_secrecy_100.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths["rolling_secrecy_100"] = p

        stable_ep = estimate_convergence_episode(r_sec, window=window)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(episodes, r_sec, color="#7f7f7f", alpha=0.35, linewidth=0.8, label="Episode secrecy")
        ax.plot(roll_ep, roll_secrecy, color="#1f77b4", linewidth=2.0, label=f"Rolling-{window}")
        ax.axvline(stable_ep, color="#d62728", linestyle="--", linewidth=1.2,
                   label=f"Approx. stable episode {stable_ep}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Secrecy Rate (Mbps)")
        ax.set_title("Secrecy Convergence")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = str(out / "convergence_secrecy.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths["convergence_secrecy"] = p

    elif len(r_sec) >= 2:
        conv_window = min(100, len(r_sec))
        roll_secrecy = np.convolve(r_sec, np.ones(conv_window) / conv_window, mode="valid")
        roll_ep = episodes[conv_window - 1:]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(episodes, r_sec, color="#7f7f7f", alpha=0.35, linewidth=0.8, label="Episode secrecy")
        ax.plot(roll_ep, roll_secrecy, color="#1f77b4", linewidth=2.0, label=f"Rolling-{conv_window}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Secrecy Rate (Mbps)")
        ax.set_title("Secrecy Convergence")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = str(out / "convergence_secrecy.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths["convergence_secrecy"] = p

    return paths


def run_evaluation(
    env_cfg: EnvConfig,
    actor_fn,
    device,
    episodes: int = 20,
    seed_offset: int = 9000,
) -> dict:
    env = UAVEnvironment(env_cfg)
    rewards = []
    rsecs = []
    rlegits = []
    reverses = []
    num_eves_list = []
    nearest_dists = []
    mean_dists = []
    max_caps = []

    for i in range(episodes):
        cfg_i = build_env_config(seed=seed_offset + i, use_multiple_eves=True,
                                  eve_density_lambda=HPPP_CONFIG["eve_density_lambda"])
        env = UAVEnvironment(cfg_i)
        state = env.reset().astype(np.float32)
        done = False
        ep_rew = 0.0
        ep_rsec = 0.0
        ep_rlegit = 0.0
        ep_reve = 0.0
        ep_ne = 0
        ep_nd = 0.0
        ep_md = 0.0
        ep_mc = 0.0
        steps = 0

        while not done:
            import torch
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = actor_fn(s_t).cpu().numpy()[0]
            from rl.advanced_rl_train import split_action as sa
            a_relay, a_jammer, jammer_power, role_switch = sa(action)
            next_state, reward, done, info = env.step(a_relay, a_jammer, jammer_power, role_switch)
            state = next_state.astype(np.float32)
            ep_rew += reward
            ep_rsec += info["R_sec"]
            ep_rlegit += info["R_legit"]
            ep_reve += info["R_eve"]
            ep_ne += info.get("num_eves", 1)
            ep_nd += info.get("nearest_eve_distance", 0.0)
            ep_md += info.get("mean_eve_distance", 0.0)
            ep_mc += info.get("max_eve_capacity", 0.0)
            steps += 1

        rewards.append(ep_rew / max(steps, 1))
        rsecs.append((ep_rsec / max(steps, 1)) / 1e6)
        rlegits.append((ep_rlegit / max(steps, 1)) / 1e6)
        reverses.append((ep_reve / max(steps, 1)) / 1e6)
        num_eves_list.append(ep_ne / max(steps, 1))
        nearest_dists.append(ep_nd / max(steps, 1))
        mean_dists.append(ep_md / max(steps, 1))
        max_caps.append((ep_mc / max(steps, 1)) / 1e6)

    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_R_sec_mbps": float(np.mean(rsecs)),
        "avg_R_legit_mbps": float(np.mean(rlegits)),
        "avg_R_eve_mbps": float(np.mean(reverses)),
        "avg_num_eves": float(np.mean(num_eves_list)),
        "avg_nearest_eve_distance": float(np.mean(nearest_dists)),
        "avg_mean_eve_distance": float(np.mean(mean_dists)),
        "avg_max_eve_capacity_mbps": float(np.mean(max_caps)),
    }


def generate_evaluation_summary(results: dict, output_path: str):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["Algorithm", "avg_reward", "avg_R_sec_mbps", "avg_R_legit_mbps",
            "avg_R_eve_mbps", "avg_num_eves", "avg_nearest_eve_distance",
            "avg_mean_eve_distance", "avg_max_eve_capacity_mbps"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for algo, data in sorted(results.items()):
            row = {"Algorithm": algo}
            for k in keys[1:]:
                row[k] = data.get(k, "")
            w.writerow(row)


def generate_convergence_summary(results: dict, output_path: str):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Convergence Study Summary\n",
        "| Algorithm | Approx Stable Episode | Final Rolling Reward | Final Rolling Secrecy (Mbps) |\n",
        "|-----------|----------------------|---------------------|------------------------------|\n",
    ]

    for algo, data in sorted(results.items()):
        stable = data.get("stable_episode", "N/A")
        final_rew = data.get("final_rolling_reward", "N/A")
        final_sec = data.get("final_rolling_secrecy", "N/A")
        if isinstance(final_rew, float):
            final_rew = f"{final_rew:.4f}"
        if isinstance(final_sec, float):
            final_sec = f"{final_sec:.4f}"
        lines.append(f"| {algo} | {stable} | {final_rew} | {final_sec} |\n")

    with out.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def estimate_convergence_episode(r_sec_series: np.ndarray, window: int = 100) -> int:
    if len(r_sec_series) < window * 2:
        return len(r_sec_series)
    roll = np.convolve(r_sec_series, np.ones(window) / window, mode="valid")
    variances = []
    for i in range(len(roll) - window):
        variances.append(float(np.var(roll[i:i + window])))
    variances = np.array(variances)
    threshold = np.median(variances[-window:]) * 1.5
    stable_idx = int(np.argmax(variances < threshold)) if np.any(variances < threshold) else len(roll) // 2
    return stable_idx + window


def _safe_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _read_csv(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key)
    if v in (None, ""):
        return default
    return float(v)
