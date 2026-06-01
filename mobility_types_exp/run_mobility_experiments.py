import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.environment as env_module

from mobility_types_exp.mobility_models import create_mobility_model
from mobility_types_exp.configs import MobilityExperimentConfig, OUTPUT_ROOT


_REQUIRED_COLS = {
    "episode": "episode",
    "reward": ["avg_shaped_reward"],
    "secrecy": ["avg_R_sec_mbps"],
    "rolling100": ["rolling100_avg_R_sec_mbps", "rolling100"],
    "algo": ["algorithm"],
    "fading": ["fading_model"],
}


class MobilityEnv(env_module.UAVEnvironment):
    _mm_class = None
    _mm_kwargs = {}

    def __init__(self, config):
        super().__init__(config)
        if MobilityEnv._mm_class is not None:
            self._mm = MobilityEnv._mm_class(**MobilityEnv._mm_kwargs)
        else:
            self._mm = None

    def _update_user_position(self):
        if not self.config.user_mobile:
            return
        mm = self._mm
        if mm is None:
            return super()._update_user_position()
        new_pos, new_vel = mm.step(
            self.user_position[:2].copy(),
            self.user_velocity.copy(),
            self.config.dt,
            self.half_area,
            self.config.user_max_speed,
        )
        self.user_position[:2] = new_pos
        self.user_velocity = new_vel

    def reset(self):
        state = super().reset()
        if self._mm is not None:
            self._mm.reset()
        return state


env_module.UAVEnvironment = MobilityEnv

from rl.dqn_train import train_dqn, DQNConfig
from rl.ddpg_train import train_ddpg, DDPGConfig
from d3qn_study.train_d3qn import train_d3qn, D3QNConfig
from PPO_study.train_ppo import train_ppo, PPOConfig
from sac_study.sac_train import train_sac
from sac_study.configs import SACStudyConfig
from td3pg_study.td3pg_train import train_td3pg
from td3pg_study.configs import TD3PGStudyConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def _read_csv(csv_path):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _get_col(row, candidates):
    for c in candidates:
        val = row.get(c)
        if val is not None and val != "":
            return float(val)
    return 0.0


def generate_plots(csv_path, output_dir, title_prefix=""):
    rows = _read_csv(csv_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    episodes = [_get_col(r, ["episode"]) for r in rows]
    rewards = [_get_col(r, _REQUIRED_COLS["reward"]) for r in rows]
    secrecy = [_get_col(r, _REQUIRED_COLS["secrecy"]) for r in rows]
    roll100 = [_get_col(r, _REQUIRED_COLS["rolling100"]) for r in rows]

    for name, ydata, ylabel, color in [
        ("reward_curve", rewards, "Avg Shaped Reward", "#1f77b4"),
        ("secrecy_curve", secrecy, "Avg Secrecy Rate (Mbps)", "#2ca02c"),
        ("rolling100_curve", roll100, "Rolling100 Secrecy (Mbps)", "#d62728"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(episodes, ydata, color=color, linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix} - {name.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3)
        fig.savefig(str(out / f"{name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    paths = [str(out / f"{name}.png") for name in ["reward_curve", "secrecy_curve", "rolling100_curve"]]
    return paths


def build_dqn_config(channel, episodes, seed, device, hidden_dim, lr):
    from convergence_study.configs import build_dqn_convergence_config
    cfg = build_dqn_convergence_config(
        fading_model=channel, episodes=episodes,
        hidden_dim=hidden_dim, seed=seed, device=device,
    )
    cfg.lr = lr
    cfg.user_mobile = True
    return cfg


def build_ddpg_config(channel, episodes, seed, device, hidden_dim, lr):
    from convergence_study.configs import build_ddpg_convergence_config
    cfg = build_ddpg_convergence_config(
        fading_model=channel, episodes=episodes,
        hidden_dim=hidden_dim, seed=seed, device=device,
    )
    cfg.actor_lr = lr
    cfg.critic_lr = lr
    cfg.user_mobile = True
    return cfg


def _summarize_log(csv_path):
    rows = _read_csv(csv_path)
    if not rows:
        return {"Final_Rolling100_Secrecy": 0, "Best_Secrecy": 0, "Average_Reward": 0, "Convergence_Episode": 0}
    last = rows[-1]
    final_roll100 = _get_col(last, _REQUIRED_COLS["rolling100"])
    best_sec = max(_get_col(r, _REQUIRED_COLS["secrecy"]) for r in rows)
    avg_rew = sum(_get_col(r, _REQUIRED_COLS["reward"]) for r in rows) / len(rows)

    # Convergence episode: first episode where rolling100 is within 95% of final
    target = 0.95 * final_roll100 if final_roll100 > 0 else 0
    conv_ep = len(rows)
    for r in rows:
        if _get_col(r, _REQUIRED_COLS["rolling100"]) >= target:
            conv_ep = int(_get_col(r, ["episode"]))
            break

    return {
        "Final_Rolling100_Secrecy": round(final_roll100, 4),
        "Best_Secrecy": round(best_sec, 4),
        "Average_Reward": round(avg_rew, 4),
        "Convergence_Episode": conv_ep,
    }


def run_experiment(
    mobility_name="random_walk",
    algorithm="dqn",
    channel="rician",
    episodes=100,
    seed=42,
    device="cpu",
    output_root=OUTPUT_ROOT,
):
    cfg = MobilityExperimentConfig(
        mobility_model=mobility_name,
        algorithm=algorithm,
        channel=channel,
        episodes=episodes,
        seed=seed,
        device=device,
        output_root=output_root,
    )
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mm_class, mm_kwargs = create_mobility_model(mobility_name, return_class=True)
    MobilityEnv._mm_class = mm_class
    MobilityEnv._mm_kwargs = mm_kwargs
    set_seed(seed)

    algo = algorithm.lower()
    print(f"\n{'='*60}")
    print(f"  Mobility: {mobility_name} | Algorithm: {algo} | Channel: {channel}")
    print(f"  Episodes: {episodes} | Output: {out_dir}")
    print(f"{'='*60}")

    training_params = {
        "dqn": {"hidden_dim": 32, "lr": 5e-4},
        "ddpg": {"hidden_dim": 32, "lr": 5e-4},
        "d3qn": {"hidden_dim": 64, "lr": 8e-4},
        "ppo": {"hidden_dim": 64, "lr": 3e-4},
        "sac": {"hidden_dim": 64, "lr": 3e-4},
        "td3pg": {"hidden_dim": 64, "lr": 1e-3},
    }
    params = training_params.get(algo, {})
    hd = params.get("hidden_dim", 64)
    lr = params.get("lr", 3e-4)

    result = {}

    if algo == "dqn":
        train_cfg = build_dqn_config(channel, episodes, seed, device, hd, lr)
        summary = train_dqn(train_cfg, output_dir=str(out_dir))
        log_csv = summary.get("training_log_csv", str(out_dir / "dqn_training_log.csv"))
    elif algo == "ddpg":
        train_cfg = build_ddpg_config(channel, episodes, seed, device, hd, lr)
        summary = train_ddpg(train_cfg, output_dir=str(out_dir))
        log_csv = summary.get("training_log_csv", str(out_dir / "ddpg_training_log.csv"))
    elif algo == "d3qn":
        train_cfg = D3QNConfig(
            episodes=episodes, fading_model=channel, hidden_dim=hd, lr=lr,
            seed=seed, device=device,
        )
        summary = train_d3qn(train_cfg, output_dir=str(out_dir))
        log_csv = summary.get("training_log_csv", str(out_dir / "training_log.csv"))
    elif algo == "ppo":
        train_cfg = PPOConfig(
            episodes=episodes, fading_model=channel, hidden_dim=hd,
            actor_lr=lr, critic_lr=lr * 2,
            seed=seed, device=device,
        )
        summary = train_ppo(train_cfg, output_dir=str(out_dir))
        log_csv = summary.get("training_log_csv", str(out_dir / "training_log.csv"))
    elif algo == "sac":
        train_cfg = SACStudyConfig(
            episodes=episodes, fading_model=channel, hidden_dim=hd,
            actor_lr=lr, critic_lr=lr,
            seed=seed, device=device, output_root=str(out_dir.parent),
        )
        summary = train_sac(train_cfg, output_dir=str(out_dir))
        log_csv = summary.get("training_log_csv", str(out_dir / "training_log.csv"))
    elif algo == "td3pg":
        train_cfg = TD3PGStudyConfig(
            episodes=episodes, fading_model=channel, hidden_dim=hd,
            actor_lr=lr, critic_lr=lr,
            seed=seed, device=device, output_root=str(out_dir.parent),
        )
        summary = train_td3pg(train_cfg, output_dir=str(out_dir))
        log_csv = summary.get("training_log_csv", str(out_dir / "training_log.csv"))
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    log_path = Path(log_csv)
    if log_path.is_file():
        title = f"{algo.upper()} + {channel.title()} + {mobility_name}"
        generate_plots(str(log_path), str(out_dir), title_prefix=title)
        stats = _summarize_log(str(log_path))
    else:
        stats = _summarize_log(str(log_csv))

    print(f"  Log: {log_csv}")
    print(f"  Final Rolling100: {stats['Final_Rolling100_Secrecy']} Mbps")
    print(f"  Best Secrecy:     {stats['Best_Secrecy']} Mbps")
    print(f"  Convergence Ep:   {stats['Convergence_Episode']}")

    result = {
        "mobility_model": mobility_name,
        "algorithm": algo,
        "channel": channel,
        **stats,
        "log_path": log_csv,
        "output_dir": str(out_dir),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Run mobility model experiments")
    parser.add_argument("--mobility", type=str, default="random_walk",
                        choices=["random_walk", "random_waypoint", "gauss_markov", "constant_velocity"])
    parser.add_argument("--algo", type=str, default="dqn",
                        choices=["dqn", "ddpg", "d3qn", "ppo", "sac", "td3pg"])
    parser.add_argument("--channel", type=str, default="rician", choices=["rician", "rayleigh"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    result = run_experiment(
        mobility_name=args.mobility,
        algorithm=args.algo,
        channel=args.channel,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
    )
    print(f"\nDone: {result['mobility_model']}/{result['algorithm']}_{result['channel']}")


if __name__ == "__main__":
    main()
