import argparse
from pathlib import Path

import numpy as np
import torch

from analysis.baselines import distance_greedy_policy, random_policy
from core.config_utils import build_env_config
from rl.ddpg_train import Actor, split_action
from rl.dqn_train import QNetwork, make_action_table
from core.environment import EnvConfig, UAVEnvironment


def _safe_import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _infer_actor_action_dim(model_path: str) -> int:
    checkpoint = torch.load(model_path, map_location="cpu")
    output_key = "net.4.weight"
    return int(checkpoint[output_key].shape[0]) if output_key in checkpoint else 5


def rollout_policy(env: UAVEnvironment, policy_name: str, model_path: str | None = None) -> dict:
    state = env.reset().astype(np.float32)
    trace = {
        "relay": [env.relay_position.copy()],
        "jammer": [env.jammer_position.copy()],
        "user": [env.user_position.copy()],
        "eve": env.eve_position.copy(),
        "bs": env.bs_position.copy(),
        "total_r_sec": 0.0,
    }

    if policy_name == "dqn":
        action_table = make_action_table()
        state_dim = state.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q_net = QNetwork(state_dim, len(action_table), hidden_dim=128).to(device)
        q_net.load_state_dict(torch.load(model_path, map_location=device))
        q_net.eval()
    elif policy_name == "ddpg":
        state_dim = state.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor = Actor(state_dim, _infer_actor_action_dim(model_path), hidden_dim=128).to(device)
        actor.load_state_dict(torch.load(model_path, map_location=device))
        actor.eval()

    done = False
    while not done:
        if policy_name == "random":
            a_relay, a_jammer, jammer_power = random_policy(env)
        elif policy_name == "greedy":
            a_relay, a_jammer, jammer_power = distance_greedy_policy(env)
        elif policy_name == "dqn":
            with torch.no_grad():
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action_id = int(torch.argmax(q_net(s_t), dim=1).item())
            a_relay, a_jammer, jammer_power = action_table[action_id]
        elif policy_name == "ddpg":
            with torch.no_grad():
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = actor(s_t).cpu().numpy()[0]
            a_relay, a_jammer, jammer_power = split_action(action)
            role_switch = bool(action.shape[0] > 5 and action[5] > 0.5)
        else:
            raise ValueError(f"Unsupported policy: {policy_name}")

        next_state, _, done, info = env.step(
            a_relay,
            a_jammer,
            jammer_power,
            role_switch if policy_name == "ddpg" else False,
        )
        state = next_state.astype(np.float32)
        trace["relay"].append(env.relay_position.copy())
        trace["jammer"].append(env.jammer_position.copy())
        trace["total_r_sec"] += info["R_sec"]
        trace["user"].append(env.user_position.copy())

    trace["relay"] = np.asarray(trace["relay"])
    trace["jammer"] = np.asarray(trace["jammer"])
    trace["user"] = np.asarray(trace["user"])
    return trace


def plot_trajectory(trace: dict, policy_name: str, fading_model: str, output_path: str) -> str:
    plt = _safe_import_matplotlib()
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    relay = trace["relay"]
    jammer = trace["jammer"]
    user = trace["user"]  # shape (T, 3) if mobile, (1, 3) if static
    eve = trace["eve"]
    bs = trace["bs"]

    ax.plot(relay[:, 0], relay[:, 1], label="Relay UAV", color="#1f77b4", linewidth=2)
    ax.plot(jammer[:, 0], jammer[:, 1], label="Jammer UAV", color="#d62728", linewidth=2)
    ax.plot(user[:, 0], user[:, 1], label="User path", color="#2ca02c", linewidth=1.5, alpha=0.7)
    ax.scatter(user[0, 0], user[0, 1], color="#2ca02c", s=60, marker="o", zorder=5)
    ax.scatter(eve[0], eve[1], label="Eavesdropper", color="#ff7f0e", s=80, marker="X")
    ax.scatter(bs[0], bs[1], label="Base Station", color="#111111", s=90, marker="s")
    ax.scatter(relay[0, 0], relay[0, 1], color="#1f77b4", s=40, marker="^")
    ax.scatter(jammer[0, 0], jammer[0, 1], color="#d62728", s=40, marker="^")

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title(f"{policy_name.upper()} Trajectory | {fading_model.capitalize()} fading")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(Path(output_path).resolve())


def generate_trajectory_suite(
    dqn_model: str,
    ddpg_actor: str,
    fading_model: str = "rician",
    seed: int = 42,
    output_dir: str = "outputs/trajectories",
    control_mode: str = "velocity",
    role_switching: bool = False,
    user_mobile: bool = False,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        ("random", None),
        ("greedy", None),
        ("dqn", dqn_model),
        ("ddpg", ddpg_actor),
    ]

    outputs = {}
    for method, model_path in methods:
        env = UAVEnvironment(
            build_env_config(
                seed=seed,
                fading_model=fading_model,
                control_mode=control_mode,
                role_switching=role_switching,
                user_mobile=user_mobile,
            )
        )
        trace = rollout_policy(env, method, model_path=model_path)
        out_path = out_dir / f"{method}_{fading_model}_trajectory.png"
        outputs[method] = plot_trajectory(trace, method, fading_model, str(out_path))
    return outputs


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate UAV trajectory plots.")
    parser.add_argument("--dqn-model", type=str, required=True, help="Path to DQN model")
    parser.add_argument("--ddpg-actor", type=str, required=True, help="Path to DDPG actor")
    parser.add_argument(
        "--channel-model",
        type=str,
        default="rician",
        choices=["rician", "rayleigh"],
        help="Fading model for rollout",
    )
    parser.add_argument("--seed", type=int, default=42, help="Episode seed")
    parser.add_argument("--output-dir", type=str, default="outputs/trajectories", help="Output directory")
    parser.add_argument("--control-mode", type=str, default="velocity", choices=["velocity", "waypoint"])
    parser.add_argument("--role-switching", action="store_true")
    parser.add_argument("--user-mobile", action="store_true", help="Enable mobile user during rollout")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    outputs = generate_trajectory_suite(
        dqn_model=args.dqn_model,
        ddpg_actor=args.ddpg_actor,
        fading_model=args.channel_model,
        seed=args.seed,
        output_dir=args.output_dir,
        control_mode=args.control_mode,
        role_switching=args.role_switching,
        user_mobile=args.user_mobile,
    )
    print("Saved trajectory plots:")
    for method, path in outputs.items():
        print(f"  {method}: {path}")
