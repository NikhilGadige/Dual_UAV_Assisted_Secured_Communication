import argparse
from pathlib import Path

import numpy as np
import torch

from analysis.baselines import distance_greedy_policy, random_policy
from core.config_utils import build_env_config
from rl.advanced_rl_train import GaussianActor
from rl.ddpg_train import Actor, split_action
from rl.dqn_train import QNetwork, make_action_table
from rl.marl_utils import (
    decode_jammer_action,
    jammer_observation,
    make_jammer_action_table,
    make_relay_action_table,
    relay_observation,
)
from core.environment import UAVEnvironment


def _safe_import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _infer_actor_action_dim(model_path: str) -> int:
    checkpoint = torch.load(model_path, map_location="cpu")
    if "mean.weight" in checkpoint:
        return int(checkpoint["mean.weight"].shape[0])
    output_key = "net.4.weight"
    return int(checkpoint[output_key].shape[0]) if output_key in checkpoint else 5


def rollout_policy(
    env: UAVEnvironment,
    policy_name: str,
    model_path: str | dict | None = None,
) -> dict:
    state = env.reset().astype(np.float32)
    trace = {
        "relay": [env.relay_position.copy()],
        "jammer": [env.jammer_position.copy()],
        "user": [env.user_position.copy()],
        "eve": env.eve_position.copy(),
        "bs": env.bs_position.copy(),
        "total_r_sec": 0.0,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if policy_name == "dqn":
        action_table = make_action_table()
        state_dim = state.shape[0]
        q_net = QNetwork(state_dim, len(action_table), hidden_dim=128).to(device)
        q_net.load_state_dict(torch.load(model_path, map_location=device))
        q_net.eval()
    elif policy_name in ("ddpg", "td3"):
        state_dim = state.shape[0]
        actor = Actor(state_dim, _infer_actor_action_dim(model_path), hidden_dim=128).to(device)
        actor.load_state_dict(torch.load(model_path, map_location=device))
        actor.eval()
    elif policy_name == "sac":
        state_dim = state.shape[0]
        action_dim = _infer_actor_action_dim(model_path)
        actor = GaussianActor(state_dim, action_dim, hidden_dim=128).to(device)
        actor.load_state_dict(torch.load(model_path, map_location=device))
        actor.eval()
    elif policy_name in ("marl_shared", "marl_split"):
        relay_path = model_path["relay"]
        jammer_path = model_path["jammer"]
        agent_obs_mode = "shared" if policy_name == "marl_shared" else "split"
        relay_action_table = make_relay_action_table()
        jammer_action_table = make_jammer_action_table()
        if agent_obs_mode == "shared":
            relay_dim = state.shape[0]
            jammer_dim = state.shape[0]
        else:
            relay_dim = relay_observation(state).shape[0]
            jammer_dim = jammer_observation(state).shape[0]
        relay_qnet = QNetwork(relay_dim, len(relay_action_table), hidden_dim=128).to(device)
        jammer_qnet = QNetwork(jammer_dim, len(jammer_action_table), hidden_dim=128).to(device)
        relay_qnet.load_state_dict(torch.load(relay_path, map_location=device))
        jammer_qnet.load_state_dict(torch.load(jammer_path, map_location=device))
        relay_qnet.eval()
        jammer_qnet.eval()

    done = False
    role_switch = False
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
        elif policy_name in ("ddpg", "td3", "sac"):
            with torch.no_grad():
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                if policy_name == "sac":
                    action = actor.deterministic(s_t).cpu().numpy()[0]
                else:
                    action = actor(s_t).cpu().numpy()[0]
            a_relay, a_jammer, jammer_power = split_action(action)
            role_switch = bool(action.shape[0] > 5 and action[5] > 0.5)
        elif policy_name in ("marl_shared", "marl_split"):
            with torch.no_grad():
                if agent_obs_mode == "shared":
                    r_obs = state
                    j_obs = state
                else:
                    r_obs = relay_observation(state)
                    j_obs = jammer_observation(state)
                rs_t = torch.tensor(r_obs, dtype=torch.float32, device=device).unsqueeze(0)
                js_t = torch.tensor(j_obs, dtype=torch.float32, device=device).unsqueeze(0)
                relay_id = int(torch.argmax(relay_qnet(rs_t), dim=1).item())
                jammer_id = int(torch.argmax(jammer_qnet(js_t), dim=1).item())
            a_relay = relay_action_table[relay_id]
            jammer_vec = jammer_action_table[jammer_id]
            a_jammer, jammer_power = decode_jammer_action(jammer_vec)
        else:
            raise ValueError(f"Unsupported policy: {policy_name}")

        next_state, _, done, info = env.step(a_relay, a_jammer, jammer_power, role_switch)
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


def _resolve_checkpoint(
    method: str,
    dqn_model: str | None = None,
    ddpg_actor: str | None = None,
) -> str | dict | None:
    training_dir = Path("outputs/training")
    if method == "dqn":
        return dqn_model or str(training_dir / "dqn" / "dqn_qnet.pt")
    if method == "ddpg":
        return ddpg_actor or str(training_dir / "ddpg" / "ddpg_actor.pt")
    if method == "td3":
        return str(training_dir / "td3" / "td3_actor.pt")
    if method == "sac":
        return str(training_dir / "sac" / "sac_actor.pt")
    if method == "greedy" or method == "random":
        return None
    if method in ("marl_shared", "marl_split"):
        return {
            "relay": str(training_dir / method / "marl_relay_qnet.pt"),
            "jammer": str(training_dir / method / "marl_jammer_qnet.pt"),
        }
    raise ValueError(f"Unknown method: {method}")


def generate_trajectory_suite(
    method: str = "dqn",
    dqn_model: str | None = None,
    ddpg_actor: str | None = None,
    fading_model: str = "rician",
    seed: int = 42,
    output_dir: str = "outputs/trajectories",
    control_mode: str = "velocity",
    role_switching: bool = False,
    user_mobile: bool = False,
) -> dict:
    model_path = _resolve_checkpoint(method, dqn_model, ddpg_actor)
    out_dir = Path(output_dir) / method
    out_dir.mkdir(parents=True, exist_ok=True)

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
    out_path = out_dir / f"{method}_trajectory_{fading_model}.png"
    saved = plot_trajectory(trace, method, fading_model, str(out_path))
    return {method: saved}


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate UAV trajectory plots.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["dqn", "ddpg", "td3", "sac", "greedy", "marl_shared", "marl_split"],
        help="RL method for trajectory rollout",
    )
    parser.add_argument(
        "--dqn-model",
        type=str,
        default=None,
        help="Path to DQN model (optional, auto-resolved if omitted)",
    )
    parser.add_argument(
        "--ddpg-actor",
        type=str,
        default=None,
        help="Path to DDPG actor (optional, auto-resolved if omitted)",
    )
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
        method=args.method,
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
