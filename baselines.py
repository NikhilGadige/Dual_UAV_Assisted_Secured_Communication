import numpy as np

from environment import EnvConfig, UAVEnvironment

BPS_TO_MBPS = 1e-6
BITS_TO_MBITS = 1e-6

def _unit_direction(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    vec = np.asarray(dst_xy) - np.asarray(src_xy)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return np.zeros(2, dtype=float)
    return (vec / norm).astype(float)

def random_policy(env: UAVEnvironment) -> tuple[np.ndarray, np.ndarray]:
    a_relay = np.random.uniform(-1.0, 1.0, size=2)
    a_jammer = np.random.uniform(-1.0, 1.0, size=2)
    return a_relay, a_jammer

def distance_greedy_policy(env: UAVEnvironment) -> tuple[np.ndarray, np.ndarray]:
    # Relay targets midpoint between user and BS to balance two-hop quality.
    relay_target_xy = 0.5 * (env.user_position[:2] + env.bs_position[:2])
    jammer_target_xy = env.eve_position[:2]

    a_relay = _unit_direction(env.relay_position[:2], relay_target_xy)
    a_jammer = _unit_direction(env.jammer_position[:2], jammer_target_xy)
    return a_relay, a_jammer

def run_episode(env: UAVEnvironment, policy_fn) -> dict:
    env.reset()
    total_reward = 0.0
    total_r_legit = 0.0
    total_r_eve = 0.0
    total_r_sec = 0.0
    total_secrecy_bits = 0.0

    done = False
    steps = 0
    while not done:
        a_relay, a_jammer = policy_fn(env)
        _, reward, done, info = env.step(a_relay, a_jammer)

        steps += 1
        total_reward += reward
        total_r_legit += info["R_legit"]
        total_r_eve += info["R_eve"]
        total_r_sec += info["R_sec"]
        total_secrecy_bits += info["R_sec"] * env.config.dt

    return {
        "steps": steps,
        "episode_reward_bps_step": float(total_reward),
        "episode_secrecy_throughput_bps_step": float(total_r_sec),
        "episode_secrecy_bits": float(total_secrecy_bits),
        "episode_secrecy_mbits": float(total_secrecy_bits * BITS_TO_MBITS),
        "avg_step_reward_bps": float(total_reward / max(steps, 1)),
        "avg_R_legit_bps": float(total_r_legit / max(steps, 1)),
        "avg_R_eve_bps": float(total_r_eve / max(steps, 1)),
        "avg_R_sec_bps": float(total_r_sec / max(steps, 1)),
        "avg_step_reward_mbps": float((total_reward / max(steps, 1)) * BPS_TO_MBPS),
        "avg_R_legit_mbps": float((total_r_legit / max(steps, 1)) * BPS_TO_MBPS),
        "avg_R_eve_mbps": float((total_r_eve / max(steps, 1)) * BPS_TO_MBPS),
        "avg_R_sec_mbps": float((total_r_sec / max(steps, 1)) * BPS_TO_MBPS),
    }

def evaluate_policy(
    policy_name: str,
    policy_fn,
    episodes: int = 20,
    seed: int = 42,
    return_episode_metrics: bool = False,
) -> dict:
    config = EnvConfig(seed=seed)
    env = UAVEnvironment(config)
    metrics = [run_episode(env, policy_fn) for _ in range(episodes)]

    summary = {
        "policy": policy_name,
        "episodes": episodes,
        "mean_episode_reward_bps_step": float(np.mean([m["episode_reward_bps_step"] for m in metrics])),
        "mean_episode_secrecy_throughput_bps_step": float(
            np.mean([m["episode_secrecy_throughput_bps_step"] for m in metrics])
        ),
        "mean_episode_secrecy_bits": float(np.mean([m["episode_secrecy_bits"] for m in metrics])),
        "mean_episode_secrecy_mbits": float(np.mean([m["episode_secrecy_mbits"] for m in metrics])),
        "mean_avg_step_reward_bps": float(np.mean([m["avg_step_reward_bps"] for m in metrics])),
        "mean_avg_R_legit_bps": float(np.mean([m["avg_R_legit_bps"] for m in metrics])),
        "mean_avg_R_eve_bps": float(np.mean([m["avg_R_eve_bps"] for m in metrics])),
        "mean_avg_R_sec_bps": float(np.mean([m["avg_R_sec_bps"] for m in metrics])),
        "mean_avg_step_reward_mbps": float(np.mean([m["avg_step_reward_mbps"] for m in metrics])),
        "mean_avg_R_legit_mbps": float(np.mean([m["avg_R_legit_mbps"] for m in metrics])),
        "mean_avg_R_eve_mbps": float(np.mean([m["avg_R_eve_mbps"] for m in metrics])),
        "mean_avg_R_sec_mbps": float(np.mean([m["avg_R_sec_mbps"] for m in metrics])),
    }
    if return_episode_metrics:
        summary["episode_metrics"] = metrics
    return summary

def print_summary(summary: dict) -> None:
    print(f"\nPolicy: {summary['policy']} | Episodes: {summary['episodes']}")
    print(
        "  Mean episode reward              : "
        f"{summary['mean_episode_reward_bps_step']:.4f} bps-step"
    )
    print(
        "  Mean episode secrecy throughput  : "
        f"{summary['mean_episode_secrecy_throughput_bps_step']:.4f} bps-step"
    )
    print(
        "  Mean episode secrecy payload     : "
        f"{summary['mean_episode_secrecy_mbits']:.4f} Mbits"
    )
    print(f"  Mean avg step reward             : {summary['mean_avg_step_reward_mbps']:.4f} Mbps")
    print(f"  Mean avg R_legit                 : {summary['mean_avg_R_legit_mbps']:.4f} Mbps")
    print(f"  Mean avg R_eve                   : {summary['mean_avg_R_eve_mbps']:.4f} Mbps")
    print(f"  Mean avg R_sec                   : {summary['mean_avg_R_sec_mbps']:.4f} Mbps")

if __name__ == "__main__":
    EPISODES = 20
    random_summary = evaluate_policy("Random Walk", random_policy, episodes=EPISODES, seed=42)
    greedy_summary = evaluate_policy(
        "Distance-Greedy", distance_greedy_policy, episodes=EPISODES, seed=42
    )

    print_summary(random_summary)
    print_summary(greedy_summary)