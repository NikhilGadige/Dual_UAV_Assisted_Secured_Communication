import numpy as np

def compute_secrecy_reward(secrecy_scale: float, R_sec: float) -> float:
    return secrecy_scale * R_sec


def compute_energy_penalty(energy_reward_weight: float, total_energy: float) -> float:
    return energy_reward_weight * total_energy


def compute_motion_penalty(
    movement_penalty_weight: float,
    relay_velocity: np.ndarray,
    jammer_velocity: np.ndarray,
    max_speed: float,
) -> float:
    r_speed = float(np.linalg.norm(relay_velocity))
    j_speed = float(np.linalg.norm(jammer_velocity))
    norm_speed_sq = (r_speed ** 2 + j_speed ** 2) / (max_speed ** 2 + 1e-10)
    return movement_penalty_weight * norm_speed_sq


def compute_smoothness_penalty(
    smoothness_penalty_weight: float,
    relay_velocity: np.ndarray,
    jammer_velocity: np.ndarray,
    prev_relay_velocity: np.ndarray,
    prev_jammer_velocity: np.ndarray,
    max_acceleration: float,
    dt: float,
) -> float:
    delta_relay = float(np.linalg.norm(relay_velocity - prev_relay_velocity))
    delta_jammer = float(np.linalg.norm(jammer_velocity - prev_jammer_velocity))
    max_delta = max_acceleration * dt
    norm_accel = (delta_relay + delta_jammer) / (2.0 * max_delta + 1e-10)
    return smoothness_penalty_weight * norm_accel


def compute_boundary_penalty(
    boundary_penalty_weight: float,
    relay_position: np.ndarray,
    jammer_position: np.ndarray,
    half_area: float,
) -> float:
    penalty = 0.0
    for pos in (relay_position, jammer_position):
        dx = min(pos[0] + half_area, half_area - pos[0])
        dy = min(pos[1] + half_area, half_area - pos[1])
        d_min = min(dx, dy)
        if d_min < half_area * 0.3:
            penalty += float(np.exp(-d_min / (half_area * 0.05)))
    return boundary_penalty_weight * penalty


def compute_sustainability_bonus(
    harvesting_reward_weight: float,
    total_harvested_energy_j: float,
) -> float:
    return harvesting_reward_weight * total_harvested_energy_j


def compute_total_reward(
    scaled_secrecy: float,
    energy_penalty: float,
    motion_penalty: float,
    smoothness_penalty: float,
    boundary_penalty: float,
    sustainability_bonus: float,
    battery_depletion_penalty: float,
    battery_depleted: bool,
) -> float:
    reward = (
        scaled_secrecy
        - energy_penalty
        - motion_penalty
        - smoothness_penalty
        - boundary_penalty
        + sustainability_bonus
    )
    if battery_depleted:
        reward -= battery_depletion_penalty
    return reward
