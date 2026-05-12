import numpy as np

def build_geometry_observation(
    relay_position: np.ndarray,
    jammer_position: np.ndarray,
    user_position: np.ndarray,
    bs_position: np.ndarray,
    eve_position: np.ndarray,
    relay_velocity: np.ndarray,
    jammer_velocity: np.ndarray,
    user_velocity: np.ndarray,
) -> np.ndarray:
    return np.concatenate([
        relay_position,
        jammer_position,
        user_position,
        bs_position,
        eve_position,
        relay_velocity,
        jammer_velocity,
        user_velocity,
    ])


def build_channel_observation(gains: dict, rates: dict) -> np.ndarray:
    return np.concatenate([
        np.array([gains["h_UR"], gains["h_RB"], gains["h_UE"], gains["h_JE"]]),
        np.array([rates["gamma_UR"], rates["gamma_RB"], rates["gamma_E"]]),
        np.array([rates["R_legit"], rates["R_eve"], rates["R_sec"]]),
        np.array([rates["jammer_power"]]),
    ]).astype(float)


def build_distances_observation(distances: dict) -> np.ndarray:
    return np.array([
        distances["d_UR"], distances["d_RB"],
        distances["d_UE"], distances["d_JE"],
    ])


def build_battery_observation(
    relay_battery: float,
    jammer_battery: float,
) -> np.ndarray:
    return np.array([relay_battery, jammer_battery])


def normalize_observation(
    state: np.ndarray,
    mode: str,
    normalize: bool,
    half_area: float,
    max_speed: float,
    user_max_speed: float,
    jammer_power_max: float,
    area_size: float,
    relay_battery_capacity: float,
    jammer_battery_capacity: float,
) -> np.ndarray:
    if not normalize:
        return state

    eps = 1e-15
    max_dist = np.sqrt(2.0) * area_size  # square diagonal
    out = state.copy()

    if mode == "geometry":            # [pos:15, uav_vel:4, user_vel:2]
        out[:15] /= half_area
        out[15:19] /= max_speed
        if len(out) > 19:
            out[19:21] /= user_max_speed

    elif mode == "channels":          # [gains:4, snrs:3, secrecy:3, j_pwr:1]
        out[:4] = np.log10(np.maximum(out[:4], eps))
        out[4:7] = np.log10(np.maximum(out[4:7], eps))
        out[7:10] = np.log10(np.maximum(out[7:10], eps))
        out[10] /= jammer_power_max

    elif mode == "full":               # [geom:21, dist:4, chan:11, batt:2]
        out[:15] /= half_area
        out[15:19] /= max_speed
        out[19:21] /= user_max_speed
        out[21:25] /= max_dist
        s = 25
        out[s:s+4] = np.log10(np.maximum(out[s:s+4], eps))
        out[s+4:s+7] = np.log10(np.maximum(out[s+4:s+7], eps))
        out[s+7:s+10] = np.log10(np.maximum(out[s+7:s+10], eps))
        out[s+10] /= jammer_power_max
        out[36] /= relay_battery_capacity
        out[37] /= jammer_battery_capacity

    else:
        raise ValueError(f"Unknown observation mode: {mode}")

    return out

def build_observation(
    mode: str,
    relay_position: np.ndarray,
    jammer_position: np.ndarray,
    user_position: np.ndarray,
    bs_position: np.ndarray,
    eve_position: np.ndarray,
    relay_velocity: np.ndarray,
    jammer_velocity: np.ndarray,
    user_velocity: np.ndarray,
    relay_battery: float,
    jammer_battery: float,
    gains: dict,
    rates: dict,
    distances: dict,
    normalize: bool,
    half_area: float,
    max_speed: float,
    user_max_speed: float,
    jammer_power_max: float,
    area_size: float,
    relay_battery_capacity: float,
    jammer_battery_capacity: float,
) -> np.ndarray:
    if mode == "geometry":
        state = build_geometry_observation(
            relay_position, jammer_position, user_position, bs_position,
            eve_position, relay_velocity, jammer_velocity, user_velocity,
        )
    elif mode == "channels":
        state = build_channel_observation(gains, rates)
    elif mode == "full":
        geom = build_geometry_observation(
            relay_position, jammer_position, user_position, bs_position,
            eve_position, relay_velocity, jammer_velocity, user_velocity,
        )
        dist = build_distances_observation(distances)
        chan = build_channel_observation(gains, rates)
        batt = build_battery_observation(relay_battery, jammer_battery)
        state = np.concatenate([geom, dist, chan, batt])
    else:
        raise ValueError(f"Unknown observation mode: {mode}")

    return normalize_observation(
        state, mode, normalize,
        half_area, max_speed, user_max_speed,
        jammer_power_max, area_size,
        relay_battery_capacity, jammer_battery_capacity,
    )
