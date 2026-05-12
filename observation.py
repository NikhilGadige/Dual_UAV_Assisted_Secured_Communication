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


def build_eh_observation(
    relay_battery: float,
    jammer_battery: float,
    relay_battery_capacity: float,
    jammer_battery_capacity: float,
    relay_harvest_power_w: float,
    jammer_harvest_power_w: float,
    relay_harvest_max_watts: float,
    jammer_harvest_max_watts: float,
    battery_saturation_event: bool,
) -> np.ndarray:
    eps = 1e-10
    relay_battery_ratio = relay_battery / max(relay_battery_capacity, eps)
    jammer_battery_ratio = jammer_battery / max(jammer_battery_capacity, eps)
    relay_harvest_power_norm = relay_harvest_power_w / max(relay_harvest_max_watts, eps)
    jammer_harvest_power_norm = jammer_harvest_power_w / max(jammer_harvest_max_watts, eps)
    saturation_flag = 1.0 if battery_saturation_event else 0.0
    return np.array([
        relay_battery_ratio,
        jammer_battery_ratio,
        relay_harvest_power_norm,
        jammer_harvest_power_norm,
        saturation_flag,
    ])


def build_ntn_observation(
    satellite_position: np.ndarray,
    relay_position: np.ndarray,
    h_sat_relay: float,
) -> np.ndarray:
    # Satellite elevation angle (radians) from relay
    dx = satellite_position[0] - relay_position[0]
    dy = satellite_position[1] - relay_position[1]
    dz = satellite_position[2] - relay_position[2]
    d_h = np.sqrt(dx * dx + dy * dy)
    elevation = np.arctan2(abs(dz), max(d_h, 1e-10))
    # Satellite-relay slant range
    sat_relay_dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    return np.array([
        float(elevation),
        float(sat_relay_dist),
        float(h_sat_relay),
    ])


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
    satellite_altitude_m: float = 500_000.0,
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

    elif mode == "full_eh":            # [full(38) + eh(5) = 43]
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
        # EH features at indices 38-42 are already in [0,1] / {0,1}

    elif mode == "full_ntn":           # [full(38) + eh(5) + ntn(3) = 46]
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
        # EH features at indices 38-42 are already in [0,1] / {0,1}
        # NTN features at 43-45
        out[43] = out[43] / (np.pi / 2.0)  # elevation angle → [0, 1]
        # sat_relay_dist at idx 44: normalise by 3× satellite altitude as max range proxy
        out[44] /= max(3.0 * satellite_altitude_m, eps)
        out[45] = np.log10(np.maximum(out[45], eps))  # h_sat_relay

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
    # EH observation params (for full_eh / full_ntn mode)
    enable_energy_harvesting: bool = False,
    relay_harvest_power_w: float = 0.0,
    jammer_harvest_power_w: float = 0.0,
    relay_harvest_max_watts: float = 8.0,
    jammer_harvest_max_watts: float = 8.0,
    battery_saturation_event: bool = False,
    # NTN observation params (for full_ntn mode)
    satellite_position: np.ndarray | None = None,
    h_sat_relay: float = 0.0,
    satellite_altitude_m: float = 500_000.0,
) -> np.ndarray:
    if mode == "geometry":
        state = build_geometry_observation(
            relay_position, jammer_position, user_position, bs_position,
            eve_position, relay_velocity, jammer_velocity, user_velocity,
        )
    elif mode == "channels":
        state = build_channel_observation(gains, rates)
    elif mode in ("full", "full_eh", "full_ntn"):
        geom = build_geometry_observation(
            relay_position, jammer_position, user_position, bs_position,
            eve_position, relay_velocity, jammer_velocity, user_velocity,
        )
        dist = build_distances_observation(distances)
        chan = build_channel_observation(gains, rates)
        batt = build_battery_observation(relay_battery, jammer_battery)
        state = np.concatenate([geom, dist, chan, batt])
        if mode in ("full_eh", "full_ntn"):
            eh = build_eh_observation(
                relay_battery, jammer_battery,
                relay_battery_capacity, jammer_battery_capacity,
                relay_harvest_power_w, jammer_harvest_power_w,
                relay_harvest_max_watts, jammer_harvest_max_watts,
                battery_saturation_event,
            )
            state = np.concatenate([state, eh])
        if mode == "full_ntn":
            ntn = build_ntn_observation(
                satellite_position, relay_position, h_sat_relay,
            )
            state = np.concatenate([state, ntn])
    else:
        raise ValueError(f"Unknown observation mode: {mode}")

    return normalize_observation(
        state, mode, normalize,
        half_area, max_speed, user_max_speed,
        jammer_power_max, area_size,
        relay_battery_capacity, jammer_battery_capacity,
        satellite_altitude_m=satellite_altitude_m,
    )
