import numpy as np

def compute_energy_usage(
    relay_velocity: np.ndarray,
    jammer_velocity: np.ndarray,
    jammer_power: float,
    relay_hover_power_watts: float,
    relay_motion_power_coeff: float,
    jammer_hover_power_watts: float,
    jammer_motion_power_coeff: float,
    jammer_rf_power_coeff: float,
    dt: float,
) -> dict:
    relay_speed = float(np.linalg.norm(relay_velocity))
    jammer_speed = float(np.linalg.norm(jammer_velocity))

    relay_power_draw = (
        relay_hover_power_watts
        + relay_motion_power_coeff * relay_speed ** 2
    )
    jammer_power_draw = (
        jammer_hover_power_watts
        + jammer_motion_power_coeff * jammer_speed ** 2
        + jammer_rf_power_coeff * jammer_power
    )

    relay_energy = relay_power_draw * dt
    jammer_energy = jammer_power_draw * dt

    return {
        "relay_speed": relay_speed,
        "jammer_speed": jammer_speed,
        "relay_energy": relay_energy,
        "jammer_energy": jammer_energy,
        "total_energy": relay_energy + jammer_energy,
    }


def compute_energy_harvesting(
    relay_harvest_efficiency: float,
    relay_harvest_max_watts: float,
    jammer_harvest_efficiency: float,
    jammer_harvest_max_watts: float,
    solar_variability: float,
    dt: float,
) -> dict:
    # --- Relay UAV harvesting ---
    relay_base_power = relay_harvest_efficiency * relay_harvest_max_watts
    relay_fluctuation = np.random.normal(0.0, solar_variability)
    relay_harvest_power = relay_base_power * (1.0 + relay_fluctuation)
    relay_harvest_power = np.clip(
        relay_harvest_power, 0.0, relay_harvest_max_watts
    )
    relay_harvested_energy = relay_harvest_power * dt

    # --- Jammer UAV harvesting ---
    jammer_base_power = jammer_harvest_efficiency * jammer_harvest_max_watts
    jammer_fluctuation = np.random.normal(0.0, solar_variability)
    jammer_harvest_power = jammer_base_power * (1.0 + jammer_fluctuation)
    jammer_harvest_power = np.clip(
        jammer_harvest_power, 0.0, jammer_harvest_max_watts
    )
    jammer_harvested_energy = jammer_harvest_power * dt

    total_harvested_energy = relay_harvested_energy + jammer_harvested_energy

    return {
        "relay_harvest_power_w": float(relay_harvest_power),
        "jammer_harvest_power_w": float(jammer_harvest_power),
        "relay_harvested_energy_j": float(relay_harvested_energy),
        "jammer_harvested_energy_j": float(jammer_harvested_energy),
        "total_harvested_energy_j": float(total_harvested_energy),
    }


def update_battery_state(
    relay_battery: float,
    jammer_battery: float,
    relay_energy_consumed: float,
    jammer_energy_consumed: float,
    relay_battery_capacity: float,
    jammer_battery_capacity: float,
    enable_harvesting: bool = False,
    relay_harvested_energy: float = 0.0,
    jammer_harvested_energy: float = 0.0,
) -> dict:
    # Step 1: Discharge (energy consumption)
    relay_battery = max(0.0, relay_battery - relay_energy_consumed)
    jammer_battery = max(0.0, jammer_battery - jammer_energy_consumed)

    # Step 2: Charge (energy harvesting)
    if enable_harvesting:
        relay_battery = min(relay_battery_capacity, relay_battery + relay_harvested_energy)
        jammer_battery = min(jammer_battery_capacity, jammer_battery + jammer_harvested_energy)

    # Saturation: battery reached capacity after harvesting
    battery_saturation = (
        relay_battery >= relay_battery_capacity - 1e-9
        or jammer_battery >= jammer_battery_capacity - 1e-9
    ) if enable_harvesting else False

    # Depletion: any battery is empty
    battery_depleted = relay_battery <= 0.0 or jammer_battery <= 0.0

    return {
        "relay_battery": relay_battery,
        "jammer_battery": jammer_battery,
        "battery_depleted": battery_depleted,
        "battery_saturation_event": battery_saturation,
    }
