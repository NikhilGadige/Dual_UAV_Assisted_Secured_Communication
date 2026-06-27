from sensing_matrix_exp.environments.sensing_matrix_env import SensingMatrixConfig

default_config = SensingMatrixConfig(
    N_tx=16,
    N_rx=16,
    antenna_spacing=0.5,
    wavelength=1.0,
    L_pilot=32,
    noise_power=1e-10,
    sensing_power=1.0,
    seed=42,
    num_targets=3,
    output_root="outputs/sensing_matrix",
)

# Scenario: monostatic sensing at UAV (same position for TX and RX)
monostatic_config = SensingMatrixConfig(
    N_tx=16,
    N_rx=16,
    antenna_spacing=0.5,
    wavelength=1.0,
    L_pilot=32,
    noise_power=1e-10,
    sensing_power=1.0,
    seed=42,
    num_targets=3,
    output_root="outputs/sensing_matrix",
)

# Override for multi-target scenario
multi_target_config = SensingMatrixConfig(
    N_tx=16,
    N_rx=16,
    antenna_spacing=0.5,
    wavelength=1.0,
    L_pilot=64,
    noise_power=1e-10,
    sensing_power=1.0,
    seed=42,
    num_targets=5,
    output_root="outputs/sensing_matrix",
)
