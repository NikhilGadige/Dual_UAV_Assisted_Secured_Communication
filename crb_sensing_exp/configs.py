from crb_sensing_exp.environments.crb_sensing_env import CRBConfig

default_config = CRBConfig(
    N_tx=16,
    N_rx=16,
    antenna_spacing=0.5,
    wavelength=1.0,
    L_pilot=32,
    noise_power=1e-10,
    seed=42,
    num_targets=3,
    output_root="outputs/crb_sensing",
)

single_target_config = CRBConfig(
    N_tx=16,
    N_rx=16,
    antenna_spacing=0.5,
    wavelength=1.0,
    L_pilot=32,
    noise_power=1e-10,
    seed=42,
    num_targets=1,
    output_root="outputs/crb_sensing",
)

multi_target_config = CRBConfig(
    N_tx=16,
    N_rx=16,
    antenna_spacing=0.5,
    wavelength=1.0,
    L_pilot=64,
    noise_power=1e-10,
    seed=42,
    num_targets=5,
    output_root="outputs/crb_sensing",
)
