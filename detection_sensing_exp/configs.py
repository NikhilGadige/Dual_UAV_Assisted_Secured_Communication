from detection_sensing_exp.environments.detection_sensing_env import DetectionConfig

default_config = DetectionConfig(
    N_tx=16,
    N_rx=16,
    antenna_spacing=0.5,
    wavelength=1.0,
    L_pilot=32,
    noise_power=1e-10,
    num_mc=500,
    seed=42,
    num_targets=3,
    output_root="outputs/detection_sensing",
)

roc_config = DetectionConfig(
    N_tx=16,
    N_rx=16,
    L_pilot=32,
    noise_power=1e-10,
    num_mc=200,
    seed=42,
    num_targets=3,
)

sweep_config = DetectionConfig(
    N_tx=16,
    N_rx=16,
    L_pilot=32,
    noise_power=1e-10,
    num_mc=200,
    seed=42,
    num_targets=3,
)
