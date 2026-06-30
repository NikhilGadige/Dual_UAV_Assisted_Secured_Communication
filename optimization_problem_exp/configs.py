from optimization_problem_exp.environments.optimization_problem_env import (
    OptimizationConfig, default_scenario,
)

default_config = OptimizationConfig(
    N_ris=16,
    N_j=4,
    N_tx_sense=16,
    N_rx_sense=16,
    L_pilot=32,
    N_time=3,
    P_bs_max=10.0,
    P_j_max=0.05,
    sigma2=1e-8,
    noise_power_sense=1e-8,
    v_max=50.0,
    dt=1.0,
    d_ant=0.5,
    wavelength=1.0,
    f_c=2e9,
    eta_ris=0.3,
    seed=42,
    output_root="outputs/optimization_problem",
)
