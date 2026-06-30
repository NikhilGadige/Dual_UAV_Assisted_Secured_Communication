from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SCABCDConfig:
    channel_model: str = "rician"
    seed: int = 0
    N_time: int = 5
    N_ris: int = 16
    N_j: int = 4
    N_tx_sense: int = 16
    N_rx_sense: int = 16
    L_pilot: int = 32
    M_bs: int = 4
    P_bs_max: float = 10.0
    P_j_max: float = 0.05
    sigma2: float = 1e-8
    noise_power_sense: float = 1e-8
    v_max: float = 50.0
    dt: float = 1.0
    d_ant: float = 0.5
    wavelength: float = 1.0
    eta_ris: float = 0.3
    alpha: float = 0.5
    jammer_mode: str = "mixed"
    include_direct_links: bool = False
    sensing_utility_mode: str = "log"

    max_bcd_iters: int = 50
    max_sca_iters: int = 20
    tol_obj: float = 1e-4
    tol_var: float = 1e-4
    trust_region_weight: float = 1.0
    reg_eps: float = 1e-8
    rho_penalty: float = 1e3
    fd_h: float = 1e-5
    sca_candidate_step_sizes: tuple = (1.0, 0.5, 0.25, 0.1, 0.05)

    q_bs: tuple = (0.0, 0.0, 30.0)
    q_user: tuple = (200.0, 0.0, 1.5)
    q_jammer: tuple = (100.0, -120.0, 50.0)
    q_eves: tuple = ((200.0, 150.0, 1.5), (150.0, -130.0, 1.5), (300.0, 80.0, 1.5))
    q_vehicles: tuple = ((200.0, 80.0, 0.0), (250.0, -60.0, 0.0), (180.0, -100.0, 0.0))
    vehicle_types: tuple = ("car", "truck", "motorcycle")
    q_min: tuple = (0.0, -150.0, 50.0)
    q_max: tuple = (400.0, 150.0, 120.0)

    def output_root(self) -> Path:
        return Path("outputs") / "sca_bcd"

    def ensure_output_dirs(self) -> dict[str, Path]:
        root = self.output_root()
        dirs = {
            "root": root,
        }
        root.mkdir(parents=True, exist_ok=True)
        return dirs

    @property
    def q_bs_arr(self) -> np.ndarray:
        return np.array(self.q_bs, dtype=float)

    @property
    def q_user_arr(self) -> np.ndarray:
        return np.array(self.q_user, dtype=float)

    @property
    def q_jammer_arr(self) -> np.ndarray:
        return np.array(self.q_jammer, dtype=float)

    @property
    def q_eves_arr(self) -> np.ndarray:
        return np.array(self.q_eves, dtype=float)

    @property
    def q_vehicles_arr(self) -> np.ndarray:
        return np.array(self.q_vehicles, dtype=float)

    @property
    def q_min_arr(self) -> np.ndarray:
        return np.array(self.q_min, dtype=float)

    @property
    def q_max_arr(self) -> np.ndarray:
        return np.array(self.q_max, dtype=float)
