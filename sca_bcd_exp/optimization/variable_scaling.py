"""Variable scaling for conditioning improvement.

Maps each optimisation block to a dimensionless space:

    w_scaled  = w / sqrt(P_bs_max)           (power)
    q_scaled  = (q - q_center) / q_scale     (trajectory)
    v_scaled  = v / sqrt(P_j_max)            (jammer)

This puts all variables in O(1) range, preventing ill-conditioned
gradients caused by vastly different physical scales.
"""

from __future__ import annotations

import numpy as np

from sca_bcd_exp.configs import SCABCDConfig


class VariableScaler:
    """Scale/unscale decision variables block-wise.

    Stores per-element scale and centre arrays so that the full flat
    vector can be transformed in a single vectorised operation.
    """

    def __init__(
        self, config: SCABCDConfig, block_slices: dict[str, slice],
    ):
        self.config = config
        self.slices = block_slices
        self._build()

    # ── initialisation ──────────────────────────────────────────────

    def _build(self) -> None:
        n_time = self.config.N_time
        n_ris = self.config.N_ris
        n_j = self.config.N_j
        m_bs = self.config.M_bs
        total_dim = (
            2 * n_time * m_bs   # power (re + im per antenna)
            + 3 * n_time        # trajectory (x, y, z per slot)
            + n_ris             # RIS phases
            + 2 * n_time * n_j  # jammer (re + im)
        )

        self._scale = np.ones(total_dim, dtype=float)
        self._center = np.zeros(total_dim, dtype=float)

        # ── Power block ──────────────────────────────────────────
        sl = self.slices["power"]
        scl = float(np.sqrt(self.config.P_bs_max))
        self._scale[sl] = scl
        # centre stays 0

        # ── Trajectory block ─────────────────────────────────────
        sl = self.slices["trajectory"]
        q_min = self.config.q_min_arr  # (3,)
        q_max = self.config.q_max_arr  # (3,)
        q_center = (q_min + q_max) / 2.0
        q_scale = (q_max - q_min) / 2.0  # half-range => scaled in [-1, 1]

        self._center[sl] = np.tile(q_center, n_time)
        self._scale[sl] = np.tile(q_scale, n_time)

        # ── Jammer block ─────────────────────────────────────────
        sl = self.slices["jammer"]
        scl = float(np.sqrt(self.config.P_j_max))
        self._scale[sl] = scl
        # centre stays 0

    # ── full-vector transforms ──────────────────────────────────────

    def scale_full(self, x: np.ndarray) -> np.ndarray:
        """Unscaled flat vector -> scaled flat vector."""
        return (x - self._center) / self._scale

    def unscale_full(self, xs: np.ndarray) -> np.ndarray:
        """Scaled flat vector -> unscaled flat vector."""
        return xs * self._scale + self._center

    # ── block-level transforms ──────────────────────────────────────

    def scale_block(self, x_block: np.ndarray, block_name: str) -> np.ndarray:
        sl = self.slices[block_name]
        return (x_block - self._center[sl]) / self._scale[sl]

    def unscale_block(self, xs_block: np.ndarray, block_name: str) -> np.ndarray:
        sl = self.slices[block_name]
        return xs_block * self._scale[sl] + self._center[sl]

    # ── per-element access ──────────────────────────────────────────

    def element_scale(self, idx: int) -> float:
        return float(self._scale[idx])

    def element_center(self, idx: int) -> float:
        return float(self._center[idx])

    # ── adaptive perturbation size ──────────────────────────────────

    def adaptive_eps(self, xs_block: np.ndarray) -> np.ndarray:
        """Per-element FD step in scaled space.

        eps_i = max(1e-6, 1e-3 * |x_scaled_i|)
        """
        return np.maximum(1e-6, 1e-3 * np.abs(xs_block))
