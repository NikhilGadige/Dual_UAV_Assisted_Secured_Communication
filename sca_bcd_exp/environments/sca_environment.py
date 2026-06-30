from __future__ import annotations

import numpy as np

from optimization_problem_exp.optimization.problem_formulation import (
    DecisionVariables,
    evaluate_objective_and_constraints,
)
from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.optimization.secrecy_optimizer import (
    SolutionState,
    clone_solution,
    make_initial_decision_vars,
)
from sca_bcd_exp.optimization.variable_scaling import VariableScaler


class SCABCDEnvironment:
    def __init__(self, config: SCABCDConfig):
        self.config = config
        self.scaler = VariableScaler(config, self.block_slices())

    def reset(self) -> SolutionState:
        initial_vars = make_initial_decision_vars(
            N_time=self.config.N_time,
            N_ris=self.config.N_ris,
            N_j=self.config.N_j,
            M_bs=self.config.M_bs,
            P_bs_max=self.config.P_bs_max,
            P_j_max=self.config.P_j_max,
            q_min=self.config.q_min_arr,
            q_max=self.config.q_max_arr,
            rng=np.random.default_rng(self.config.seed),
        )
        return SolutionState(decision_vars=initial_vars)

    def evaluate(self, solution: SolutionState, sensing_u_ref: float | None = None) -> dict:
        return evaluate_objective_and_constraints(
            decision_vars=solution.decision_vars,
            q_bs=self.config.q_bs_arr,
            q_user=self.config.q_user_arr,
            q_eves=self.config.q_eves_arr,
            q_jammer=self.config.q_jammer_arr,
            q_vehicles=self.config.q_vehicles_arr,
            vehicle_types=list(self.config.vehicle_types),
            N_ris=self.config.N_ris,
            N_j=self.config.N_j,
            N_tx_sense=self.config.N_tx_sense,
            N_rx_sense=self.config.N_rx_sense,
            L_pilot=self.config.L_pilot,
            P_bs_max=self.config.P_bs_max,
            P_j_max=self.config.P_j_max,
            sigma2=self.config.sigma2,
            noise_power_sense=self.config.noise_power_sense,
            v_max=self.config.v_max,
            dt=self.config.dt,
            q_min=self.config.q_min_arr,
            q_max=self.config.q_max_arr,
            d_ant=self.config.d_ant,
            wavelength=self.config.wavelength,
            eta_ris=self.config.eta_ris,
            alpha=self.config.alpha,
            jammer_mode=self.config.jammer_mode,
            include_direct_links=self.config.include_direct_links,
            seed=self.config.seed,
            sensing_utility_mode=self.config.sensing_utility_mode,
            sensing_u_ref=sensing_u_ref,
            M_bs=self.config.M_bs,
        )

    def evaluate_objective(self, solution: SolutionState) -> float:
        result = self.evaluate(solution)
        obj = float(result["objective"])
        viol = float(result["violations"]["total_violation"])
        penalty = self.config.rho_penalty * viol
        return obj - penalty

    def _pack_decision_vars(
        self, vars_solution: np.ndarray, template: SolutionState,
    ) -> DecisionVariables:
        n_time = self.config.N_time
        n_ris = self.config.N_ris
        n_j = self.config.N_j
        m_bs = self.config.M_bs

        w_dim = 2 * n_time * m_bs
        off_q = w_dim + 3 * n_time
        off_phi = off_q + n_ris
        off_v = off_phi + 2 * n_time * n_j

        w_re = vars_solution[:n_time * m_bs].reshape(n_time, m_bs)
        w_im = vars_solution[n_time * m_bs:w_dim].reshape(n_time, m_bs)
        w_bs = w_re + 1j * w_im

        q_uav = vars_solution[w_dim:off_q].reshape(n_time, 3)

        phi_rad = vars_solution[off_q:off_phi]

        v_flat = vars_solution[off_phi:off_v]
        v_re = v_flat[:n_time * n_j].reshape(n_time, n_j)
        v_im = v_flat[n_time * n_j:].reshape(n_time, n_j)
        v_jammer = v_re + 1j * v_im

        return DecisionVariables(
            phi_rad=phi_rad, q_uav=q_uav, w_bs=w_bs, v_jammer=v_jammer,
        )

    def _unpack_decision_vars(
        self, dv: DecisionVariables,
    ) -> np.ndarray:
        n_ris = self.config.N_ris

        w_re = np.real(dv.w_bs).ravel()
        w_im = np.imag(dv.w_bs).ravel()
        q_flat = dv.q_uav.ravel()
        v_re = np.real(dv.v_jammer).ravel()
        v_im = np.imag(dv.v_jammer).ravel()
        return np.concatenate([w_re, w_im, q_flat, dv.phi_rad, v_re, v_im])

    def _flat_obj(self, x: np.ndarray, template: SolutionState) -> float:
        dv = self._pack_decision_vars(x, template)
        sol = SolutionState(decision_vars=dv)
        return self.evaluate_objective(sol)

    def finite_diff_gradient(
        self, x: np.ndarray, template: SolutionState,
    ) -> np.ndarray:
        h = self.config.fd_h
        grad = np.zeros_like(x)
        f0 = self._flat_obj(x, template)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += h
            fp = self._flat_obj(xp, template)
            grad[i] = (fp - f0) / h
        return grad

    def finite_diff_gradient_for_block(
        self, x_block: np.ndarray, block_sl: slice, template: SolutionState,
    ) -> np.ndarray:
        h = self.config.fd_h
        grad = np.zeros_like(x_block)
        full = self._unpack_decision_vars(template.decision_vars)
        f0 = self._flat_block_obj(full, block_sl, template)
        for i in range(len(x_block)):
            full_p = full.copy()
            full_p[block_sl][i] += h
            fp = self._flat_block_obj(full_p, block_sl, template)
            grad[i] = (fp - f0) / h
        return grad

    # ── Scaled-coordinate finite differences ────────────────────
    # These work in the normalised space produced by VariableScaler
    # and use adaptive perturbation sizes.

    def finite_diff_gradient_scaled(
        self, x_scaled: np.ndarray, template: SolutionState,
    ) -> np.ndarray:
        x_unscaled = self.scaler.unscale_full(x_scaled)
        f0 = self._flat_obj(x_unscaled, template)
        grad = np.zeros_like(x_scaled)
        eps = self.scaler.adaptive_eps(x_scaled)
        for i in range(len(x_scaled)):
            pert = eps[i] * self.scaler.element_scale(i)
            xp = x_unscaled.copy()
            xp[i] += pert
            fp = self._flat_obj(xp, template)
            grad[i] = (fp - f0) / eps[i]
        return grad

    def finite_diff_gradient_for_block_scaled(
        self, x_block_scaled: np.ndarray,
        block_sl: slice, template: SolutionState,
    ) -> np.ndarray:
        full = self._unpack_decision_vars(template.decision_vars)
        f0 = self._flat_block_obj(full, block_sl, template)
        grad = np.zeros_like(x_block_scaled)
        eps = self.scaler.adaptive_eps(x_block_scaled)
        start = block_sl.start
        for i in range(len(x_block_scaled)):
            idx = start + i
            pert = eps[i] * self.scaler.element_scale(idx)
            full_p = full.copy()
            full_p[idx] += pert
            fp = self._flat_block_obj(full_p, block_sl, template)
            grad[i] = (fp - f0) / eps[i]
        return grad

    def _flat_block_obj(
        self, full: np.ndarray, block_sl: slice, template: SolutionState,
    ) -> float:
        dv = self._pack_decision_vars(full, template)
        sol = SolutionState(decision_vars=dv)
        return self.evaluate_objective(sol)

    def block_slices(self) -> dict[str, slice]:
        n_time = self.config.N_time
        n_ris = self.config.N_ris
        n_j = self.config.N_j
        m_bs = self.config.M_bs

        power_dim = 2 * n_time * m_bs
        traj_dim = 3 * n_time
        ris_dim = n_ris
        jammer_dim = 2 * n_time * n_j

        power_sl = slice(0, power_dim)
        traj_sl = slice(power_dim, power_dim + traj_dim)
        ris_sl = slice(power_dim + traj_dim, power_dim + traj_dim + ris_dim)
        jammer_sl = slice(
            power_dim + traj_dim + ris_dim,
            power_dim + traj_dim + ris_dim + jammer_dim,
        )
        return {
            "power": power_sl,
            "trajectory": traj_sl,
            "ris": ris_sl,
            "jammer": jammer_sl,
        }

    def total_dim(self) -> int:
        n_time = self.config.N_time
        n_ris = self.config.N_ris
        n_j = self.config.N_j
        m_bs = self.config.M_bs
        return 2 * n_time * m_bs + 3 * n_time + n_ris + 2 * n_time * n_j
