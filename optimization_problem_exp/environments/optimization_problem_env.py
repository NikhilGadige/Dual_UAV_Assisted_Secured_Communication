"""Optimization problem environment: scenario config, objective evaluation, alpha sweep."""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from optimization_problem_exp.optimization.problem_formulation import (
    DecisionVariables,
    compute_secrecy_rate,
    compute_sensing_utility,
    evaluate_weighted_objective,
    check_constraints,
    compute_constraint_violations,
    compute_normalization_constants,
    design_ris_phases,
    compute_bs_ris_channel,
    compute_ris_user_channel,
    compute_ris_eve_channel,
    compute_jammer_user_channel,
    compute_jammer_eve_channel,
    compute_user_sinr,
    compute_eve_sinr,
    compute_direct_bs_user_channel,
    compute_direct_bs_eve_channel,
    design_heuristic_jammer_beam,
    R_S_REF,
    U_SENSE_REF,
    get_u_ref,
)
from vehicle_reflection_exp.channels.vehicle_channel import compute_rcs
from ris_uav_exp.channels.ris_channel import (
    compute_ris_reflection_matrix,
    compute_effective_channel,
    compute_effective_channel_gain,
)
from fd_jammer_exp.channels.fd_jammer_channel import compute_jammer_gain as jg

SPEED_OF_LIGHT = 3e8


@dataclass
class OptimizationConfig:
    N_ris: int = 16
    N_j: int = 4
    N_tx_sense: int = 16
    N_rx_sense: int = 16
    L_pilot: int = 32
    N_time: int = 5
    M_bs: int = 4
    P_bs_max: float = 10.0
    P_j_max: float = 0.05
    sigma2: float = 1e-8
    noise_power_sense: float = 1e-8
    v_max: float = 50.0
    dt: float = 1.0
    d_ant: float = 0.5
    wavelength: float = 1.0
    f_c: float = 2e9
    eta_ris: float = 0.3
    seed: int | None = 42
    output_root: str = "outputs/optimization_problem"
    sensing_utility_mode: str = "log"


@dataclass
class Scenario:
    """Fixed scenario geometry.

    User is directly under the UAV midpoint for best RIS channel.
    Eves are laterally offset so path loss and RIS misalignment hurt them.
    Vehicles are spread in y to create a sensing-communication trade-off.
    """
    q_bs: np.ndarray
    q_user: np.ndarray
    q_jammer: np.ndarray
    q_eves: np.ndarray
    q_vehicles: np.ndarray
    q_uav_start: np.ndarray
    q_uav_end: np.ndarray
    q_min: np.ndarray
    q_max: np.ndarray
    vehicle_types: list


def default_scenario() -> Scenario:
    return Scenario(
        q_bs=np.array([0.0, 0.0, 30.0]),
        q_user=np.array([200.0, 0.0, 1.5]),
        q_jammer=np.array([100.0, -120.0, 50.0]),
        q_eves=np.array([
            [200.0, 150.0, 1.5],
            [150.0, -130.0, 1.5],
            [300.0, 80.0, 1.5],
        ]),
        q_vehicles=np.array([
            [200.0, 80.0, 0.0],
            [250.0, -60.0, 0.0],
            [180.0, -100.0, 0.0],
        ]),
        q_uav_start=np.array([50.0, -50.0, 60.0]),
        q_uav_end=np.array([350.0, 50.0, 60.0]),
        q_min=np.array([0.0, -150.0, 50.0]),
        q_max=np.array([400.0, 150.0, 120.0]),
        vehicle_types=["car", "truck", "motorcycle"],
    )


class OptimizationProblemEnv:
    def __init__(self, config: OptimizationConfig | None = None):
        self.config = config or OptimizationConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        self.scenario: Scenario = default_scenario()
        self.decision_vars: DecisionVariables | None = None

    # ── Heuristic alpha-dependent decision variables ──────

    def _design_alpha_vars(self, alpha: float, rng_seed: int = 0) -> DecisionVariables:
        """Design decision variables that depend smoothly on alpha.

        No hard thresholds — all policies vary continuously with alpha.
        """
        np.random.seed(rng_seed)
        N_time = self.config.N_time
        N_ris = self.config.N_ris
        N_j = self.config.N_j

        phi_rad = np.random.uniform(-np.pi, np.pi, size=N_ris)

        # UAV trajectory: smooth y-shift between user (alpha=1)
        # and vehicle centroid (alpha=0)
        user_y = float(self.scenario.q_user[1])
        veh_y = float(np.mean(self.scenario.q_vehicles[:, 1]))
        target_y = alpha * user_y + (1.0 - alpha) * veh_y

        q_uav = np.zeros((N_time, 3))
        start = self.scenario.q_uav_start
        end = self.scenario.q_uav_end
        for n in range(N_time):
            t = n / max(N_time - 1, 1)
            q_uav[n] = start + t * (end - start)
            q_uav[n, 1] = target_y + np.random.uniform(-3.0, 3.0)
        for d in range(3):
            q_uav[:, d] = np.clip(
                q_uav[:, d],
                self.scenario.q_min[d],
                self.scenario.q_max[d],
            )

        # BS power: scales smoothly from 50% (alpha=0) to 100% (alpha=1)
        M_bs = self.config.M_bs
        bs_power_scale = 0.5 + 0.5 * alpha
        power_per_antenna = self.config.P_bs_max * bs_power_scale / max(M_bs, 1)
        w_bs = np.sqrt(power_per_antenna) * np.exp(
            1j * np.random.uniform(0, 2 * np.pi, size=(N_time, M_bs)),
        )

        # Jammer: placeholder — actual beam built inside compute_secrecy_rate
        v_jammer = np.zeros((N_time, N_j), dtype=complex)
        for n in range(N_time):
            v_raw = np.random.randn(N_j) + 1j * np.random.randn(N_j)
            v_raw = v_raw / float(np.linalg.norm(v_raw))
            v_jammer[n] = v_raw * np.sqrt(self.config.P_j_max)

        return DecisionVariables(
            phi_rad=phi_rad,
            q_uav=q_uav,
            w_bs=w_bs,
            v_jammer=v_jammer,
        )

    # ── Main evaluation ─────────────────────────────────

    def get_phase_aligned_phi(self, dv: DecisionVariables) -> np.ndarray | None:
        """Return phase-aligned phi; None means compute fresh in secrecy routine."""
        return None

    def design_phases_for_user(self, q_uav_slot: np.ndarray) -> np.ndarray:
        h_BR = compute_bs_ris_channel(
            self.scenario.q_bs, q_uav_slot, self.config.N_ris, seed=0,
        )
        h_RU = compute_ris_user_channel(
            q_uav_slot, self.scenario.q_user, self.config.N_ris, seed=1,
        )
        return design_ris_phases(h_BR, h_RU)

    def random_decision_vars(self, alpha: float = 0.85) -> DecisionVariables:
        """Generate random feasible decision variables."""
        return self._design_alpha_vars(alpha=alpha, rng_seed=42)

    def evaluate(
        self,
        dv: DecisionVariables | None = None,
        jammer_mode: str = "mixed",
        include_direct_links: bool = False,
        alpha: float = 0.5,
    ) -> dict:
        """Full evaluation: secrecy, sensing, objective, constraints.

        All alpha-dependent policies vary smoothly — no hard thresholds.
        """
        if dv is None:
            dv = self.random_decision_vars()

        rcs_list = [
            compute_rcs(vtype) for vtype in self.scenario.vehicle_types
        ]

        # Smooth alpha-dependent parameters
        ris_align = alpha
        jammer_mix = alpha
        jp_factor = max(0.01, alpha)
        bs_power_scale = 0.5 + 0.5 * alpha

        # Adjust w_bs power to match scaled value
        M_bs = self.config.M_bs
        w_bs_scaled = dv.w_bs.copy()
        target_power = self.config.P_bs_max * bs_power_scale
        if M_bs > 1:
            for n in range(len(w_bs_scaled)):
                current_norm = float(np.linalg.norm(w_bs_scaled[n]))
                scale = np.sqrt(target_power) / max(current_norm, 1e-30)
                w_bs_scaled[n] = w_bs_scaled[n] * scale
        else:
            for n in range(len(w_bs_scaled)):
                current_phase = np.angle(w_bs_scaled[n])
                w_bs_scaled[n] = np.sqrt(target_power) * np.exp(1j * current_phase)

        sec_result = compute_secrecy_rate(
            q_bs=self.scenario.q_bs,
            q_user=self.scenario.q_user,
            q_eves=self.scenario.q_eves,
            q_jammer=self.scenario.q_jammer,
            N_ris=self.config.N_ris,
            N_j=self.config.N_j,
            Phi=None,
            q_uav=dv.q_uav,
            w_bs=w_bs_scaled,
            v_jammer=dv.v_jammer,
            P_bs_max=self.config.P_bs_max,
            P_j_max=self.config.P_j_max,
            sigma2=self.config.sigma2,
            seed=self.config.seed or 0,
            jammer_mode=jammer_mode,
            jammer_mix_alpha=jammer_mix,
            jammer_power_factor=jp_factor,
            include_direct_links=include_direct_links,
            eta_ris=self.config.eta_ris,
            ris_alignment_alpha=ris_align,
            M_bs=M_bs,
        )

        # Sensing
        mode = self.config.sensing_utility_mode
        sense_result = compute_sensing_utility(
            q_uav=dv.q_uav,
            q_vehicles=self.scenario.q_vehicles,
            rcs_list=rcs_list,
            N_tx=self.config.N_tx_sense,
            N_rx=self.config.N_rx_sense,
            L_pilot=self.config.L_pilot,
            noise_power=self.config.noise_power_sense,
            d_ant=self.config.d_ant,
            wavelength=self.config.wavelength,
            seed=self.config.seed or 0,
            mode=mode,
        )

        # Weighted objective with mode-specific normalisation
        u_ref = get_u_ref(mode)
        f = evaluate_weighted_objective(
            alpha, sec_result["R_s_total"],
            sense_result["U_sense_total"],
            U_sense_ref=u_ref,
        )

        constraints = check_constraints(
            phi_rad=dv.phi_rad,
            q_uav=dv.q_uav,
            w_bs=dv.w_bs,
            v_jammer=dv.v_jammer,
            P_bs_max=self.config.P_bs_max,
            P_j_max=self.config.P_j_max,
            v_max=self.config.v_max,
            dt=self.config.dt,
            q_min=self.scenario.q_min,
            q_max=self.scenario.q_max,
        )

        violations = compute_constraint_violations(
            phi_rad=dv.phi_rad,
            q_uav=dv.q_uav,
            w_bs=dv.w_bs,
            v_jammer=dv.v_jammer,
            P_bs_max=self.config.P_bs_max,
            P_j_max=self.config.P_j_max,
            v_max=self.config.v_max,
            dt=self.config.dt,
            q_min=self.scenario.q_min,
            q_max=self.scenario.q_max,
            R_s_total=sec_result["R_s_total"],
            U_sense_total=sense_result["U_sense_total"],
        )

        return {
            "secrecy": sec_result,
            "sensing": sense_result,
            "objective": {
                "alpha": alpha,
                "f": f,
                "R_s_total": sec_result["R_s_total"],
                "U_sense_total": sense_result["U_sense_total"],
                "R_s_norm": sec_result["R_s_total"] / R_S_REF,
                "U_sense_norm": sense_result["U_sense_total"] / u_ref,
                "sensing_utility_mode": mode,
            },
            "constraints": constraints,
            "violations": violations,
        }

    # ── Alpha sweep ─────────────────────────────────────

    def sweep_alpha(
        self,
        dv: DecisionVariables | None = None,
        alpha_range: list[float] | None = None,
    ) -> dict:
        """Sweep alpha with smooth parameter variation — no hard thresholds."""
        if alpha_range is None:
            alpha_range = np.linspace(0.0, 1.0, 11).tolist()

        alphas = []
        Rs_vals = []
        Us_vals = []
        f_vals = []
        Rs_norm_vals = []
        Us_norm_vals = []

        for a in alpha_range:
            dv_a = self._design_alpha_vars(alpha=a, rng_seed=int(a * 1000))
            result = self.evaluate(
                dv_a, jammer_mode="mixed", alpha=a,
            )
            alphas.append(a)
            Rs_vals.append(result["secrecy"]["R_s_total"])
            Us_vals.append(result["sensing"]["U_sense_total"])
            f_vals.append(result["objective"]["f"])
            Rs_norm_vals.append(result["objective"]["R_s_norm"])
            Us_norm_vals.append(result["objective"]["U_sense_norm"])

        return {
            "alpha": np.array(alphas),
            "R_s_total": np.array(Rs_vals),
            "U_sense_total": np.array(Us_vals),
            "f_weighted": np.array(f_vals),
            "R_s_norm": np.array(Rs_norm_vals),
            "U_sense_norm": np.array(Us_norm_vals),
        }

    # ── Monte Carlo secrecy ─────────────────────────────

    def run_monte_carlo_secrecy(
        self,
        num_realizations: int = 500,
        jammer_mode: str = "mixed",
        include_direct_links: bool = False,
        ris_phase_noise_std: float = 1.5,
    ) -> dict:
        """Run MC secrecy statistics with per-realization randomisation.

        Each realisation draws a random ris_alignment_alpha ~ Unif(0, 1)
        and a random jammer_mode (80% mixed, 20% blast) for realistic spread.
        """
        dv = self.random_decision_vars()
        Rs_all = np.zeros(num_realizations)
        Ru_all = np.zeros(num_realizations)
        Re_all = np.zeros(num_realizations)
        SINRu_all = []
        SINRe_all = []

        for r in range(num_realizations):
            rng = np.random.RandomState(r)
            rand_alpha = rng.uniform(0.0, 1.0)
            rand_jammer = "blast" if rng.rand() < 0.2 else "mixed"

            sec = compute_secrecy_rate(
                q_bs=self.scenario.q_bs,
                q_user=self.scenario.q_user,
                q_eves=self.scenario.q_eves,
                q_jammer=self.scenario.q_jammer,
                N_ris=self.config.N_ris, N_j=self.config.N_j,
                Phi=None,
                q_uav=dv.q_uav, w_bs=dv.w_bs, v_jammer=dv.v_jammer,
                P_bs_max=self.config.P_bs_max,
                P_j_max=self.config.P_j_max,
                sigma2=self.config.sigma2,
                seed=r,
                jammer_mode=rand_jammer,
                jammer_mix_alpha=rand_alpha,
                jammer_power_factor=rand_alpha,
                include_direct_links=include_direct_links,
                eta_ris=self.config.eta_ris,
                ris_phase_noise_std=ris_phase_noise_std,
                ris_alignment_alpha=rand_alpha,
                M_bs=self.config.M_bs,
            )
            Rs_all[r] = sec["R_s_total"]
            Ru_all[r] = float(np.mean(sec["R_user"]))
            Re_all[r] = float(np.max(sec["R_eve_max"]))
            SINRu_all.extend(sec["SINR_user"].tolist())
            SINRe_all.extend(sec["SINR_eve"].flatten().tolist())

        prob_gt_0 = float(np.mean(Rs_all > 0.01))
        sorted_rs = np.sort(Rs_all)
        cdf_probs = np.linspace(0.0, 1.0, num_realizations)

        return {
            "prob_rs_gt_0": prob_gt_0,
            "avg_secrecy": float(np.mean(Rs_all)),
            "median_secrecy": float(np.median(Rs_all)),
            "std_secrecy": float(np.std(Rs_all)),
            "min_secrecy": float(np.min(Rs_all)),
            "max_secrecy": float(np.max(Rs_all)),
            "secrecy_cdf_vals": sorted_rs,
            "secrecy_cdf_probs": cdf_probs,
            "avg_user_sinr": float(np.mean(SINRu_all)),
            "avg_eve_sinr": float(np.mean(SINRe_all)),
            "avg_user_rate": float(np.mean(Ru_all)),
            "avg_eve_rate": float(np.mean(Re_all)),
            "all_Rs": Rs_all,
        }

    # ── Sanity experiments ─────────────────────────────

    def run_secrecy_vs_user_distance(
        self, num_points: int = 8, num_trials: int = 3,
    ) -> dict:
        distances = np.linspace(50.0, 350.0, num_points)
        Rs_vals = []
        Ru_vals = []
        Re_vals = []

        for d in distances:
            scenario = default_scenario()
            scenario.q_user = np.array([d, 0.0, 1.5])
            env = OptimizationProblemEnv(self.config)
            env.scenario = scenario
            slot_Rs = []
            slot_Ru = []
            slot_Re = []
            for _ in range(num_trials):
                dv = env.random_decision_vars()
                sec = compute_secrecy_rate(
                    q_bs=scenario.q_bs, q_user=scenario.q_user,
                    q_eves=scenario.q_eves, q_jammer=scenario.q_jammer,
                    N_ris=self.config.N_ris, N_j=self.config.N_j,
                    Phi=None, q_uav=dv.q_uav, w_bs=dv.w_bs,
                    v_jammer=dv.v_jammer,
                    P_bs_max=self.config.P_bs_max,
                    P_j_max=self.config.P_j_max,
                    sigma2=self.config.sigma2, seed=0,
                    jammer_mode="mixed", jammer_mix_alpha=0.85,
                    jammer_power_factor=0.85,
                    eta_ris=self.config.eta_ris,
                    ris_alignment_alpha=0.85,
                    M_bs=self.config.M_bs,
                )
                slot_Rs.append(sec["R_s_total"])
                slot_Ru.append(float(np.mean(sec["R_user"])))
                slot_Re.append(float(np.max(sec["R_eve_max"])))
            Rs_vals.append(float(np.mean(slot_Rs)))
            Ru_vals.append(float(np.mean(slot_Ru)))
            Re_vals.append(float(np.mean(slot_Re)))

        return {
            "user_dist": distances,
            "R_s": np.array(Rs_vals),
            "R_user": np.array(Ru_vals),
            "R_eve_max": np.array(Re_vals),
        }

    def run_secrecy_vs_jammer_power(
        self, num_points: int = 8, num_trials: int = 3,
    ) -> dict:
        pj_vals = np.logspace(-3, 0, num_points)
        Rs_vals = []

        for pj in pj_vals:
            cfg = OptimizationConfig(
                N_ris=self.config.N_ris, N_j=self.config.N_j,
                P_bs_max=self.config.P_bs_max, P_j_max=pj,
                sigma2=self.config.sigma2,
                seed=self.config.seed,
            )
            slot_Rs = []
            for _ in range(num_trials):
                env = OptimizationProblemEnv(cfg)
                dv = env.random_decision_vars()
                sec = compute_secrecy_rate(
                    q_bs=env.scenario.q_bs, q_user=env.scenario.q_user,
                    q_eves=env.scenario.q_eves,
                    q_jammer=env.scenario.q_jammer,
                    N_ris=cfg.N_ris, N_j=cfg.N_j,
                    Phi=None, q_uav=dv.q_uav, w_bs=dv.w_bs,
                    v_jammer=dv.v_jammer,
                    P_bs_max=cfg.P_bs_max,
                    P_j_max=cfg.P_j_max,
                    sigma2=cfg.sigma2, seed=0,
                    jammer_mode="mixed", jammer_mix_alpha=0.85,
                    jammer_power_factor=0.85,
                    eta_ris=self.config.eta_ris,
                    ris_alignment_alpha=0.85,
                    M_bs=self.config.M_bs,
                )
                slot_Rs.append(sec["R_s_total"])
            Rs_vals.append(float(np.mean(slot_Rs)))

        return {"P_j": pj_vals, "R_s": np.array(Rs_vals)}

    def run_secrecy_vs_eve_distance(
        self, num_points: int = 8, num_trials: int = 3,
    ) -> dict:
        offsets = np.linspace(20.0, 250.0, num_points)
        Rs_vals = []
        Ru_vals = []
        Re_vals = []

        for off in offsets:
            scenario = default_scenario()
            scenario.q_eves = np.array([
                [200.0, off, 1.5],
                [200.0, -off, 1.5],
            ])
            slot_Rs = []
            slot_Ru = []
            slot_Re = []
            for _ in range(num_trials):
                env = OptimizationProblemEnv(self.config)
                env.scenario = scenario
                dv = env.random_decision_vars()
                sec = compute_secrecy_rate(
                    q_bs=scenario.q_bs, q_user=scenario.q_user,
                    q_eves=scenario.q_eves, q_jammer=scenario.q_jammer,
                    N_ris=self.config.N_ris, N_j=self.config.N_j,
                    Phi=None, q_uav=dv.q_uav, w_bs=dv.w_bs,
                    v_jammer=dv.v_jammer,
                    P_bs_max=self.config.P_bs_max,
                    P_j_max=self.config.P_j_max,
                    sigma2=self.config.sigma2, seed=0,
                    jammer_mode="mixed", jammer_mix_alpha=0.85,
                    jammer_power_factor=0.85,
                    eta_ris=self.config.eta_ris,
                    ris_alignment_alpha=0.85,
                    M_bs=self.config.M_bs,
                )
                slot_Rs.append(sec["R_s_total"])
                slot_Ru.append(float(np.mean(sec["R_user"])))
                slot_Re.append(float(np.max(sec["R_eve_max"])))
            Rs_vals.append(float(np.mean(slot_Rs)))
            Ru_vals.append(float(np.mean(slot_Ru)))
            Re_vals.append(float(np.mean(slot_Re)))

        return {
            "eve_offset": offsets,
            "R_s": np.array(Rs_vals),
            "R_user": np.array(Ru_vals),
            "R_eve_max": np.array(Re_vals),
        }

    def compute_channel_debug(self) -> dict:
        dv = self.random_decision_vars()
        sec = compute_secrecy_rate(
            q_bs=self.scenario.q_bs,
            q_user=self.scenario.q_user,
            q_eves=self.scenario.q_eves,
            q_jammer=self.scenario.q_jammer,
            N_ris=self.config.N_ris,
            N_j=self.config.N_j,
            Phi=None,
            q_uav=dv.q_uav,
            w_bs=dv.w_bs,
            v_jammer=dv.v_jammer,
            P_bs_max=self.config.P_bs_max,
            P_j_max=self.config.P_j_max,
            sigma2=self.config.sigma2,
            seed=self.config.seed or 0,
            jammer_mode="mixed", jammer_mix_alpha=0.85,
            jammer_power_factor=0.85,
            eta_ris=self.config.eta_ris,
            ris_alignment_alpha=0.85,
            M_bs=self.config.M_bs,
        )

        debug = {
            "decision_vars": {
                "phi_rad_mean": float(np.mean(dv.phi_rad)),
                "phi_rad_std": float(np.std(dv.phi_rad)),
                "q_uav": dv.q_uav.tolist(),
                "w_bs_power": [float(np.linalg.norm(w)**2) for w in dv.w_bs],
                "v_jammer_power": [
                    float(np.linalg.norm(v)**2) for v in dv.v_jammer
                ],
            },
            "secrecy": {
                "R_s_total": sec["R_s_total"],
                "R_s_per_slot": sec["R_s_per_slot"].tolist(),
                "R_user": sec["R_user"].tolist(),
                "R_eve_max": sec["R_eve_max"].tolist(),
                "SINR_user": sec["SINR_user"].tolist(),
                "SINR_eve": sec["SINR_eve"].tolist(),
                "gain_user_avg": sec["gain_user_avg"],
                "gain_eve_avg": sec["gain_eve_avg"],
            },
            "path_losses": {
                name: float(_compute_pl(pos_a, pos_b))
                for name, pos_a, pos_b in _path_loss_table(self.scenario)
            },
        }
        return debug


def _compute_pl(pos_a, pos_b):
    d = float(np.linalg.norm(np.array(pos_a) - np.array(pos_b)))
    return 1.0 / (d**2) if d > 0 else 1.0


def _path_loss_table(scenario):
    entries = []
    entries.append(("BS-UAV_mid", scenario.q_bs, (scenario.q_uav_start + scenario.q_uav_end) / 2))
    entries.append(("UAVmid-User", (scenario.q_uav_start + scenario.q_uav_end) / 2, scenario.q_user))
    entries.append(("UAVmid-Eve0", (scenario.q_uav_start + scenario.q_uav_end) / 2, scenario.q_eves[0]))
    entries.append(("UAVmid-Eve1", (scenario.q_uav_start + scenario.q_uav_end) / 2, scenario.q_eves[1]))
    entries.append(("UAVmid-Eve2", (scenario.q_uav_start + scenario.q_uav_end) / 2, scenario.q_eves[2]))
    entries.append(("Jammer-User", scenario.q_jammer, scenario.q_user))
    entries.append(("Jammer-Eve0", scenario.q_jammer, scenario.q_eves[0]))
    entries.append(("Jammer-Eve1", scenario.q_jammer, scenario.q_eves[1]))
    return entries
