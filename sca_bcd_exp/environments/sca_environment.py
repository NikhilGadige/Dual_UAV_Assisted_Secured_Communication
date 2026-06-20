from __future__ import annotations

from dataclasses import asdict

import numpy as np

from core.channel import generate_fading
from core.environment import UAVEnvironment, compute_distance
from hppp_training_utils import build_hppp_env_config
from sca_bcd_exp.configs import SCABCDConfig
from sca_bcd_exp.optimization.secrecy_optimizer import SolutionState
from vehicle_receiver_exp.vehicle_models import VehicleUAVEnvironment


class SCABCDEnvironment:
    def __init__(self, config: SCABCDConfig):
        self.config = config
        self.base_env = None
        self.user_positions = None
        self.destination_position = np.array([0.0, 0.0], dtype=float)
        self.eve_positions = None
        self.relay_start = None
        self.relay_end = None
        self.jammer_start = None
        self.jammer_end = None
        self.fading = {}

    def reset(self) -> SolutionState:
        env_cfg = build_hppp_env_config(
            seed=self.config.seed,
            fading_model=self.config.channel_model,
            rician_k=self.config.rician_k,
            use_los_model=False,
            use_multiple_eves=self.config.use_multiple_eves,
            eve_density_lambda=self.config.eve_density_lambda,
        )
        env_cfg.max_steps = self.config.horizon
        env_cfg.area_size = self.config.area_size
        env_cfg.relay_altitude = self.config.relay_altitude
        env_cfg.jammer_altitude = self.config.jammer_altitude
        env_cfg.max_speed = self.config.max_speed
        env_cfg.dt = self.config.slot_duration
        env_cfg.bandwidth = self.config.bandwidth
        env_cfg.noise_psd = self.config.noise_psd
        env_cfg.alpha = self.config.alpha
        env_cfg.beta0 = self.config.beta0

        if self.config.use_vehicle_receiver:
            self.base_env = VehicleUAVEnvironment(
                env_cfg,
                mobility_mode=self.config.vehicle_mobility_mode,
                vehicle_max_speed=self.config.vehicle_max_speed,
            )
        else:
            self.base_env = UAVEnvironment(env_cfg)
        self.base_env.reset()

        self.eve_positions = self.base_env.eve_positions.copy()
        self.user_positions = self._build_user_trajectory()
        self.relay_start = self.base_env.relay_position[:2].copy()
        self.jammer_start = self.base_env.jammer_position[:2].copy()
        self.relay_end = self._clip_radius(np.array([self.config.max_flight_radius, -0.2 * self.config.max_flight_radius]))
        self.jammer_end = self._clip_radius(np.array([self.config.max_flight_radius, 0.2 * self.config.max_flight_radius]))
        self.fading = self._sample_fading()
        return self.initial_solution()

    def initial_solution(self) -> SolutionState:
        return SolutionState(
            relay_trajectory=self._linear_path(self.relay_start, self.relay_end),
            jammer_trajectory=self._linear_path(self.jammer_start, self.jammer_end),
            source_power=np.full(self.config.horizon, self.config.avg_user_power_budget, dtype=float),
            relay_power=np.full(self.config.horizon, self.config.avg_relay_power_budget, dtype=float),
            jammer_power=np.full(self.config.horizon, self.config.avg_jammer_power_budget, dtype=float),
            alpha_trajectory=np.full(self.config.horizon, 0.5, dtype=float),
        )

    def _build_user_trajectory(self) -> np.ndarray:
        positions = [self.base_env.user_position[:2].copy()]
        if isinstance(self.base_env, VehicleUAVEnvironment) and self.base_env.vehicle is not None:
            for _ in range(1, self.config.horizon):
                self.base_env.vehicle.update(dt=self.config.slot_duration, half_area=self.config.half_area)
                positions.append(self.base_env.vehicle.position.copy())
        else:
            positions.extend([positions[0].copy() for _ in range(1, self.config.horizon)])
        return np.asarray(positions, dtype=float)

    def _sample_fading(self) -> dict[str, np.ndarray]:
        model = self.config.channel_model
        k = self.config.rician_k
        horizon = self.config.horizon
        num_eves = self.eve_positions.shape[0]
        return {
            "SR": np.array([generate_fading(model, k) for _ in range(horizon)], dtype=float),
            "RD": np.array([generate_fading(model, k) for _ in range(horizon)], dtype=float),
            "SE": np.array([[generate_fading(model, k) for _ in range(num_eves)] for _ in range(horizon)], dtype=float),
            "RE": np.array([[generate_fading(model, k) for _ in range(num_eves)] for _ in range(horizon)], dtype=float),
            "JE": np.array([[generate_fading(model, k) for _ in range(num_eves)] for _ in range(horizon)], dtype=float),
        }

    def _linear_path(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        grid = np.linspace(0.0, 1.0, self.config.horizon)
        return np.stack([(1.0 - t) * start + t * end for t in grid], axis=0)

    def _clip_radius(self, point: np.ndarray) -> np.ndarray:
        point = np.clip(np.asarray(point, dtype=float), -self.config.half_area, self.config.half_area)
        norm = float(np.linalg.norm(point))
        if norm > self.config.max_flight_radius > 0.0:
            point = point * (self.config.max_flight_radius / norm)
        return point

    def _gain(self, distance_sq: float, fading: float) -> float:
        return self.config.beta0 * fading * (max(distance_sq, 1e-6) ** (-0.5 * self.config.alpha))

    def _gain_grad_from_sq(self, vector: np.ndarray, distance_sq: float, fading: float) -> np.ndarray:
        power = -0.5 * self.config.alpha
        coeff = self.config.beta0 * fading * power * (max(distance_sq, 1e-6) ** (power - 1.0)) * 2.0
        return coeff * vector

    def _ground_to_ground_gain(self, tx_xy: np.ndarray, rx_xy: np.ndarray, fading: float, shrink: float = 0.0) -> float:
        distance = max(np.linalg.norm(tx_xy - rx_xy) - shrink, 1e-3)
        return self.config.beta0 * fading * (distance ** (-self.config.alpha))

    def _relay_eve_gain(self, q_relay: np.ndarray, eve_xy: np.ndarray, fading: float) -> tuple[float, np.ndarray]:
        diff = q_relay - eve_xy
        radius = float(np.linalg.norm(diff))
        reduced = max(radius - self.config.eve_uncertainty_radius, 1e-3)
        distance_sq = reduced ** 2 + self.config.relay_altitude ** 2
        gain = self._gain(distance_sq, fading)
        if radius <= 1e-6 or radius <= self.config.eve_uncertainty_radius:
            return gain, np.zeros(2, dtype=float)
        grad_sq = 2.0 * reduced * (diff / radius)
        power = -0.5 * self.config.alpha
        coeff = self.config.beta0 * fading * power * (distance_sq ** (power - 1.0))
        return gain, coeff * grad_sq

    def _jammer_eve_gain(self, q_jammer: np.ndarray, eve_xy: np.ndarray, fading: float) -> tuple[float, np.ndarray]:
        diff = q_jammer - eve_xy
        radius = max(float(np.linalg.norm(diff)), 1e-6)
        inflated = radius + self.config.eve_uncertainty_radius
        distance_sq = inflated ** 2 + self.config.jammer_altitude ** 2
        gain = self._gain(distance_sq, fading)
        grad_sq = 2.0 * inflated * (diff / radius)
        power = -0.5 * self.config.alpha
        coeff = self.config.beta0 * fading * power * (distance_sq ** (power - 1.0))
        return gain, coeff * grad_sq

    def _slot_terms(self, solution: SolutionState, m: int) -> dict:
        src_xy = self.user_positions[m]
        relay_xy = solution.relay_trajectory[m]
        jammer_xy = solution.jammer_trajectory[m]
        src_pow = float(solution.source_power[m])
        relay_pow = float(solution.relay_power[m])
        jam_pow = float(solution.jammer_power[m])

        sr_vec = relay_xy - src_xy
        rd_vec = relay_xy - self.destination_position
        sr_sq = float(np.dot(sr_vec, sr_vec) + self.config.relay_altitude ** 2)
        rd_sq = float(np.dot(rd_vec, rd_vec) + self.config.relay_altitude ** 2)
        h_sr = self._gain(sr_sq, self.fading["SR"][m])
        h_rd = self._gain(rd_sq, self.fading["RD"][m])

        slot_factor = 0.5 * solution.alpha_trajectory[m] * self.config.slot_duration
        gamma_sr = src_pow * h_sr / self.config.noise_power
        gamma_rd = relay_pow * h_rd / self.config.noise_power
        r_sr_raw = np.log2(1.0 + gamma_sr)
        r_rd_raw = np.log2(1.0 + gamma_rd)
        r_sr = slot_factor * r_sr_raw
        r_rd = slot_factor * r_rd_raw
        r_leg = min(r_sr, r_rd)

        eve_sum = 0.0
        worst_terms = []
        nearest_dist = 0.0
        if self.eve_positions.shape[0] > 0:
            nearest_dist = float(np.min(np.linalg.norm(self.eve_positions - src_xy, axis=1)))
        for idx, eve_xy in enumerate(self.eve_positions):
            g_se = self._ground_to_ground_gain(src_xy, eve_xy, self.fading["SE"][m, idx], shrink=self.config.eve_uncertainty_radius)
            g_re, grad_g_re = self._relay_eve_gain(relay_xy, eve_xy, self.fading["RE"][m, idx])
            g_je, grad_g_je = self._jammer_eve_gain(jammer_xy, eve_xy, self.fading["JE"][m, idx])
            den = self.config.noise_power + jam_pow * g_je
            frac = (src_pow * g_se + relay_pow * g_re) / den
            eve_sum += frac
            worst_terms.append(
                {
                    "g_se": g_se,
                    "g_re": g_re,
                    "grad_g_re": grad_g_re,
                    "g_je": g_je,
                    "grad_g_je": grad_g_je,
                    "frac": frac,
                    "den": den,
                }
            )

        r_wir_raw = np.log2(1.0 + eve_sum)
        r_wir = slot_factor * r_wir_raw
        secrecy_lb = r_leg - r_wir
        secrecy_clip = max(secrecy_lb, 0.0)
        return {
            "h_sr": h_sr,
            "h_rd": h_rd,
            "sr_vec": sr_vec,
            "rd_vec": rd_vec,
            "sr_sq": sr_sq,
            "rd_sq": rd_sq,
            "gamma_sr": gamma_sr,
            "gamma_rd": gamma_rd,
            "r_sr": r_sr,
            "r_rd": r_rd,
            "r_leg": r_leg,
            "r_wir": r_wir,
            "r_sr_raw": r_sr_raw,
            "r_rd_raw": r_rd_raw,
            "r_wir_raw": r_wir_raw,
            "slot_factor": slot_factor,
            "secrecy_lb": secrecy_lb,
            "secrecy_clip": secrecy_clip,
            "eve_sum": eve_sum,
            "eve_terms": worst_terms,
            "nearest_eve_distance": nearest_dist,
        }

    def evaluate_solution(self, solution: SolutionState) -> dict:
        raw_secrecy = []
        clipped_secrecy = []
        legit_rates = []
        wiretap_rates = []
        nearest_eve_distances = []

        for m in range(self.config.horizon):
            terms = self._slot_terms(solution, m)
            raw_secrecy.append(float(terms["secrecy_lb"]))
            clipped_secrecy.append(float(terms["secrecy_clip"]))
            legit_rates.append(float(terms["r_leg"]))
            wiretap_rates.append(float(terms["r_wir"]))
            nearest_eve_distances.append(float(terms["nearest_eve_distance"]))

        raw_objective = float(np.mean(raw_secrecy))
        average_secrecy = float(np.mean(clipped_secrecy))
        average_wiretap = float(np.mean(wiretap_rates))
        return {
            "objective": raw_objective,
            "raw_objective": raw_objective,
            "average_secrecy_rate": average_secrecy,
            "average_legit_rate": float(np.mean(legit_rates)),
            "average_wiretap_rate_upper": average_wiretap,
            "average_num_eves": float(self.eve_positions.shape[0]),
            "average_nearest_eve_distance": float(np.mean(nearest_eve_distances)) if nearest_eve_distances else 0.0,
            "average_max_eve_capacity": average_wiretap,
            "slot_raw_secrecy": np.asarray(raw_secrecy, dtype=float),
            "slot_clipped_secrecy": np.asarray(clipped_secrecy, dtype=float),
            "scenario": {
                "config": asdict(self.config),
                "num_eves": int(self.eve_positions.shape[0]),
                "relay_start": self.relay_start.tolist(),
                "relay_end": self.relay_end.tolist(),
                "jammer_start": self.jammer_start.tolist(),
                "jammer_end": self.jammer_end.tolist(),
                "destination": self.destination_position.tolist(),
            },
        }

    def power_gradient(self, solution: SolutionState) -> np.ndarray:
        grad_src = np.zeros(self.config.horizon, dtype=float)
        grad_rel = np.zeros(self.config.horizon, dtype=float)
        grad_jam = np.zeros(self.config.horizon, dtype=float)

        for m in range(self.config.horizon):
            terms = self._slot_terms(solution, m)
            sf = terms["slot_factor"]
            common = sf / (np.log(2.0) * (1.0 + terms["eve_sum"]))

            if terms["r_sr"] <= terms["r_rd"]:
                grad_src[m] += sf * (terms["h_sr"] / self.config.noise_power) / (np.log(2.0) * (1.0 + terms["gamma_sr"]))
            if terms["r_rd"] <= terms["r_sr"]:
                grad_rel[m] += sf * (terms["h_rd"] / self.config.noise_power) / (np.log(2.0) * (1.0 + terms["gamma_rd"]))

            for eve_terms in terms["eve_terms"]:
                den = eve_terms["den"]
                num = solution.source_power[m] * eve_terms["g_se"] + solution.relay_power[m] * eve_terms["g_re"]
                grad_src[m] -= common * (eve_terms["g_se"] / den)
                grad_rel[m] -= common * (eve_terms["g_re"] / den)
                grad_jam[m] += common * (num * eve_terms["g_je"] / (den ** 2))

        return np.concatenate([grad_src, grad_rel, grad_jam]) / self.config.horizon

    def relay_gradient(self, solution: SolutionState) -> np.ndarray:
        grad = np.zeros((self.config.horizon, 2), dtype=float)
        for m in range(self.config.horizon):
            terms = self._slot_terms(solution, m)
            sf = terms["slot_factor"]
            grad_h_sr = self._gain_grad_from_sq(terms["sr_vec"], terms["sr_sq"], self.fading["SR"][m])
            grad_h_rd = self._gain_grad_from_sq(terms["rd_vec"], terms["rd_sq"], self.fading["RD"][m])
            if terms["r_sr"] <= terms["r_rd"]:
                grad[m] += sf * (solution.source_power[m] / self.config.noise_power) * grad_h_sr / (np.log(2.0) * (1.0 + terms["gamma_sr"]))
            if terms["r_rd"] <= terms["r_sr"]:
                grad[m] += sf * (solution.relay_power[m] / self.config.noise_power) * grad_h_rd / (np.log(2.0) * (1.0 + terms["gamma_rd"]))

            common = sf / (np.log(2.0) * (1.0 + terms["eve_sum"]))
            for eve_terms in terms["eve_terms"]:
                grad[m] -= common * (solution.relay_power[m] * eve_terms["grad_g_re"] / eve_terms["den"])
        return grad.reshape(-1) / self.config.horizon

    def jammer_gradient(self, solution: SolutionState) -> np.ndarray:
        grad = np.zeros((self.config.horizon, 2), dtype=float)
        for m in range(self.config.horizon):
            terms = self._slot_terms(solution, m)
            common = terms["slot_factor"] / (np.log(2.0) * (1.0 + terms["eve_sum"]))
            for eve_terms in terms["eve_terms"]:
                num = solution.source_power[m] * eve_terms["g_se"] + solution.relay_power[m] * eve_terms["g_re"]
                grad[m] += common * (num * solution.jammer_power[m] * eve_terms["grad_g_je"] / (eve_terms["den"] ** 2))
        return grad.reshape(-1) / self.config.horizon

    def alpha_gradient(self, solution: SolutionState) -> np.ndarray:
        grad = np.zeros(self.config.horizon, dtype=float)
        for m in range(self.config.horizon):
            terms = self._slot_terms(solution, m)
            legitimate_factor = terms["r_sr_raw"] if terms["r_sr"] <= terms["r_rd"] else terms["r_rd_raw"]
            net_factor = legitimate_factor - terms["r_wir_raw"]
            grad[m] = 0.5 * self.config.slot_duration * net_factor
        return grad / self.config.horizon
