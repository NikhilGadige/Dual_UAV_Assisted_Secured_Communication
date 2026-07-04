"""Multi-agent environment for ISAC with MIMO BS — updated 4-agent version.

Adds three pieces that exist elsewhere in the repo but were never wired
into the MADRL environment: Random-Walk user mobility, HPPP-distributed
eavesdroppers, and RIS phase-shift as a real optimized decision variable
(previously a fixed closed-form heuristic).
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from optimization_problem_exp.optimization.problem_formulation import (
    compute_secrecy_rate,
    compute_sensing_utility,
    check_constraints,
    compute_constraint_violations,
    evaluate_weighted_objective,
    generate_mimo_rician_channel,
    compute_bs_ris_channel,
    compute_ris_user_channel,
    compute_ris_eve_channel,
    compute_jammer_user_channel,
    compute_jammer_eve_channel,
    design_ris_phases,
    compute_effective_channel,
    compute_effective_channel_gain,
    get_u_ref,
    R_S_REF,
)
from ris_uav_exp.channels.ris_channel import compute_ris_reflection_matrix
from vehicle_reflection_exp.channels.vehicle_channel import compute_rcs
from mobility_types_exp.mobility_models import RandomWalkMobility
from madrl_updated_exp.configs import EnvConfig, RewardConfig
from madrl_updated_exp.reward import (
    RewardNormalizer,
    compute_reward_original,
    compute_reward_normalized,
    compute_reward_full,
)


N_EVE = 3  # cap on simultaneously-modeled eavesdroppers; actual active
           # count each episode is HPPP-sampled and <= N_EVE (see
           # _sample_hppp_eves). Unused slots are pushed far outside the
           # scene so they contribute ~0 channel gain.
N_VEH = 3

SCENE_MIN = np.array([0.0, -150.0, 0.0])
SCENE_MAX = np.array([400.0, 150.0, 120.0])
FAR_AWAY_EVE = np.array([1.0e5, 1.0e5, 1.5])


class ISACMultiAgentEnv:
    """Multi-agent environment for ISAC — 4 agents.

    Agent 1: bs_beamformer     (action: w_bs)
    Agent 2: uav_trajectory    (action: q_uav increments)
    Agent 3: jammer_beamformer (action: v_jammer)
    Agent 4: ris_phase         (action: RIS phase-shift vector phi, per slot)

    The mobile user follows a Random-Walk mobility model each step, and
    eavesdroppers are re-sampled from a Homogeneous Poisson Point Process
    (HPPP) on every reset().

    All agents share the global weighted objective reward.
    Gymnasium-compatible: reset(), step(), observation_space, action_space.
    """

    def __init__(
        self,
        env_cfg: EnvConfig | None = None,
        reward_cfg: RewardConfig | None = None,
        seed: int = 42,
    ):
        self.cfg = env_cfg or EnvConfig()
        self.r_cfg = reward_cfg or RewardConfig()
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.np_rng = np.random.default_rng(seed)

        self._setup_scenario()
        self._setup_spaces()
        self._load_constants()
        self._init_reward_computer()
        self._bs_projection_log = []
        self.reset()

    # ── Scenario ────────────────────────────────────────────

    def _setup_scenario(self):
        c = self.cfg
        self.q_bs = np.array([0.0, 0.0, 30.0])
        self.q_user_start = np.array([200.0, 0.0, 1.5])
        self.q_user = self.q_user_start.copy()
        self.user_velocity = np.zeros(2)
        self.mobility = RandomWalkMobility()
        self.q_jammer = np.array([100.0, -120.0, 50.0])
        self.q_eves = np.tile(FAR_AWAY_EVE, (N_EVE, 1)).astype(float)
        self.n_active_eves = 0
        self.q_vehicles = np.array([
            [200.0, 80.0, 0.0],
            [250.0, -60.0, 0.0],
            [180.0, -100.0, 0.0],
        ])
        self.vehicle_types = ["car", "truck", "motorcycle"]
        self.rcs_list = [compute_rcs(vt) for vt in self.vehicle_types]
        self.q_min = np.array([0.0, -150.0, 50.0])
        self.q_max = np.array([400.0, 150.0, 120.0])
        self.q_uav_start = np.array([50.0, -50.0, 60.0])
        self.q_uav_end = np.array([200.0, 0.0, 60.0])
        self.N_eve = N_EVE
        self.N_veh = N_VEH

    def _sample_hppp_eves(self) -> np.ndarray:
        """Sample eavesdropper positions from a Homogeneous Poisson Point
        Process over the scene footprint, capped at N_EVE slots.

        Slots beyond the sampled count are pushed far outside the scene
        (FAR_AWAY_EVE) so their channel gain is negligible — this keeps the
        observation/action tensor shapes fixed while still giving each
        episode a genuinely Poisson-distributed number and placement of
        active eavesdroppers, matching the HPPP sampling already used in
        fd_jammer_exp/environments/fd_jammer_env.py.
        """
        c = self.cfg
        area = (SCENE_MAX[0] - SCENE_MIN[0]) * (SCENE_MAX[1] - SCENE_MIN[1])
        n_active = int(self.rng.poisson(c.eve_density_lambda * area))
        n_active = min(n_active, N_EVE)

        q_eves = np.tile(FAR_AWAY_EVE, (N_EVE, 1)).astype(float)
        if n_active > 0:
            xs = self.rng.uniform(SCENE_MIN[0], SCENE_MAX[0], size=n_active)
            ys = self.rng.uniform(SCENE_MIN[1], SCENE_MAX[1], size=n_active)
            q_eves[:n_active] = np.column_stack(
                [xs, ys, np.full(n_active, 1.5)],
            )
        self.n_active_eves = n_active
        return q_eves

    def _load_constants(self):
        self.u_ref = get_u_ref(self.cfg.sensing_utility_mode)

    def _init_reward_computer(self):
        self._norm_R = RewardNormalizer()
        self._norm_U = RewardNormalizer()

    def _step_user_mobility(self):
        """Advance the mobile user by one Random-Walk step.

        RandomWalkMobility.step() reflects at a square boundary centered on
        the origin; our scene is an off-center rectangle
        (SCENE_MIN..SCENE_MAX), so we pass a very large half_area to
        disable its internal reflection and instead clamp/reflect against
        the true scene bounds ourselves, the same way UAV trajectory
        actions are clamped in _apply_uav_trajectory_action.
        """
        c = self.cfg
        xy = self.q_user[:2].copy()
        new_xy, self.user_velocity = self.mobility.step(
            xy, self.user_velocity, dt=c.dt,
            half_area=1.0e6, user_max_speed=c.user_speed_max,
        )
        for d in range(2):
            if new_xy[d] < SCENE_MIN[d]:
                new_xy[d] = SCENE_MIN[d]
                self.user_velocity[d] *= -1.0
            elif new_xy[d] > SCENE_MAX[d]:
                new_xy[d] = SCENE_MAX[d]
                self.user_velocity[d] *= -1.0
        self.q_user[:2] = new_xy

    def _heuristic_ris_phase_hint(self) -> np.ndarray:
        """Closed-form conjugate-alignment phase for slot 0 — given to the
        ris_phase agent as an observation feature so it has a reasonable
        reference point to learn around, rather than optimizing blind."""
        c = self.cfg
        h_BR = compute_bs_ris_channel(self.q_bs, self.q_uav[0], c.N_ris, 0, M_bs=c.M_bs)
        h_RU = compute_ris_user_channel(self.q_uav[0], self.q_user, c.N_ris, 1)
        return design_ris_phases(h_BR, h_RU)

    # ── Spaces ──────────────────────────────────────────────

    def _setup_spaces(self):
        c = self.cfg
        a1_act = 2 * c.N_time * c.M_bs
        if c.beamform_mode == "power_direction":
            a1_act += c.N_time
        a2_act = 3 * c.N_time
        a3_act = 2 * c.N_time * c.N_j
        a4_act = c.N_time * c.N_ris

        self.agent_names = [
            "bs_beamformer", "uav_trajectory", "jammer_beamformer", "ris_phase",
        ]
        ar = c.action_range
        self.action_spaces = {
            "bs_beamformer": spaces.Box(-ar, ar, shape=(a1_act,), dtype=np.float32),
            "uav_trajectory": spaces.Box(-ar, ar, shape=(a2_act,), dtype=np.float32),
            "jammer_beamformer": spaces.Box(-ar, ar, shape=(a3_act,), dtype=np.float32),
            "ris_phase": spaces.Box(-ar, ar, shape=(a4_act,), dtype=np.float32),
        }
        self.observation_spaces = {
            "bs_beamformer": self._make_obs_space_bs(),
            "uav_trajectory": self._make_obs_space_uav_trajectory(),
            "jammer_beamformer": self._make_obs_space_jammer(),
            "ris_phase": self._make_obs_space_ris_phase(),
        }

    def _make_obs_space_bs(self) -> spaces.Box:
        c = self.cfg
        n = (
            2 * c.M_bs          +  # h_eff_user (complex -> 2 real)
            2 * N_EVE * c.M_bs  +  # h_eff_eve per eve
            2 * c.N_time * c.M_bs +  # previous w_bs
            1                    +  # secrecy rate
            1                    +  # sensing utility
            1                    +  # secrecy outage
            1                    +  # constraint violation
            1                       # alpha
        )
        return spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)

    def _make_obs_space_uav_trajectory(self) -> spaces.Box:
        c = self.cfg
        n = (
            3 * N_VEH           +  # vehicle positions (9)
            3 * c.N_time        +  # previous UAV position (15)
            1                    +  # remaining motion budget
            1                    +  # secrecy rate
            1                    +  # sensing utility
            1                    +  # constraint violation
            1                    +  # alpha
            3                    +  # user position
            3                       # eve mean position
        )
        return spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)

    def _make_obs_space_jammer(self) -> spaces.Box:
        c = self.cfg
        n = (
            2 * c.N_time * c.N_j +  # previous v_jammer (real + imag)
            N_EVE                +  # SINR_eve per eve
            1                    +  # jammer power utilization
            1                    +  # secrecy rate
            1                    +  # sensing utility
            1                    +  # constraint violation
            1                       # alpha
        )
        return spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)

    def _make_obs_space_ris_phase(self) -> spaces.Box:
        c = self.cfg
        n = (
            c.N_time * c.N_ris  +  # previous phi (per slot)
            c.N_ris              +  # heuristic alignment phase hint (slot 0)
            1                    +  # secrecy rate
            1                    +  # sensing utility
            1                    +  # constraint violation
            1                       # alpha
        )
        return spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)

    # ── Channel computation helpers ────────────────────────

    def _compute_channels(self, q_uav: np.ndarray, w_bs: np.ndarray, v_jammer: np.ndarray):
        c = self.cfg
        N_time = c.N_time

        sec_result = compute_secrecy_rate(
            q_bs=self.q_bs, q_user=self.q_user, q_eves=self.q_eves,
            q_jammer=self.q_jammer,
            N_ris=c.N_ris, N_j=c.N_j, Phi=None,
            q_uav=q_uav, w_bs=w_bs, v_jammer=v_jammer,
            P_bs_max=c.P_bs_max, P_j_max=c.P_j_max,
            sigma2=c.sigma2, seed=self.seed,
            jammer_mode="mixed", jammer_mix_alpha=c.alpha,
            jammer_power_factor=max(0.01, c.alpha),
            include_direct_links=c.include_direct_links,
            eta_ris=c.eta_ris, ris_alignment_alpha=c.alpha,
            M_bs=c.M_bs,
            phi_override=self.phi_ris,
        )
        sense_result = compute_sensing_utility(
            q_uav=q_uav, q_vehicles=self.q_vehicles,
            rcs_list=self.rcs_list,
            N_tx=c.N_tx_sense, N_rx=c.N_rx_sense,
            L_pilot=c.L_pilot,
            noise_power=c.noise_power_sense,
            d_ant=c.d_ant, wavelength=c.wavelength,
            seed=self.seed, mode=c.sensing_utility_mode,
        )
        return sec_result, sense_result

    def _compute_effective_channels_slot(self, q_uav_slot, seed_offset=0, phi_slot=None):
        c = self.cfg
        h_BR = compute_bs_ris_channel(
            self.q_bs, q_uav_slot, c.N_ris, seed_offset, M_bs=c.M_bs,
        )
        h_RU = compute_ris_user_channel(
            q_uav_slot, self.q_user, c.N_ris, seed_offset + 1,
        )
        if phi_slot is None:
            phi_slot = design_ris_phases(h_BR, h_RU)
        Phi = compute_ris_reflection_matrix(phi_slot)
        h_eff_user = compute_effective_channel(h_RU, Phi, h_BR)

        h_eff_eves = []
        for ke in range(N_EVE):
            h_RE = compute_ris_eve_channel(
                q_uav_slot, self.q_eves[ke], c.N_ris, seed_offset + 2 + ke,
            )
            h_eff_eves.append(compute_effective_channel(h_RE, Phi, h_BR))
        return h_eff_user, h_eff_eves

    # ── Observation builders ───────────────────────────────

    def _norm_pos(self, pos: np.ndarray) -> np.ndarray:
        return (pos - SCENE_MIN) / (SCENE_MAX - SCENE_MIN + 1e-30) * 2 - 1

    def _maybe_clip_obs(self, obs: np.ndarray) -> np.ndarray:
        clip = self.r_cfg.obs_clip
        if clip > 0:
            return np.clip(obs, -clip, clip).astype(np.float32)
        return obs

    def _build_obs_bs(self, q_uav, w_bs, v_jammer, sec, sense, cons_viol) -> np.ndarray:
        c = self.cfg
        parts = []

        h_eff_user_avg = np.zeros(c.M_bs, dtype=complex)
        for n in range(min(1, c.N_time)):
            h_eff_user, _ = self._compute_effective_channels_slot(
                q_uav[n], n * 10, phi_slot=self.phi_ris[n],
            )
            h_eff_user_avg = h_eff_user
        parts.append(np.real(h_eff_user_avg).astype(np.float32))
        parts.append(np.imag(h_eff_user_avg).astype(np.float32))

        _, h_eff_eves = self._compute_effective_channels_slot(
            q_uav[0], 0, phi_slot=self.phi_ris[0],
        )
        for ke in range(N_EVE):
            parts.append(np.real(h_eff_eves[ke]).astype(np.float32))
            parts.append(np.imag(h_eff_eves[ke]).astype(np.float32))

        parts.append(np.real(w_bs).ravel().astype(np.float32))
        parts.append(np.imag(w_bs).ravel().astype(np.float32))

        parts.append(np.array([float(sec["R_s_total"])], dtype=np.float32))
        parts.append(np.array([float(sense["U_sense_total"])], dtype=np.float32))
        outage = float(np.mean(sec["R_s_per_slot"]) < self.r_cfg.secrecy_outage_threshold)
        parts.append(np.array([outage], dtype=np.float32))
        parts.append(np.array([float(cons_viol)], dtype=np.float32))
        parts.append(np.array([c.alpha], dtype=np.float32))

        return self._maybe_clip_obs(np.concatenate(parts).astype(np.float32))

    def _build_obs_uav_trajectory(self, q_uav, w_bs, v_jammer, sec, sense, cons_viol) -> np.ndarray:
        c = self.cfg
        parts = []

        # Vehicle positions (normalized to scene bounds)
        veh_flat = self.q_vehicles.ravel()
        parts.append(veh_flat.astype(np.float32))

        # Previous UAV position
        parts.append(q_uav.ravel().astype(np.float32))

        # Remaining motion budget
        max_total_dist = c.v_max * c.dt * (c.N_time - 1)
        if max_total_dist > 0 and c.N_time > 1:
            used = sum(float(np.linalg.norm(q_uav[n] - q_uav[n-1])) for n in range(1, c.N_time))
            remaining = max(0.0, max_total_dist - used) / max_total_dist
        else:
            remaining = 1.0
        parts.append(np.array([remaining], dtype=np.float32))

        # Scalar metrics
        parts.append(np.array([float(sec["R_s_total"])], dtype=np.float32))
        parts.append(np.array([float(sense["U_sense_total"])], dtype=np.float32))
        parts.append(np.array([float(cons_viol)], dtype=np.float32))
        parts.append(np.array([c.alpha], dtype=np.float32))

        # Scene context: user position + mean position of active eves only
        # (inactive HPPP slots sit at FAR_AWAY_EVE and would otherwise
        # dominate the mean).
        parts.append(self.q_user.ravel().astype(np.float32))
        if self.n_active_eves > 0:
            eve_mean = np.mean(self.q_eves[:self.n_active_eves], axis=0)
        else:
            eve_mean = np.zeros(3)
        parts.append(eve_mean.astype(np.float32))

        return self._maybe_clip_obs(np.concatenate(parts).astype(np.float32))

    def _build_obs_jammer(self, q_uav, w_bs, v_jammer, sec, sense, cons_viol) -> np.ndarray:
        c = self.cfg
        parts = []

        # Previous v_jammer (real + imag)
        parts.append(np.real(v_jammer).ravel().astype(np.float32))
        parts.append(np.imag(v_jammer).ravel().astype(np.float32))

        # SINR_eve per eve (average across time slots)
        sinr_eve = sec["SINR_eve"]
        sinr_eve_avg = np.mean(sinr_eve, axis=0)
        parts.append(sinr_eve_avg.astype(np.float32))

        # Jammer power utilization
        j_powers = np.array([float(np.linalg.norm(v_jammer[n]) ** 2) for n in range(c.N_time)])
        avg_j_power = float(np.mean(j_powers))
        j_power_util = avg_j_power / max(c.P_j_max, 1e-30)
        parts.append(np.array([j_power_util], dtype=np.float32))

        # Scalar metrics
        parts.append(np.array([float(sec["R_s_total"])], dtype=np.float32))
        parts.append(np.array([float(sense["U_sense_total"])], dtype=np.float32))
        parts.append(np.array([float(cons_viol)], dtype=np.float32))
        parts.append(np.array([c.alpha], dtype=np.float32))

        return self._maybe_clip_obs(np.concatenate(parts).astype(np.float32))

    def _build_obs_ris_phase(self, phi_ris, sec, sense, cons_viol) -> np.ndarray:
        c = self.cfg
        parts = []

        parts.append(phi_ris.ravel().astype(np.float32))
        parts.append(self._heuristic_ris_phase_hint().astype(np.float32))

        parts.append(np.array([float(sec["R_s_total"])], dtype=np.float32))
        parts.append(np.array([float(sense["U_sense_total"])], dtype=np.float32))
        parts.append(np.array([float(cons_viol)], dtype=np.float32))
        parts.append(np.array([c.alpha], dtype=np.float32))

        return self._maybe_clip_obs(np.concatenate(parts).astype(np.float32))

    # ── Action application ────────────────────────────────

    def _apply_bs_action(self, action_bs: np.ndarray, prev_w_bs: np.ndarray) -> np.ndarray:
        c = self.cfg
        if c.beamform_mode == "reim":
            return self._apply_bs_reim(action_bs)
        elif c.beamform_mode == "direction":
            return self._apply_bs_direction(action_bs)
        elif c.beamform_mode == "power_direction":
            return self._apply_bs_power_direction(action_bs)
        else:
            raise ValueError(f"Unknown beamform_mode: {c.beamform_mode}")

    def _log_bs_projection(self, w_raw: np.ndarray, w_actual: np.ndarray):
        c = self.cfg
        for n in range(c.N_time):
            pre_power = float(np.linalg.norm(w_raw[n]) ** 2)
            post_power = float(np.linalg.norm(w_actual[n]) ** 2)
            dist = float(np.linalg.norm(w_raw[n] - w_actual[n]))
            self._bs_projection_log.append({
                "slot": n,
                "pre_projection_power": pre_power,
                "post_projection_power": post_power,
                "projection_distance": dist,
            })

    def _apply_bs_reim(self, action_bs: np.ndarray) -> np.ndarray:
        c = self.cfg
        half = c.N_time * c.M_bs
        re = action_bs[:half].reshape(c.N_time, c.M_bs).astype(float)
        im = action_bs[half:].reshape(c.N_time, c.M_bs).astype(float)
        w_raw = re + 1j * im
        w_out = w_raw.copy()
        for n in range(c.N_time):
            power = float(np.linalg.norm(w_out[n]) ** 2)
            if power > c.P_bs_max:
                w_out[n] = w_out[n] * np.sqrt(c.P_bs_max) / max(np.sqrt(power), 1e-30)
        self._log_bs_projection(w_raw, w_out)
        return w_out

    def _apply_bs_direction(self, action_bs: np.ndarray) -> np.ndarray:
        c = self.cfg
        half = c.N_time * c.M_bs
        re = action_bs[:half].reshape(c.N_time, c.M_bs).astype(float)
        im = action_bs[half:].reshape(c.N_time, c.M_bs).astype(float)
        u = re + 1j * im
        w_out = np.zeros_like(u)
        for n in range(c.N_time):
            norm_u = float(np.linalg.norm(u[n]))
            if norm_u < 1e-30:
                w_out[n] = np.sqrt(c.P_bs_max / c.M_bs) * np.ones(c.M_bs, dtype=complex)
            else:
                w_out[n] = np.sqrt(c.P_bs_max) * u[n] / norm_u
        self._log_bs_projection(u, w_out)
        return w_out

    def _apply_bs_power_direction(self, action_bs: np.ndarray) -> np.ndarray:
        c = self.cfg
        ar = c.action_range
        N_t = c.N_time
        M = c.M_bs
        p_raw = action_bs[:N_t].astype(float)
        p = (p_raw + ar) / (2.0 * ar)  # map [-ar, ar] to [0, 1]
        p = np.clip(p, 0.0, 1.0)
        half_u = N_t * M
        re = action_bs[N_t:N_t + half_u].reshape(N_t, M).astype(float)
        im = action_bs[N_t + half_u:].reshape(N_t, M).astype(float)
        u = re + 1j * im
        w_out = np.zeros_like(u)
        for n in range(N_t):
            norm_u = float(np.linalg.norm(u[n]))
            if norm_u < 1e-30:
                w_out[n] = np.sqrt(p[n] * c.P_bs_max / M) * np.ones(M, dtype=complex)
            else:
                w_out[n] = np.sqrt(p[n] * c.P_bs_max) * u[n] / norm_u
        self._log_bs_projection(u, w_out)
        return w_out

    def get_bs_projection_stats(self) -> list[dict]:
        return self._bs_projection_log

    def clear_bs_projection_log(self):
        self._bs_projection_log = []

    def _apply_uav_trajectory_action(self, action_uav: np.ndarray, prev_q: np.ndarray) -> np.ndarray:
        c = self.cfg
        N_t = c.N_time

        traj_delta = action_uav.reshape(N_t, 3).astype(float)
        max_step = c.v_max * c.dt
        traj_delta = np.clip(traj_delta, -1.0, 1.0) * max_step * 0.5

        new_q = prev_q + traj_delta
        for d in range(3):
            new_q[:, d] = np.clip(new_q[:, d], self.q_min[d], self.q_max[d])
        for n in range(1, N_t):
            seg = new_q[n] - new_q[n - 1]
            dist = float(np.linalg.norm(seg))
            max_dist = c.v_max * c.dt
            if dist > max_dist:
                new_q[n] = new_q[n - 1] + seg * max_dist / max(dist, 1e-30)

        return new_q

    def _apply_jammer_action(self, action_jammer: np.ndarray) -> np.ndarray:
        c = self.cfg
        N_t = c.N_time
        half = N_t * c.N_j
        j_re = action_jammer[:half].reshape(N_t, c.N_j).astype(float)
        j_im = action_jammer[half:].reshape(N_t, c.N_j).astype(float)
        new_v = (j_re + 1j * j_im) * np.sqrt(c.P_j_max)
        for n in range(N_t):
            norm = float(np.linalg.norm(new_v[n]))
            if norm > np.sqrt(c.P_j_max):
                new_v[n] = new_v[n] * np.sqrt(c.P_j_max) / max(norm, 1e-30)
        return new_v

    def _apply_ris_phase_action(self, action_ris: np.ndarray) -> np.ndarray:
        """Map raw action in [-action_range, action_range] to a phase
        vector in [-pi, pi] per time slot — this is the actual RIS
        phase-shift decision variable phi_n from the proposed model,
        replacing the fixed conjugate-alignment heuristic."""
        c = self.cfg
        ar = c.action_range
        phi = action_ris.reshape(c.N_time, c.N_ris).astype(float)
        phi = np.clip(phi, -ar, ar) / ar * np.pi
        return phi

    # ── Reward ─────────────────────────────────────────────

    def _compute_reward(self, sec, sense, cons_viol, actions=None) -> dict[str, float]:
        c = self.cfg
        r_cfg = self.r_cfg

        U_total = float(sense["U_sense_total"])
        R_total = float(sec["R_s_total"])

        secrecy_outage = float(np.mean(
            [max(r_th - r, 0.0) for r, r_th in
             zip(sec["R_s_per_slot"], [self.r_cfg.secrecy_outage_threshold] * c.N_time)]
        ))

        # Update running statistics for normalization
        self._norm_R.update(R_total)
        self._norm_U.update(U_total)

        mode = r_cfg.reward_mode
        if mode == "original":
            result = compute_reward_original(
                R_total, U_total, self.u_ref,
                cons_viol, secrecy_outage,
                c.alpha, r_cfg.lambda_constraint, r_cfg.lambda_outage,
            )
        elif mode == "normalized":
            result = compute_reward_normalized(
                R_total, U_total,
                self._norm_R, self._norm_U,
                cons_viol, secrecy_outage,
                c.alpha, r_cfg.lambda_constraint, r_cfg.lambda_outage,
            )
        else:
            # normalized_penalty or full
            result = compute_reward_full(
                R_total, U_total,
                self._norm_R, self._norm_U,
                cons_viol, secrecy_outage,
                c.alpha, r_cfg.lambda_constraint, r_cfg.lambda_outage,
                r_cfg.lambda_secret, r_cfg.R_target,
                actions,
                r_cfg.lambda_action if mode == "full" else 0.0,
            )

        result["secrecy"] = R_total
        result["sensing"] = U_total
        result["outage"] = secrecy_outage
        result["violation"] = cons_viol
        return result

    # ── Gymnasium interface ────────────────────────────────

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed = seed
            self.rng = np.random.RandomState(seed)

        c = self.cfg
        N_t = c.N_time

        t = np.linspace(0.0, 1.0, N_t)
        self.q_uav = np.zeros((N_t, 3))
        for d in range(3):
            self.q_uav[:, d] = self.q_uav_start[d] + t * (self.q_uav_end[d] - self.q_uav_start[d])

        p_per_ant = np.sqrt(c.P_bs_max / (N_t * c.M_bs))
        self.w_bs = np.full((N_t, c.M_bs), p_per_ant, dtype=complex)

        self.v_jammer = np.zeros((N_t, c.N_j), dtype=complex)
        self.phi_ris = np.zeros((N_t, c.N_ris))

        self.mobility.reset()
        self.q_user = self.q_user_start.copy()
        self.user_velocity = np.zeros(2)

        self.q_eves = self._sample_hppp_eves()

        self.sec, self.sense = self._compute_channels(
            self.q_uav, self.w_bs, self.v_jammer,
        )
        self.cons_viol = float(self._compute_constraint_violations(
            self.q_uav, self.w_bs, self.v_jammer,
        ))

        obs = {
            "bs_beamformer": self._build_obs_bs(
                self.q_uav, self.w_bs, self.v_jammer,
                self.sec, self.sense, self.cons_viol,
            ),
            "uav_trajectory": self._build_obs_uav_trajectory(
                self.q_uav, self.w_bs, self.v_jammer,
                self.sec, self.sense, self.cons_viol,
            ),
            "jammer_beamformer": self._build_obs_jammer(
                self.q_uav, self.w_bs, self.v_jammer,
                self.sec, self.sense, self.cons_viol,
            ),
            "ris_phase": self._build_obs_ris_phase(
                self.phi_ris, self.sec, self.sense, self.cons_viol,
            ),
        }
        return obs, {}

    def step(self, actions: dict[str, np.ndarray]):
        c = self.cfg

        if "bs_beamformer" in actions:
            self.w_bs = self._apply_bs_action(actions["bs_beamformer"], self.w_bs)

        if "uav_trajectory" in actions:
            self.q_uav = self._apply_uav_trajectory_action(
                actions["uav_trajectory"], self.q_uav,
            )

        if "jammer_beamformer" in actions:
            self.v_jammer = self._apply_jammer_action(actions["jammer_beamformer"])

        if "ris_phase" in actions:
            self.phi_ris = self._apply_ris_phase_action(actions["ris_phase"])

        self._step_user_mobility()

        self.sec, self.sense = self._compute_channels(
            self.q_uav, self.w_bs, self.v_jammer,
        )
        self.cons_viol = float(self._compute_constraint_violations(
            self.q_uav, self.w_bs, self.v_jammer,
        ))

        r_info = self._compute_reward(self.sec, self.sense, self.cons_viol, actions)

        obs = {
            "bs_beamformer": self._build_obs_bs(
                self.q_uav, self.w_bs, self.v_jammer,
                self.sec, self.sense, self.cons_viol,
            ),
            "uav_trajectory": self._build_obs_uav_trajectory(
                self.q_uav, self.w_bs, self.v_jammer,
                self.sec, self.sense, self.cons_viol,
            ),
            "jammer_beamformer": self._build_obs_jammer(
                self.q_uav, self.w_bs, self.v_jammer,
                self.sec, self.sense, self.cons_viol,
            ),
            "ris_phase": self._build_obs_ris_phase(
                self.phi_ris, self.sec, self.sense, self.cons_viol,
            ),
        }

        rewards = {
            "bs_beamformer": r_info["reward"],
            "uav_trajectory": r_info["reward"],
            "jammer_beamformer": r_info["reward"],
            "ris_phase": r_info["reward"],
        }
        terminated = {name: False for name in self.agent_names}
        truncated = {name: False for name in self.agent_names}
        terminated["__all__"] = False
        truncated["__all__"] = False

        return obs, rewards, terminated, truncated, r_info

    def _compute_constraint_violations(self, q_uav, w_bs, v_jammer) -> float:
        c = self.cfg
        viol = 0.0
        N_t = c.N_time

        bs_power = [float(np.linalg.norm(w_bs[n]) ** 2) for n in range(N_t)]
        viol += float(np.max(np.maximum(np.array(bs_power) - c.P_bs_max, 0.0)))

        j_power = [float(np.linalg.norm(v_jammer[n]) ** 2) for n in range(N_t)]
        viol += float(np.max(np.maximum(np.array(j_power) - c.P_j_max, 0.0)))

        speeds = [0.0]
        for n in range(1, N_t):
            dist = float(np.linalg.norm(q_uav[n] - q_uav[n-1]))
            speeds.append(dist / c.dt)
        viol += float(np.max(np.maximum(np.array(speeds) - c.v_max, 0.0)))

        return viol

    # ── Properties ─────────────────────────────────────────

    @property
    def num_agents(self) -> int:
        return len(self.agent_names)

    def close(self):
        pass
