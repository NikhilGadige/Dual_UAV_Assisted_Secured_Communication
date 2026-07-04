"""Multi-agent sensing environment for the 2026-07-07 presentation.

N sensing UAVs independently control 2D trajectories around K vehicle
targets. Reward is a shared team signal combining:
  - CRB-derived sensing utility (minimize Cramer-Rao Bound), and
  - Pd, detection probability (maximize),
per the proposed ISAC model's sensing objectives. The target reflection
channel is Rician-faded (see common/sensing_metrics.py).

This is the *sensing-only* counterpart to madrl_updated_exp's secrecy+
sensing environment: no BS/RIS/jammer/secrecy here, just cooperative
multi-agent trajectory control for CRB + Pd.

Simplification (documented, not hidden): each agent's CRB/Pd is computed
independently from its own monostatic vantage point and then averaged
across agents into one team reward. This rewards agents for individually
finding good sensing geometry (and implicitly for not collapsing onto the
same spot, since the reward is a mean and duplicated viewpoints don't
help each other). It is not a full multistatic Fisher-information fusion
across sensors (that would require reparametrizing the per-sensor angle
FIMs into a shared position-domain FIM) — a natural next step, not
implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vehicle_reflection_exp.channels.vehicle_channel import compute_rcs
from common.sensing_metrics import agent_sensing_metrics, crb_to_utility

SCENE_MIN = np.array([0.0, -150.0])
SCENE_MAX = np.array([400.0, 150.0])


@dataclass
class SensingEnvConfig:
    n_agents: int = 2
    n_vehicles: int = 3
    N_tx: int = 8
    N_rx: int = 8
    L_pilot: int = 16
    noise_power: float = 1.0e-2
    d_ant: float = 0.5
    wavelength: float = 1.0
    rician_k_db: float = 5.0
    pfa: float = 0.05
    num_mc_train: int = 40
    num_mc_eval: int = 200
    v_max: float = 20.0
    dt: float = 1.0
    steps_per_episode: int = 40
    alpha_pd: float = 0.5  # reward weight: alpha_pd * Pd + (1-alpha_pd) * z(U_crb)
    action_range: float = 1.0
    seed: int = 42


class RunningNormalizer:
    """Welford's online mean/std — used to z-score the CRB utility term
    so it combines sensibly with Pd (already in [0, 1])."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return float(np.sqrt(self.M2 / (self.count - 1)))

    def zscore(self, x: float) -> float:
        s = self.std
        if s < 1e-8:
            return 0.0
        return (x - self.mean) / s


class MultiAgentSensingEnv:
    """Gymnasium-style multi-agent env: N independent 2D-trajectory agents,
    shared team reward from mean CRB-utility + mean Pd across agents."""

    def __init__(self, cfg: SensingEnvConfig | None = None):
        self.cfg = cfg or SensingEnvConfig()
        self.rng = np.random.RandomState(self.cfg.seed)
        self.agent_names = [f"uav_{i+1}" for i in range(self.cfg.n_agents)]

        vehicle_types = ["car", "truck", "motorcycle"][: self.cfg.n_vehicles]
        while len(vehicle_types) < self.cfg.n_vehicles:
            vehicle_types.append("car")
        self.vehicle_types = vehicle_types
        self.rcs_lin_list = [compute_rcs(vt) for vt in vehicle_types]

        self._norm_U = RunningNormalizer()
        self.obs_dim = self._obs_dim()
        self.act_dim = 2  # (vx, vy) per agent

        self.t = 0
        self.q_agents = np.zeros((self.cfg.n_agents, 2))
        self.q_vehicles = np.zeros((self.cfg.n_vehicles, 2))
        self._last_crb = np.zeros(self.cfg.n_agents)
        self._last_pd = np.zeros(self.cfg.n_agents)

    def _obs_dim(self) -> int:
        n_a, n_v = self.cfg.n_agents, self.cfg.n_vehicles
        return 2 + 2 * (n_a - 1) + 2 * n_v + 1 + 1 + 1

    def _norm_pos(self, xy: np.ndarray) -> np.ndarray:
        return (xy - SCENE_MIN) / (SCENE_MAX - SCENE_MIN) * 2.0 - 1.0

    def _build_obs(self, i: int) -> np.ndarray:
        parts = [self._norm_pos(self.q_agents[i])]
        for j in range(self.cfg.n_agents):
            if j != i:
                parts.append(self._norm_pos(self.q_agents[j]))
        for k in range(self.cfg.n_vehicles):
            parts.append(self._norm_pos(self.q_vehicles[k]))
        parts.append(np.array([crb_to_utility(self._last_crb[i])]))
        parts.append(np.array([self._last_pd[i]]))
        parts.append(np.array([self.cfg.alpha_pd]))
        return np.concatenate(parts).astype(np.float32)

    def _sample_vehicles(self) -> np.ndarray:
        xs = self.rng.uniform(SCENE_MIN[0] + 50, SCENE_MAX[0] - 50, size=self.cfg.n_vehicles)
        ys = self.rng.uniform(SCENE_MIN[1] + 30, SCENE_MAX[1] - 30, size=self.cfg.n_vehicles)
        return np.column_stack([xs, ys])

    def _sample_agent_starts(self) -> np.ndarray:
        xs = self.rng.uniform(SCENE_MIN[0], SCENE_MAX[0], size=self.cfg.n_agents)
        ys = self.rng.uniform(SCENE_MIN[1], SCENE_MAX[1], size=self.cfg.n_agents)
        return np.column_stack([xs, ys])

    def _evaluate_sensing(self, num_mc: int) -> tuple[np.ndarray, np.ndarray]:
        crbs = np.zeros(self.cfg.n_agents)
        pds = np.zeros(self.cfg.n_agents)
        for i in range(self.cfg.n_agents):
            m = agent_sensing_metrics(
                agent_xy=self.q_agents[i],
                vehicle_xy=self.q_vehicles,
                rcs_lin_list=self.rcs_lin_list,
                N_tx=self.cfg.N_tx, N_rx=self.cfg.N_rx, L_pilot=self.cfg.L_pilot,
                noise_power=self.cfg.noise_power,
                d_ant=self.cfg.d_ant, wavelength=self.cfg.wavelength,
                rician_k_db=self.cfg.rician_k_db, pfa=self.cfg.pfa,
                num_mc=num_mc, rng=self.rng,
            )
            crbs[i] = m["crb_trace"]
            pds[i] = m["pd"]
        return crbs, pds

    def _compute_reward(self, crbs: np.ndarray, pds: np.ndarray) -> dict:
        u_mean = float(np.mean([crb_to_utility(c) for c in crbs]))
        pd_mean = float(np.mean(pds))
        self._norm_U.update(u_mean)
        z_u = self._norm_U.zscore(u_mean)
        alpha = self.cfg.alpha_pd
        reward = alpha * pd_mean + (1.0 - alpha) * z_u
        return {
            "reward": reward, "crb_mean": float(np.mean(crbs)),
            "u_mean": u_mean, "pd_mean": pd_mean,
        }

    def reset(self, seed: int | None = None, num_mc: int | None = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.t = 0
        self.q_vehicles = self._sample_vehicles()
        self.q_agents = self._sample_agent_starts()
        mc = num_mc if num_mc is not None else self.cfg.num_mc_train
        self._last_crb, self._last_pd = self._evaluate_sensing(mc)
        obs = {name: self._build_obs(i) for i, name in enumerate(self.agent_names)}
        return obs, {}

    def step(self, actions: dict[str, np.ndarray], num_mc: int | None = None):
        c = self.cfg
        for i, name in enumerate(self.agent_names):
            a = np.clip(actions[name], -c.action_range, c.action_range) / c.action_range
            delta = a * c.v_max * c.dt
            new_xy = self.q_agents[i] + delta
            self.q_agents[i] = np.clip(new_xy, SCENE_MIN, SCENE_MAX)

        mc = num_mc if num_mc is not None else c.num_mc_train
        self._last_crb, self._last_pd = self._evaluate_sensing(mc)
        r_info = self._compute_reward(self._last_crb, self._last_pd)

        self.t += 1
        done = self.t >= c.steps_per_episode
        obs = {name: self._build_obs(i) for i, name in enumerate(self.agent_names)}
        rewards = {name: r_info["reward"] for name in self.agent_names}
        terminated = {name: False for name in self.agent_names}
        truncated = {name: done for name in self.agent_names}
        terminated["__all__"] = False
        truncated["__all__"] = done

        return obs, rewards, terminated, truncated, r_info
