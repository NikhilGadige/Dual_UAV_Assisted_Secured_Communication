"""
SCA_study/train_sca.py
======================
Successive Convex Approximation (SCA) baseline for UAV-relay / UAV-jammer
secrecy-rate maximisation under HPPP eavesdroppers and a mobile user.

Optimisations over the original version
----------------------------------------
1.  env.reset() is properly silenced in the training loop (was uncommented).
2.  Anchor EMA is warmed up quickly in the first 10 % of training so the
    anchor actually tracks a decent policy before the trust region shrinks.
3.  Multi-start SCA: each step tries 3 restarts (geometry heuristic, random
    perturbation, pure geometry) and keeps the best, avoiding shallow local
    optima.
4.  Trust-region restart: if no improvement occurs for PATIENCE_EPISODES
    consecutive episodes the trust region is reset to 60 % of its start value,
    preventing premature convergence.
5.  Convergence trace uses a *real* SCA run on a fixed frozen environment
    instead of a purely synthetic curve, giving honest convergence data.
6.  Power-dimension is searched with finer resolution (5 candidates) instead
    of 3 in the original.
7.  Rolling statistics use both 20- and 100-episode windows; the gap between
    them is logged as a convergence diagnostic.
8.  eval_interval default lowered to 50 (unchanged) but the fixed-eval now
    uses 12 episodes (was 8) for a more stable estimate.
9.  All matplotlib figures use consistent colour palette so they can be
    overlaid with DRL plots in post-processing.
"""

import contextlib
import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from core.config_utils import build_env_config
from core.environment import EnvConfig, UAVEnvironment
from hppp_training_utils import HPPP_CONFIG, plot_training_curves

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (consistent with BCD and DRL plots)
# ──────────────────────────────────────────────────────────────────────────────
_C_RAW    = "#9e9e9e"   # raw episode samples
_C_BEST   = "#2ca02c"   # best-so-far / best feasible
_C_SMOOTH = "#ff7f0e"   # smoothed SCA curve  ← orange so it differs from BCD blue
_C_EVAL   = "#1f77b4"   # fixed-eval curve

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SCAConfig:
    episodes: int = 3000
    seed: int = 42
    fading_model: str = "rician"
    rician_k: float = 5.0
    control_mode: str = "velocity"
    user_mobile: bool = True
    use_los_model: bool = False
    observation_mode: str = "full"
    normalize_observations: bool = True
    max_steps: int = 50

    # SCA inner loop
    inner_iterations: int = 3          # was 1 — more inner iters = better per-step quality
    n_restarts: int = 3                # multi-start: number of x0 candidates per step
    trust_region_start: float = 0.85
    trust_region_end: float = 0.06     # was 0.08 — allow finer refinement at the end
    trust_decay_on_stuck: float = 0.60 # fraction of start trust to restore after patience
    patience_episodes: int = 120       # episodes with no rolling improvement before restart

    # Exploration / smoothing
    smoothing_start: float = 0.28
    smoothing_end: float = 0.72
    exploration_start: float = 0.35
    exploration_end: float = 0.02

    # Anchor warm-up: for the first `anchor_warmup_frac` of training use a
    # faster EMA so the anchor reflects a decent policy earlier.
    anchor_warmup_frac: float = 0.10
    anchor_ema_fast: float = 0.030     # EMA rate during warm-up
    anchor_ema_slow: float = 0.005     # EMA rate after warm-up (original value)

    # Objective penalties
    power_penalty: float = 0.015
    motion_penalty: float = 0.025
    improvement_tol: float = 1e-5

    # Logging / evaluation
    log_interval: int = 100
    eval_interval: int = 50
    eval_episodes: int = 12            # was 8 — more stable fixed-eval estimate
    eval_seed: int = 9000
    eval_smoothing: float = 0.75

    # Convergence-trace fixed scenario
    convergence_seed: int = 12000
    convergence_episodes: int = 300    # inner iters for the trace (separate from episodes)


# ──────────────────────────────────────────────────────────────────────────────
# Environment factory
# ──────────────────────────────────────────────────────────────────────────────

def make_env_config(seed: int, cfg: SCAConfig) -> EnvConfig:
    env_cfg = build_env_config(
        seed=seed,
        fading_model=cfg.fading_model,
        rician_k=cfg.rician_k,
        control_mode=cfg.control_mode,
        role_switching=False,
        user_mobile=cfg.user_mobile,
        use_los_model=cfg.use_los_model,
        observation_mode=cfg.observation_mode,
        normalize_observations=cfg.normalize_observations,
        use_multiple_eves=True,
        eve_density_lambda=HPPP_CONFIG["eve_density_lambda"],
    )
    env_cfg.max_steps = cfg.max_steps
    return env_cfg


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_sca(cfg: SCAConfig, output_dir: str) -> dict:
    rng = np.random.default_rng(cfg.seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    eval_rows: list[dict] = []
    rolling: list[float] = []
    latest_fixed_eval: float | str = ""
    global_step = 0

    # Policy anchor — represents the "current best policy" across episodes.
    policy_anchor = np.array([0.0, 0.0, 0.0, 0.0, 0.55], dtype=float)

    # Patience / stuck-detection state
    best_roll100 = -np.inf
    patience_counter = 0
    current_trust_start = cfg.trust_region_start  # can be reset on patience trigger

    warmup_cutoff = int(cfg.anchor_warmup_frac * cfg.episodes)

    for ep in range(1, cfg.episodes + 1):
        progress = (ep - 1) / max(cfg.episodes - 1, 1)
        trust = _linear_decay(current_trust_start, cfg.trust_region_end, progress)
        noise = _linear_decay(cfg.exploration_start, cfg.exploration_end, progress)
        smooth = _linear_decay(cfg.smoothing_start, cfg.smoothing_end, progress)
        anchor_ema = cfg.anchor_ema_fast if ep <= warmup_cutoff else cfg.anchor_ema_slow

        env = UAVEnvironment(make_env_config(cfg.seed + ep, cfg))
        with contextlib.redirect_stdout(io.StringIO()):  # FIX: was uncommented in original
            env.reset()

        ep_reward = 0.0
        ep_rsec = 0.0
        ep_rlegit = 0.0
        ep_reve = 0.0
        ep_num_eves = 0.0
        ep_nearest_eve_dist = 0.0
        ep_mean_eve_dist = 0.0
        ep_max_eve_cap = 0.0
        ep_inner_gain = 0.0
        steps = 0
        done = False
        local_action = policy_anchor.copy()

        while not done:
            nominal = _geometry_action(env)

            # ── Multi-start: build several x0 candidates and pick the best ──
            x0_base = _clip_action(0.65 * local_action + 0.35 * nominal)
            x0_candidates = [x0_base]
            if cfg.n_restarts >= 2:
                x0_candidates.append(_clip_action(nominal))
            if cfg.n_restarts >= 3 and noise > 0.01:
                x0_candidates.append(
                    _clip_action(x0_base + rng.normal(0.0, noise * 1.5, size=5))
                )

            best_action = x0_base
            best_gain = -np.inf
            for x0 in x0_candidates:
                if noise > 0.0:
                    x0 = _clip_action(x0 + rng.normal(0.0, noise, size=5))
                cand_action, cand_gain = _sca_refine_action(env, x0, trust, cfg)
                if cand_gain > best_gain:
                    best_gain = cand_gain
                    best_action = cand_action

            action = best_action
            surrogate_gain = best_gain if best_gain > 0.0 else 0.0

            local_action = _clip_action((1.0 - smooth) * local_action + smooth * action)
            policy_anchor = _clip_action(
                (1.0 - anchor_ema) * policy_anchor + anchor_ema * local_action
            )

            _, reward, done, info = env.step(
                local_action[:2],
                local_action[2:4],
                float(local_action[4]),
                False,
            )
            global_step += 1
            steps += 1
            ep_reward += float(reward)
            ep_rsec += float(info["R_sec"])
            ep_rlegit += float(info["R_legit"])
            ep_reve += float(info["R_eve"])
            ep_num_eves += float(info.get("num_eves", 1))
            ep_nearest_eve_dist += float(info.get("nearest_eve_distance", 0.0))
            ep_mean_eve_dist += float(info.get("mean_eve_distance", 0.0))
            ep_max_eve_cap += float(info.get("max_eve_capacity", 0.0))
            ep_inner_gain += surrogate_gain

        avg_sec = float((ep_rsec / max(steps, 1)) / 1e6)
        rolling.append(avg_sec)

        # ── Patience / trust-region restart ──
        roll100 = float(np.mean(rolling[-min(100, len(rolling)):]))
        if roll100 > best_roll100 + cfg.improvement_tol:
            best_roll100 = roll100
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= cfg.patience_episodes:
            current_trust_start = cfg.trust_decay_on_stuck * cfg.trust_region_start
            patience_counter = 0  # reset so we don't fire every episode

        row = _row(
            cfg=cfg,
            ep=ep,
            global_step=global_step,
            ep_reward=ep_reward,
            ep_rsec=ep_rsec,
            ep_rlegit=ep_rlegit,
            ep_reve=ep_reve,
            steps=steps,
            rolling=rolling,
            trust=trust,
            exploration=noise,
            ep_inner_gain=ep_inner_gain,
            ep_num_eves=ep_num_eves,
            ep_nearest_eve_dist=ep_nearest_eve_dist,
            ep_mean_eve_dist=ep_mean_eve_dist,
            ep_max_eve_cap=ep_max_eve_cap,
        )

        if ep == 1 or ep % max(cfg.eval_interval, 1) == 0 or ep == cfg.episodes:
            fixed_eval = _evaluate_policy_anchor(cfg, policy_anchor)
            eval_row = {
                "episode": ep,
                "fading_model": cfg.fading_model,
                "fixed_eval_R_sec_mbps": fixed_eval["avg_R_sec_mbps"],
                "fixed_eval_R_legit_mbps": fixed_eval["avg_R_legit_mbps"],
                "fixed_eval_R_eve_mbps": fixed_eval["avg_R_eve_mbps"],
                "fixed_eval_reward": fixed_eval["avg_reward"],
                "fixed_eval_avg_num_eves": fixed_eval["avg_num_eves"],
                "fixed_eval_avg_inner_gain": fixed_eval["avg_inner_gain"],
                "policy_anchor_relay_x": float(policy_anchor[0]),
                "policy_anchor_relay_y": float(policy_anchor[1]),
                "policy_anchor_jammer_x": float(policy_anchor[2]),
                "policy_anchor_jammer_y": float(policy_anchor[3]),
                "policy_anchor_power": float(policy_anchor[4]),
            }
            eval_rows.append(eval_row)
            latest_fixed_eval = fixed_eval["avg_R_sec_mbps"]

        row["fixed_eval_R_sec_mbps"] = latest_fixed_eval
        rows.append(row)

        if ep == 1 or ep % max(cfg.log_interval, 1) == 0 or ep == cfg.episodes:
            fixed_text = "" if latest_fixed_eval == "" else f" fixed_eval={latest_fixed_eval:8.4f}"
            print(
                f"SCA {cfg.fading_model} ep={ep:5d}/{cfg.episodes} "
                f"avg_R_sec={avg_sec:8.4f} Mbps roll100={roll100:8.4f} "
                f"trust={trust:.3f} patience={patience_counter}{fixed_text}"
            )

    # ── Save logs ──
    log_path = out_dir / "training_log.csv"
    eval_log_path = out_dir / "fixed_eval_log.csv"
    _write_csv(rows, log_path)
    _write_csv(eval_rows, eval_log_path)
    plot_paths = plot_training_curves(str(log_path), str(out_dir / "plots"))
    fixed_eval_plot = _plot_fixed_eval_convergence(eval_rows, out_dir / "plots")
    if fixed_eval_plot:
        plot_paths["fixed_eval_convergence"] = fixed_eval_plot

    # ── Real SCA convergence trace on a frozen scenario ──
    sca_trace_rows = _generate_sca_objective_trace(cfg)
    sca_trace_path = out_dir / "sca_convergence_log.csv"
    _write_csv(sca_trace_rows, sca_trace_path)
    sca_trace_plot = _plot_sca_objective_convergence(sca_trace_rows, out_dir / "plots")
    if sca_trace_plot:
        plot_paths["sca_objective_convergence"] = sca_trace_plot

    # ── Checkpoint ──
    checkpoint_path = out_dir / "sca_checkpoint.npz"
    np.savez_compressed(
        checkpoint_path,
        policy_anchor=policy_anchor,
        episodes=np.array([cfg.episodes], dtype=np.int64),
        seed=np.array([cfg.seed], dtype=np.int64),
        fading_model=np.array([cfg.fading_model]),
        final_rolling100_R_sec_mbps=np.array([float(np.mean(rolling[-min(100, len(rolling)):]))]),
        best_episode_R_sec_mbps=np.array([float(np.max(rolling))]),
    )

    metadata = {
        "algorithm": "sca",
        "config": asdict(cfg),
        "training_log_csv": str(log_path.resolve()),
        "fixed_eval_log_csv": str(eval_log_path.resolve()),
        "sca_convergence_log_csv": str(sca_trace_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "plot_paths": plot_paths,
        "final_rolling100_R_sec_mbps": float(np.mean(rolling[-min(100, len(rolling)):])),
        "final_fixed_eval_R_sec_mbps": float(eval_rows[-1]["fixed_eval_R_sec_mbps"]) if eval_rows else None,
        "final_sca_objective_mbps": float(sca_trace_rows[-1]["smoothed_objective_mbps"]) if sca_trace_rows else None,
        "best_episode_R_sec_mbps": float(np.max(rolling)),
        "mean_last100_reward": float(np.mean([r["avg_shaped_reward"] for r in rows[-min(100, len(rows)):]]))
    }
    metadata_path = out_dir / "sca_summary.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "algorithm": "sca",
        "training_log_csv": str(log_path.resolve()),
        "fixed_eval_log_csv": str(eval_log_path.resolve()),
        "sca_convergence_log_csv": str(sca_trace_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "summary_json": str(metadata_path.resolve()),
        "mean_avg_rsec_mbps": metadata["final_rolling100_R_sec_mbps"],
        "best_episode_R_sec_mbps": metadata["best_episode_R_sec_mbps"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# SCA inner solver
# ──────────────────────────────────────────────────────────────────────────────

def _sca_refine_action(
    env: UAVEnvironment,
    x0: np.ndarray,
    trust: float,
    cfg: SCAConfig,
) -> tuple[np.ndarray, float]:
    """One SCA linearisation pass.

    Builds a set of first-order candidates around x0 (geometry direction,
    trust-region steps in each coordinate, power extremes) and iteratively
    accepts improvements.  Returns (best_action, total_surrogate_gain).
    """
    x = _clip_action(x0)
    best = _surrogate_objective(env, x, cfg)
    total_gain = 0.0

    for _ in range(max(cfg.inner_iterations, 1)):
        improved = False
        geom = _geometry_action(env)
        direction = geom - x
        direction_norm = np.linalg.norm(direction[:4])
        if direction_norm > 1e-9:
            direction[:4] = direction[:4] / max(direction_norm, 1.0)

        # Build candidate set
        candidates = [
            x,
            _clip_action(0.55 * x + 0.45 * geom),          # geometry blend
            _clip_action(x + trust * direction),             # step toward geometry
            _clip_action(x - 0.5 * trust * direction),      # step away
        ]
        # Power candidates — 5 levels for finer search (was 3)
        for pwr in (-1.0, -0.5, 0.0, 0.5, 1.0):
            pc = x.copy(); pc[4] = pwr
            candidates.append(_clip_action(pc))
        # ±step in each position coordinate
        for idx in range(4):
            plus = x.copy(); minus = x.copy()
            plus[idx] += trust; minus[idx] -= trust
            candidates.extend([_clip_action(plus), _clip_action(minus)])

        for cand in candidates:
            val = _surrogate_objective(env, cand, cfg)
            if val > best + cfg.improvement_tol:
                total_gain += val - best
                best = val
                x = cand
                improved = True

        trust *= 0.55
        if not improved:
            break

    return x, float(total_gain)


# ──────────────────────────────────────────────────────────────────────────────
# Surrogate objective
# ──────────────────────────────────────────────────────────────────────────────

def _surrogate_objective(env: UAVEnvironment, action: np.ndarray, cfg: SCAConfig) -> float:
    relay_pos, jammer_pos = _predict_positions(env, action)
    gains = _predicted_gains(env, relay_pos, jammer_pos)
    rates = _rates_from_gains(env, gains, action[4])
    secrecy_mbps = rates["R_sec"] / 1e6
    motion_cost = cfg.motion_penalty * (np.linalg.norm(action[:2]) + np.linalg.norm(action[2:4]))
    power_cost = cfg.power_penalty * ((action[4] + 1.0) * 0.5)
    return float(secrecy_mbps - motion_cost - power_cost)


# ──────────────────────────────────────────────────────────────────────────────
# Fixed-scenario evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate_policy_anchor(cfg: SCAConfig, policy_anchor: np.ndarray) -> dict:
    rewards, rsecs, rlegits, reverses, num_eves, inner_gains = [], [], [], [], [], []

    for idx in range(max(cfg.eval_episodes, 1)):
        env = UAVEnvironment(make_env_config(cfg.eval_seed + idx, cfg))
        with contextlib.redirect_stdout(io.StringIO()):
            env.reset()

        done = False
        steps = 0
        local_action = policy_anchor.copy()
        ep_reward = ep_rsec = ep_rlegit = ep_reve = ep_num_eves = ep_inner_gain = 0.0

        while not done:
            x0 = _clip_action(0.70 * local_action + 0.30 * _geometry_action(env))
            action, surrogate_gain = _sca_refine_action(env, x0, cfg.trust_region_end, cfg)
            local_action = _clip_action(
                (1.0 - cfg.eval_smoothing) * local_action + cfg.eval_smoothing * action
            )
            _, reward, done, info = env.step(
                local_action[:2], local_action[2:4], float(local_action[4]), False,
            )
            steps += 1
            ep_reward += float(reward)
            ep_rsec += float(info["R_sec"])
            ep_rlegit += float(info["R_legit"])
            ep_reve += float(info["R_eve"])
            ep_num_eves += float(info.get("num_eves", 1))
            ep_inner_gain += surrogate_gain

        rewards.append(ep_reward / max(steps, 1))
        rsecs.append((ep_rsec / max(steps, 1)) / 1e6)
        rlegits.append((ep_rlegit / max(steps, 1)) / 1e6)
        reverses.append((ep_reve / max(steps, 1)) / 1e6)
        num_eves.append(ep_num_eves / max(steps, 1))
        inner_gains.append(ep_inner_gain / max(steps, 1))

    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_R_sec_mbps": float(np.mean(rsecs)),
        "avg_R_legit_mbps": float(np.mean(rlegits)),
        "avg_R_eve_mbps": float(np.mean(reverses)),
        "avg_num_eves": float(np.mean(num_eves)),
        "avg_inner_gain": float(np.mean(inner_gains)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Real SCA convergence trace (fixed frozen scenario)
# ──────────────────────────────────────────────────────────────────────────────

def _generate_sca_objective_trace(cfg: SCAConfig) -> list[dict]:
    """Run SCA iterations on a single frozen environment to get honest convergence.

    Unlike the original (which was synthetic), this actually calls
    _sca_refine_action repeatedly on a fixed env and records the objective.
    The number of iterations is cfg.convergence_episodes (default 300) which
    is much smaller than cfg.episodes but sufficient for a convergence curve.
    The x-axis of the plot is 'SCA iteration' not 'training episode'.
    """
    rng = np.random.default_rng(cfg.seed + 99999)
    trace_cfg = make_env_config(cfg.convergence_seed, cfg)
    trace_cfg.user_mobile = False   # freeze the scene
    trace_cfg.max_steps = 1
    env = UAVEnvironment(trace_cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        env.reset()

    # Bad initialisation so we can see convergence from below
    x = _clip_action(np.array([-0.85, 0.65, -0.75, -0.55, -0.65], dtype=float))
    best_objective = _surrogate_objective(env, x, cfg)
    smooth_objective = best_objective
    rows = []

    n_iters = max(cfg.convergence_episodes, 50)
    trust = cfg.trust_region_start

    for it in range(1, n_iters + 1):
        progress = (it - 1) / max(n_iters - 1, 1)
        noise = _linear_decay(0.30, 0.0, progress)
        trust = _linear_decay(cfg.trust_region_start, cfg.trust_region_end, progress)

        # Multi-start: try geometry init + noisy perturbation
        candidates_x0 = [x, _clip_action(_geometry_action(env))]
        if noise > 0.01:
            candidates_x0.append(_clip_action(x + rng.normal(0.0, noise, size=5)))

        best_x_iter = x
        best_obj_iter = best_objective
        best_gain_iter = 0.0
        for x0 in candidates_x0:
            cand, gain = _sca_refine_action(env, x0, trust, cfg)
            cand_obj = _surrogate_objective(env, cand, cfg)
            if cand_obj > best_obj_iter + cfg.improvement_tol:
                best_obj_iter = cand_obj
                best_x_iter = cand
                best_gain_iter = gain

        if best_obj_iter > best_objective + cfg.improvement_tol:
            best_objective = best_obj_iter
            x = best_x_iter
        else:
            # Slow drift toward geometry to avoid getting stuck
            x = _clip_action(0.998 * x + 0.002 * _geometry_action(env))

        # Smooth the objective for plotting
        smooth_alpha = 0.04 + 0.06 * (1.0 - progress)
        smooth_objective = (1.0 - smooth_alpha) * smooth_objective + smooth_alpha * best_objective

        rows.append({
            "iteration": it,
            "fading_model": cfg.fading_model,
            "sca_objective_mbps": float(best_obj_iter),
            "best_objective_mbps": float(best_objective),
            "smoothed_objective_mbps": float(smooth_objective),
            "inner_gain_mbps": float(best_gain_iter),
            "trust_region": float(trust),
            "convergence_noise": float(noise),
            "action_relay_x": float(x[0]),
            "action_relay_y": float(x[1]),
            "action_jammer_x": float(x[2]),
            "action_jammer_y": float(x[3]),
            "action_power": float(x[4]),
        })

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Geometry / action helpers  (shared with BCD via import)
# ──────────────────────────────────────────────────────────────────────────────

def _predict_positions(env: UAVEnvironment, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    relay_velocity = env._update_velocity(env.relay_velocity, action[:2])
    jammer_velocity = env._update_velocity(env.jammer_velocity, action[2:4])
    relay_pos = env._clip_to_bounds(
        env.relay_position + np.append(relay_velocity * env.config.dt, 0.0),
        env.config.relay_altitude,
    )
    jammer_pos = env._clip_to_bounds(
        env.jammer_position + np.append(jammer_velocity * env.config.dt, 0.0),
        env.config.jammer_altitude,
    )
    return relay_pos, jammer_pos


def _predicted_gains(env: UAVEnvironment, relay_pos: np.ndarray, jammer_pos: np.ndarray) -> dict:
    gains = {
        "h_UR": env.compute_channel_gain(env.user_position, relay_pos, env.fading["UR"]),
        "h_RB": env.compute_channel_gain(relay_pos, env.bs_position, env.fading["RB"]),
    }
    if env.config.use_multiple_eves:
        n = env.num_eves
        if n == 0:
            gains["h_UE"] = np.array([], dtype=float)
            gains["h_JE"] = np.array([], dtype=float)
        else:
            gains["h_UE"] = np.array([
                env.compute_channel_gain(env.user_position, np.append(env.eve_positions[i], 0.0), env.fading["UE"][i])
                for i in range(n)
            ])
            gains["h_JE"] = np.array([
                env.compute_channel_gain(jammer_pos, np.append(env.eve_positions[i], 0.0), env.fading["JE"][i])
                for i in range(n)
            ])
    else:
        gains["h_UE"] = env.compute_channel_gain(env.user_position, env.eve_position, env.fading["UE"])
        gains["h_JE"] = env.compute_channel_gain(jammer_pos, env.eve_position, env.fading["JE"])
    return gains


def _rates_from_gains(env: UAVEnvironment, gains: dict, action_power: float) -> dict:
    noise_power = env.config.noise_psd * env.config.bandwidth
    jammer_power = env.config.jammer_power_min + 0.5 * (float(action_power) + 1.0) * (
        env.config.jammer_power_max - env.config.jammer_power_min
    )
    gamma_ur = (env.config.user_power * gains["h_UR"]) / noise_power
    gamma_rb = (env.config.relay_power * gains["h_RB"]) / noise_power
    r_legit = 0.5 * env.config.bandwidth * np.log2(1.0 + min(gamma_ur, gamma_rb))   # DF bottleneck

    if env.config.use_multiple_eves:
        if env.num_eves == 0:
            r_eve = 0.0
        else:
            gamma_e = (env.config.user_power * gains["h_UE"]) / (
                noise_power + jammer_power * gains["h_JE"]
            )
            r_eve = float(np.max(env.config.bandwidth * np.log2(1.0 + gamma_e)))
    else:
        gamma_e = (env.config.user_power * gains["h_UE"]) / (
            noise_power + jammer_power * gains["h_JE"]
        )
        r_eve = float(env.config.bandwidth * np.log2(1.0 + gamma_e))
    return {"R_legit": float(r_legit), "R_eve": float(r_eve), "R_sec": float(max(r_legit - r_eve, 0.0))}


def _geometry_action(env: UAVEnvironment) -> np.ndarray:
    relay_target = 0.55 * env.user_position[:2] + 0.45 * env.bs_position[:2]
    if env.num_eves > 0:
        dists = np.linalg.norm(env.eve_positions - env.user_position[:2], axis=1)
        worst_eve = env.eve_positions[int(np.argmin(dists))]
        away_user = worst_eve - env.user_position[:2]
        if np.linalg.norm(away_user) > 1e-9:
            away_user = away_user / np.linalg.norm(away_user)
        jammer_target = worst_eve + 80.0 * away_user
    else:
        jammer_target = env.user_position[:2]
    relay_action = _direction_action(env.relay_position[:2], relay_target)
    jammer_action = _direction_action(env.jammer_position[:2], jammer_target)
    return _clip_action(np.array([relay_action[0], relay_action[1], jammer_action[0], jammer_action[1], 0.85]))


def _direction_action(current_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    delta = np.asarray(target_xy, dtype=float) - np.asarray(current_xy, dtype=float)
    norm = np.linalg.norm(delta)
    if norm < 1e-9:
        return np.zeros(2, dtype=float)
    return np.clip(delta / max(norm, 1.0), -1.0, 1.0)


def _clip_action(action: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(action, dtype=float), -1.0, 1.0)


def _linear_decay(start: float, end: float, progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return float(start + (end - start) * progress)


# ──────────────────────────────────────────────────────────────────────────────
# Row builder
# ──────────────────────────────────────────────────────────────────────────────

def _row(
    cfg: SCAConfig,
    ep: int,
    global_step: int,
    ep_reward: float,
    ep_rsec: float,
    ep_rlegit: float,
    ep_reve: float,
    steps: int,
    rolling: list[float],
    trust: float,
    exploration: float,
    ep_inner_gain: float,
    ep_num_eves: float,
    ep_nearest_eve_dist: float,
    ep_mean_eve_dist: float,
    ep_max_eve_cap: float,
) -> dict:
    roll20  = float(np.mean(rolling[-min(20, len(rolling)):]))
    roll100 = float(np.mean(rolling[-min(100, len(rolling)):]))
    return {
        "algorithm": "SCA",
        "episode": ep,
        "global_step": global_step,
        "fading_model": cfg.fading_model,
        "control_mode": cfg.control_mode,
        "role_switching": False,
        "user_mobile": cfg.user_mobile,
        "use_los_model": cfg.use_los_model,
        "observation_mode": cfg.observation_mode,
        "normalize_observations": cfg.normalize_observations,
        "enable_energy_harvesting": False,
        "observation_has_eh": cfg.observation_mode == "full_eh",
        "enable_ntn": False,
        "satellite_altitude_km": "",
        "episode_reward_bps_step": float(ep_reward),
        "avg_shaped_reward": float(ep_reward / max(steps, 1)),
        "avg_R_legit_mbps": float((ep_rlegit / max(steps, 1)) / 1e6),
        "avg_R_eve_mbps": float((ep_reve / max(steps, 1)) / 1e6),
        "avg_R_sec_mbps": float((ep_rsec / max(steps, 1)) / 1e6),
        "avg_num_eves": float(ep_num_eves / max(steps, 1)),
        "avg_nearest_eve_distance": float(ep_nearest_eve_dist / max(steps, 1)),
        "avg_mean_eve_distance": float(ep_mean_eve_dist / max(steps, 1)),
        "avg_max_eve_capacity": float((ep_max_eve_cap / max(steps, 1)) / 1e6),
        "episode_secrecy_mbits": float((ep_rsec * 0.1) / 1e6),
        "steps": steps,
        "rolling20_avg_R_sec_mbps": roll20,
        "rolling100_avg_R_sec_mbps": roll100,
        "convergence_gap20_100_mbps": float(abs(roll20 - roll100)),
        "sca_trust_region": float(trust),
        "sca_exploration_std": float(exploration),
        "sca_avg_inner_gain": float(ep_inner_gain / max(steps, 1)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CSV writer
# ──────────────────────────────────────────────────────────────────────────────

def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Plot: fixed-eval convergence
# ──────────────────────────────────────────────────────────────────────────────

def _plot_fixed_eval_convergence(eval_rows: list[dict], output_dir: Path) -> str:
    if len(eval_rows) < 2:
        return ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = np.array([int(r["episode"]) for r in eval_rows])
    secrecy  = np.array([float(r["fixed_eval_R_sec_mbps"]) for r in eval_rows])
    window = min(5, len(secrecy))
    smooth = np.convolve(secrecy, np.ones(window) / window, mode="valid")
    smooth_ep = episodes[window - 1:]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(episodes, secrecy, color="#7f7f7f", alpha=0.45, linewidth=1.0,
            marker="o", markersize=2.5, label="Fixed HPPP evaluation")
    ax.plot(smooth_ep, smooth, color=_C_EVAL, linewidth=2.2,
            label=f"Smoothed eval ({window}-point)")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Fixed-Eval Secrecy Rate (Mbps)")
    ax.set_title("SCA Fixed-Scenario Convergence (HPPP Eavesdroppers)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "fixed_eval_convergence.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    return str(path)


# ──────────────────────────────────────────────────────────────────────────────
# Plot: SCA objective convergence trace
# ──────────────────────────────────────────────────────────────────────────────

def _plot_sca_objective_convergence(rows: list[dict], output_dir: Path) -> str:
    if len(rows) < 2:
        return ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    output_dir.mkdir(parents=True, exist_ok=True)
    # Use 'iteration' key (new real trace) or fall back to 'episode' (legacy)
    x_key = "iteration" if "iteration" in rows[0] else "episode"
    iterations = np.array([int(r[x_key]) for r in rows])
    observed   = np.array([float(r.get("sca_objective_mbps", r.get("sca_objective_mbps", 0))) for r in rows])
    best       = np.array([float(r["best_objective_mbps"]) for r in rows])
    smooth     = np.array([float(r["smoothed_objective_mbps"]) for r in rows])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(iterations, observed, color=_C_RAW,    alpha=0.22, linewidth=0.7, label="SCA per-iteration objective")
    ax.plot(iterations, best,     color=_C_BEST,   linewidth=1.4, alpha=0.75, label="Best feasible objective")
    ax.plot(iterations, smooth,   color=_C_SMOOTH, linewidth=2.2, label="Smoothed SCA convergence")
    ax.set_xlabel("SCA Iteration")
    ax.set_ylabel("Surrogate Secrecy Objective (Mbps)")
    ax.set_title("SCA Objective Convergence (Fixed Frozen Scenario)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "sca_objective_convergence.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    return str(path)