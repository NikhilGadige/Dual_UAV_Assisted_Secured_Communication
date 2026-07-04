"""
BCD_study/train_bcd.py
======================
Block Coordinate Descent (BCD) baseline for UAV-relay / UAV-jammer
secrecy-rate maximisation under HPPP eavesdroppers and a mobile user.

Optimisations over the original version
-----------------------------------------
1.  Inner step shrinkage (`* 0.55` inside block cycles) now happens only
    after the OUTER block cycle completes, not inside each block.  This
    prevents the step from collapsing too early within a single step.
2.  Multi-start BCD: each step tries a geometry initialisation and
    the current local_action, keeping the better result.
3.  Patience / trust-region restart: if rolling-100 stagnates for
    PATIENCE_EPISODES episodes the step sizes are partially restored.
4.  Anchor warm-up: faster EMA during the first 10 % of training so the
    anchor tracks a good policy before step sizes shrink.
5.  Real convergence trace: _generate_bcd_objective_trace now runs actual
    BCD iterations on a frozen env instead of using synthetic math.
6.  Power block now searched at 7 levels (was 3) for finer granularity.
7.  Consistent colour palette with SCA plots (BCD = blue, SCA = orange).
8.  `block_cycles` default raised to 3 (was 2) for better per-step quality.
9.  eval_episodes raised to 12 (was 8) for a more stable fixed-eval estimate.
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
from SCA_study.train_sca import (
    _clip_action,
    _direction_action,
    _geometry_action,
    _linear_decay,
    _predicted_gains,
    _predict_positions,
    _rates_from_gains,
    _surrogate_objective,
    _write_csv,
)

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette  (BCD = blue so it differs from SCA = orange)
# ──────────────────────────────────────────────────────────────────────────────
_C_RAW    = "#9e9e9e"
_C_BEST   = "#2ca02c"
_C_SMOOTH = "#1f77b4"   # blue for BCD
_C_EVAL   = "#1f77b4"


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BCDConfig:
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

    # BCD inner loop
    block_cycles: int = 3              # was 2 — more cycles per step
    n_restarts: int = 2                # multi-start: geometry + local_action init
    relay_step_start: float = 0.80
    relay_step_end: float = 0.05       # was 0.06
    jammer_step_start: float = 0.90
    jammer_step_end: float = 0.06      # was 0.07
    power_step_start: float = 0.70
    power_step_end: float = 0.04
    inner_step_decay: float = 0.65     # decay PER BLOCK CYCLE (was applied per block)

    # Patience / restart
    patience_episodes: int = 120
    restart_fraction: float = 0.60    # fraction of start steps to restore on patience

    # Exploration / smoothing
    smoothing_start: float = 0.25
    smoothing_end: float = 0.80
    exploration_start: float = 0.30
    exploration_end: float = 0.01

    # Anchor warm-up
    anchor_warmup_frac: float = 0.10
    anchor_ema_fast: float = 0.030
    anchor_ema_slow: float = 0.005

    # Objective penalties
    power_penalty: float = 0.012
    motion_penalty: float = 0.020
    improvement_tol: float = 1e-5

    # Logging / evaluation
    log_interval: int = 100
    eval_interval: int = 50
    eval_episodes: int = 12            # was 8
    eval_seed: int = 9100
    eval_smoothing: float = 0.75

    # Convergence-trace fixed scenario
    convergence_seed: int = 13000
    convergence_episodes: int = 300    # actual BCD iters on frozen env


# ──────────────────────────────────────────────────────────────────────────────
# Environment factory
# ──────────────────────────────────────────────────────────────────────────────

def make_env_config(seed: int, cfg: BCDConfig) -> EnvConfig:
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

def train_bcd(cfg: BCDConfig, output_dir: str) -> dict:
    rng = np.random.default_rng(cfg.seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    eval_rows: list[dict] = []
    rolling: list[float] = []
    latest_fixed_eval: float | str = ""
    global_step = 0
    policy_anchor = np.array([0.0, 0.0, 0.0, 0.0, 0.60], dtype=float)

    best_roll100 = -np.inf
    patience_counter = 0
    # Current step starts; can be reset on patience trigger
    cur_relay_start   = cfg.relay_step_start
    cur_jammer_start  = cfg.jammer_step_start
    cur_power_start   = cfg.power_step_start

    warmup_cutoff = int(cfg.anchor_warmup_frac * cfg.episodes)

    for ep in range(1, cfg.episodes + 1):
        progress   = (ep - 1) / max(cfg.episodes - 1, 1)
        relay_step  = _linear_decay(cur_relay_start,  cfg.relay_step_end,  progress)
        jammer_step = _linear_decay(cur_jammer_start, cfg.jammer_step_end, progress)
        power_step  = _linear_decay(cur_power_start,  cfg.power_step_end,  progress)
        smooth      = _linear_decay(cfg.smoothing_start, cfg.smoothing_end, progress)
        exploration = _linear_decay(cfg.exploration_start, cfg.exploration_end, progress)
        anchor_ema  = cfg.anchor_ema_fast if ep <= warmup_cutoff else cfg.anchor_ema_slow

        env = UAVEnvironment(make_env_config(cfg.seed + ep, cfg))
        with contextlib.redirect_stdout(io.StringIO()):
            env.reset()

        ep_reward = ep_rsec = ep_rlegit = ep_reve = 0.0
        ep_num_eves = ep_nearest_eve_dist = ep_mean_eve_dist = ep_max_eve_cap = 0.0
        ep_block_gain = 0.0
        steps = 0
        done = False
        local_action = policy_anchor.copy()

        while not done:
            nominal = _geometry_action(env)

            # ── Multi-start: try geometry init + policy init ──
            x0_policy   = _clip_action(0.60 * local_action + 0.40 * nominal)
            x0_geometry = _clip_action(nominal)

            if exploration > 0.0:
                x0_policy   = _clip_action(x0_policy   + rng.normal(0.0, exploration, size=5))
                x0_geometry = _clip_action(x0_geometry + rng.normal(0.0, exploration * 0.5, size=5))

            best_action = x0_policy
            best_gain   = -np.inf
            for x0 in ([x0_policy, x0_geometry] if cfg.n_restarts >= 2 else [x0_policy]):
                cand_action, cand_gain = _bcd_refine_action(
                    env, x0, cfg,
                    relay_step=relay_step,
                    jammer_step=jammer_step,
                    power_step=power_step,
                )
                if cand_gain > best_gain:
                    best_gain   = cand_gain
                    best_action = cand_action

            action     = best_action
            block_gain = best_gain if best_gain > 0.0 else 0.0

            local_action  = _clip_action((1.0 - smooth) * local_action + smooth * action)
            policy_anchor = _clip_action(
                (1.0 - anchor_ema) * policy_anchor + anchor_ema * local_action
            )

            _, reward, done, info = env.step(
                local_action[:2], local_action[2:4], float(local_action[4]), False,
            )
            global_step += 1
            steps += 1
            ep_reward         += float(reward)
            ep_rsec           += float(info["R_sec"])
            ep_rlegit         += float(info["R_legit"])
            ep_reve           += float(info["R_eve"])
            ep_num_eves       += float(info.get("num_eves", 1))
            ep_nearest_eve_dist += float(info.get("nearest_eve_distance", 0.0))
            ep_mean_eve_dist  += float(info.get("mean_eve_distance", 0.0))
            ep_max_eve_cap    += float(info.get("max_eve_capacity", 0.0))
            ep_block_gain     += block_gain

        avg_sec = float((ep_rsec / max(steps, 1)) / 1e6)
        rolling.append(avg_sec)

        # ── Patience / step-size restart ──
        roll100 = float(np.mean(rolling[-min(100, len(rolling)):]))
        if roll100 > best_roll100 + cfg.improvement_tol:
            best_roll100  = roll100
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= cfg.patience_episodes:
            cur_relay_start   = cfg.restart_fraction * cfg.relay_step_start
            cur_jammer_start  = cfg.restart_fraction * cfg.jammer_step_start
            cur_power_start   = cfg.restart_fraction * cfg.power_step_start
            patience_counter  = 0

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
            relay_step=relay_step,
            jammer_step=jammer_step,
            power_step=power_step,
            exploration=exploration,
            ep_block_gain=ep_block_gain,
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
                "fixed_eval_avg_block_gain": fixed_eval["avg_block_gain"],
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
                f"BCD {cfg.fading_model} ep={ep:5d}/{cfg.episodes} "
                f"avg_R_sec={avg_sec:8.4f} Mbps roll100={roll100:8.4f} "
                f"relay_step={relay_step:.3f} power_step={power_step:.3f} "
                f"patience={patience_counter}{fixed_text}"
            )

    # ── Save logs ──
    log_path      = out_dir / "training_log.csv"
    eval_log_path = out_dir / "fixed_eval_log.csv"
    _write_csv(rows, log_path)
    _write_csv(eval_rows, eval_log_path)
    plot_paths = plot_training_curves(str(log_path), str(out_dir / "plots"))
    fixed_eval_plot = _plot_fixed_eval_convergence(eval_rows, out_dir / "plots")
    if fixed_eval_plot:
        plot_paths["fixed_eval_convergence"] = fixed_eval_plot

    # ── Real BCD convergence trace on a frozen scenario ──
    bcd_trace_rows = _generate_bcd_objective_trace(cfg)
    bcd_trace_path = out_dir / "bcd_convergence_log.csv"
    _write_csv(bcd_trace_rows, bcd_trace_path)
    bcd_trace_plot = _plot_bcd_objective_convergence(bcd_trace_rows, out_dir / "plots")
    if bcd_trace_plot:
        plot_paths["bcd_objective_convergence"] = bcd_trace_plot

    # ── Checkpoint ──
    checkpoint_path = out_dir / "bcd_checkpoint.npz"
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
        "algorithm": "bcd",
        "config": asdict(cfg),
        "training_log_csv": str(log_path.resolve()),
        "fixed_eval_log_csv": str(eval_log_path.resolve()),
        "bcd_convergence_log_csv": str(bcd_trace_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "plot_paths": plot_paths,
        "final_rolling100_R_sec_mbps": float(np.mean(rolling[-min(100, len(rolling)):])),
        "final_fixed_eval_R_sec_mbps": float(eval_rows[-1]["fixed_eval_R_sec_mbps"]) if eval_rows else None,
        "final_bcd_objective_mbps": float(bcd_trace_rows[-1]["smoothed_objective_mbps"]) if bcd_trace_rows else None,
        "best_episode_R_sec_mbps": float(np.max(rolling)),
        "mean_last100_reward": float(np.mean([r["avg_shaped_reward"] for r in rows[-min(100, len(rows)):]]))
    }
    metadata_path = out_dir / "bcd_summary.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "algorithm": "bcd",
        "training_log_csv": str(log_path.resolve()),
        "fixed_eval_log_csv": str(eval_log_path.resolve()),
        "bcd_convergence_log_csv": str(bcd_trace_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "summary_json": str(metadata_path.resolve()),
        "mean_avg_rsec_mbps": metadata["final_rolling100_R_sec_mbps"],
        "best_episode_R_sec_mbps": metadata["best_episode_R_sec_mbps"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# BCD inner solver
# ──────────────────────────────────────────────────────────────────────────────

def _bcd_refine_action(
    env: UAVEnvironment,
    x0: np.ndarray,
    cfg: BCDConfig,
    relay_step: float,
    jammer_step: float,
    power_step: float,
) -> tuple[np.ndarray, float]:
    """BCD: cycle over (relay, jammer, power) blocks, accepting improvements.

    Key fix: step sizes are decayed ONCE per full cycle (not per block),
    so the relay block still has a non-trivial step when jammer and power
    are being optimised in the same cycle.
    """
    x = _clip_action(x0)
    best = _bcd_objective(env, x, cfg)
    total_gain = 0.0

    for _ in range(max(cfg.block_cycles, 1)):
        for block, step in (("relay", relay_step), ("jammer", jammer_step), ("power", power_step)):
            x, best, gain = _optimize_block(env, x, best, cfg, block, step)
            total_gain += gain
        # Decay AFTER the full cycle, not inside each block (FIX)
        relay_step  *= cfg.inner_step_decay
        jammer_step *= cfg.inner_step_decay
        power_step  *= cfg.inner_step_decay

    return x, float(total_gain)


def _optimize_block(
    env: UAVEnvironment,
    x: np.ndarray,
    best: float,
    cfg: BCDConfig,
    block: str,
    step: float,
) -> tuple[np.ndarray, float, float]:
    candidates = [x]
    geom = _geometry_action(env)

    if block == "relay":
        candidates.append(_replace_block(x, slice(0, 2), geom[:2]))
        indices = (0, 1)
    elif block == "jammer":
        candidates.append(_replace_block(x, slice(2, 4), geom[2:4]))
        indices = (2, 3)
    else:
        indices = (4,)
        # 7 power levels (was 3) for finer grid search
        for pwr in (-1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0):
            candidates.append(_replace_power(x, pwr))

    for idx in indices:
        plus = x.copy(); minus = x.copy()
        plus[idx] += step; minus[idx] -= step
        candidates.extend([_clip_action(plus), _clip_action(minus)])

    best_x = x
    gain = 0.0
    for cand in candidates:
        val = _bcd_objective(env, cand, cfg)
        if val > best + cfg.improvement_tol:
            gain += val - best
            best  = val
            best_x = cand
    return _clip_action(best_x), best, float(gain)


def _replace_block(x: np.ndarray, block_slice: slice, values: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[block_slice] = values
    return _clip_action(y)


def _replace_power(x: np.ndarray, value: float) -> np.ndarray:
    y = x.copy()
    y[4] = value
    return _clip_action(y)


def _bcd_objective(env: UAVEnvironment, action: np.ndarray, cfg: BCDConfig) -> float:
    relay_pos, jammer_pos = _predict_positions(env, action)
    gains = _predicted_gains(env, relay_pos, jammer_pos)
    rates = _rates_from_gains(env, gains, action[4])
    secrecy_mbps = rates["R_sec"] / 1e6
    motion_cost  = cfg.motion_penalty * (np.linalg.norm(action[:2]) + np.linalg.norm(action[2:4]))
    power_cost   = cfg.power_penalty  * ((action[4] + 1.0) * 0.5)
    return float(secrecy_mbps - motion_cost - power_cost)


# ──────────────────────────────────────────────────────────────────────────────
# Fixed-scenario evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate_policy_anchor(cfg: BCDConfig, policy_anchor: np.ndarray) -> dict:
    rewards, rsecs, rlegits, reverses, num_eves, block_gains = [], [], [], [], [], []

    for idx in range(max(cfg.eval_episodes, 1)):
        env = UAVEnvironment(make_env_config(cfg.eval_seed + idx, cfg))
        with contextlib.redirect_stdout(io.StringIO()):
            env.reset()

        done = False
        steps = 0
        local_action = policy_anchor.copy()
        ep_reward = ep_rsec = ep_rlegit = ep_reve = ep_num_eves = ep_block_gain = 0.0

        while not done:
            x0 = _clip_action(0.70 * local_action + 0.30 * _geometry_action(env))
            action, bg = _bcd_refine_action(
                env, x0, cfg,
                relay_step=cfg.relay_step_end,
                jammer_step=cfg.jammer_step_end,
                power_step=cfg.power_step_end,
            )
            local_action = _clip_action(
                (1.0 - cfg.eval_smoothing) * local_action + cfg.eval_smoothing * action
            )
            _, reward, done, info = env.step(
                local_action[:2], local_action[2:4], float(local_action[4]), False,
            )
            steps += 1
            ep_reward    += float(reward)
            ep_rsec      += float(info["R_sec"])
            ep_rlegit    += float(info["R_legit"])
            ep_reve      += float(info["R_eve"])
            ep_num_eves  += float(info.get("num_eves", 1))
            ep_block_gain += bg

        rewards.append(ep_reward / max(steps, 1))
        rsecs.append((ep_rsec / max(steps, 1)) / 1e6)
        rlegits.append((ep_rlegit / max(steps, 1)) / 1e6)
        reverses.append((ep_reve / max(steps, 1)) / 1e6)
        num_eves.append(ep_num_eves / max(steps, 1))
        block_gains.append(ep_block_gain / max(steps, 1))

    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_R_sec_mbps": float(np.mean(rsecs)),
        "avg_R_legit_mbps": float(np.mean(rlegits)),
        "avg_R_eve_mbps": float(np.mean(reverses)),
        "avg_num_eves": float(np.mean(num_eves)),
        "avg_block_gain": float(np.mean(block_gains)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Real BCD convergence trace (fixed frozen scenario)
# ──────────────────────────────────────────────────────────────────────────────

def _generate_bcd_objective_trace(cfg: BCDConfig) -> list[dict]:
    """Run actual BCD iterations on a single frozen environment.

    Provides an honest convergence curve that reflects real BCD behaviour
    rather than synthetic math.  The x-axis of the resulting plot is
    'BCD iteration', not 'training episode'.
    """
    rng = np.random.default_rng(cfg.seed + 88888)
    trace_cfg = make_env_config(cfg.convergence_seed, cfg)
    trace_cfg.user_mobile = False   # freeze the scene
    trace_cfg.max_steps = 1
    env = UAVEnvironment(trace_cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        env.reset()

    x = _clip_action(np.array([-0.75, 0.55, -0.65, -0.50, -0.45], dtype=float))
    best_objective = _bcd_objective(env, x, cfg)
    smooth_objective = best_objective
    rows = []

    n_iters = max(cfg.convergence_episodes, 50)

    for it in range(1, n_iters + 1):
        progress    = (it - 1) / max(n_iters - 1, 1)
        relay_step  = _linear_decay(cfg.relay_step_start,  cfg.relay_step_end,  progress)
        jammer_step = _linear_decay(cfg.jammer_step_start, cfg.jammer_step_end, progress)
        power_step  = _linear_decay(cfg.power_step_start,  cfg.power_step_end,  progress)
        noise       = _linear_decay(0.30, 0.0, progress)

        # Multi-start: geometry + noisy perturbation
        x0_geom  = _clip_action(_geometry_action(env))
        x0_noisy = _clip_action(x + rng.normal(0.0, max(noise, 0.02), size=5))

        best_x_iter   = x
        best_obj_iter = best_objective
        best_gain_iter = 0.0
        for x0 in [x, x0_geom, x0_noisy]:
            cand, gain = _bcd_refine_action(env, x0, cfg, relay_step, jammer_step, power_step)
            cand_obj = _bcd_objective(env, cand, cfg)
            if cand_obj > best_obj_iter + cfg.improvement_tol:
                best_obj_iter  = cand_obj
                best_x_iter    = cand
                best_gain_iter = gain

        if best_obj_iter > best_objective + cfg.improvement_tol:
            best_objective = best_obj_iter
            x = best_x_iter
        else:
            x = _clip_action(0.998 * x + 0.002 * _geometry_action(env))

        smooth_alpha = 0.04 + 0.06 * (1.0 - progress)
        smooth_objective = (1.0 - smooth_alpha) * smooth_objective + smooth_alpha * best_objective

        rows.append({
            "iteration": it,
            "fading_model": cfg.fading_model,
            "bcd_objective_mbps": float(best_obj_iter),
            "best_objective_mbps": float(best_objective),
            "smoothed_objective_mbps": float(smooth_objective),
            "block_gain_mbps": float(best_gain_iter),
            "relay_step": float(relay_step),
            "jammer_step": float(jammer_step),
            "power_step": float(power_step),
            "convergence_noise": float(noise),
            "action_relay_x": float(x[0]),
            "action_relay_y": float(x[1]),
            "action_jammer_x": float(x[2]),
            "action_jammer_y": float(x[3]),
            "action_power": float(x[4]),
        })

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Row builder
# ──────────────────────────────────────────────────────────────────────────────

def _row(
    cfg: BCDConfig,
    ep: int,
    global_step: int,
    ep_reward: float,
    ep_rsec: float,
    ep_rlegit: float,
    ep_reve: float,
    steps: int,
    rolling: list[float],
    relay_step: float,
    jammer_step: float,
    power_step: float,
    exploration: float,
    ep_block_gain: float,
    ep_num_eves: float,
    ep_nearest_eve_dist: float,
    ep_mean_eve_dist: float,
    ep_max_eve_cap: float,
) -> dict:
    roll20  = float(np.mean(rolling[-min(20,  len(rolling)):]))
    roll100 = float(np.mean(rolling[-min(100, len(rolling)):]))
    return {
        "algorithm": "BCD",
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
        "bcd_relay_step": float(relay_step),
        "bcd_jammer_step": float(jammer_step),
        "bcd_power_step": float(power_step),
        "bcd_exploration_std": float(exploration),
        "bcd_avg_block_gain": float(ep_block_gain / max(steps, 1)),
    }


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
    ax.set_title("BCD Fixed-Scenario Convergence (HPPP Eavesdroppers)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "fixed_eval_convergence.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    return str(path)


# ──────────────────────────────────────────────────────────────────────────────
# Plot: BCD objective convergence trace
# ──────────────────────────────────────────────────────────────────────────────

def _plot_bcd_objective_convergence(rows: list[dict], output_dir: Path) -> str:
    if len(rows) < 2:
        return ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    output_dir.mkdir(parents=True, exist_ok=True)
    x_key      = "iteration" if "iteration" in rows[0] else "episode"
    iterations = np.array([int(r[x_key]) for r in rows])
    observed   = np.array([float(r.get("bcd_objective_mbps", 0)) for r in rows])
    best       = np.array([float(r["best_objective_mbps"]) for r in rows])
    smooth     = np.array([float(r["smoothed_objective_mbps"]) for r in rows])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(iterations, observed, color=_C_RAW,    alpha=0.22, linewidth=0.7, label="BCD per-iteration objective")
    ax.plot(iterations, best,     color=_C_BEST,   linewidth=1.4, alpha=0.75, label="Best block objective")
    ax.plot(iterations, smooth,   color=_C_SMOOTH, linewidth=2.2, label="Smoothed BCD convergence")
    ax.set_xlabel("BCD Iteration")
    ax.set_ylabel("Surrogate Secrecy Objective (Mbps)")
    ax.set_title("BCD Objective Convergence (Fixed Frozen Scenario)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "bcd_objective_convergence.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    return str(path)