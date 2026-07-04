# madrl_updated_exp — Updated ISAC MADRL Model

This is an updated copy of `madrl_exp/` (the original 3-agent MADRL
implementation) that wires in three pieces of physics that already
existed elsewhere in the repo but were never connected to the MADRL
training loop. `madrl_exp/` is left untouched; this folder is the one to
use going forward.

## What changed vs. `madrl_exp/`

1. **Random-Walk mobile user.** The user position was static
   (`q_user = [200, 0, 1.5]` forever). It now moves every step using
   `mobility_types_exp.mobility_models.RandomWalkMobility`
   (correlated-random-walk turning + reflecting at the scene boundary),
   capped at `EnvConfig.user_speed_max` (default 3 m/s).

2. **HPPP-distributed eavesdroppers.** The 3 eavesdropper positions were
   hardcoded. They are now re-sampled every `reset()` from a Homogeneous
   Poisson Point Process over the scene footprint, using the same
   `eve_density_lambda` (default `2e-5`, matching `fd_jammer_exp`).
   The active count is genuinely Poisson-distributed (typically 0–3);
   unused slots (when the draw is below the 3-slot cap) are pushed far
   outside the scene so they contribute ~0 channel gain instead of
   distorting observations or the secrecy rate. `env.n_active_eves` tells
   you the true count each episode.

3. **RIS phase-shift as a real decision variable.** Previously `phi_n`
   was fixed by a closed-form conjugate-alignment heuristic
   (`design_ris_phases`), never learned. There is now a 4th agent,
   `ris_phase`, whose action *is* the per-slot RIS phase vector
   (`N_time * N_ris` = 80 dims by default), matching the proposed model's
   `Phi = diag(e^{jφ_1}, ..., e^{jφ_16})` with `β_n = 1`. The closed-form
   heuristic phase is still fed to the agent as an observation hint so it
   has something sensible to anchor its exploration around.

   `compute_secrecy_rate()` in
   `optimization_problem_exp/optimization/problem_formulation.py` gained
   a new optional `phi_override` parameter for this (default `None`,
   so every other caller — `sca_bcd_exp`, `madrl_exp`, etc. — is
   unaffected).

All four agents still share the single global weighted-objective reward
(`alpha * secrecy_norm + (1 - alpha) * sensing_norm - penalties`).

### Incidental fix

While smoke-testing, `fd_jammer_exp/channels/fd_jammer_channel.py:compute_jammer_gain`
crashed under NumPy 2.x (`float()` on a non-0-d array is now a hard
error, it used to silently work). Fixed with `.item()`. This affected
`madrl_exp` too — it was never actually runnable in this NumPy version
before this fix.

## What this does *not* address

Per the proposed-model gap analysis, these are still not implemented
(out of scope for this pass — they need new physics, not just wiring):
OFDM/subcarrier-level waveform processing, FD-jammer self-interference
and monostatic echo reception, and Selection Combining between bistatic
and monostatic echoes.

## Running training (epochs)

Each "episode" in the code is one epoch of interaction (`--steps` steps
per episode, all 4 agents act every step). From the repo root, using
whichever Python environment has `numpy`, `torch`, and `gymnasium`
installed:

```bash
python -m madrl_updated_exp.run --algorithm mappo --episodes 200 --steps 100 --alpha 0.5 --mode all
```

Arguments (`madrl_updated_exp/run.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--algorithm` | `mappo` | `mappo` or `matd3` |
| `--episodes` | `100` | number of training epochs |
| `--steps` | `100` | steps per episode |
| `--alpha` | `0.5` | secrecy/sensing reward weight (`alpha * secrecy + (1-alpha) * sensing`) |
| `--seed` | `42` | RNG seed |
| `--mode` | `train` | `train`, `eval`, or `all` (train then compare vs. random/SCA-BCD baselines) |

Example — quick smoke run (fast, just to confirm everything moves):

```bash
python -m madrl_updated_exp.run --algorithm mappo --episodes 3 --steps 5 --mode train
```

Example — a real training run with evaluation against baselines:

```bash
python -m madrl_updated_exp.run --algorithm mappo --episodes 300 --steps 100 --alpha 0.5 --mode all
```

Outputs land in `outputs/reinforcement_learning/training_runs/<algorithm>_<timestamp>/`:
- `checkpoints/{bs_beamformer,uav_trajectory,jammer_beamformer,ris_phase}_ep<N>.pt`
- `csv/episode_{rewards,secrecy,sensing,constraints}.csv`, `csv/loss_history.csv`
- `logs/history.json`

To change environment/training/reward hyperparameters beyond the CLI
flags (e.g. `user_speed_max`, `eve_density_lambda`, `N_time`, `P_bs_max`),
edit `madrl_updated_exp/configs.py` or construct a custom `MADRLConfig`
and call `MARLTrainer(cfg).train()` directly (see `run.py` for the
pattern).

### Dependencies

If `gymnasium` isn't installed in your environment:

```bash
pip install gymnasium
```

(`numpy`, `torch` are already required by the rest of the repo.)
