# 7th July Presentation — Multi-Agent Sensing (CRB + Pd)

This folder is the sensing-only companion to `madrl_updated_exp/`: instead
of the secrecy+sensing ISAC environment, it isolates the **sensing**
half of the proposed model — minimize Cramer-Rao Bound (CRB), maximize
detection probability (Pd) — and compares four control strategies for it:
**MAPPO**, **MATD3PG**, **MADDPG**, and a non-learning **Random-Walk**
baseline. Each gets its own subfolder with training code, logs, and
convergence plots for both CRB and Pd, so results are easy to pull
straight into presentation slides.

## Scenario

- `n_agents` (default 2) sensing UAVs independently control 2D velocity
  to position themselves around `n_vehicles` (default 3: car/truck/
  motorcycle, RCS from `vehicle_reflection_exp.compute_rcs`) vehicle
  targets in a 400m x 300m scene.
- Each agent runs a monostatic ULA (8 Tx/Rx elements) sensing pilot
  against every target every step. Per-agent:
  - **CRB**: Fisher Information Matrix over target angle-of-arrival →
    `CRB = J^-1` (`crb_sensing_exp`), reduced to a log-utility
    (`optimization_problem_exp`'s convention: `U = -log10(CRB_trace)`).
  - **Pd**: energy-detector test statistic vs. an analytically-derived
    Neyman-Pearson threshold (`detection_sensing_exp`), Pd estimated by
    Monte Carlo (40 trials during training, 200 for final eval).
  - Target reflection channel is **Rician-faded**
    (`vehicle_reflection_exp._generate_scalar_rician`, K = 5 dB), per the
    proposed ISAC model's channel-model section.
- Reward (shared team signal): `alpha_pd * mean(Pd) + (1-alpha_pd) * zscore(mean(U_CRB))`,
  `alpha_pd = 0.5` by default.
- Episode = 40 steps; agents reset to random positions and targets
  re-sample each episode.

**Documented simplification**: each agent's CRB/Pd is computed from its
own independent monostatic vantage point and then averaged into the team
reward — this rewards agents for individually finding good sensing
geometry (and implicitly discourages collapsing onto the same spot,
since duplicated viewpoints don't raise the mean). It is **not** a full
multistatic Fisher-information fusion across sensors (that needs
reparametrizing each sensor's angle-domain FIM into a shared
position-domain FIM and summing) — flagged here as a natural next step,
not implemented.

## What's reused vs. new

| Piece | Source |
|---|---|
| CRB / FIM math | `crb_sensing_exp` (unchanged) |
| Detection statistic | `detection_sensing_exp` (unchanged) |
| Rician target channel | `vehicle_reflection_exp._generate_scalar_rician` (unchanged) |
| MAPPO agent | `madrl_updated_exp.agents.mappo.MAPPOAgent` (unchanged) — same actor-critic design as `PPO_study/train_ppo.py` |
| MATD3PG agent | `madrl_updated_exp.agents.matd3.MATD3Agent` (unchanged) — same twin-critic/TD3 design as `td3pg_study/td3pg_train.py` |
| MADDPG agent | **new** `common/ddpg_agent.py`, modeled directly on `rl/ddpg_train.py`'s Actor/Critic/OU-noise design, repackaged behind the same per-agent interface |
| Random Walk | `mobility_types_exp.mobility_models.RandomWalkMobility` (unchanged) |
| Environment, trainer, plotting | **new** (`common/sensing_env.py`, `common/trainer.py`, `common/plotting.py`, `common/comparison.py`) — this is the sensing-focused environment that didn't exist before |

All three learned algorithms are **independent per-agent learners**
(each UAV has its own actor/critic) sharing one global team reward — the
same decentralized-actor pattern already used in `madrl_updated_exp`.

## Folder layout

```
7th july presentation/
  common/            sensing_metrics.py, sensing_env.py, ddpg_agent.py,
                      trainer.py, plotting.py, comparison.py
  mappo/             train_mappo.py            -> output/mappo_<ts>/
  matd3pg/           train_matd3pg.py          -> output/matd3pg_<ts>/
  maddpg/            train_maddpg.py           -> output/maddpg_<ts>/
  random_walk/       run_random_walk.py        -> output/random_walk_<ts>/
  run_all.py         runs all 4 + cross-algorithm comparison plots
  comparison_plots/  overlay plots from run_all.py (created on first run)
```

Each `output/<algo>_<timestamp>/` contains:
- `csv/training_log.csv` — episode, avg_crb, avg_pd, avg_reward, loss stats
- `plots/crb_convergence.png`, `pd_convergence.png`,
  `combined_convergence.png` (dual-axis CRB+Pd), `reward_convergence.png`
- `checkpoints/*.pt` per agent (MAPPO/MATD3PG/MADDPG only — no
  checkpoints for Random Walk, there's nothing to learn)
- `history.json`

## Running it

Note: the top folder name has a space, so run these as **scripts**
(`python "7th july presentation/.../train_x.py"`), not as `-m` modules.

Run everything and get the comparison plots in one go:
```bash
python "7th july presentation/run_all.py" --episodes 300 --steps 40
```

Or run one algorithm at a time:
```bash
python "7th july presentation/mappo/train_mappo.py"       --episodes 300 --steps 40
python "7th july presentation/matd3pg/train_matd3pg.py"   --episodes 300 --steps 40
python "7th july presentation/maddpg/train_maddpg.py"     --episodes 300 --steps 40
python "7th july presentation/random_walk/run_random_walk.py" --episodes 300 --steps 40
```

Shared flags: `--episodes` (default 300), `--steps` per episode (default
40), `--n-agents` (default 2), `--alpha-pd` (Pd vs. CRB reward weight,
default 0.5), `--seed` (default 42), `--output-root` (per-script only).

Dependencies: `numpy`, `scipy`, `torch`, `matplotlib` (matplotlib needed
a rebuild against NumPy 2.x in this environment — if you hit
`AttributeError: _ARRAY_API not found`, run `pip install --upgrade matplotlib`).
