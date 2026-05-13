# Secure Dual-UAV Cooperative Communication using Reinforcement Learning

## 1. Project Overview

This project simulates a dual-UAV cooperative communication system where two unmanned aerial vehicles — a **relay UAV** and a **jammer UAV** — work together to maintain a secure link between a ground user and a base station, in the presence of a ground eavesdropper.

The relay UAV relays data from the user to the base station. The jammer UAV transmits interference to degrade the eavesdropper's channel. Both UAVs learn their trajectories and transmit power jointly via reinforcement learning. The objective is to maximize the **secrecy rate** (the difference between the legitimate link rate and the eavesdropper's rate).

The simulation supports **Rayleigh** and **Rician** fading channels, **energy harvesting** with battery-aware rewards, and **NTN (non-terrestrial network)** satellite-assisted relay links. The environment is built as a Gym-style discrete-time simulation with configurable observation and reward structures.

### Implemented algorithms

| Algorithm | Type | Action space |
|-----------|------|-------------|
| DQN | Single-agent discrete | 21 discrete actions (velocity + power) |
| DDPG | Single-agent continuous | 5-dim continuous (positions + power) |
| TD3 | Single-agent continuous | 5-dim continuous |
| SAC | Single-agent continuous | 5-dim continuous |
| PPO | Single-agent continuous | 5-dim continuous |
| MARL-DQN (shared) | Multi-agent, shared observation | Relay (21) + Jammer (168) discrete |
| MARL-DQN (split) | Multi-agent, split observation | Same action spaces, partial obs |

---

## 2. System Architecture

### Roles

- **Ground User (U)**: Source of data. Transmits to the relay UAV.
- **Relay UAV**: Decodes and forwards user data to the base station. Learns to position itself for best link quality.
- **Jammer UAV**: Transmits artificial interference toward the eavesdropper. Learns to position and set power level.
- **Base Station (BS)**: Destination of the legitimate data.
- **Eavesdropper (Eve)**: Passive adversary trying to intercept the transmission.
- **NTN Satellite** (optional): Provides an alternative relay path via satellite link.

### Communication flow

```
User  ---(Uplink)--->  Relay UAV  ---(Relay)--->  Base Station
                         |
                    [Jammer UAV] --(Jamming)-->  Eve
                         |
                    (optional) NTN satellite ---> Relay UAV
```

The relay UAV receives from the user and forwards to the BS. The jammer UAV transmits interference targeted at Eve's location. Both UAVs move in 2D space (fixed altitude) and can adjust their transmit power. The secrecy rate at each step is `max(R_legit - R_eve, 0)`.

### ASCII diagram

```
                              +------------------+
                              |   NTN Satellite  |
                              +--------+---------+
                                       | (optional sat link)
    +-------+      Uplink      +--------v--------+      Relay       +-----------+
    | User  | -------------->  |   Relay UAV     | ---------------> |    BS     |
    | (GND) |                  |  (learns traj)  |                  | (GND)     |
    +-------+                  +--------+--------+                  +-----------+
                                       |
                                       | (cooperative jamming)
                                       |
                              +--------v--------+
                              |  Jammer UAV     | - - - - - >  +-----------+
                              |  (learns traj)  |              |    Eve    |
                              +-----------------+              | (adversary|
                                                               +-----------+
```

---

## 3. Project Structure

```
├── core/                    # Simulation environment
│   ├── environment.py       # UAVEnvironment (step, reset, rates, gains)
│   ├── config_utils.py      # EnvConfig factory
│   ├── channel.py           # Path loss, fading (Rayleigh/Rician), LoS model
│   ├── ntn_channel.py       # Satellite channel gain model
│   ├── observation.py       # build_observation for all 5 modes
│   ├── observation_schema.py# Slice indices and feature offsets
│   ├── energy.py            # Energy usage, harvesting, battery state
│   └── reward.py            # Reward components
│
├── rl/                      # RL training scripts
│   ├── dqn_train.py         # DQN training (epsilon-greedy, replay buffer)
│   ├── ddpg_train.py        # DDPG training (actor-critic, OU noise)
│   ├── advanced_rl_train.py # TD3, SAC, PPO training
│   ├── marl_dqn_train.py    # MARL DQN (shared/split observation modes)
│   ├── marl_utils.py        # Observation splitting and action tables
│   ├── dqn_evaluate.py      # Multi-seed DQN vs baseline evaluation
│   └── ddpg_evaluate.py     # Multi-seed DDPG vs baseline evaluation
│
├── analysis/                # Analysis and plotting scripts
│   ├── dqn_analysis.py      # DQN/MARL training curve plots
│   ├── ddpg_analysis.py     # DDPG/TD3/SAC/PPO training curve plots
│   ├── trajectory_plots.py  # 2D trajectory rollout + visualization
│   ├── final_comparison.py  # Cross-method comparison runner
│   ├── paper_reproduction.py# Full pipeline orchestrator
│   ├── paper_plots.py       # Paper-grade comparison figures
│   ├── rl_channel_experiments.py  # DQN + DDPG x Rician + Rayleigh matrix
│   ├── baselines.py         # Random and distance-greedy policies
│   └── experiments.py       # Baseline experiment runner
│
├── outputs/                 # All generated artifacts
│
├── .gitignore
└── README.md
```

### What each major file does

- **core/environment.py** — Gym-style `UAVEnvironment` class. Manages positions, velocities, power, rates, energy, and rewards per step.
- **core/observation.py** — Builds observation vectors from environment state. Supports 5 observation modes.
- **core/reward.py** — Computes per-step reward from secrecy rate, energy usage, motion penalties, and bonuses.
- **rl/dqn_train.py** — Trains a DQN agent on the environment. Saves model weights and training log.
- **rl/ddpg_train.py** — Trains a DDPG agent with actor-critic and OU noise.
- **rl/advanced_rl_train.py** — Unified trainer for TD3, SAC, and PPO.
- **rl/marl_dqn_train.py** — Multi-agent DQN with either shared or split observations between relay and jammer.
- **analysis/final_comparison.py** — Evaluates all trained models + baselines and generates comparison plots.
- **analysis/trajectory_plots.py** — Rolls out a trained policy and plots the 2D flight path.
- **analysis/paper_reproduction.py** — Runs the full pipeline from training through final plots.

---

## 4. Features Implemented

- Rayleigh fading channel model
- Rician fading channel model (K-factor configurable)
- LoS-aware path loss (sigmoid LoS probability)
- Energy harvesting (solar-based, configurable efficiency)
- Battery-aware reward with depletion penalty
- Trajectory learning (UAVs learn to position themselves)
- Cooperative jamming (jammer targets Eve's channel)
- NTN satellite-assisted relay link
- MARL with shared observation (both agents see full state)
- MARL with split observation (each agent sees only its relevant state)
- Replay buffers (experience replay for all off-policy algorithms)
- Convergence logging (per-episode CSV logs)
- Trajectory visualization (2D flight path plots)
- Role switching (relay/jammer can swap roles)
- Mobile user (user moves during episode)

---

## 5. Observation Modes

The environment supports five observation modes, selected via `--observation-mode`.

| Mode | Dimension | Contents |
|------|-----------|----------|
| `geometry` | 21 | Relay/jammer/user/BS/Eve positions + relay/jammer/user velocities |
| `channels` | 11 | Channel gains (4), SNRs (3), rates (3), jammer power (1) |
| `full` | 38 | Geometry (21) + distances (4) + channels (11) + battery levels (2) |
| `full_eh` | 43 | Full (38) + energy harvesting features (5): battery ratios, harvest power, saturation flag |
| `full_ntn` | 46 | Full EH (43) + NTN features (3): satellite elevation, slant range, sat-relay gain |

All modes support optional observation normalization.

---

## 6. Reward Design

The per-step reward is a weighted sum:

```
reward = secrecy_reward - energy_penalty - motion_penalty
        - smoothness_penalty - boundary_penalty
        + harvesting_bonus - depletion_penalty
```

| Component | Description |
|-----------|-------------|
| **Secrecy reward** | `secrecy_scale * max(R_legit - R_eve, 0)` |
| **Energy penalty** | `energy_reward_weight * total_energy_consumed` |
| **Motion penalty** | Penalizes squared UAV speeds to discourage unnecessary movement |
| **Smoothness penalty** | Penalizes acceleration (change in velocity between steps) |
| **Boundary penalty** | Exponential penalty when UAVs approach area edges |
| **Harvesting bonus** | Rewards harvested energy when energy harvesting is enabled |
| **Depletion penalty** | Large negative reward when a UAV's battery hits zero |

All penalty weights are configurable in `EnvConfig`.

---

## 7. Training Instructions

All training scripts save model weights (`.pt`) and per-episode logs (`.csv`) to `outputs/training/<method>/`.

### DQN

```
python -m rl.dqn_train --episodes 500
```

### DDPG

```
python -m rl.ddpg_train --episodes 500
```

### Advanced RL (TD3 / SAC / PPO)

```
python -m rl.advanced_rl_train --method td3 --episodes 500
python -m rl.advanced_rl_train --method sac --episodes 500
python -m rl.advanced_rl_train --method ppo --episodes 500
```

### MARL DQN (shared vs split observation)

```
python -m rl.marl_dqn_train --episodes 500 --agent-obs-mode shared
python -m rl.marl_dqn_train --episodes 500 --agent-obs-mode split
```

### Common optional flags

```
--channel-model rician|rayleigh    # fading model (default: rician)
--control-mode velocity|waypoint   # control strategy (default: velocity)
--enable-ntn                       # enable NTN satellite relay
--role-switching                   # enable relay/jammer role switching
--seed <N>                         # random seed
```

---

## 8. Analysis & Plot Generation

### Training curves

```
python -m analysis.dqn_analysis --csv-path outputs/training/dqn/dqn_training_log.csv
python -m analysis.ddpg_analysis --csv-path outputs/training/ddpg/ddpg_training_log.csv
```

Supports `--algorithm dqn|marl_shared|marl_split` or `--algorithm ddpg|td3|sac|ppo` to override auto-detection.

### Trajectory plots

```
python -m analysis.trajectory_plots --method dqn
python -m analysis.trajectory_plots --method ddpg
python -m analysis.trajectory_plots --method td3
python -m analysis.trajectory_plots --method sac
python -m analysis.trajectory_plots --method marl_shared
python -m analysis.trajectory_plots --method marl_split
python -m analysis.trajectory_plots --method greedy
```

Optional: `--channel-model rayleigh`, `--seed <N>`, `--role-switching`, `--user-mobile`.

### Final comparison

```
python -m analysis.final_comparison --seeds 7,21,42,84,168
```

Discovers all available trained models from `outputs/training/` and evaluates against random and greedy baselines.

### Paper reproduction (full pipeline)

```
python -m analysis.paper_reproduction --train-episodes 60 --include-advanced
```

Runs baselines, RL training, channel matrix, trajectories, comparison, and paper plots. Results organized in `outputs/manifests/`.

---

## 9. Output Folder Structure

```
outputs/
├── training/            # Trained models (.pt) + training logs (.csv) + curve plots
│   ├── dqn/
│   ├── ddpg/
│   ├── td3/
│   ├── sac/
│   ├── ppo/
│   ├── marl_shared/
│   └── marl_split/
├── evaluations/         # Multi-seed baseline comparisons
│   ├── dqn/
│   ├── ddpg/
│   ├── marl/
│   ├── eh/
│   ├── mobility/
│   ├── ntn/
│   └── channel_matrix/
├── trajectories/        # 2D trajectory PNGs per method/channel
│   ├── dqn/
│   ├── ddpg/
│   ├── td3/
│   ├── sac/
│   ├── marl_shared/
│   ├── marl_split/
│   └── greedy/
├── comparisons/         # Cross-method comparison CSVs and bar/ranking plots
├── plots/               # Standalone analysis plots
├── analysis/            # Training curve plots by algorithm and channel
└── manifests/           # Organized reproduction of results (from paper_reproduction.py)
```

---

## 10. Convergence Summary

Based on observed training behavior across runs:

- **DQN**: Converges but with noticeable variance. Evaluation secrecy rate sometimes collapses after reward spikes. Sensitive to hyperparameters.
- **DDPG**: More stable than DQN. Continuous action space allows finer control. Converges to higher secrecy rates on average.
- **TD3**: Smoother convergence than DDPG. Clipped double-Q learning reduces overestimation. Generally reliable.
- **SAC**: Best stability and exploration among the continuous methods. Entropy regularization prevents premature convergence. Most consistent across seeds.
- **PPO**: Stable but slower to converge. Clipping prevents large policy updates, which helps reliability but limits peak performance.
- **MARL-DQN (shared)**: Both agents see the full state. Learns reasonable coordination, though individual Q-networks sometimes compete.
- **MARL-DQN (split)**: Each agent sees only its own relevant observation. This causes coordination difficulty — the jammer does not observe the relay's state, leading to suboptimal joint behavior. Generally underperforms shared observation.

The continuous algorithms (particularly SAC and TD3) outperform discrete DQN approaches on this task, likely because UAV movement benefits from fine-grained continuous control.

---

## 11. Example Outputs

Generated artifacts include:

- **Training curves**: Per-episode secrecy rate, shaped reward, path length, and (for DQN) epsilon decay plots.
- **Rolling-average secrecy plots**: Smoothed secrecy rate over training.
- **Trajectory plots**: 2D top-down view showing UAV flight paths, user/BS/Eve positions, and final locations.
- **Comparison bar charts**: Mean secrecy rate across methods with error bars over multiple seeds.
- **Method ranking plots**: Sorted mean secrecy rates with variance.
- **Evaluation CSVs**: Per-episode secrecy rates, path lengths, and energy consumption for each method.

Output formats: PNG (plots), CSV (tabular data), PT (model weights).

---

## 12. Requirements

- Python 3.10+
- PyTorch (>= 2.0 recommended)
- NumPy
- Matplotlib
- Pandas
- Gymnasium (for environment interface)

No external UAV or network simulators are required. Everything runs in simulation.

---

## 13. Notes

- This is a **simulation framework** for convergence analysis and RL experimentation.
- No real UAV deployment, hardware-in-the-loop, or real-time control is involved.
- Channel models are statistical (Rayleigh/Rician with path loss), not ray-traced.
- The environment is designed for single-episode training with random initialization per episode.
- MARL split observation mode is experimental — coordination between agents is limited by partial observability.

---

## 14. Future Improvements

- 3D UAV movement (variable altitude as a learnable action)
- Multi-eavesdropper and multi-user scenarios
- Realistic irradiance-based energy harvesting (rather than configurable max power)
- Collision avoidance between relay and jammer UAVs
- Transformer-based policy architectures for sequence-aware decision making
- Centralized critic / CTDE (centralized training with decentralized execution) for MARL
- Continuous-action MARL (MADDPG, MATD3)
- Realistic air-to-ground channel models with terrain occlusion
