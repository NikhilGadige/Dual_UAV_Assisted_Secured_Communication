# Mobility Model Analysis Report

Generated: 2026-06-02 00:15:00

Models: 4 | Algorithms: 6 | Channels: 2

## 1. Ranking of Mobility Models (by avg Final_Rolling100_Secrecy)

| Rank | Mobility Model | Avg Final Rolling100 Secrecy (Mbps) |
|------|---------------|-------------------------------------|
| 1 | random_waypoint | 4.3487 |
| 2 | random_walk | 4.3016 |
| 3 | gauss_markov | 4.2851 |
| 4 | constant_velocity | 4.2431 |

## 2. Hardest Mobility Model
**constant_velocity** - lowest average secrecy (4.2431 Mbps).

## 3. Highest Secrecy Mobility Model
**random_waypoint** - highest average secrecy (4.3487 Mbps).

## 4. Fastest Converging Mobility Model
**random_walk** - average convergence at episode 5.

| Mobility Model | Avg Convergence Episode |
|---------------|------------------------|
| random_walk | 5 |
| gauss_markov | 8 |
| random_waypoint | 8 |
| constant_velocity | 20 |

## 5. Per-Algorithm Observations

### DQN
- Avg Final Rolling100: 3.2324 Mbps (over 8 runs)
- Best mobility: **random_walk** (3.2591 Mbps)
- Worst mobility: **constant_velocity** (3.2055 Mbps)

### DDPG
- Avg Final Rolling100: 5.0376 Mbps (over 8 runs)
- Best mobility: **random_waypoint** (5.1771 Mbps)
- Worst mobility: **random_walk** (4.9841 Mbps)

### D3QN
- Avg Final Rolling100: 3.0904 Mbps (over 8 runs)
- Best mobility: **random_waypoint** (3.1726 Mbps)
- Worst mobility: **gauss_markov** (3.0390 Mbps)

### PPO
- Avg Final Rolling100: 4.4416 Mbps (over 8 runs)
- Best mobility: **random_waypoint** (4.5434 Mbps)
- Worst mobility: **constant_velocity** (4.3666 Mbps)

### SAC
- Avg Final Rolling100: 5.1037 Mbps (over 8 runs)
- Best mobility: **random_walk** (5.3312 Mbps)
- Worst mobility: **constant_velocity** (4.8319 Mbps)

### TD3PG
- Avg Final Rolling100: 4.8621 Mbps (over 8 runs)
- Best mobility: **constant_velocity** (5.0255 Mbps)
- Worst mobility: **random_walk** (4.7313 Mbps)

## 6. Channel Model Effect

- **Rician**: 5.1980 Mbps avg over 24 runs
- **Rayleigh**: 3.3913 Mbps avg over 24 runs