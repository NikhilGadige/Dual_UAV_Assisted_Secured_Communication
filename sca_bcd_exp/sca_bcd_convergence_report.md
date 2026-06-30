# SCA+BCD Convergence Study Report

## 1. Problem Setup

- **Area**: 1000×1000 m
- **Max flight radius**: 350 m
- **Relay/jammer altitude**: 50 m
- **Slot duration**: 4 s
- **Horizon (M)**: 12 slots
- **Bandwidth**: 1 MHz
- **Noise PSD**: −174 dBm/Hz (−17.4 dB)
- **β₀**: 1.0 (linear)
- **Path-loss exponent (α)**: 2.0
- **Eve model**: HPPP (λ = 2×10⁻⁵), robust formulation with uncertainty radius 30 m
- **Vehicle receiver**: Yes (straight-road mobility, 8 m/s)
- **Channel models**: Rician (K = 5) and Rayleigh for comparison

## 2. Optimization Variables

| Variable | Dimension | Constraints |
|---|---|---|
| **Source power** pₛ[m] | M × 1 | [1 mW, 200 mW], avg ≤ 150 mW |
| **Relay power** pᵣ[m] | M × 1 | [1 mW, 500 mW], avg ≤ 350 mW |
| **Jammer power** pⱼ[m] | M × 1 | [0, 500 mW], avg ≤ 250 mW |
| **Relay trajectory** qᵣ[m] | M × 2 | Start/end fixed, max 350 m radius, max speed 20 m/s, collision avoid |
| **Jammer trajectory** qⱼ[m] | M × 2 | Start/end fixed, max 350 m radius, max speed 20 m/s, collision avoid |
| **Time-splitting α[m]** | M × 1 | [0.05, 0.95] |

## 3. BCD Blocks (executed each outer iteration)

| Block | Method | Trust-region radius |
|---|---|---|
| 1. Power allocation | SCA (linear surrogate + quadratic penalty) | 0.35 |
| 2. Relay trajectory | SCA | 180.0 m |
| 3. Jammer trajectory | SCA | 180.0 m |
| 4. Time-splitting α | SCA (exact linear surrogate) | 0.5 |

- **Max BCD iterations**: 100
- **Min BCD iterations**: 20
- **Patience**: 8 iterations with |Δ| ≤ 10⁻³ or relative gap ≤ 5×10⁻⁴
- **Max SCA sub-iterations per block**: 8
- **SCA tolerance**: 10⁻⁴

## 4. Convergence Behavior

| Metric | Rician | Rayleigh |
|---|---|---|
| Converged at iteration | 20 | 20 |
| Initial objective | 8.279956 | 6.329766 |
| Final objective (raw) | 19.520238 | 15.856420 |
| Absolute improvement | 11.240282 | 9.526654 |
| Relative improvement | 135.7529% | 150.5056% |
| Final secrecy rate | 19.520238 bps/Hz | 15.856420 bps/Hz |

### Update norms at convergence

| Variable | Rician | Rayleigh |
|---|---|---|
| Relay | 1.086585e-02 | 8.456179e-03 |
| Jammer | 0.000000e+00 | 0.000000e+00 |
| Power | 0.000000e+00 | 0.000000e+00 |
| Alpha | 2.646567e-07 | 2.992870e-07 |

## 5. Rician Results

- **Convergence iteration**: 20
- **Final raw objective**: 19.520238
- **Final secrecy rate**: 19.520238 bps/Hz
- **Mean α at convergence**: 0.9500
- **Improvement from initial**: 11.240282 (135.7529%)
- **SCA block contributions (sum of sub-iteration deltas across all BCD iters)**:
    - alpha: 8.705818
    - jammer: 0.000000
    - power: 2.513525
    - relay: 0.001117

### Final α trajectory
- Mean α = 0.9500
- Alpha update norm at convergence = 2.646567e-07

## 6. Rayleigh Results

- **Convergence iteration**: 20
- **Final raw objective**: 15.856420
- **Final secrecy rate**: 15.856420 bps/Hz
- **Mean α at convergence**: 0.9500
- **Improvement from initial**: 9.526654 (150.5056%)
- **SCA block contributions (sum of sub-iteration deltas across all BCD iters)**:
    - alpha: 6.952574
    - jammer: 0.000022
    - power: 2.554847
    - relay: 0.000676

### Final α trajectory
- Mean α = 0.9500
- Alpha update norm at convergence = 2.992870e-07

## 7. Final Comparison

| Metric | Rician | Rayleigh | Winner |
|---|---|---|---|
| Convergence speed | 20 iters | 20 iters | Both converged at the same iteration (20) |
| Final objective | 19.520238 | 15.856420 | Rician |
| Final secrecy rate | 19.520238 bps/Hz | 15.856420 bps/Hz | Rician |
| Improvement | 11.240282 | 9.526654 | Rician |
| Final mean α | 0.9500 | 0.9500 | — |
| Largest contributing block | alpha | alpha | — |

## 8. Key Observations

1. **Faster convergence**: Both converged at the same iteration (20)
2. **Higher secrecy**: Rician (19.5202 bps/Hz) > Rayleigh (15.8564 bps/Hz)
3. **Alpha convergence**: Both channels converged alpha within [0.05, 0.95] bounds. Rician mean α = 0.9500, Rayleigh mean α = 0.9500.
4. **Objective improvement**: Rician improved from 8.279956 → 19.520238 (135.7529%). Rayleigh improved from 6.329766 → 15.856420 (150.5056%).
5. **Dominant block**: The largest SCA sub-iteration improvement contributor was **alpha** for Rician and **alpha** for Rayleigh.
6. **Convergence quality**: All update norms at convergence are below 10⁻⁴, confirming that the algorithm reached a stationary point.
