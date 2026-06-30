# Conditioning Report

## Summary

| Metric | Before (Unscaled) | After (Scaled) |
|--------|:-----------------:|:--------------:|
| Condition ratio (sensitivity) | 1236.43 | 3.64 |
| Condition ratio (gradient)    | 41.54 | 5342.97 |
| Success (ratio_after < 50)    | | YES |

---

## Per-Block Mean Sensitivity

| Block | |f/x| (unscaled) | |f/x_scaled| (scaled) |
|-------|:---------------------------:|:----------------------------:|
| Power (w_bs) | 3.671992e-03 | 1.161186e-02 |
| Trajectory (q_uav) | 1.478248e-04 | 1.122435e-02 |
| Jammer (v_jammer) | 1.827756e-01 | 4.086987e-02 |

### Detailed (unscaled)

| Block | Pert (%) | df/dx |
|-------|:--------:|:-----:|
| power        |   +1.0 | +3.368e-10 |
| power        |   -1.0 | -7.068e-03 |
| power        |   +5.0 | +6.737e-11 |
| power        |   -5.0 | -7.317e-03 |
| power        |  +10.0 | +3.368e-11 |
| power        |  -10.0 | -7.647e-03 |
| trajectory   |   +1.0 | -1.500e-04 |
| trajectory   |   -1.0 | +1.459e-04 |
| trajectory   |   +5.0 | -1.580e-04 |
| trajectory   |   -5.0 | +1.377e-04 |
| trajectory   |  +10.0 | -1.677e-04 |
| trajectory   |  -10.0 | +1.277e-04 |
| jammer       |   +1.0 | -1.196e-01 |
| jammer       |   -1.0 | +2.421e-01 |
| jammer       |   +5.0 | -1.169e-01 |
| jammer       |   -5.0 | +2.483e-01 |
| jammer       |  +10.0 | -1.138e-01 |
| jammer       |  -10.0 | +2.560e-01 |

### Detailed (scaled)

| Block | Pert (%) | df/dx_scaled |
|-------|:--------:|:------------:|
| power        |   +1.0 | +1.065e-09 |
| power        |   -1.0 | -2.235e-02 |
| power        |   +5.0 | +2.130e-10 |
| power        |   -5.0 | -2.314e-02 |
| power        |  +10.0 | +1.065e-10 |
| power        |  -10.0 | -2.418e-02 |
| trajectory   |   +1.0 | -1.139e-02 |
| trajectory   |   -1.0 | +1.108e-02 |
| trajectory   |   +5.0 | -1.200e-02 |
| trajectory   |   -5.0 | +1.046e-02 |
| trajectory   |  +10.0 | -1.274e-02 |
| trajectory   |  -10.0 | +9.692e-03 |
| jammer       |   +1.0 | -2.674e-02 |
| jammer       |   -1.0 | +5.413e-02 |
| jammer       |   +5.0 | -2.615e-02 |
| jammer       |   -5.0 | +5.551e-02 |
| jammer       |  +10.0 | -2.544e-02 |
| jammer       |  -10.0 | +5.724e-02 |

---

## Gradient Norms

| Block | ||g|| (unscaled) | ||g|| (scaled) |
|-------|:------------------------:|:-----------------------:|
| Power (w_bs) | 1.264772e+04 | 4.001999e+04 |
| Trajectory (q_uav) | 2.000000e+03 | 3.639377e+05 |
| Jammer (v_jammer) | 3.044975e+02 | 6.811525e+01 |

---

## Scaling Details

```
Power:      w_scaled = w / sqrt(P_bs_max)
Trajectory: q_scaled = (q - q_center) / q_scale
Jammer:     v_scaled = v / sqrt(P_j_max)

Adaptive FD step: eps_i = max(1e-6, 1e-3 * |x_scaled_i|)
```
