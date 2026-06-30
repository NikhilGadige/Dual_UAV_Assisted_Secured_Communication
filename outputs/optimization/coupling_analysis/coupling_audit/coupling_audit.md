# Coupling Audit -- SCA-BCD

## Overview

- **Objective**:  weighted ISAC objective  f = alpha.R_s/R_s_ref + (1-alpha).U_sense/U_sense_ref  (alpha = 0.5)
- **Secrecy**:  total secrecy rate R_s_total (sum over time slots)
- **Perturbations**:  each block scaled by (1 +/- p) for p in {1%, 5%, 10%}
- **Sensitivity**:  Deltaf / Deltax_norm  and  DeltaR_s / Deltax_norm
- **Config**:  channel_model = rician,  jammer_mode = given

---

## Nominal (Feasible) Solution

| Metric | Value |
|--------|-------|
| Converged | False (51 iterations) |
| Objective f | 0.928202 |
| Secrecy R_s_total | 6.640808 |
| Max Eve SINR | 1.767147e+00 |

---

## Per-Block Sensitivity

### Summary (mean +/- std of |sensitivity|)

| Block | |df/dx| (obj) | |dR_s/dx| (sec) | Status |
|-------|:----:|:----:|:----:|
| Power (w_bs) | 3.671992e-03 +/- 3.7e-03 | 2.570395e-01 +/- 2.6e-01 | ACTIVE |
| Trajectory (q_uav) | 1.478248e-04 +/- 1.3e-05 | 1.034773e-02 +/- 9.1e-04 | ACTIVE |
| Jammer (v_jammer) | 1.827756e-01 +/- 6.6e-02 | 1.279429e+01 +/- 4.6e+00 | ACTIVE |

### Raw sensitivity (signed)

| Block | Deltaf/Deltax (mean +/- std) | DeltaR_s/Deltax (mean +/- std) |
|-------|:----:|:----:|
| Power (w_bs) | -3.671992e-03 +/- 3.7e-03 | -2.570394e-01 +/- 2.6e-01 |
| Trajectory (q_uav) | -1.074628e-05 +/- 1.5e-04 | -7.522382e-04 +/- 1.0e-02 |
| Jammer (v_jammer) | 6.600242e-02 +/- 1.8e-01 | 4.620170e+00 +/- 1.3e+01 |

### Detailed perturbation results

| Block | Pert (%) | Deltax_norm | Deltaf | DeltaR_s | df/dx | dR_s/dx |
|-------|:--------:|:-------:|:---:|:----:|:-----:|:-------:|
| power        |   +1.0 | 6.324555e-02 | +0.000000 | +0.000000 | +3.368e-10 | +2.358e-08 |
| power        |   -1.0 | 6.324555e-02 | -0.000447 | -0.031289 | -7.068e-03 | -4.947e-01 |
| power        |   +5.0 | 3.162278e-01 | +0.000000 | +0.000000 | +6.737e-11 | +4.716e-09 |
| power        |   -5.0 | 3.162278e-01 | -0.002314 | -0.161969 | -7.317e-03 | -5.122e-01 |
| power        |  +10.0 | 6.324555e-01 | +0.000000 | +0.000000 | +3.368e-11 | +2.358e-09 |
| power        |  -10.0 | 6.324555e-01 | -0.004837 | -0.338565 | -7.647e-03 | -5.353e-01 |
| trajectory   |   +1.0 | 3.432224e+00 | -0.000515 | -0.036029 | -1.500e-04 | -1.050e-02 |
| trajectory   |   -1.0 | 3.432224e+00 | +0.000501 | +0.035048 | +1.459e-04 | +1.021e-02 |
| trajectory   |   +5.0 | 1.716112e+01 | -0.002712 | -0.189831 | -1.580e-04 | -1.106e-02 |
| trajectory   |   -5.0 | 1.716112e+01 | +0.002363 | +0.165424 | +1.377e-04 | +9.639e-03 |
| trajectory   |  +10.0 | 3.432224e+01 | -0.005757 | -0.402976 | -1.677e-04 | -1.174e-02 |
| trajectory   |  -10.0 | 3.432224e+01 | +0.004381 | +0.306687 | +1.277e-04 | +8.936e-03 |
| jammer       |   +1.0 | 5.000000e-03 | -0.000598 | -0.041857 | -1.196e-01 | -8.371e+00 |
| jammer       |   -1.0 | 5.000000e-03 | +0.001210 | +0.084727 | +2.421e-01 | +1.695e+01 |
| jammer       |   +5.0 | 2.500000e-02 | -0.002924 | -0.204648 | -1.169e-01 | -8.186e+00 |
| jammer       |   -5.0 | 2.500000e-02 | +0.006206 | +0.434450 | +2.483e-01 | +1.738e+01 |
| jammer       |  +10.0 | 5.000000e-02 | -0.005689 | -0.398257 | -1.138e-01 | -7.965e+00 |
| jammer       |  -10.0 | 5.000000e-02 | +0.012800 | +0.896001 | +2.560e-01 | +1.792e+01 |

---

## Specific Verification

### 1. Jammer beamforming variables change eavesdropper SINR

- When jammer variables are perturbed, max |DeltaSINR_eve| = 4.379558e-01
- **Result**: [OK] Jammer variables DO affect Eve SINR

### 2. Jammer power changes secrecy rate

- When jammer variables are perturbed, max |DeltaR_s| = 8.960011e-01
- **Result**: [OK] Jammer power DOES affect secrecy rate

### 3. Finite-difference gradients for jammer variables are nonzero

- ||g_jammer|| = 3.044975e+02
- max|g_i| = 1.784513e+02
- Non-zero entries: 32 / 40
- **Result**: [OK] FD gradients are nonzero

### FD gradients across all blocks

| Block | ||g|| | max|g_i| | nonzero/total |
|-------|:-----:|:--------:|:-------------:|
| Power (w_bs) | 1.264772e+04 | 6.324512e+03 | 10/10 |
| Trajectory (q_uav) | 2.000000e+03 | 7.922734e+02 | 15/15 |
| Jammer (v_jammer) | 3.044975e+02 | 1.784513e+02 | 32/40 |

---

## Conditioning Assessment

- **Condition ratio** (max |df/dx| / min |df/dx| among active blocks): **1236.43**
- Threshold: well-conditioned if ratio < 100
- **Verdict**: Ill-conditioned -- sensitivity varies significantly across blocks

---

## Coupling Matrix

| Block | Objective Sensitivity (mean +/- std) | Secrecy Sensitivity (mean +/- std) |
|-------|:-------------------------------:|:-------------------------------:|
| Power (w_bs) | 3.671992e-03 +/- 3.7e-03 | 2.570395e-01 +/- 2.6e-01 |
| Trajectory (q_uav) | 1.478248e-04 +/- 1.3e-05 | 1.034773e-02 +/- 9.1e-04 |
| Jammer (v_jammer) | 1.827756e-01 +/- 6.6e-02 | 1.279429e+01 +/- 4.6e+00 |

---

## Notes

1. The default configuration uses `jammer_mode = "mixed"`, which causes the jammer beamforming
   variables (v_jammer) to be **ignored** during evaluation -- heuristic beams are designed instead.
   For this audit, `jammer_mode` was set to `"given"` so that the optimizer's jammer decisions
   actually affect the objective.  Under the default configuration, the jammer block would appear
   **completely inactive** (zero sensitivity, zero gradients, no effect on secrecy or SINR).

2. The RIS phase variables (phi_rad) are not included as an optimisation block in the current BCD loop.
   They remain at their initial value (zeros) throughout and are not optimised.  A full coupling
   analysis would include an RIS block.

---

## Conclusion

**B. Partially coupled optimisation problem**
