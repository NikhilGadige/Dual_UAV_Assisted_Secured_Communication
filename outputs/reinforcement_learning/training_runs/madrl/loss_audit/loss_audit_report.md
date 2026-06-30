# MADRL Loss Audit Report

## MAPPO

### Root Cause Detection

| Check | Status | First Episode |
|-------|--------|---------------|
| nan_loss | PASS |  |
| inf_loss | PASS |  |
| exploding_loss | PASS |  |
| nan_grad | PASS |  |
| inf_grad | PASS |  |
| exploding_grad | PASS |  |
| nan_reward | PASS |  |
| inf_reward | PASS |  |
| zero_batch | PASS |  |
| zero_std_advantage | PASS |  |

### Validation

- all_actor_losses_finite: PASS
- all_critic_losses_finite: PASS
- all_gradient_norms_finite: PASS

### Loss Statistics

#### bs_beamformer

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| policy_loss | 1.248688e-08 | 2.656781e-07 | -9.529758e-07 | 7.591676e-07 |
| value_loss | 6.597345e+02 | 1.206672e+03 | 3.381516e+02 | 1.155081e+04 |
| entropy | 5.897464e+01 | 8.079796e-01 | 5.651196e+01 | 6.030631e+01 |
| approx_kl | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| grad_norm | 1.179266e+02 | 2.714591e+02 | 1.174643e+01 | 2.635420e+03 |
| reward_mean | 6.924236e+00 | 5.531875e-02 | 6.738745e+00 | 6.987611e+00 |
| reward_std | 1.078120e-01 | 4.125050e-02 | 5.795649e-02 | 4.137298e-01 |

#### uav_jammer

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| policy_loss | 1.641258e-09 | 2.720748e-07 | -7.960480e-07 | 8.097850e-07 |
| value_loss | 6.613986e+02 | 1.210063e+03 | 3.380198e+02 | 1.137250e+04 |
| entropy | 7.743405e+01 | 8.248359e-01 | 7.486103e+01 | 7.860743e+01 |
| approx_kl | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| grad_norm | 1.510864e+02 | 2.618663e+02 | 1.518700e+01 | 2.474192e+03 |
| reward_mean | 6.923816e+00 | 5.657509e-02 | 6.735517e+00 | 6.990675e+00 |
| reward_std | 1.080623e-01 | 4.224157e-02 | 6.243822e-02 | 4.084985e-01 |

## MATD3

### Root Cause Detection

| Check | Status | First Episode |
|-------|--------|---------------|
| nan_loss | PASS |  |
| inf_loss | PASS |  |
| exploding_loss | PASS |  |
| nan_grad | PASS |  |
| inf_grad | PASS |  |
| exploding_grad | PASS |  |
| nan_reward | PASS |  |
| inf_reward | PASS |  |
| zero_batch | PASS |  |
| zero_std_advantage | PASS |  |

### Validation

- all_actor_losses_finite: PASS
- all_critic_losses_finite: PASS
- all_gradient_norms_finite: PASS

### Loss Statistics

#### bs_beamformer

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| critic_loss | 2.352733e-01 | 2.819499e+00 | 8.618793e-03 | 7.435460e+01 |
| actor_loss | -7.064995e+00 | 7.964146e+00 | -2.309369e+01 | 2.620992e-01 |
| critic_grad_norm | 3.130108e+01 | 6.398641e+01 | 1.202867e+00 | 1.348139e+03 |
| actor_grad_norm | 1.250404e-01 | 7.898419e-01 | 0.000000e+00 | 1.479177e+01 |
| reward_mean | 7.035989e+00 | 2.731749e-02 | 7.002526e+00 | 7.265053e+00 |
| reward_std | 6.144037e-02 | 2.913428e-02 | 2.753065e-02 | 2.266944e-01 |

#### uav_jammer

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| critic_loss | 4.611529e+00 | 1.721112e+01 | 8.930120e-02 | 3.370310e+02 |
| actor_loss | 5.299383e+00 | 7.904320e+00 | -1.732295e+00 | 3.031866e+01 |
| critic_grad_norm | 1.427654e+03 | 1.156028e+03 | 5.484995e+01 | 1.499943e+04 |
| actor_grad_norm | 2.848653e-01 | 1.978251e+00 | 0.000000e+00 | 4.014993e+01 |
| reward_mean | 7.036291e+00 | 2.705065e-02 | 7.000632e+00 | 7.267427e+00 |
| reward_std | 5.947987e-02 | 2.531555e-02 | 2.763345e-02 | 2.314932e-01 |


## Final Decision: NUMERICAL_STABILITY_CONFIRMED
