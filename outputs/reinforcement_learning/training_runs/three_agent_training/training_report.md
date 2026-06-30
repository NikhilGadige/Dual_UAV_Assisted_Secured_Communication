# — Training Report

**Date**: 2026-06-29 23:13:41

## Agent Decomposition

| Agent | Action | Observation |
|-------|--------|-------------|
| bs\_beamformer | w\_bs | channel state, secrecy, sensing, power utilization |
| uav\_trajectory | q\_uav increments | UAV state, positions, motion budget, objective |
| jammer\_beamformer | v\_jammer | eve channels, interference, power utilization |

## Learning Check

- **mappo_100**: first_20=6.9197 -> last_20=6.9318 (improved=True)
- **matd3_100**: first_20=6.8758 -> last_20=6.9875 (improved=True)

## Baseline Comparison

| Method | Reward | Secrecy | Sensing | Violation |
|--------|--------|---------|---------|-----------|
| random_feasible | 0.0000 | 2.1708 | 42.0078 | 0.0000 |
| mappo_100 | 7.1812 | 1.8366 | 42.7662 | 0.0000 |
| matd3_100 | 6.9949 | 1.7025 | 41.8321 | 0.0000 |

## Numerical Checks

- mappo_100_bs_beamformer_rewards_finite: PASS
- mappo_100_bs_beamformer_losses_finite: PASS
- mappo_100_bs_beamformer_grad_norm_finite: PASS
- mappo_100_uav_trajectory_rewards_finite: PASS
- mappo_100_uav_trajectory_losses_finite: PASS
- mappo_100_uav_trajectory_grad_norm_finite: PASS
- mappo_100_jammer_beamformer_rewards_finite: PASS
- mappo_100_jammer_beamformer_losses_finite: PASS
- mappo_100_jammer_beamformer_grad_norm_finite: PASS
- mappo_100_all_numerical_pass: PASS
- matd3_100_bs_beamformer_rewards_finite: PASS
- matd3_100_bs_beamformer_losses_finite: PASS
- matd3_100_bs_beamformer_grad_norm_finite: PASS
- matd3_100_uav_trajectory_rewards_finite: PASS
- matd3_100_uav_trajectory_losses_finite: PASS
- matd3_100_uav_trajectory_grad_norm_finite: PASS
- matd3_100_jammer_beamformer_rewards_finite: PASS
- matd3_100_jammer_beamformer_losses_finite: PASS
- matd3_100_jammer_beamformer_grad_norm_finite: PASS
- matd3_100_all_numerical_pass: PASS

## Decision


## Final Decision

**THREE_AGENT_MARL_FAILED**

| Criterion | Status |
|-----------|--------|
| validation_passed | PASS |
| mappo_improves | PASS |
| matd3_improves | PASS |
| mappo_beats_random | FAIL |
| matd3_beats_random | FAIL |

### Summary

- Validation: 16/16 passed
- MAPPO improvement: True
- MATD3 improvement: True
- MAPPO secrecy (1.8366) vs random (2.1708)
- MATD3 secrecy (1.7025) vs random (2.1708)
