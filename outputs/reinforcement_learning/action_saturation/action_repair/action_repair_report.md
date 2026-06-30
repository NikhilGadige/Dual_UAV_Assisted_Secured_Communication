# Phase 6C6 — Action Saturation Repair Report

**Date**: 2026-06-29 16:57:02

**Random baseline**: secrecy=2.6779, sensing=42.9760, reward=0.3846

## Configuration Results

| Config | Action R | la | Entropy | ObsClip | Secrecy | Reward | corr(R,S) | Max Sat | Rew Impr |
|--------|----------|------|---------|---------|---------|--------|-----------|---------|----------|
| baseline | 1.0 | 0.0 | 0.01 | 0.0 | 2.8928 | 0.3697 | 0.5615 | 97.20% | -0.0149 |
| ar0.5 | 0.5 | 0.0 | 0.01 | 0.0 | 3.2686 | 0.0564 | 0.1863 | 99.60% | +0.3704 |
| ar0.5_la1e-2 | 0.5 | 0.01 | 0.01 | 0.0 | 3.5007 | 1.0038 | 0.6941 | 100.00% | +0.0674 |
| ar0.5_la5e-2 | 0.5 | 0.05 | 0.01 | 0.0 | 2.1605 | 0.0715 | 0.8239 | 93.33% | -0.0696 |
| ar0.5_la5e-2_e0.05 | 0.5 | 0.05 | 0.05 | 0.0 | 2.3790 | -0.2068 | 0.1798 | 100.00% | -0.0840 |
| ar0.25_la1e-2 | 0.25 | 0.01 | 0.01 | 0.0 | 1.5317 | -0.8943 | 0.7475 | 100.00% | -0.8311 |
| ar0.25_la5e-2_e0.05 | 0.25 | 0.05 | 0.05 | 0.0 | 3.5679 | -0.4693 | 0.3712 | 100.00% | +0.2293 |
| ar0.5_la1e-1_e0.10_oc5 | 0.5 | 0.1 | 0.1 | 5.0 | 3.2520 | 0.9720 | 0.7486 | 87.55% | -0.1433 |

## Per-Agent Saturation Details

### baseline
| Agent | Saturation | Pre-Tanh | Post-Tanh |
|-------|------------|----------|-----------|
| bs_beamformer | 77.20% | 3.0393 | 0.9546 |
| uav_trajectory | 97.20% | 5.8459 | 0.9966 |
| jammer_beamformer | 86.60% | 17.3057 | 0.9462 |
### ar0.5
| Agent | Saturation | Pre-Tanh | Post-Tanh |
|-------|------------|----------|-----------|
| bs_beamformer | 99.60% | 25.1664 | 0.9988 |
| uav_trajectory | 91.33% | 11.6409 | 0.9743 |
| jammer_beamformer | 91.85% | 12.9995 | 0.9783 |
### ar0.5_la1e-2
| Agent | Saturation | Pre-Tanh | Post-Tanh |
|-------|------------|----------|-----------|
| bs_beamformer | 73.20% | 16.7753 | 0.8980 |
| uav_trajectory | 100.00% | 4.8153 | 0.9989 |
| jammer_beamformer | 85.10% | 21.4743 | 0.9450 |
### ar0.5_la5e-2
| Agent | Saturation | Pre-Tanh | Post-Tanh |
|-------|------------|----------|-----------|
| bs_beamformer | 58.45% | 8.9897 | 0.8264 |
| uav_trajectory | 93.33% | 20.7989 | 0.9796 |
| jammer_beamformer | 90.95% | 15.7945 | 0.9704 |
### ar0.5_la5e-2_e0.05
| Agent | Saturation | Pre-Tanh | Post-Tanh |
|-------|------------|----------|-----------|
| bs_beamformer | 100.00% | 4.7182 | 0.9988 |
| uav_trajectory | 100.00% | 12.8732 | 1.0000 |
| jammer_beamformer | 94.20% | 21.3552 | 0.9841 |
### ar0.25_la1e-2
| Agent | Saturation | Pre-Tanh | Post-Tanh |
|-------|------------|----------|-----------|
| bs_beamformer | 100.00% | 34.3859 | 1.0000 |
| uav_trajectory | 83.47% | 4.3350 | 0.9530 |
| jammer_beamformer | 98.10% | 31.4895 | 0.9918 |
### ar0.25_la5e-2_e0.05
| Agent | Saturation | Pre-Tanh | Post-Tanh |
|-------|------------|----------|-----------|
| bs_beamformer | 100.00% | 4.9840 | 0.9991 |
| uav_trajectory | 100.00% | 12.4687 | 1.0000 |
| jammer_beamformer | 87.40% | 11.0866 | 0.9469 |
### ar0.5_la1e-1_e0.10_oc5
| Agent | Saturation | Pre-Tanh | Post-Tanh |
|-------|------------|----------|-----------|
| bs_beamformer | 87.55% | 7.5174 | 0.9413 |
| uav_trajectory | 51.73% | 2.1469 | 0.8054 |
| jammer_beamformer | 64.05% | 6.3284 | 0.8651 |

## Acceptance Criteria

- **C1 (max saturation < 70%)**: **FAIL**
  - Best: ar0.5_la1e-1_e0.10_oc5 (87.55%)
  - baseline: 97.20% (FAIL)
  - ar0.5: 99.60% (FAIL)
  - ar0.5_la1e-2: 100.00% (FAIL)
  - ar0.5_la5e-2: 93.33% (FAIL)
  - ar0.5_la5e-2_e0.05: 100.00% (FAIL)
  - ar0.25_la1e-2: 100.00% (FAIL)
  - ar0.25_la5e-2_e0.05: 100.00% (FAIL)
  - ar0.5_la1e-1_e0.10_oc5: 87.55% (FAIL)
- **C2 (corr reward-secrecy > 0.5)**: **PASS**
  - Best: ar0.5_la5e-2 (0.8239)
  - baseline: 0.5615 (PASS)
  - ar0.5: 0.1863 (FAIL)
  - ar0.5_la1e-2: 0.6941 (PASS)
  - ar0.5_la5e-2: 0.8239 (PASS)
  - ar0.5_la5e-2_e0.05: 0.1798 (FAIL)
  - ar0.25_la1e-2: 0.7475 (PASS)
  - ar0.25_la5e-2_e0.05: 0.3712 (FAIL)
  - ar0.5_la1e-1_e0.10_oc5: 0.7486 (PASS)
- **C3 (trained secrecy >= random)**: **PASS**
  - Random: 2.6779, Best: ar0.25_la5e-2_e0.05 (3.5679)
  - baseline: 2.8928 (PASS)
  - ar0.5: 3.2686 (PASS)
  - ar0.5_la1e-2: 3.5007 (PASS)
  - ar0.5_la5e-2: 2.1605 (FAIL)
  - ar0.5_la5e-2_e0.05: 2.3790 (FAIL)
  - ar0.25_la1e-2: 1.5317 (FAIL)
  - ar0.25_la5e-2_e0.05: 3.5679 (PASS)
  - ar0.5_la1e-1_e0.10_oc5: 3.2520 (PASS)

## Decision: ACTION_POLICY_STILL_SATURATED
