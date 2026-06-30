# Phase 6C5 — Reward Repair Report

**Date**: 2026-06-29 16:23:40

**Random secrecy baseline**: 2.7303

## Reward Ablation Results

| Variant | Secrecy | Sensing | Reward | corr(R,sec) | Max Sat | Sec Impr |
|---------|---------|---------|--------|-------------|---------|----------|
| Original Reward | 2.0924 | 43.1009 | 7.2101 | -0.4770 | 86.67% | -0.6380 |
| Normalized Reward | 4.3595 | 40.5094 | 0.3210 | 0.8436 | 90.75% | +1.6292 |
| Norm + Secrecy Penalty | 3.4031 | 41.0959 | 0.3610 | 0.8947 | 96.00% | +0.6728 |
| Full (Norm + Penalty + Action Reg) | 3.7948 | 37.8812 | 0.1642 | 0.9228 | 100.00% | +1.0644 |

## Action Saturation Details

### Original Reward
| Agent | Saturation | LB Fraction | UB Fraction | Mean |
|-------|------------|-------------|-------------|------|
| bs_beamformer | 70.00% | 30.00% | 40.00% | 0.1359 |
| uav_trajectory | 86.67% | 46.67% | 40.00% | -0.0714 |
| jammer_beamformer | 65.00% | 37.50% | 27.50% | -0.0591 |
### Normalized Reward
| Agent | Saturation | LB Fraction | UB Fraction | Mean |
|-------|------------|-------------|-------------|------|
| bs_beamformer | 79.95% | 44.20% | 35.75% | -0.0984 |
| uav_trajectory | 68.00% | 30.67% | 37.33% | 0.0579 |
| jammer_beamformer | 90.75% | 53.65% | 37.10% | -0.1781 |
### Norm + Secrecy Penalty
| Agent | Saturation | LB Fraction | UB Fraction | Mean |
|-------|------------|-------------|-------------|------|
| bs_beamformer | 79.55% | 44.10% | 35.45% | -0.0945 |
| uav_trajectory | 96.00% | 56.00% | 40.00% | -0.1943 |
| jammer_beamformer | 90.05% | 49.90% | 40.15% | -0.1418 |
### Full (Norm + Penalty + Action Reg)
| Agent | Saturation | LB Fraction | UB Fraction | Mean |
|-------|------------|-------------|-------------|------|
| bs_beamformer | 95.35% | 35.30% | 60.05% | 0.2493 |
| uav_trajectory | 100.00% | 46.67% | 53.33% | 0.0667 |
| jammer_beamformer | 86.45% | 46.65% | 39.80% | -0.0741 |

## Normalization Statistics

| Variant | R count | R mean | R std | U count | U mean | U std |
|---------|---------|--------|-------|---------|--------|-------|
| original | 6000 | 1.7868 | 1.4000 | 6000 | 41.0280 | 1.5326 |
| normalized | 6000 | 3.3657 | 2.5591 | 6000 | 39.7641 | 2.8677 |
| normalized_penalty | 6000 | 1.2792 | 1.4150 | 6000 | 40.8900 | 2.7591 |
| full | 6000 | 1.2610 | 1.7076 | 6000 | 40.1312 | 4.8037 |

## Acceptance Criteria

- **C1 (corr reward-secrecy > 0.3)**: **PASS**
  - Original Reward: -0.4770
  - Normalized Reward: 0.8436
  - Norm + Secrecy Penalty: 0.8947
  - Full (Norm + Penalty + Action Reg): 0.9228
- **C2 (trained secrecy >= random)**: **PASS**
  - Random: 2.7303, Best trained: 4.3595
  - Original Reward: 2.0924 (FAIL)
  - Normalized Reward: 4.3595 (PASS)
  - Norm + Secrecy Penalty: 3.4031 (PASS)
  - Full (Norm + Penalty + Action Reg): 3.7948 (PASS)
- **C3 (<70% action saturated)**: **FAIL**
  - Min max-saturation: 86.67%
  - Original Reward: 86.67% (FAIL)
  - Normalized Reward: 90.75% (FAIL)
  - Norm + Secrecy Penalty: 96.00% (FAIL)
  - Full (Norm + Penalty + Action Reg): 100.00% (FAIL)
- **C4 (reward-secrecy positive corr)**: **PASS**
  - Best correlation: 0.9228
  - Original Reward: -0.4770 (FAIL)
  - Normalized Reward: 0.8436 (PASS)
  - Norm + Secrecy Penalty: 0.8947 (PASS)
  - Full (Norm + Penalty + Action Reg): 0.9228 (PASS)
- **C5 (reward improvement during training)**: **FAIL**
  - Best improvement: -0.0332
  - Original Reward: 6.9231 -> 6.8063 (-0.1168) (FAIL)
  - Normalized Reward: 0.0879 -> -0.2609 (-0.3488) (FAIL)
  - Norm + Secrecy Penalty: -1.0521 -> -1.5646 (-0.5125) (FAIL)
  - Full (Norm + Penalty + Action Reg): -1.7855 -> -1.8187 (-0.0332) (FAIL)

## Decision: REWARD_STILL_BROKEN
