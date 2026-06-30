# Phase 6C — Reward and Policy Audit Report

**Date**: 2026-06-29 16:04:01

## Part 1: Reward Decomposition

### mappo
- corr(total, secrecy): -0.2749
- corr(total, sensing): 0.9955
- corr(total, penalty): 0.0000
- mean reward: 7.2213
- mean secrecy: 1.4572
- mean sensing: 43.2244

### matd3
- corr(total, secrecy): -0.3595
- corr(total, sensing): 0.9968
- corr(total, penalty): 0.0000
- mean reward: 6.9943
- mean secrecy: 1.6450
- mean sensing: 41.8454

## Part 2: Policy Behaviour Audit

| Policy | Secrecy | Sensing | BS Power | Jammer Power | UAV Speed |
|--------|---------|---------|----------|--------------|-----------|
| random_feasible | 2.4502 | 42.4005 | 2.6774 | 0.0498 | 35.0368 |
| mappo | 1.4572 | 43.2244 | 7.9521 | 0.0500 | 39.0346 |
| matd3 | 1.6450 | 41.8454 | 7.7850 | 0.0500 | 36.8436 |

## Part 3: Action Saturation

### mappo
| Agent | Saturation | LB Fraction | UB Fraction | Clipped | Mean |
|-------|------------|-------------|-------------|---------|------|
| bs_beamformer | 95.00% | 55.00% | 40.00% | 0.00% | -0.1513 |
| uav_trajectory | 53.33% | 26.67% | 26.67% | 0.00% | 0.0353 |
| jammer_beamformer | 87.50% | 42.50% | 45.00% | 0.00% | 0.0614 |
### matd3
| Agent | Saturation | LB Fraction | UB Fraction | Clipped | Mean |
|-------|------------|-------------|-------------|---------|------|
| bs_beamformer | 86.75% | 46.75% | 40.00% | 0.00% | -0.0647 |
| uav_trajectory | 100.00% | 54.67% | 45.33% | 0.00% | -0.0933 |
| jammer_beamformer | 86.55% | 38.65% | 47.90% | 0.00% | 0.1151 |

## Part 4: Observation Importance (Ablation)

| Condition | Secrecy | Sensing | Degradation |
|-----------|---------|---------|-------------|
| full_obs | 1.6550 | 43.3796 | 0.0% |
| no_secrecy | 2.5507 | 43.7358 | -54.1% |
| no_sensing | 2.8417 | 43.5115 | -71.7% |
| no_channels | 2.2519 | 40.3156 | -36.1% |

## Part 5: Reward Weight Sweep

| Alpha | Secrecy | Sensing | Reward |
|-------|---------|---------|--------|
| 0.00 | 1.3037 | 42.7881 | 14.2597 |
| 0.25 | 3.9668 | 39.4627 | 9.8920 |
| 0.50 | 2.1991 | 43.5980 | 7.2951 |
| 0.75 | 7.9633 | 39.5054 | 3.4619 |
| 1.00 | 18.3632 | 43.3800 | 0.5246 |

## Acceptance Criteria

- C1 (corr reward-secrecy > 0.3): FAIL
  - mappo: -0.2749
  - matd3: -0.3595
- C2 (trained secrecy >= random): FAIL
  - Random secrecy: 2.4502, Trained max: 1.6450
- C3 (<50% actions saturated): FAIL
  - Max saturation: 100.00%
- C4 (no-secrecy ablation degrades): FAIL
  - Full obs secrecy: 1.6550, No secrecy obs: 2.5507
- C5 (Pareto trade-off): PASS
  - Secrecy range: 17.0595, Sensing range: 4.1353

## Decision: REWARD_DESIGN_BROKEN
