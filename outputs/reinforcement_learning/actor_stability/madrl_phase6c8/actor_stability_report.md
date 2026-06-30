# Phase 6C8 — Actor Stabilization Report

**Date**: 2026-06-29 17:19:26

**Random baseline**: secrecy=2.5880

## Configuration Results

| Config | T | Clip | LR | Secrecy | Reward | corr(R,S) | BS Sat | PT Mean | PT Max |
|--------|---|------|----|---------|--------|-----------|--------|---------|--------|
| baseline | 1.0 | False | 0.0003 | 2.8974 | 0.0505 | 0.5196 | 39.35% | 3.1889 | 47.3767 |

## Per-Agent Saturation Details

### baseline
| Agent | Saturation | Pre-Tanh | Post-Tanh |
|-------|------------|----------|-----------|
| bs_beamformer | 39.35% | 5.0719 | 0.7006 |
| uav_trajectory | 54.80% | 2.2392 | 0.8576 |
| jammer_beamformer | 14.20% | 1.1934 | 0.5728 |

## Acceptance Criteria

Accept if ANY of: BS saturation < 80% OR pre_tanh_mean < 5 OR pre_tanh_max < 10

- **C1 (BS sat < 80%)**: **PASS**
  - baseline: 39.35% (PASS)
- **C2 (pre_tanh_mean < 5)**: **PASS**
  - baseline: 3.1889 (PASS)
- **C3 (pre_tanh_max < 10)**: **FAIL**
  - baseline: 47.3767 (FAIL)

## Decision: ACTOR_STABILIZED
