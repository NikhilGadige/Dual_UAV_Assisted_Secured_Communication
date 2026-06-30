# Phase 6C9 — Post-Fix Re-evaluation Report

**Date**: 2026-06-29 17:57:04

## Comparison Table

| Method | Secrecy | Sensing | Reward | BS Sat | corr(R,S) |
|--------|---------|---------|--------|--------|-----------|
| Random Feasible | 3.0977 | 42.6136 | 0.0000 | N/A | N/A |
| SCA-BCD | 7.0313 | 44.1617 | 0.0000 | N/A | N/A |
| MAPPO (new init) | 1.8659 | 43.4239 | 0.1100 | 86.05% | 0.3358 |
| MATD3 (new init) | 1.9026 | 43.0545 | 2.6688 | 93.15% | 0.2102 |
| MAPPO (previous) | 2.7884 | 42.2637 | N/A | N/A | N/A |
| MATD3 (previous) | 0.0500 | 41.8938 | N/A | N/A | N/A |

## Acceptance Criteria

- **C1 (BS sat < 50%)**: MAPPO=FAIL (86.05%), MATD3=FAIL (93.15%)
- **C2 (secrecy > previous)**: MAPPO=FAIL (1.8659 > 2.7884), MATD3=PASS (1.9026 > 0.0500)
- **C3 (secrecy > random)**: MAPPO=FAIL (1.8659 > 3.0977), MATD3=FAIL (1.9026 > 3.0977)
- **C4 (corr > 0.5)**: MAPPO=FAIL (0.3358), MATD3=FAIL (0.2102)

## Decision: FURTHER_TUNING_REQUIRED
