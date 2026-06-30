# Phase 6C10 — Output Layer Regularization Report

**Date**: 2026-06-29 19:43:36

## Comparison Table

| Config | LR | WD | Secrecy | Sensing | Reward | BS Sat | Max Sat | corr(R,S) | |pre_tanh| | WNorm |
|--------|-----|------|---------|---------|--------|--------|---------|-----------|----------|-------|
| Random Feasible | - | - | 3.2382 | 42.4561 | - | - | - | - | - | - |
| Baseline | 3e-04 | 0e+00 | 1.4493 | 43.4723 | -0.1733 | 67.42% | 67.42% | 0.6160 | 129.7385 | 52.4230 |
| A | 1e-04 | 0e+00 | 0.5855 | 43.3020 | 0.6114 | 78.00% | 85.33% | 0.6383 | 51.5462 | 20.7513 |
| B | 3e-04 | 1e-04 | 3.2536 | 44.0741 | 1.1640 | 97.95% | 97.95% | 0.8488 | 47.4522 | 48.9558 |
| C | 1e-04 | 1e-04 | 4.6827 | 44.4748 | 0.5673 | 97.92% | 100.00% | 0.8427 | 37.7901 | 19.3059 |
| D | 5e-05 | 5e-04 | 2.0006 | 39.7108 | -0.5006 | 76.10% | 76.10% | 0.5597 | 34.1362 | 12.0140 |

## Acceptance Criteria

### Baseline (lr=3e-04, wd=0e+00)

- **C1 (BS sat < 70%)**: PASS (67.42%)
- **C2 (weight norm < 2)**: FAIL (52.4230)
- **C3 (|pre_tanh| < 5)**: FAIL (129.7385)
- **C4 (secrecy >= random 3.2382)**: FAIL (1.4493)
- **C5 (corr > 0.5)**: PASS (0.6160)

**Overall**: FAIL

### A (lr=1e-04, wd=0e+00)

- **C1 (BS sat < 70%)**: FAIL (78.00%)
- **C2 (weight norm < 2)**: FAIL (20.7513)
- **C3 (|pre_tanh| < 5)**: FAIL (51.5462)
- **C4 (secrecy >= random 3.2382)**: FAIL (0.5855)
- **C5 (corr > 0.5)**: PASS (0.6383)

**Overall**: FAIL

### B (lr=3e-04, wd=1e-04)

- **C1 (BS sat < 70%)**: FAIL (97.95%)
- **C2 (weight norm < 2)**: FAIL (48.9558)
- **C3 (|pre_tanh| < 5)**: FAIL (47.4522)
- **C4 (secrecy >= random 3.2382)**: PASS (3.2536)
- **C5 (corr > 0.5)**: PASS (0.8488)

**Overall**: FAIL

### C (lr=1e-04, wd=1e-04)

- **C1 (BS sat < 70%)**: FAIL (97.92%)
- **C2 (weight norm < 2)**: FAIL (19.3059)
- **C3 (|pre_tanh| < 5)**: FAIL (37.7901)
- **C4 (secrecy >= random 3.2382)**: PASS (4.6827)
- **C5 (corr > 0.5)**: PASS (0.8427)

**Overall**: FAIL

### D (lr=5e-05, wd=5e-04)

- **C1 (BS sat < 70%)**: FAIL (76.10%)
- **C2 (weight norm < 2)**: FAIL (12.0140)
- **C3 (|pre_tanh| < 5)**: FAIL (34.1362)
- **C4 (secrecy >= random 3.2382)**: FAIL (2.0006)
- **C5 (corr > 0.5)**: PASS (0.5597)

**Overall**: FAIL

## Decision: SATURATION_PERSISTS
