# Phase 6C9 — Action Saturation Report

**Date**: 2026-06-29 17:57:04

## MAPPO

| Agent | Saturation | Pre-Tanh | Post-Tanh |
|-------|------------|----------|-----------|
| bs_beamformer | 86.05% | 90.2484 | 0.9306 |
| uav_trajectory | 67.07% | 4.0333 | 0.8648 |
| jammer_beamformer | 55.20% | 25.5269 | 0.8012 |

## MATD3

| Agent | Saturation | Pre-Tanh | Post-Tanh |
|-------|------------|----------|-----------|
| bs_beamformer | 93.15% | 8.5478 | 0.9739 |
| uav_trajectory | 100.00% | 71.2781 | 1.0000 |
| jammer_beamformer | 92.80% | 8.7590 | 0.9768 |

## Acceptance

C1: BS saturation < 50%

- **MAPPO**: BS saturation = 86.05% (FAIL)
- **MATD3**: BS saturation = 93.15% (FAIL)
