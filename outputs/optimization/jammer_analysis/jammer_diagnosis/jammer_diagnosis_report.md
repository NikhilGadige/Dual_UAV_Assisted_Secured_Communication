# Jammer Block — Root Cause Diagnosis Report

## Summary

**Overall Severity: CRITICAL**

The jammer block contributes **0.0% of total improvement** in the SCA-BCD solver.  This diagnosis identifies the root cause and confirms the fix.

### Primary Root Cause (Part 1)

jammer_mode='mixed' overrides v_jammer with heuristic design. The objective is unchanged when v_jammer is scrambled under 'mixed', but changes under 'given'.

- Objective under 'mixed' with original v_jammer: 0.720588
- Objective under 'mixed' with scrambled v_jammer: 0.720588
- Objective under 'given' with original v_jammer: 0.776423
- Objective under 'given' with scrambled v_jammer: 0.765645
- Mixed mode is invariant to v_jammer: True
- Given mode is sensitive to v_jammer: True

### Consequence (Part 2)

The objective changes by at most 0.000e+00 under random jammer perturbations in 'mixed' mode. This confirms that finite-difference gradients are effectively zero, so the SCA/QP solver cannot find a meaningful step direction.
- Max objective change under random jammer perturbation: 0.000e+00

### Secondary Bug (Part 3)

Line 56 uses `norm > P_j_max` (threshold=0.05) instead of `norm**2 > P_j_max` i.e. `norm > sqrt(P_j_max)` (correct threshold ≈ 0.2236). For per-element-max jammer, norm=0.2236, power=0.0500 W, P_j_max=0.05 W. Incorrect threshold triggers: True. Correct threshold would trigger: False. The incorrect threshold constrains power to 0.002500 W instead of 0.05 W — a factor of 20× too restrictive.

### Trust Region Analysis (Part 4)

Trust region radius = 1.1180, variable range (2 × bound) = 0.2236, ratio = 5.0×. The trust region is 5.0× larger than the feasible space, so the SCA solver always respects bounds first. With zero gradient (Part 2), the solver has no information to move.

### SCA Solver Behaviour (Part 5)

SCA solver status: 'max_iters', iterations: 10, step norm: 2.500e-01, max element change: 8.589e-02. CLARABEL warnings detected! The non-zero step confirms some movement, but it may be noise.

### Corrected Sensitivity (Part 6)

With jammer_mode='given', the jammer optimizer improves objective from 0.776423 to 0.725039 (Δ = -0.051384). Under 'mixed', the objective is stuck at 0.720588. This confirms the jammer block IS effective — it is only paralyzed by the heuristic override.

## Recommended Fix

```python
# In bcd_solver.py or jammer_optimizer.py, temporarily set
# jammer_mode='given' when optimizing the jammer block.
#
# Option A (minimal): In env.evaluate() calls triggered by
#   jammer_optimizer.block_objective(), override jammer_mode
#   to 'given'.  The heuristic 'mixed' mode should still be
#   used for the initial solution and final evaluation.
#
# Option B (conceptual fix): Remove the heuristic override
#   entirely and let the optimizer learn both the phase
#   AND the power allocation end-to-end.
```

### Validation

After fix, re-run the BCD solver and verify:
1. Jammer block contribution > 1%
2. Objective improves after jammer block
3. Jammer SCA solver produces non-zero steps
4. All existing validation tests still pass

## Block Diagram

```
BCD Iteration:
  Power block  → w_bs changes → evaluate() reads w_bs  ✓ works
  Traj block   → q_uav changes → evaluate() reads q_uav ✓ works
  Jammer block → v_jammer changes → evaluate() IGNORES  ✗ BROKEN
                  because jammer_mode='mixed' overrides with heuristic

Fix: env.config.jammer_mode = 'given' for jammer optimizer
```
