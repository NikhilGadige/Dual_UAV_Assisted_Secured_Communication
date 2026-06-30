# MADRL Stage 1 Training Report

## Learning Check

- **mappo_100**: first_20=6.6701 -> last_20=6.7257 (improved=True)
- **mappo_1000**: first_20=6.7169 -> last_20=7.0794 (improved=True)
- **matd3_100**: first_20=6.8293 -> last_20=7.0592 (improved=True)
- **matd3_1000**: first_20=6.6686 -> last_20=6.9385 (improved=True)

## Policy Stability

- Mean objective: 7.3229
- Std objective: 0.0747
- CV: 1.02%
- Per seed: [7.434892003435858, 7.257680878974907, 7.329992768192023, 7.226650292520527, 7.365084067861558]

## Numerical Checks

- nan_rewards: PASS
- nan_obs: PASS
- exploding_actions: PASS
- exploding_gradients: PASS
- critic_losses_finite: FAIL
- actor_losses_finite: FAIL

## Generalization

- seed 2: objective=7.0057, drop=-0.47%
- seed 3: objective=6.9658, drop=-1.03%
- seed 4: objective=7.2278, drop=2.69%
- seed 5: objective=7.0985, drop=0.85%

## Acceptance Criteria

- C1: PASS
- C2: PASS
- C3: PASS
- C4: PASS
- all_pass: PASS
- decision: PASS

## Decision: MADRL_STAGE1_COMPLETE
