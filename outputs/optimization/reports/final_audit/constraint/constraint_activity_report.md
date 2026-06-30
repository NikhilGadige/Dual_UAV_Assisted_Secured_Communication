# Constraint Activity Analysis

Total runs: 5, Converged: 100.0%, Mean iters: 4.00

| Constraint | Active count | Frequency | Mean viol | Max viol |
|------------|-------------|-----------|-----------|----------|
| bs_power_excess | 0/5 | 0.00% | 0.0000e+00 | 0.0000e+00 |
| bs_power_negative | 0/5 | 0.00% | 0.0000e+00 | 0.0000e+00 |
| jammer_power_excess | 0/5 | 0.00% | 0.0000e+00 | 0.0000e+00 |
| uav_speed_excess | 0/5 | 0.00% | 7.1054e-15 | 7.1054e-15 |
| uav_boundary_violation | 0/5 | 0.00% | 0.0000e+00 | 0.0000e+00 |
| secrecy_rate_shortfall | 0/5 | 0.00% | 0.0000e+00 | 0.0000e+00 |
| sensing_utility_shortfall | 0/5 | 0.00% | 0.0000e+00 | 0.0000e+00 |

## Dead Constraints (never active)
  bs_power_excess
  bs_power_negative
  jammer_power_excess
  uav_speed_excess
  uav_boundary_violation
  secrecy_rate_shortfall
  sensing_utility_shortfall

## Always Active Constraints
  None.

## Interpretation

- Dead constraints (0% activation): can potentially be removed
- Always active (100%): likely binding at optimum
- Low activation (<5%): check if constraint is needed
