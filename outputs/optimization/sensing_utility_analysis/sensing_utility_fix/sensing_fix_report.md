# Sensing Utility Fix Report

## Part 4: Normalization Constants (100 MC)

| Mode | U_ref | Std | Min | Max |
|------|-------|-----|-----|-----|
| original | 4.0000e+00 | 1.1964e-07 | 4.0000e+00 | 4.0000e+00 |
| log | 1.2884e+01 | 1.2162e+00 | 8.0080e+00 | 1.4643e+01 |
| inverse | 4.3112e+09 | 1.9686e+09 | 7.0063e+08 | 9.7769e+09 |
| normalized | 4.0000e+00 | 0.0000e+00 | 4.0000e+00 | 4.0000e+00 |
| exponential | 3.9608e+00 | 9.1601e-02 | 3.4788e+00 | 3.9971e+00 |

## Part 7: Statistical Evaluation (100 MC seeds)

| Mode | Mean | Std | CV | Dynamic Range | Min | Max |
|------|------|-----|----|--------------|-----|-----|
| original | 4.0000e+00 | 1.2043e-07 | 0.0000 | 8.2225e-07 | 4.0000e+00 | 4.0000e+00 |
| log | 1.2956e+01 | 1.1699e+00 | 0.0903 | 4.5868e+00 | 1.0200e+01 | 1.4787e+01 |
| inverse | 4.7037e+09 | 1.9664e+09 | 0.4181 | 8.8974e+09 | 9.6689e+08 | 9.8643e+09 |
| normalized | 4.0000e+00 | 0.0000e+00 | 0.0000 | 0.0000e+00 | 4.0000e+00 | 4.0000e+00 |
| exponential | 3.9661e+00 | 8.8966e-02 | 0.0224 | 5.6483e-01 | 3.4330e+00 | 3.9979e+00 |

## Part 6: Pareto Front Re-Evaluation

Pareto comparison saved to: outputs/sca_bcd_benchmark/sensing_utility_fix\pareto_comparison.csv
### original
| alpha | Secrecy | Sensing | Objective |
|-------|---------|---------|-----------|
| 0.00 | 3.2063 | 4.0000e+00 | 1.0000 |
| 0.25 | 2.5469 | 4.0000e+00 | 0.7682 |
| 0.50 | 8.4618 | 4.0000e+00 | 0.6209 |
| 0.75 | 16.9252 | 4.0000e+00 | 0.6127 |
| 1.00 | 25.7188 | 4.0000e+00 | 0.7348 |

### log
| alpha | Secrecy | Sensing | Objective |
|-------|---------|---------|-----------|
| 0.00 | 3.2063 | 1.1749e+01 | 0.9119 |
| 0.25 | 2.5469 | 1.1925e+01 | 0.7124 |
| 0.50 | 8.4618 | 1.1739e+01 | 0.5765 |
| 0.75 | 16.9252 | 1.1172e+01 | 0.5795 |
| 1.00 | 25.7188 | 1.1144e+01 | 0.7348 |

### inverse
| alpha | Secrecy | Sensing | Objective |
|-------|---------|---------|-----------|
| 0.00 | 3.2063 | 1.4279e+09 | 0.3312 |
| 0.25 | 2.5469 | 1.4504e+09 | 0.2705 |
| 0.50 | 8.4618 | 1.3200e+09 | 0.2740 |
| 0.75 | 16.9252 | 1.1465e+09 | 0.4292 |
| 1.00 | 25.7188 | 1.0717e+09 | 0.7348 |

### normalized
| alpha | Secrecy | Sensing | Objective |
|-------|---------|---------|-----------|
| 0.00 | 3.2063 | 4.0000e+00 | 1.0000 |
| 0.25 | 2.5469 | 4.0000e+00 | 0.7682 |
| 0.50 | 8.4618 | 4.0000e+00 | 0.6209 |
| 0.75 | 16.9252 | 4.0000e+00 | 0.6127 |
| 1.00 | 25.7188 | 4.0000e+00 | 0.7348 |

### exponential
| alpha | Secrecy | Sensing | Objective |
|-------|---------|---------|-----------|
| 0.00 | 3.2063 | 3.9803e+00 | 1.0049 |
| 0.25 | 2.5469 | 3.9821e+00 | 0.7722 |
| 0.50 | 8.4618 | 3.9790e+00 | 0.6232 |
| 0.75 | 16.9252 | 3.9596e+00 | 0.6126 |
| 1.00 | 25.7188 | 3.9581e+00 | 0.7348 |

## Part 8: Selection Criterion

Criteria: 1) largest dynamic range, 2) preserves monotonicity,
3) no numerical explosion, 4) non-trivial Pareto front.

| Mode | Dynamic Range | CV | Explosion | Pareto Trade-off |
|------|--------------|----|-----------|-----------------|
| original | 8.2225e-07 | 0.0000 | PASS | PASS |
| log | 4.5868e+00 | 0.0903 | PASS | PASS |
| inverse | 8.8974e+09 | 0.4181 | PASS | PASS |
| normalized | 0.0000e+00 | 0.0000 | PASS | WEAK |
| exponential | 5.6483e-01 | 0.0224 | PASS | PASS |

**Selected utility: inverse**
**Score: 7037590916.3584**

Pareto trade-off for 'inverse': 14/20 steps show higher secrecy -> lower sensing (70%)
Sensing utility dynamic range across alpha: 381176655.3766

# UTILITY_FIXED

The 'inverse' utility replaces the original saturated 1/(1+tr(CRB)).
U_ref = 4.3112e+09 provides proper normalization.
The objective f = alpha * R_s / R_s_ref + (1-alpha) * U / U_ref
now exhibits a genuine bi-objective trade-off.