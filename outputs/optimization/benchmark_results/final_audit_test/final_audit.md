# Final Audit Report

## 1. Numerical Robustness

  Issues found:
  - Pareto secrecy is not monotonic.
  - Pareto sensing is not monotonic.

## 2. Statistical Robustness

  7/8 comparisons statistically significant (p<0.05)
  7/8 with large effect size (|d|>0.8)

## 3. Benchmark Validity

  Expected ordering: Random < Single-block < Ablation < Full SCA-BCD
  VALID: Ranking is as expected.

## 4. Constraint Activity

  Dead constraints (0% activation): []
  Always active (100%): []
  Converged fraction: 0.0%

## 5. BCD Block Activity

  power: 0.0% of total improvement
  trajectory: 0.0% of total improvement
  jammer: 0.0% of total improvement
  WARNING: Inactive blocks:
    FLAG: power contributes only 0.00% of total improvement
    FLAG: trajectory contributes only 0.00% of total improvement
    FLAG: jammer contributes only 0.00% of total improvement

## 6. Recommendation

  Issues requiring attention:
  - Pareto secrecy non-monotonic
  - Pareto sensing non-monotonic
  - FLAG: N_RIS has R^2=0.5708 < 0.9
  - FLAG: power contributes only 0.00% of total improvement
  - FLAG: trajectory contributes only 0.00% of total improvement
  - FLAG: jammer contributes only 0.00% of total improvement
