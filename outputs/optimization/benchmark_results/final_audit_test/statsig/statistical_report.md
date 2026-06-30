# Statistical Significance Report

Target method: sca_bcd_full vs each baseline (N_mc=5)

| Comparison | N_pairs | t_stat | t_pval | wilcoxon_stat | wilcoxon_pval | Cohen's d | Mean diff |
|-----------|---------|--------|--------|---------------|----------------|-----------|-----------|
| sca_bcd_full_vs_random_feasible | 5 | 15.7756 | 9.4333e-05 | 0.0000 | 6.2500e-02 | 9.0088 | 0.1958 |
| sca_bcd_full_vs_power_only | 5 | 15.4934 | 1.0130e-04 | 0.0000 | 6.2500e-02 | 7.4770 | 0.1655 |
| sca_bcd_full_vs_trajectory_only | 5 | 0.4562 | 6.7194e-01 | 7.0000 | 1.0000e+00 | 0.2244 | 0.0052 |
| sca_bcd_full_vs_jammer_only | 5 | 15.7756 | 9.4333e-05 | 0.0000 | 6.2500e-02 | 9.0088 | 0.1958 |
| sca_bcd_full_vs_no_ris | 5 | 15.7756 | 9.4333e-05 | 0.0000 | 6.2500e-02 | 9.0088 | 0.1958 |
| sca_bcd_full_vs_no_jammer | 5 | 15.7756 | 9.4333e-05 | 0.0000 | 6.2500e-02 | 9.0088 | 0.1958 |
| sca_bcd_full_vs_no_secrecy | 5 | -55.5734 | 6.2769e-07 | 0.0000 | 6.2500e-02 | -35.1477 | -0.7546 |
| sca_bcd_full_vs_no_sensing | 5 | 16.9341 | 7.1297e-05 | 0.0000 | 6.2500e-02 | 2.9688 | 0.0862 |

## Interpretation

- p < 0.05: statistically significant difference
- |Cohen's d| > 0.8: large effect size
- |Cohen's d| > 0.5: medium effect size
- |Cohen's d| > 0.2: small effect size
