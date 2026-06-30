# Benchmark Ranking Sanity Report

Config: N_mc=5, max_bcd_iters=3, max_sca_iters=2

## Rankings (mean objective)

  random_feasible            0.716226 ± 0.004309
  power_only                 0.746515 ± 0.006842
  trajectory_only            0.906897 ± 0.010446
  jammer_only                0.716226 ± 0.004309
  no_ris                     0.716226 ± 0.004309
  no_jammer                  0.716226 ± 0.004309
  no_secrecy                 1.666667 ± 0.000000
  no_sensing                 0.825869 ± 0.024719
  sca_bcd_full               0.912060 ± 0.027157

## Expected Ordering

  Random < Single-block < Ablation variants < Full SCA-BCD

## No flags raised. Ordering is as expected.
