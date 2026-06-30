# Conditioning Validation Report

**Overall**: ALL TESTS PASSED

## Test Results

| Test | Status | Details |
|------|--------|---------|
| test_scaled_gradients_finite | PASS | {"power": {"g_unscaled_norm": 12647.72198334403, "g_scaled_norm": 40019.98589795497, "max_chain_rel_err": 2.1808792934634877, "median_chain_rel_err": 1.8928984520799614, "n_significant_elements": 10, "g_unscaled_finite": true, "g_scaled_finite": true, "chain_rule_ok": true}, "trajectory": {"g_unscaled_norm": 1999.9999364630291, "g_scaled_norm": 363937.6612871534, "max_chain_rel_err": 0.0019806765652832, "median_chain_rel_err": 0.0005660792821183753, "n_significant_elements": 15, "g_unscaled_finite": true, "g_scaled_finite": true, "chain_rule_ok": true}, "jammer": {"g_unscaled_norm": 304.4961524963345, "g_scaled_norm": 68.11495024807365, "max_chain_rel_err": 0.0035520103256219994, "median_chain_rel_err": 0.000326941749086587, "n_significant_elements": 32, "g_unscaled_finite": true, "g_scaled_finite": true, "chain_rule_ok": true}} |
| test_conditioning_improved | PASS | {"ratio_before": 1236.4402263617217, "ratio_after": 3.6411985162088216, "improved": true, "under_threshold_50": true, "mean_sens_unscaled": {"power": 0.0036719565314435785, "trajectory": 0.00014782324215108328, "jammer": 0.182774602986809}, "mean_sens_scaled": {"power": 0.011611746108493401, "trajectory": 0.011224228369305492, "jammer": 0.04086964368390412}} |
| test_adaptive_fd_stability | PASS | {"power": {"adaptive_norm": 40019.98589795497, "fixed_norm": 39998.791235519355, "norm_ratio": 1.0005298825734712, "sign_agreement_pct": 90.0, "n_sign_checked": 10, "all_finite": true}, "trajectory": {"adaptive_norm": 363937.6612871534, "fixed_norm": 363738.64900902146, "norm_ratio": 1.0005471298655617, "sign_agreement_pct": 100.0, "n_sign_checked": 15, "all_finite": true}, "jammer": {"adaptive_norm": 68.11495024807365, "fixed_norm": 68.08237058513134, "norm_ratio": 1.000478533027894, "sign_agreement_pct": 100.0, "n_sign_checked": 32, "all_finite": true}} |
