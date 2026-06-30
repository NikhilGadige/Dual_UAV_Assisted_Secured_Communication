# MADRL Baseline Comparison Report

## Metrics

| Method | Reward | Secrecy | Sensing | Runtime |
|--------|--------|---------|---------|--------|
| random_feasible | 7.0097 | 1.2752 | 41.3096 | 3.636s |
| sca_bcd | 0.0000 | 9.4397 | 43.6146 | 20.159s |
| mappo_1000 | 7.0796 | 2.7884 | 42.2637 | 2.337s |
| matd3_1000 | 6.8683 | 0.0500 | 41.8938 | 2.672s |

## Learning Check

- **mappo_100**: first_20=6.6701, last_20=6.7257, improved=True
- **mappo_1000**: first_20=6.7169, last_20=7.0794, improved=True
- **matd3_100**: first_20=6.8293, last_20=7.0592, improved=True
- **matd3_1000**: first_20=6.6686, last_20=6.9385, improved=True

## Summary

- Random feasible baseline: secrecy=1.2752, sensing=41.3096
- SCA-BCD baseline: secrecy=9.4397, sensing=43.6146
- MAPPO (1000 ep): secrecy=2.7884, sensing=42.2637
- MATD3 (1000 ep): secrecy=0.0500, sensing=41.8938
