# TD3PG Convergence Study

This folder contains the TD3PG convergence-study code for the dual-UAV secure uplink setup.
It uses the repository's `UAVEnvironment`, with a relay UAV, jammer UAV, one random-walk mobile user,
one eavesdropper, and a static base station. The objective remains maximization of secrecy rate.

Outputs are written only under:

```text
td3pg_study/output/
```

Recommended study runs:

```bash
python -m td3pg_study.td3pg_train --channel-model rician --episodes 4000 --hidden-dim 64 --seed 42
python -m td3pg_study.td3pg_train --channel-model rayleigh --episodes 4000 --hidden-dim 64 --seed 42
```

A combined run is also available:

```bash
python -m td3pg_study.run_both
```

For a quicker draft run, use `3000` episodes. For a smoke test, use `--episodes 5`.
