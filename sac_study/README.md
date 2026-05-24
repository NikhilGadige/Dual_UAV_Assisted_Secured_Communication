# SAC Convergence Study

This folder contains the Soft Actor-Critic convergence-study code for the dual-UAV secure uplink setup.
It uses the repository's `UAVEnvironment`, with a relay UAV, jammer UAV, one random-walk mobile user,
one eavesdropper, and a static base station. The objective is still secrecy-rate maximization.

Outputs are written only under:

```text
sac_study/output/
```

Recommended final convergence runs:

```bash
python -m sac_study.sac_train --channel-model rician --episodes 4000 --hidden-dim 64 --seed 42
python -m sac_study.sac_train --channel-model rayleigh --episodes 4000 --hidden-dim 64 --seed 42
```

A combined run is also available:

```bash
python -m sac_study.run_both
```

For a quicker draft run, use `3000` episodes. For a smoke test, use `--episodes 5`.

