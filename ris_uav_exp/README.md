# RIS-Mounted UAV Communication Model

Phase 1: Establish the RIS-mounted UAV communication model for a BS → RIS-UAV → User system with HPPP eavesdroppers.

## System

```
BS -----> RIS-UAV -----> User
              |
              v
         Eavesdroppers (HPPP)
```

## Files

| File / Directory | Description |
|---|---|
| `channels/ris_channel.py` | RIS channel model: Rician fading vectors, reflection matrix Φ, effective channel computation |
| `environments/ris_uav_env.py` | `RISUAVConfig` dataclass + `RISUAVEnvironment` with HPPP eves, rate/secrecy computation |
| `configs.py` | `RISExperimentConfig` + factory helpers |
| `validate_ris_uav.py` | Validation suite + plot generation |

## RIS Properties

| Property | Value |
|---|---|
| Duplex | Half |
| Type | Passive |
| Array | 4×4 planar |
| Elements | 16 |
| Altitude | Configurable (default 50 m) |

## Reflection Matrix

```
Φ = diag(e^{jφ₁}, e^{jφ₂}, ..., e^{jφ₁₆})
βₙ = 1  (ideal passive RIS)
```

Only phase shifts are optimised.

## Channel Links

| Link | Fading |
|---|---|
| BS ↔ RIS UAV | Rician |
| RIS UAV ↔ User | Rician |
| RIS UAV ↔ Eavesdroppers | Rician |

All reuse `core.channel.path_loss` for large-scale path loss.

## Effective Channel

```
h_eff = h_RUᴴ Φ h_BR
```

## Secrecy Rate

```
Rs = [R_legit − max(R_eve_i)]⁺
```

## Usage

```bash
# Run validation + generate plots
python -m ris_uav_exp.validate_ris_uav

# Outputs saved to outputs/ris_uav/
```

## Output Structure

```
outputs/ris_uav/
├── ris_phase_histogram.png
├── secrecy_rate_bar.png
├── effective_channel_gain.png
├── user_sinr.png
└── validation_summary.txt
```
