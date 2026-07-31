# Week 9 Work: Semantic-Aware ISAC Network with PD-NOMA and Elevation-Dependent Path Loss

## Overview
This module implements the Week 9 updated system model specifications for the Integrated Sensing and Communication (ISAC) network assisted by an RIS-mounted UAV ($R$) and subject to a UAV Jammer ($J$).

The system incorporates three major modeling enhancements:
1. **Semantic Awareness Across All Nodes**: All communication and sensing entities operate under semantic-aware metrics (Semantic Similarity, Semantic Rate, Semantic Distortion, and Semantic Sensing Accuracy).
2. **Power-Domain NOMA (PD-NOMA)**: Enables simultaneous transmission to near users (e.g., Target $T$) and distant users (e.g., Mobile User $U$) over superposed power signals with semantic-aware Successive Interference Cancellation (SIC) decoding.
3. **Elevation-Dependent Path Loss Model**: Replaces standard distance-only path loss with 3D Air-to-Ground (A2G) elevation-angle-dependent Line-of-Sight (LoS) probability $P_{\text{LoS}}(\theta)$ and excessive attenuation factors.

---

## 1. System Architecture & Nodes

| Node Name | Node Type | Role / Capabilities |
| :--- | :--- | :--- |
| **Base Station ($B$)** | Base Station | Semantic-aware ISAC Beamformer transmitting superposed NOMA signal to RIS UAV. |
| **RIS-mounted UAV ($R$)** | RIS UAV | Aerial reflecting surface facilitating cascaded signal transfer to users and targets. |
| **UAV Jammer ($J$)** | UAV Jammer | Aerial jammer transmitting directional jamming signal and monostatic radar sensing reflection. |
| **Target ($T$)** | Near User & Active Communicator | Ground target near RIS UAV; acts as Downlink NOMA receiver (with semantic SIC) AND active semantic transmitter (Uplink V2I via RIS to BS, and V2X to Mobile User $U$). |
| **Mobile User ($U$)** | Distant User | Ground mobile user far from RIS UAV; decodes far-user NOMA signal directly. |
| **Eavesdroppers ($E_1, E_2, E_3$)** | Wiretap Nodes | Ground eavesdropping nodes attempting to wiretap NOMA semantic transmissions. |

---

## 2. Mathematical Formulations

### A. Elevation-Dependent Path Loss Model
For any link with 3D distance $d_{ij}$ and vertical elevation difference $\Delta z = |z_i - z_j|$:

- **Elevation Angle ($\theta_{ij}$)**:
  $$\theta_{ij} = \arcsin\left(\frac{\Delta z}{d_{ij}}\right) \quad (\text{in degrees})$$

- **Sigmoidal LoS Probability $P_{\text{LoS}}(\theta_{ij})$**:
  $$P_{\text{LoS}}(\theta_{ij}) = \frac{1}{1 + a \cdot \exp\left(-b(\theta_{ij} - a)\right)}$$
  *(Default parameters for Urban A2G: $a = 9.61, b = 0.16$)*

- **Mean Path Loss $\overline{\text{PL}}(d_{ij}, \theta_{ij})$**:
  $$\overline{\text{PL}}(d_{ij}, \theta_{ij}) = P_{\text{LoS}}(\theta_{ij}) \cdot \text{PL}_{\text{LoS}}(d_{ij}) + (1 - P_{\text{LoS}}(\theta_{ij})) \cdot \text{PL}_{\text{NLoS}}(d_{ij})$$
  $$\text{PL}_{\text{LoS}}(d_{ij}) = 20\log_{10}\left(\frac{4\pi d_{ij} f_c}{c}\right) + \eta_{\text{LoS}}$$
  $$\text{PL}_{\text{NLoS}}(d_{ij}) = 20\log_{10}\left(\frac{4\pi d_{ij} f_c}{c}\right) + \eta_{\text{NLoS}}$$

- **Channel Power Gain ($g_{ij}$)**:
  $$g_{ij} = \text{PL}_{\text{linear}}(d_{ij}, \theta_{ij}) \cdot h_{ij}$$
  *(where $h_{ij}$ represents small-scale fading power gain)*

---

### B. Power-Domain NOMA & Semantic-Aware SIC Decoding
Superposed signal at transmitter (Base Station $B$):
$$s = \sqrt{a_N P_B} s_N + \sqrt{a_F P_B} s_F \quad (a_F + a_N = 1, \; a_F > a_N)$$

- **Distant User ($U$) Decoding**:
  $$\text{SINR}_F = \frac{a_F P_B g_U}{a_N P_B g_U + P_{J, U} + \sigma^2}$$

- **Near User / Target ($T$) SIC Decoding**:
  Decodes own signal $s_N$ after subtracting the already decoded far user signal $s_F$:
  $$\text{SINR}_N = \frac{a_N P_B g_T}{P_{J, T} + \sigma^2}$$
  *(where $P_{J, T}$ is the received jamming power at Target $T$, and $\sigma^2$ is the noise variance)*

---

### C. Semantic Communication & Sensing Metrics
- **Semantic Similarity Score $S(\text{SINR})$**:
  $$S(\text{SINR}) = \frac{1}{1 + \exp\left(-(\lambda_1 \cdot \text{SINR}_{\text{dB}} + \lambda_2)\right)}$$

- **Semantic Rate $R_{\text{sem}}$ (suts/s)**:
  $$R_{\text{sem}} = B \cdot \left(\frac{M}{K}\right) \cdot S(\text{SINR}) \cdot \log_2(1 + \text{SINR})$$
  *(where $M$ is semantic concepts, $K$ is channel symbols, $B$ is bandwidth)*

- **Semantic Sensing Accuracy $A_{\text{sensing}, k}$**:
  Evaluated for both the Target ($k = T$) and Eavesdroppers ($k = E_i$) via direct monostatic sensing at UAV Jammer $J$:
  $$\text{SNR}_{\text{radar}, k} = \frac{P_J \cdot g_{J, k}^2}{\sigma^2}$$
  $$A_{\text{sensing}, k} = \frac{1}{1 + \exp\left(-\kappa (\text{SNR}_{\text{radar}, k, \text{dB}} - \tau)\right)}$$

---

## 3. Directory Structure
```
Week9 work/
│── __init__.py               # Package initialization & exports
│── elevation_path_loss.py    # Elevation-dependent path loss & channel gain module
│── noma_module.py            # Power-Domain NOMA and SIC decoding engine
│── semantic_node.py          # Semantic-aware node dataclass & metric functions
│── system_model.py           # Full system integration for Week 9 ISAC network
│── test_week9_model.py       # Verification test suite
└── README.md                 # Module documentation
```

---

## 4. How to Run Verification
To run the verification suite:
```bash
python "Week9 work/test_week9_model.py"
```
