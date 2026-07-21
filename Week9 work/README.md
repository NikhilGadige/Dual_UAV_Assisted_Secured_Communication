# Week 9 Work: Semantic-Aware ISAC Network with PD-NOMA and Elevation-Dependent Path Loss

## Overview
This module implements the Week 9 updated system model specifications for the Integrated Sensing and Communication (ISAC) network assisted by an RIS-mounted UAV and subject to a UAV Jammer.

The system incorporates three major modeling enhancements:
1. **Semantic Awareness Across All Nodes**: All communication and sensing entities operate under semantic-aware metrics (Semantic Similarity, Semantic Rate, Semantic Distortion, and Semantic Sensing Accuracy).
2. **Power-Domain NOMA (PD-NOMA)**: Enables simultaneous transmission to near users (e.g., Vehicle $V$) and distant users (e.g., Mobile User $U$) over superposed power signals with Successive Interference Cancellation (SIC) decoding.
3. **Elevation-Dependent Path Loss Model**: Replaces standard distance-only path loss with 3D Air-to-Ground (A2G) elevation-angle-dependent Line-of-Sight (LoS) probability $P_{\text{LoS}}(\theta)$ and excessive attenuation factors.

---

## 1. System Architecture & Nodes

| Node Name | Node Type | Role / Capabilities |
| :--- | :--- | :--- |
| **Base Station ($BS$)** | Base Station | Semantic-aware ISAC Beamformer transmitting superposed NOMA signal to RIS UAV. |
| **RIS-mounted UAV ($R$)** | RIS UAV | Aerial reflecting surface facilitating cascaded signal transfer to users and targets. |
| **UAV Jammer ($J$)** | UAV Jammer | Aerial jammer transmitting directional jamming signal and monostatic radar sensing reflection. |
| **Vehicle ($V$)** | Near User & Active Communicator | Ground vehicle near RIS UAV; acts as Downlink NOMA receiver (with SIC) AND active semantic transmitter (Uplink V2I via RIS to BS, and V2X to Mobile User $U$). |
| **Mobile User ($U$)** | Distant User | Ground mobile user far from RIS UAV; decodes far-user NOMA signal directly. |
| **Eavesdroppers ($E_1, E_2, E_3$)** | Intercept Nodes | Ground eavesdropping nodes attempting to intercept NOMA semantic transmissions. |

---

## 2. Mathematical Formulations

### A. Elevation-Dependent Path Loss Model
For any link with 3D distance $d = \|\mathbf{p}_{\text{tx}} - \mathbf{p}_{\text{rx}}\|_2$ and vertical elevation difference $\Delta z = |z_{\text{tx}} - z_{\text{rx}}|$:

- **Elevation Angle ($\theta$)**:
  $$\theta = \arcsin\left(\frac{\Delta z}{d}\right) \quad (\text{in degrees})$$

- **Sigmoidal LoS Probability $P_{\text{LoS}}(\theta)$**:
  $$P_{\text{LoS}}(\theta) = \frac{1}{1 + a \cdot \exp\left(-b(\theta - a)\right)}$$
  *(Default parameters for Urban A2G: $a = 9.61, b = 0.16$)*

- **Mean Path Loss $\overline{\text{PL}}(d, \theta)$**:
  $$\overline{\text{PL}}(d, \theta) = P_{\text{LoS}}(\theta) \cdot \text{PL}_{\text{LoS}}(d) + (1 - P_{\text{LoS}}(\theta)) \cdot \text{PL}_{\text{NLoS}}(d)$$
  $$\text{PL}_{\text{LoS}}(d) = 20\log_{10}\left(\frac{4\pi d f_c}{c}\right) + \eta_{\text{LoS}}$$
  $$\text{PL}_{\text{NLoS}}(d) = 20\log_{10}\left(\frac{4\pi d f_c}{c}\right) + \eta_{\text{NLoS}}$$

---

### B. Power-Domain NOMA & SIC Decoding
Superposed signal at transmitter:
$$s = \sqrt{a_N P_t} s_N + \sqrt{a_F P_t} s_F \quad (a_F + a_N = 1, \; a_F > a_N)$$

- **Distant User ($U_F$) Decoding**:
  $$\text{SINR}_F = \frac{a_F P_t |h_F|^2}{a_N P_t |h_F|^2 + P_{\text{jam}} |h_{J,F}|^2 + \sigma^2}$$

- **Near User ($U_N$) SIC Decoding**:
  1. *Step 1*: Decodes Far signal $s_F$:
     $$\text{SINR}_{N \to F} = \frac{a_F P_t |h_N|^2}{a_N P_t |h_N|^2 + P_{\text{jam}} |h_{J,N}|^2 + \sigma^2}$$
  2. *Step 2*: If $\text{SINR}_{N \to F} \ge \gamma_{\text{SIC}}$, subtracts $s_F$ and decodes $s_N$:
     $$\text{SINR}_N = \frac{a_N P_t |h_N|^2}{P_{\text{jam}} |h_{J,N}|^2 + \sigma^2}$$

---

### C. Semantic Communication & Sensing Metrics
- **Semantic Similarity Score $S(\text{SINR})$**:
  $$S(\text{SINR}) = \frac{1}{1 + \exp\left(-(\lambda_1 \cdot \text{SINR}_{\text{dB}} + \lambda_2)\right)}$$

- **Semantic Rate $R_{\text{sem}}$ (suts/s)**:
  $$R_{\text{sem}} = B \cdot \left(\frac{M}{K}\right) \cdot S(\text{SINR}) \cdot \log_2(1 + \text{SINR})$$
  *(where $M$ is semantic concepts, $K$ is channel symbols, $B$ is bandwidth)*

- **Semantic Sensing Accuracy $A_{\text{sensing}}$**:
  $$A_{\text{sensing}}(\text{SNR}_{\text{radar}}) = \frac{1}{1 + \exp\left(-\kappa (\text{SNR}_{\text{radar, dB}} - \tau)\right)}$$

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
