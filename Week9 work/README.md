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
  Computed by interpolating an SINR $\to$ similarity table produced by evaluating an actual **trained DeepSC model** (Xie et al., 2021 - Transformer semantic encoder/decoder + dense channel encoder/decoder + AWGN channel) across a grid of SINR values, rather than a hand-fitted closed-form curve. See Section 5 below.

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
│── deepsc_model.py           # DeepSC architecture (Transformer semantic encoder/decoder,
│                              #   dense channel encoder/decoder, AWGN channel)
│── deepsc_corpus.py          # Small synthetic text corpus used to train DeepSC
│── train_deepsc.py           # Trains DeepSC and generates deepsc_lookup_table.csv
│── deepsc_lookup.py          # Torch-free loader/interpolator for the lookup table,
│                              #   used by semantic_node.py / noma_module.py at runtime
│── deepsc_lookup_table.csv   # Generated SINR(dB) -> semantic similarity / word accuracy table
│── deepsc_model.pt           # Trained DeepSC checkpoint (weights + vocab)
│── test_week9_model.py       # Verification test suite
└── README.md                 # Module documentation
```

---

## 4. How to Run Verification
To run the verification suite:
```bash
python "Week9 work/test_week9_model.py"
```

---

## 5. DeepSC Semantic Similarity Model

Rather than approximating $S(\text{SINR})$ with a hand-tuned sigmoid, this module trains and evaluates an actual **DeepSC** model (Xie et al., "Deep Learning Enabled Semantic Communication Systems", IEEE TSP 2021) and uses its measured performance to drive every semantic-similarity computation in the system model.

**Pipeline**: `Semantic Source -> Transformer Semantic Encoder -> Dense Channel Encoder (power-normalized) -> AWGN Channel -> Dense Channel Decoder -> Transformer Semantic Decoder -> Recovered Sentence`

- `deepsc_model.py` implements this architecture in PyTorch, plus a `Vocabulary` helper.
- `deepsc_corpus.py` generates a small, reproducible synthetic sentence corpus (template combinations over an ISAC-flavored vocabulary) to train on, so the pipeline runs in minutes without downloading a large external dataset.
- `train_deepsc.py` trains DeepSC end-to-end (cross-entropy reconstruction loss, SINR sampled randomly per batch over a realistic training range), then evaluates the trained model at a grid of fixed SINR values. At each grid point it greedily decodes the validation sentences through the full noisy pipeline and measures:
  - **semantic_similarity**: cosine similarity between the semantic encoder's sentence embedding of the original sentence and of the reconstructed sentence (used here as a self-contained proxy for the BERT-based sentence similarity used in the original paper, avoiding an external model download), rescaled to $[0, 1]$.
  - **word_accuracy**: token-level reconstruction accuracy (diagnostic).

  The result is written to `deepsc_lookup_table.csv`, and the trained weights to `deepsc_model.pt`.
- `deepsc_lookup.py` loads that CSV and linearly interpolates it (flat outside the trained range) — this is what `semantic_node.py` and `noma_module.py` call at simulation time. It has no torch dependency, so running the ISAC system model does **not** require loading DeepSC itself, only its precomputed performance table.

**Regenerating the table** (only needed if the corpus, architecture, or training range changes):
```bash
python "Week9 work/train_deepsc.py"
```

**Note on training range**: the model is trained/evaluated over SINRs from -20 dB to +20 dB. Training across an unrealistically wide range (e.g. down to -100 dB, to try to match some of this system's very low-SINR eavesdropper/sensing links) was tried and rejected — at those SINRs no scheme can recover the signal, so those batches injected near-random gradients that destabilized the shared encoder/decoder weights and hurt convergence even at high SINR. Links whose SINR falls below -20 dB are flat-extrapolated to the table's near-total-failure floor, which is a physically reasonable stand-in (the channel is effectively unusable there) rather than a modeling artifact.
