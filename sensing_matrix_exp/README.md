# Phase 4B: Sensing Matrix Framework

Extends Phase 4A by introducing the sensing channel matrices required for future CRB formulation and joint communication-sensing optimization.

---

## 1. Mathematical Derivation

We model a monostatic sensing system co-located at the RIS-UAV. The UAV is equipped with N_tx transmit antennas and N_rx receive antennas, both forming uniform linear arrays (ULAs). Each vehicle target reflects the transmitted signal, and the received echo is modeled as a matrix whose structure captures the target directions and reflection coefficients.

---

## 2. Steering Vector Equations

For a ULA with N elements, inter-element spacing d, and wavelength lambda, the steering vector for a target at angle theta (relative to array broadside) is:

    a(theta) = [1, exp(-j * 2*pi * d * sin(theta) / lambda),
                exp(-j * 2*pi * 2 * d * sin(theta) / lambda),
                ...
                exp(-j * 2*pi * (N-1) * d * sin(theta) / lambda)]^T

The transmit and receive steering vectors are identical when the same array is used for both functions (monostatic case):

    a_tx(theta) = a_rx(theta) = a(theta)  in (N x 1)

Each element has unit magnitude, i.e. |a_n| = 1, giving ||a|| = sqrt(N).

---

## 3. Target Response Matrix

For each target i at angle theta_i with complex reflection coefficient alpha_i, the rank-1 response matrix is:

    A_i = a_rx(theta_i) * a_tx(theta_i)^H

Dimensions: A_i in C^{N_r x N_t}

Properties:
- Rank = 1 (outer product of two vectors)
- Frobenius norm ||A_i||_F = ||a_rx|| * ||a_tx|| = sqrt(N_r * N_t)

---

## 4. Composite Sensing Channel

The total channel matrix is the superposition of all K target responses:

    H_sense = sum_{i=1}^{K} alpha_i * A_i

where alpha_i captures the combined effect of radar cross-section, path loss, and phase shift (scalar reflection coefficient from Phase 4A).

Dimensions: H_sense in C^{N_r x N_t}

Properties:
- rank(H_sense) <= min(K, N_t, N_r)
- For K >= min(N_t, N_r), the matrix becomes full rank in general
- ||H_sense||_F = sqrt( sum_{i,j} |H_{i,j}|^2 )

---

## 5. Echo Matrix Model

The UAV transmits L pilot symbols from N_tx antennas, forming the pilot matrix X in C^{N_t x L}. The received echo matrix is:

    Y = H_sense * X + N

Components:
- H_sense: Composite sensing channel (N_r x N_t)
- X: Pilot matrix (N_t x L), columns are unit-norm pilot vectors
- N: Complex Gaussian noise (N_r x L), each element ~ CN(0, sigma^2)
- Y: Received echo matrix (N_r x L)

We use column-normalized random pilots:

    X_{:,l} = x_l / ||x_l||,   x_l ~ CN(0, I)

---

## 6. Covariance Model

Three key covariance matrices:

| Covariance | Expression | Dimensions | Description |
|---|---|---|---|
| Pilot covariance | R_x = (1/L) * X * X^H | N_t x N_t | Sample covariance of pilots |
| Signal covariance | R_sig = H_sense * R_x * H_sense^H | N_r x N_r | Contribution of reflected echoes |
| Noise covariance | R_n = sigma^2 * I_{N_r} | N_r x N_r | White noise covariance |
| Total covariance | R_y = E[Y * Y^H] = (1/L) * Y * Y^H | N_r x N_r | Sample estimate of total received covariance |

The total covariance satisfies the PSD property:

    R_y = R_sig + R_n   (in expectation)
    R_y >= 0  (positive semidefinite)

Eigenvalues of R_y reveal the effective rank (signal subspace dimension):
- Signal eigenvalues correspond to the K dominant modes (targets)
- Noise floor eigenvalues cluster around sigma^2

---

## 7. File Structure

```
sensing_matrix_exp/
|-- __init__.py
|-- configs.py
|-- README.md
|-- validate_sensing_matrix.py
|-- channels/
|   |-- __init__.py
|   |-- sensing_matrix_channel.py
|-- environments/
|   |-- __init__.py
|   |-- sensing_matrix_env.py
```

### Key modules

| File | Responsibility |
|---|---|
| `channels/sensing_matrix_channel.py` | Steering vectors, target response, composite channel, echo matrix, covariances |
| `environments/sensing_matrix_env.py` | Config dataclass, environment class with reset/step |
| `validate_sensing_matrix.py` | 8 tests + 5 plots, output to `outputs/sensing_matrix/` |

---

## 8. How to Run

```bash
python -m sensing_matrix_exp.validate_sensing_matrix
```

Or from the project root:

```bash
python sensing_matrix_exp/validate_sensing_matrix.py
```

Outputs are written to `outputs/sensing_matrix/`:
- `validation_summary.txt`
- `steering_vector_magnitude.png`
- `sensing_matrix_heatmap.png`
- `covariance_eigenvalues.png`
- `target_matrix_rank.png`

---

## Assumptions

1. N_tx = N_rx = N_ris = 16 in default configuration (can be changed independently).
2. Uniform linear arrays (ULA) with d = 0.5 lambda.
3. Monostatic: TX and RX arrays co-located at the RIS-UAV position.
4. Narrowband model: delay across array is negligible.
5. Point-target: each vehicle is a single reflector at one angle.
6. Pilots are unit-norm per column (power control across symbols).
7. Complex Gaussian noise, independent across receive antennas and time.
8. Reflection coefficient alpha_i is a complex scalar (free-space RCS + path loss aggregated).
9. No mutual coupling between antenna elements.
10. Far-field assumption: planar wavefront at the array.
