# Phase 4D: Target Detection Metrics for RIS-UAV Sensing

Implements binary hypothesis testing, energy detector, GLRT, and Monte-Carlo estimation of Pd and Pfa for the RIS-UAV sensing subsystem.

---

## 1. Binary Hypothesis Formulation

    H0:  Y = N          (noise only)
    H1:  Y = H_sense*X + N   (target echo present)

where N ~ CN(0, sigma^2 I_Nr) and X in C^{Nt x L} is the known pilot matrix.

---

## 2. Energy Detector

Test statistic:

    T_ED(Y) = ||Y||_F^2 = sum_{i,j} |Y_{i,j}|^2

Decision:

    T_ED(Y) > gamma  =>  H1

Under H0: T_ED ~ (sigma^2/2) * chi^2(2*Nr*L)
Under H1: T_ED ~ non-central chi^2

---

## 3. GLRT Detector

For unknown H_sense, the MLE under H1 is:

    H_hat = Y * X^H * (X*X^H)^{-1}

The GLRT statistic:

    Lambda(Y) = ||Y*P||_F^2 / ||Y - Y*P||_F^2

where P = X^H * (X*X^H)^{-1} * X is the projection onto the row space of X.

Large Lambda(Y) favours H1.

---

## 4. Detection Metrics (Monte Carlo)

For threshold gamma:

    Pfa = P(T(Y) > gamma | H0)   (estimated via N_mc H0 realizations)
    Pd  = P(T(Y) > gamma | H1)   (estimated via N_mc H1 realizations)

N_mc = 200-500 used in validation.

---

## 5. Assumptions

1. N_tx = N_rx = 16 (monostatic, configurable).
2. Known pilot matrix X (unit-norm columns).
3. Complex Gaussian noise with known variance.
4. Unknown H_sense under H1 (GLRT handles this implicitly).
5. No prior information on target parameters.
6. No SCA/BCD, MADRL, or joint optimisation.

---

## 6. File Structure

detection_sensing_exp/
|-- __init__.py
|-- configs.py
|-- README.md
|-- validate_detection_sensing.py
|-- channels/
|   |-- __init__.py
|   |-- detection_channel.py
|-- environments/
|   |-- __init__.py
|   |-- detection_sensing_env.py

### Key modules

| File | Responsibility |
|---|---|
| detection_channel.py | H0/H1 generation, ED/GLRT statistics, detect(), Monte Carlo |
| detection_sensing_env.py | Config, environment, ROC curves, parameter sweeps |
| validate_detection_sensing.py | 10 tests + 6 plots -> outputs/detection_sensing/ |

---

## 7. How to Run

python -m detection_sensing_exp.validate_detection_sensing

Outputs: outputs/detection_sensing/
- validation_summary.txt
- roc_curves.png
- pd_vs_snr.png
- pd_vs_pilots.png
- pd_vs_targets.png
- detector_comparison.png
