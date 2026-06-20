# SCA-BCD Optimization Problem Formulation

## 1. System Model

### 1.1 Network Topology

We consider a dual-UAV cooperative communication system consisting of the following entities:

- **Source node (S)**: A ground user (or vehicle-mounted transmitter) located at time-varying two-dimensional positions $\mathbf{u}[m] \in \mathbb{R}^2$, $m = 1,\dots,M$. When the vehicle receiver model is disabled, the source is stationary; when enabled, it follows a straight-road mobility model with maximum speed $8$ m/s.

- **Legitimate receiver (D)**: A fixed destination (base station) located at the origin $\mathbf{d} = [0,0]^\top \in \mathbb{R}^2$.

- **Relay UAV (R)**: A decode-and-forward relay UAV that flies at fixed altitude $H_r = 50$ m. Its horizontal position at time slot $m$ is denoted $\mathbf{q}_r[m] \in \mathbb{R}^2$. The relay assists the source–destination link.

- **Jammer UAV (J)**: A cooperative jammer UAV flying at fixed altitude $H_j = 50$ m. Its horizontal position at time slot $m$ is $\mathbf{q}_j[m] \in \mathbb{R}^2$. The jammer transmits artificial noise to degrade eavesdropper channels.

- **Eavesdroppers (Eves)**: Multiple passive eavesdroppers located on the ground. Their positions are generated as a homogeneous Poisson point process (HPPP) with density $\lambda = 2 \times 10^{-5}$ m$^{-2}$ over the $1000 \times 1000$ m area. The set of eavesdropper positions is denoted $\{\mathbf{e}_k\}_{k=1}^{K}$, where $K \sim \text{Pois}(\lambda A)$ and $A = 10^6$ m$^2$. A single fixed HPPP realization is used per optimization run.

### 1.2 Coordinate System and Time Discretization

The operating area is $A_s \times A_s$ with $A_s = 1000$ m, centered at the origin:
$$
-\frac{A_s}{2} \le x,y \le \frac{A_s}{2}.
$$

The optimization horizon is divided into $M = 12$ equal time slots of duration $T_{\text{slot}} = 4$ s. Each slot $m$ is associated with a time-splitting factor $\alpha[m] \in [0.05, 0.95]$ that governs the fraction of the slot devoted to cooperative transmission.

Within each slot, communication proceeds in two phases following a half-duplex relaying protocol:
1. **Phase 1** (duration $\alpha[m] T_{\text{slot}}$): The source transmits to the relay.
2. **Phase 2** (duration $\alpha[m] T_{\text{slot}}$): The relay forwards to the destination, while the jammer transmits noise.
   
The total slot duration $T_{\text{slot}}$ is shared equally between the two phases, hence the factor $0.5$ appearing in rate expressions.

### 1.3 Trajectory Representation

The relay and jammer trajectories are discrete sequences of $M$ waypoints:
$$
\mathbf{q}_r = [\mathbf{q}_r[1], \dots, \mathbf{q}_r[M]]^\top \in \mathbb{R}^{M \times 2}, \qquad
\mathbf{q}_j = [\mathbf{q}_j[1], \dots, \mathbf{q}_j[M]]^\top \in \mathbb{R}^{M \times 2}.
$$

Both UAVs start from given initial positions $\mathbf{q}_r[1]$, $\mathbf{q}_j[1]$ (randomly sampled in the area during environment reset) and must end at specified terminal positions $\mathbf{q}_r[M]$, $\mathbf{q}_j[M]$. The initial solution is a linear interpolation between start and end.

---

## 2. Channel Model

### 2.1 Path Loss Model

Two path-loss formulations are used depending on the link type.

**Air-to-ground links** (source–relay, relay–destination, relay–eve, jammer–eve):

$$
G_{\text{ag}}(\mathbf{x}, \mathbf{y}; \beta_0, \alpha, H, \xi) = \beta_0 \, \xi \, \bigl(d_{\text{ag}}^2(\mathbf{x}, \mathbf{y}, H)\bigr)^{-\alpha/2},
$$

where $\xi$ is the small-scale fading power, and the slant distance squared is
$$
d_{\text{ag}}^2(\mathbf{x}, \mathbf{y}, H) = \|\mathbf{x} - \mathbf{y}\|_2^2 + H^2.
$$

**Ground-to-ground links** (source–eve):

$$
G_{\text{gg}}(\mathbf{x}, \mathbf{y}; \beta_0, \alpha, \xi, \delta) = \beta_0 \, \xi \, \bigl(\max\{\|\mathbf{x} - \mathbf{y}\|_2 - \delta,\; \varepsilon\}\bigr)^{-\alpha},
$$

where $\varepsilon = 10^{-3}$ is a numerical safeguard and $\delta$ is the uncertainty shrinkage parameter (Section 6).

The path-loss parameters are:
- $\beta_0 = 1.0$ (channel power gain at reference distance $1$ m)
- $\alpha = 2.0$ (path-loss exponent)

### 2.2 Fading Models

Two statistical fading models are implemented, selected via `config.channel_model`:

**Rayleigh fading** (line $10$ of `core/channel.py`):
$$
h = h_{\text{real}} + j h_{\text{imag}}, \quad h_{\text{real}}, h_{\text{imag}} \sim \mathcal{N}\!\left(0, \tfrac{1}{\sqrt{2}}\right),
\qquad \xi = |h|^2 \sim \text{Exp}(1).
$$

**Rician fading** with $K$-factor $K = 5$ (line $14$ of `core/channel.py`):
$$
s = \sqrt{\frac{K}{K+1}}, \quad \sigma = \sqrt{\frac{1}{2(K+1)}},
$$
$$
h_{\text{real}} = s + \mathcal{N}(0, \sigma), \quad h_{\text{imag}} \sim \mathcal{N}(0, \sigma),
\qquad \xi = |h|^2.
$$

The Rician $K$-factor parameterizes the ratio of the dominant LOS power to scattered power: $K = 5$ (linear).

### 2.3 Effective Channel Gains

The following channel gains are computed in the environment (`sca_environment.py` lines $119$–$121$, $123$–$134$, $136$–$145$):

**Source–Relay link** ($h_{\text{SR}}[m]$):
$$
h_{\text{SR}}[m] = \beta_0 \, \xi_{\text{SR}}[m] \, \bigl(\|\mathbf{q}_r[m] - \mathbf{u}[m]\|^2 + H_r^2\bigr)^{-\alpha/2}.
$$

**Relay–Destination link** ($h_{\text{RD}}[m]$):
$$
h_{\text{RD}}[m] = \beta_0 \, \xi_{\text{RD}}[m] \, \bigl(\|\mathbf{q}_r[m] - \mathbf{d}\|^2 + H_r^2\bigr)^{-\alpha/2}.
$$

**Source–$k$-th Eve link** ($g_{\text{SE},k}[m]$):
$$
g_{\text{SE},k}[m] = \beta_0 \, \xi_{\text{SE},k}[m] \, \bigl(\max\{\|\mathbf{u}[m] - \mathbf{e}_k\| - \delta_{\text{eve}},\; \varepsilon\}\bigr)^{-\alpha}.
$$

**Relay–$k$-th Eve link** ($g_{\text{RE},k}[m]$):
$$
g_{\text{RE},k}[m] = \beta_0 \, \xi_{\text{RE},k}[m] \, \bigl((\max\{\|\mathbf{q}_r[m] - \mathbf{e}_k\| - \delta_{\text{eve}},\; \varepsilon\})^2 + H_r^2\bigr)^{-\alpha/2}.
$$

**Jammer–$k$-th Eve link** ($g_{\text{JE},k}[m]$):
$$
g_{\text{JE},k}[m] = \beta_0 \, \xi_{\text{JE},k}[m] \, \bigl((\|\mathbf{q}_j[m] - \mathbf{e}_k\| + \delta_{\text{eve}})^2 + H_j^2\bigr)^{-\alpha/2}.
$$

Here $\xi_{\text{SR}}[m]$, $\xi_{\text{RD}}[m]$, $\xi_{\text{SE},k}[m]$, $\xi_{\text{RE},k}[m]$, $\xi_{\text{JE},k}[m]$ are independent fading samples drawn once during environment reset and fixed for the entire optimization run. The small-scale fading is sampled per time slot and per link.

---

## 3. Secrecy Rate Formulation

### 3.1 Legitimate Capacity

The instantaneous signal-to-noise ratio (SNR) on the source–relay and relay–destination links are:

$$
\gamma_{\text{SR}}[m] = \frac{P_s[m] \, h_{\text{SR}}[m]}{N_0}, \qquad
\gamma_{\text{RD}}[m] = \frac{P_r[m] \, h_{\text{RD}}[m]}{N_0},
$$

where $N_0 = B \cdot N_{\text{PSD}}$ is the noise power, $B = 10^6$ Hz is the bandwidth, and $N_{\text{PSD}} = 10^{-17.4}$ W/Hz ($-174$ dBm/Hz).

The achievable rates (in bps/Hz after normalization by bandwidth) are:

$$
R_{\text{SR}}^{\text{(raw)}}[m] = \log_2\bigl(1 + \gamma_{\text{SR}}[m]\bigr), \qquad
R_{\text{RD}}^{\text{(raw)}}[m] = \log_2\bigl(1 + \gamma_{\text{RD}}[m]\bigr).
$$

These raw rates are scaled by the slot factor:

$$
R_{\text{SR}}[m] = \frac{1}{2}\, \alpha[m] \, T_{\text{slot}} \cdot R_{\text{SR}}^{\text{(raw)}}[m], \qquad
R_{\text{RD}}[m] = \frac{1}{2}\, \alpha[m] \, T_{\text{slot}} \cdot R_{\text{RD}}^{\text{(raw)}}[m].
$$

The effective legitimate rate is the bottleneck of the two-hop DF relay channel:

$$
R_{\text{leg}}[m] = \min\bigl(R_{\text{SR}}[m],\; R_{\text{RD}}[m]\bigr).
$$

### 3.2 Eavesdropper Capacity

For each eavesdropper $k$, the combined received signal power (source + relay) and the jamming noise at the eavesdropper produce the SINR:

$$
\Gamma_{k}[m] = \frac{P_s[m] \, g_{\text{SE},k}[m] + P_r[m] \, g_{\text{RE},k}[m]}{N_0 + P_j[m] \, g_{\text{JE},k}[m]}.
$$

The wiretap rate is computed using a sum-SINR aggregation across all $K$ eavesdroppers. This represents the worst-case assumption that eavesdroppers can cooperate or that the secrecy capacity is limited by the most harmful combination:

$$
R_{\text{wiretap}}^{\text{(raw)}}[m] = \log_2\!\left(1 + \sum_{k=1}^{K} \Gamma_k[m]\right),
$$

$$
R_{\text{wiretap}}[m] = \frac{1}{2}\, \alpha[m] \, T_{\text{slot}} \cdot R_{\text{wiretap}}^{\text{(raw)}}[m].
$$

### 3.3 Instantaneous Secrecy Rate

The instantaneous secrecy rate per slot is:

$$
R_{\text{sec}}[m] = \max\!\bigl(R_{\text{leg}}[m] - R_{\text{wiretap}}[m],\; 0\bigr).
$$

**Important implementation note**: The raw (unclipped) secrecy rate used as the optimization objective is the *unclipped* difference:

$$
R_{\text{sec}}^{\text{(raw)}}[m] = R_{\text{leg}}[m] - R_{\text{wiretap}}[m].
$$

Negative values are allowed during optimization to permit gradient flow; clipping to zero is applied only for display and final metric reporting.

### 3.4 Average Secrecy Rate

The average secrecy rate over the horizon is the objective function:

$$
\bar{R}_{\text{sec}} = \frac{1}{M} \sum_{m=1}^{M} R_{\text{sec}}^{\text{(raw)}}[m].
$$

---

## 4. Optimization Variables

### 4.1 Variable Definitions

| Variable | Symbol | Description | Dimension |
|---|---|---|---|
| Source transmit power | $P_s[m]$ | Transmit power of the source in slot $m$ | $\mathbb{R}^M$ |
| Relay transmit power | $P_r[m]$ | Transmit power of the relay in slot $m$ | $\mathbb{R}^M$ |
| Jammer transmit power | $P_j[m]$ | Artificial noise power of the jammer in slot $m$ | $\mathbb{R}^M$ |
| Relay horizontal trajectory | $\mathbf{q}_r[m]$ | $xy$-position of the relay in slot $m$ | $\mathbb{R}^{M \times 2}$ |
| Jammer horizontal trajectory | $\mathbf{q}_j[m]$ | $xy$-position of the jammer in slot $m$ | $\mathbb{R}^{M \times 2}$ |
| Time-splitting factor | $\alpha[m]$ | Fraction of half-slot devoted to active transmission | $\mathbb{R}^M$ |

For convenience, we define the aggregate power vector:
$$
\mathbf{p} = \bigl[P_s[1],\dots,P_s[M],\; P_r[1],\dots,P_r[M],\; P_j[1],\dots,P_j[M]\bigr]^\top \in \mathbb{R}^{3M}.
$$

### 4.2 Variable Bounds

$$
\begin{aligned}
0.001 \le P_s[m] \le 0.2 \quad \text{(W)}, &\qquad \frac{1}{M}\sum_{m=1}^{M} P_s[m] \le 0.15, \\[4pt]
0.001 \le P_r[m] \le 0.5 \quad \text{(W)}, &\qquad \frac{1}{M}\sum_{m=1}^{M} P_r[m] \le 0.35, \\[4pt]
0.0 \le P_j[m] \le 0.5 \quad \text{(W)}, &\qquad \frac{1}{M}\sum_{m=1}^{M} P_j[m] \le 0.25, \\[4pt]
0.05 \le \alpha[m] \le 0.95,
\end{aligned}
$$

### 4.3 Trajectory Constraints

**Area bounds** ($A_s = 1000$ m):
$$
-\frac{A_s}{2} \le q_{r,x}[m], q_{r,y}[m], q_{j,x}[m], q_{j,y}[m] \le \frac{A_s}{2}.
$$

**Flight radius** (maximum distance from origin):
$$
\|\mathbf{q}_r[m]\|_2 \le R_{\max} = 350,\qquad
\|\mathbf{q}_j[m]\|_2 \le R_{\max} = 350.
$$

**Start and end anchoring**:
$$
\mathbf{q}_r[1] = \mathbf{q}_{r,\text{start}},\quad
\mathbf{q}_r[M] = \mathbf{q}_{r,\text{end}},\qquad
\mathbf{q}_j[1] = \mathbf{q}_{j,\text{start}},\quad
\mathbf{q}_j[M] = \mathbf{q}_{j,\text{end}}.
$$

**Mobility constraint** (maximum per-slot displacement):
$$
\|\mathbf{q}_r[m+1] - \mathbf{q}_r[m]\|_2 \le v_{\max} T_{\text{slot}} = 80\ \text{m},
$$
$$
\|\mathbf{q}_j[m+1] - \mathbf{q}_j[m]\|_2 \le v_{\max} T_{\text{slot}} = 80\ \text{m},
$$
where $v_{\max} = 20$ m/s.

**Collision avoidance** (separation between UAVs $\ge D_{\text{coll}} = 30$ m):

The constraint is linearized around the current relay position while keeping the jammer position fixed (and vice versa). For relay optimization:
$$
\|\mathbf{q}_r[m] - \mathbf{q}_j[m]\|_2^2 + 2\, \bigl(\mathbf{q}_{r,\text{current}}[m] - \mathbf{q}_j[m]\bigr)^\top \bigl(\mathbf{q}_r[m] - \mathbf{q}_{r,\text{current}}[m]\bigr) \ge D_{\text{coll}}^2,
$$
which is a first-order Taylor expansion of the squared distance around the current iterate. A symmetric constraint is applied during jammer optimization.

---

## 5. Optimization Objective

The complete optimization problem is:

$$
\boxed{
\begin{aligned}
& \underset{\mathbf{p},\,\mathbf{q}_r,\,\mathbf{q}_j,\,\boldsymbol{\alpha}}{\text{maximize}}
&& \frac{1}{M}\sum_{m=1}^{M} \Bigl(R_{\text{leg}}[m] - R_{\text{wiretap}}[m]\Bigr) \\[6pt]
& \text{subject to}
&& 0.001 \le P_s[m] \le 0.2,\quad \frac{1}{M}\sum_{m=1}^{M} P_s[m] \le 0.15, \\[4pt]
&&& 0.001 \le P_r[m] \le 0.5,\quad \frac{1}{M}\sum_{m=1}^{M} P_r[m] \le 0.35, \\[4pt]
&&& 0.0 \le P_j[m] \le 0.5,\quad \frac{1}{M}\sum_{m=1}^{M} P_j[m] \le 0.25, \\[4pt]
&&& -\frac{A_s}{2} \le \mathbf{q}_r[m], \mathbf{q}_j[m] \le \frac{A_s}{2}, \\[4pt]
&&& \|\mathbf{q}_r[m]\|_2 \le R_{\max},\quad \|\mathbf{q}_j[m]\|_2 \le R_{\max}, \\[4pt]
&&& \mathbf{q}_r[1] = \mathbf{q}_{r,\text{start}},\quad \mathbf{q}_r[M] = \mathbf{q}_{r,\text{end}}, \\[4pt]
&&& \mathbf{q}_j[1] = \mathbf{q}_{j,\text{start}},\quad \mathbf{q}_j[M] = \mathbf{q}_{j,\text{end}}, \\[4pt]
&&& \|\mathbf{q}_r[m+1] - \mathbf{q}_r[m]\|_2 \le v_{\max} T_{\text{slot}}, \\[4pt]
&&& \|\mathbf{q}_j[m+1] - \mathbf{q}_j[m]\|_2 \le v_{\max} T_{\text{slot}}, \\[4pt]
&&& \|\mathbf{q}_r[m] - \mathbf{q}_j[m]\|_2 \ge D_{\text{coll}}, \\[4pt]
&&& 0.05 \le \alpha[m] \le 0.95, \\[4pt]
&&& m = 1,\dots,M.
\end{aligned}
}
$$

The problem is non-convex due to the fractional terms in the wiretap rate, the coupling between trajectory and power variables in the channel gains, and the min-bottleneck structure of the legitimate rate.

---

## 6. Robust Eavesdropper Model

### 6.1 Eve Uncertainty Radius

A circular uncertainty region of radius $\delta = 30$ m is assumed around each eavesdropper position. This models the practical scenario where the exact eavesdropper location is not perfectly known but lies within a disk of radius $\delta$.

### 6.2 Distance Inflation and Shrinkage

To ensure a worst-case secrecy formulation, each link gain to/from an eavesdropper is adjusted:

- **Source–Eve links**: The ground distance is *shrunk* to maximize the eavesdropper's received signal power:
  $$
  d_{\text{SE},k}[m] = \max\{\|\mathbf{u}[m] - \mathbf{e}_k\| - \delta,\; 10^{-3}\}.
  $$

- **Relay–Eve links**: The air-to-ground distance is shrunk to maximize the signal reaching the eavesdropper:
  $$
  d_{\text{RE},k}[m] = \max\{\|\mathbf{q}_r[m] - \mathbf{e}_k\| - \delta,\; 10^{-3}\},
  $$
  with the slant distance $d_{\text{RE},\text{slant},k}^2[m] = d_{\text{RE},k}^2[m] + H_r^2$.

- **Jammer–Eve links**: The ground distance is *inflated* to minimize the effectiveness of jamming:
  $$
  d_{\text{JE},k}[m] = \|\mathbf{q}_j[m] - \mathbf{e}_k\| + \delta,
  $$
  with the slant distance $d_{\text{JE},\text{slant},k}^2[m] = d_{\text{JE},k}^2[m] + H_j^2$.

### 6.3 Worst-Case Rationale

The shrinkage of source/eavesdropper and relay/eavesdropper distances increases their effective channel gains, making the wiretap capacity larger (pessimistic from the legitimate perspective). Simultaneously, the inflation of jammer/eavesdropper distances reduces the jamming effectiveness, again making the wiretap capacity larger. This combination guarantees that the computed secrecy rate is a lower bound on the true secrecy rate under any eavesdropper position within the uncertainty disks.

---

## 7. Successive Convex Approximation (SCA)

### 7.1 Sources of Non-Convexity

The objective is non-convex in the optimization variables due to:
1. Logarithm of sum of fractional SINR terms in the wiretap rate.
2. The min-bottleneck structure coupling $R_{\text{SR}}$ and $R_{\text{RD}}$.
3. Channel gains that depend inversely on distances raised to $\alpha$.
4. Coupling between power and trajectory variables in the eavesdropper SINR.

### 7.2 SCA Surrogate Function

SCA solves each block by constructing a concave surrogate that locally approximates the non-concave objective. At iteration $\ell$ with current iterate $\mathbf{x}^{(\ell)}$, the surrogate is:

$$
\tilde{f}(\mathbf{x}; \mathbf{x}^{(\ell)}) = \nabla f(\mathbf{x}^{(\ell)})^\top (\mathbf{x} - \mathbf{x}^{(\ell)}) - \frac{\rho}{2} \|\mathbf{x} - \mathbf{x}^{(\ell)}\|_2^2,
$$

where:
- $f$ is the objective function evaluated on a cloned solution with only the current block's variables changed.
- $\nabla f(\mathbf{x}^{(\ell)})$ is the exact gradient computed via the chain rule through the secrecy rate expression (environment methods `power_gradient`, `relay_gradient`, `jammer_gradient`, `alpha_gradient`).
- $\rho =$ `config.trust_region_weight` $= 1.0$ is the quadratic penalty parameter.
- The first term is a first-order Taylor expansion that is linear in $\mathbf{x}$.
- The quadratic penalty term $\frac{\rho}{2} \|\mathbf{x} - \mathbf{x}^{(\ell)}\|_2^2$ penalizes large deviations from the current iterate.

The surrogate is maximized subject to block-specific constraints:

$$
\mathbf{x}^{(\ell+1/2)} = \underset{\mathbf{x} \in \mathcal{X}}{\arg\max}\; \tilde{f}(\mathbf{x}; \mathbf{x}^{(\ell)}),
$$

where $\mathcal{X}$ denotes the feasible set for the current block.

### 7.3 Trust-Region Constraint

Each block enforces an explicit trust-region constraint via the constraint builder:

$$
\|\mathbf{x} - \mathbf{x}^{(\ell)}\|_2 \le \Delta_{\text{block}},
$$

where:
- $\Delta_{\text{power}} = 0.35$ (power block)
- $\Delta_{\text{trajectory}} = 180.0$ m (relay and jammer trajectory blocks)
- $\Delta_{\alpha} = 0.5$ (alpha block)

This constraint ensures that the linearization remains a good approximation and prevents aggressive updates.

### 7.4 Step-Size Backtracking

After solving the quadratic surrogate, a backtracking line search over candidate step sizes $\eta \in \{0.25, 0.20, 0.15, 0.10, 0.05, 0.02\}$ is performed:

$$
\mathbf{x}^{(\ell+1)} = \begin{cases}
\mathbf{x}^{(\ell)} + \eta (\mathbf{x}^{(\ell+1/2)} - \mathbf{x}^{(\ell)}), & \text{if } f(\mathbf{x}^{(\ell)} + \eta \Delta \mathbf{x}) \ge f(\mathbf{x}^{(\ell)}) - 10^{-9}, \\[4pt]
\mathbf{x}^{(\ell)}, & \text{otherwise (no update)}.
\end{cases}
$$

The largest $\eta$ satisfying the acceptance condition is selected. If no step size improves the objective (within numerical tolerance), the SCA subproblem is terminated.

### 7.5 SCA Convergence

SCA within each block terminates when:
$$
\|\mathbf{x}^{(\ell+1)} - \mathbf{x}^{(\ell)}\|_2 \le \varepsilon_{\text{SCA}} \quad \text{and} \quad |f(\mathbf{x}^{(\ell+1)}) - f(\mathbf{x}^{(\ell)})| \le \varepsilon_{\text{SCA}},
$$
with $\varepsilon_{\text{SCA}} = 10^{-4}$, or when $\max\_iters = 8$ SCA iterations are reached.

---

## 8. Block Coordinate Descent (BCD)

### 8.1 Four Optimization Blocks

The BCD solver decomposes the joint optimization into four blocks executed sequentially in each outer iteration:

| Block | Variable | Type | Dimension | Trust-Region Radius $\Delta$ |
|---|---|---|---|---|
| **1. Power** | $\mathbf{p} = \{P_s, P_r, P_j\}$ | Continuous | $3M$ | $0.35$ |
| **2. Relay Trajectory** | $\mathbf{q}_r$ | Continuous | $2M$ | $180.0$ m |
| **3. Jammer Trajectory** | $\mathbf{q}_j$ | Continuous | $2M$ | $180.0$ m |
| **4. Alpha** | $\boldsymbol{\alpha}$ | Continuous | $M$ | $0.5$ |

### 8.2 Block 1: Power Optimization

Maximizes the surrogate over power variables with:
- Box constraints ($P_{\min}, P_{\max}$ per node)
- Average power budget constraints ($\frac{1}{M}\sum P_s \le \bar{P}_s$, etc.)
- Trust-region constraint $\|\mathbf{p} - \mathbf{p}^{(\ell)}\|_2 \le \Delta_{\text{power}}$

A projection operator `_project_average_bounded` enforces the average budget post-iteration by scaling if the mean exceeds the budget.

The gradient $\nabla_{\mathbf{p}} f$ is computed in `power_gradient` (lines $263$–$285$ of `sca_environment.py`):
- For the legitimate part, it uses the chain rule through $\log_2(1+\gamma)$ and the min selection.
- For the wiretap part, it differentiates $\log_2(1+\sum_k \Gamma_k)$.

### 8.3 Block 2: Relay Trajectory Optimization

Maximizes the surrogate over $\mathbf{q}_r$ with:
- Area bounds, radius bounds, start/end anchoring
- Mobility constraint (linear: $\|\mathbf{q}_r[m+1] - \mathbf{q}_r[m]\| \le v_{\max} T_{\text{slot}}$)
- Linearized collision avoidance with the current jammer trajectory
- Trust-region constraint $\|\mathbf{q}_r - \mathbf{q}_r^{(\ell)}\|_2 \le \Delta_{\text{trajectory}}$

The gradient $\nabla_{\mathbf{q}_r} f$ is computed in `relay_gradient` (lines $287$–$302$). The air-to-ground distance gradients are computed via:
$$
\frac{\partial h}{\partial \mathbf{q}_r[m]} = \beta_0 \xi (-\alpha/2) (d^2)^{-\alpha/2 - 1} \cdot 2(\mathbf{q}_r[m] - \text{point}),
$$
implemented in `_gain_grad_from_sq`.

### 8.4 Block 3: Jammer Trajectory Optimization

Symmetrical to Block 2, optimizing over $\mathbf{q}_j$ with the same constraint types. The jammer gradient accounts for the effect of jammer position on all eavesdropper links via the chain rule through the inflation mechanism and the denominator $N_0 + P_j g_{\text{JE},k}$.

### 8.5 Block 4: Alpha Optimization

Maximizes the surrogate over $\boldsymbol{\alpha}$ with:
- Box constraints $\alpha_{\min}=0.05$, $\alpha_{\max}=0.95$
- Trust-region constraint $\|\boldsymbol{\alpha} - \boldsymbol{\alpha}^{(\ell)}\|_2 \le \Delta_\alpha = 0.5$

The alpha gradient has a closed form (line $314$–$321$ of `sca_environment.py`):
$$
\frac{\partial \bar{R}_{\text{sec}}}{\partial \alpha[m]} = \frac{1}{M} \cdot \frac{1}{2} T_{\text{slot}} \cdot \bigl(R_{\text{legit\_factor}}[m] - R_{\text{wiretap}}^{\text{(raw)}}[m]\bigr),
$$
where
$$
R_{\text{legit\_factor}}[m] = \begin{cases}
R_{\text{SR}}^{\text{(raw)}}[m], & \text{if } R_{\text{SR}}[m] \le R_{\text{RD}}[m], \\
R_{\text{RD}}^{\text{(raw)}}[m], & \text{otherwise}.
\end{cases}
$$

### 8.6 Update Order and Rationale

The execution order is: **Power → Relay → Jammer → Alpha**.

**Rationale**: Power variables directly affect both legitimate and wiretap rates and provide the strongest immediate improvement. Trajectories are optimized while keeping the (optimized) powers fixed, which prevents conflicting updates. Alpha is placed last because its effect is multiplicative across all rate terms; optimizing it after the other blocks are tuned to specific power/trajectory values yields the best final adjustment. This ordering was validated empirically: the alpha block contributes the largest objective improvement during BCD iterations.

---

## 9. Convergence Criteria

### 9.1 BCD Outer Loop

The BCD outer loop converges when **all three** conditions are met:

1. **Minimum iterations satisfied**:
   $$
   \text{iteration} \ge M_{\text{min}} = 20.
   $$

2. **Sufficient patience** (consecutive iterations without improvement):
   $$
   \text{patience} \ge P_{\text{max}} = 8,
   $$
   where patience increments when either condition below holds and resets to zero otherwise.

3. **Stagnation condition** (either absolute or relative):
   $$
   |\bar{R}_{\text{sec}}^{(t)} - \bar{R}_{\text{sec}}^{(t-1)}| \le \varepsilon_{\text{abs}} = 10^{-3}
   \quad\text{or}\quad
   \frac{|\bar{R}_{\text{sec}}^{(t)} - \bar{R}_{\text{sec}}^{(t-1)}|}{\max(|\bar{R}_{\text{sec}}^{(t-1)}|,\, 10^{-12})} \le \varepsilon_{\text{rel}} = 5 \times 10^{-4}.
   $$

The maximum number of BCD iterations is $M_{\text{max}} = 100$.

### 9.2 SCA Inner Loop

Each SCA subproblem converges (or terminates) when:

1. **Step norm and improvement below tolerance**:
   $$
   \|\mathbf{x}^{(\ell+1)} - \mathbf{x}^{(\ell)}\|_2 \le \varepsilon_{\text{SCA}} = 10^{-4}
   \quad\text{and}\quad
   |f^{(\ell+1)} - f^{(\ell)}| \le \varepsilon_{\text{SCA}}.
   $$

2. **Maximum SCA iterations reached**: $\ell_{\max} = 8$.

3. **No step size improves objective**: The backtracking line search yields no acceptable $\eta > 0$.

---

## 10. Algorithm

---

**Algorithm 1** Proposed SCA-BCD Method for Dual-UAV Secrecy Rate Maximization

---

**Input:** Channel fading realizations $\{\xi\}$, HPPP eavesdropper positions $\{\mathbf{e}_k\}$, configuration parameters (horizon $M$, bounds, tolerances)

**Output:** Optimized variables $\mathbf{p}^\ast$, $\mathbf{q}_r^\ast$, $\mathbf{q}_j^\ast$, $\boldsymbol{\alpha}^\ast$

1. **Initialize:** $\mathbf{p}^{(0)}$, $\mathbf{q}_r^{(0)}$, $\mathbf{q}_j^{(0)}$, $\boldsymbol{\alpha}^{(0)}$ with linear trajectories and average power budgets
2. Evaluate initial objective $f^{(0)} \gets \bar{R}_{\text{sec}}(\mathbf{p}^{(0)}, \mathbf{q}_r^{(0)}, \mathbf{q}_j^{(0)}, \boldsymbol{\alpha}^{(0)})$
3. $\text{patience} \gets 0$
4. **for** $t = 1$ **to** $M_{\text{max}}$ **do**
5. &nbsp;&nbsp; $\mathbf{p}_{\text{prev}} \gets \mathbf{p}^{(t-1)}$, $\mathbf{q}_{r,\text{prev}} \gets \mathbf{q}_r^{(t-1)}$, $\mathbf{q}_{j,\text{prev}} \gets \mathbf{q}_j^{(t-1)}$, $\boldsymbol{\alpha}_{\text{prev}} \gets \boldsymbol{\alpha}^{(t-1)}$
6. &nbsp;&nbsp; // ---- Block 1: Power ----
7. &nbsp;&nbsp; $\mathbf{p}^{(t)} \gets \text{SCA-Power}(\mathbf{p}^{(t-1)}, \mathbf{q}_r^{(t-1)}, \mathbf{q}_j^{(t-1)}, \boldsymbol{\alpha}^{(t-1)})$
8. &nbsp;&nbsp; // ---- Block 2: Relay Trajectory ----
9. &nbsp;&nbsp; $\mathbf{q}_r^{(t)} \gets \text{SCA-Relay}(\mathbf{p}^{(t)}, \mathbf{q}_r^{(t-1)}, \mathbf{q}_j^{(t-1)}, \boldsymbol{\alpha}^{(t-1)})$
10. &nbsp;&nbsp; // ---- Block 3: Jammer Trajectory ----
11. &nbsp;&nbsp; $\mathbf{q}_j^{(t)} \gets \text{SCA-Jammer}(\mathbf{p}^{(t)}, \mathbf{q}_r^{(t)}, \mathbf{q}_j^{(t-1)}, \boldsymbol{\alpha}^{(t-1)})$
12. &nbsp;&nbsp; // ---- Block 4: Alpha ----
13. &nbsp;&nbsp; $\boldsymbol{\alpha}^{(t)} \gets \text{SCA-Alpha}(\mathbf{p}^{(t)}, \mathbf{q}_r^{(t)}, \mathbf{q}_j^{(t)}, \boldsymbol{\alpha}^{(t-1)})$
14. &nbsp;&nbsp; Evaluate $f^{(t)} \gets \bar{R}_{\text{sec}}(\mathbf{p}^{(t)}, \mathbf{q}_r^{(t)}, \mathbf{q}_j^{(t)}, \boldsymbol{\alpha}^{(t)})$
15. &nbsp;&nbsp; $\Delta f \gets f^{(t)} - f^{(t-1)}$
16. &nbsp;&nbsp; **if** $|\Delta f| \le \varepsilon_{\text{abs}}$ **or** $\frac{|\Delta f|}{\max(|f^{(t-1)}|, 10^{-12})} \le \varepsilon_{\text{rel}}$ **then**
17. &nbsp;&nbsp;&nbsp;&nbsp; $\text{patience} \gets \text{patience} + 1$
18. &nbsp;&nbsp; **else**
19. &nbsp;&nbsp;&nbsp;&nbsp; $\text{patience} \gets 0$
20. &nbsp;&nbsp; **end if**
21. &nbsp;&nbsp; **if** $t \ge M_{\text{min}}$ **and** $\text{patience} \ge P_{\text{max}}$ **then**
22. &nbsp;&nbsp;&nbsp;&nbsp; **break**
23. &nbsp;&nbsp; **end if**
24. **end for**
25. **return** $\mathbf{p}^\ast = \mathbf{p}^{(t)}$, $\mathbf{q}_r^\ast = \mathbf{q}_r^{(t)}$, $\mathbf{q}_j^\ast = \mathbf{q}_j^{(t)}$, $\boldsymbol{\alpha}^\ast = \boldsymbol{\alpha}^{(t)}$

---

**Procedure SCA-Block($\mathbf{x}^{(0)}$, objective $f$, gradient $\nabla f$, constraints $\mathcal{X}$, trust-radius $\Delta$, weight $\rho$)**

1. $\mathbf{x} \gets \mathbf{x}^{(0)}$, $f_0 \gets f(\mathbf{x})$
2. **for** $\ell = 1$ **to** $L_{\text{max}}$ **do**
3. &nbsp;&nbsp; $\mathbf{g} \gets \nabla f(\mathbf{x})$
4. &nbsp;&nbsp; Solve: $\tilde{\mathbf{x}} = \arg\max_{\mathbf{z} \in \mathcal{X},\; \|\mathbf{z} - \mathbf{x}\|_2 \le \Delta} \mathbf{g}^\top(\mathbf{z} - \mathbf{x}) - \frac{\rho}{2}\|\mathbf{z} - \mathbf{x}\|_2^2$
5. &nbsp;&nbsp; **for** $\eta \in [0.25, 0.20, 0.15, 0.10, 0.05, 0.02]$ **do**
6. &nbsp;&nbsp;&nbsp;&nbsp; $\mathbf{x}_{\text{trial}} \gets \mathbf{x} + \eta(\tilde{\mathbf{x}} - \mathbf{x})$
7. &nbsp;&nbsp;&nbsp;&nbsp; **if** $f(\mathbf{x}_{\text{trial}}) \ge f(\mathbf{x}) - 10^{-9}$ **then**
8. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\mathbf{x} \gets \mathbf{x}_{\text{trial}}$, accept $\eta$
9. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **break**
10. &nbsp;&nbsp;&nbsp;&nbsp; **end if**
11. &nbsp;&nbsp; **end for**
12. &nbsp;&nbsp; **if** no step accepted **then** **break**
13. &nbsp;&nbsp; **if** $\|\mathbf{x} - \mathbf{x}_{\text{prev}}\|_2 \le \varepsilon_{\text{SCA}}$ **and** $|f(\mathbf{x}) - f(\mathbf{x}_{\text{prev}})| \le \varepsilon_{\text{SCA}}$ **then** **break**
14. **end for**
15. **return** $\mathbf{x}$

---

## 11. Computational Complexity

### 11.1 Power Block

- **Variables**: $3M$
- **Gradient computation**: $O(MK)$, where $K$ is the number of eavesdroppers. Each slot requires iterating over all $K$ eavesdroppers to compute the wiretap gradient terms.
- **Surrogate optimization**: A quadratic program (QP) with $3M$ variables, box constraints, $3$ average-budget linear constraints, and one second-order cone (SOC) trust-region constraint. Solved by ECOS/SCS: $O(M^3)$ in the worst case.
- **Dominant term**: $O(M^3 + MK)$.

### 11.2 Relay Trajectory Block

- **Variables**: $2M$
- **Gradient computation**: $O(M K)$. Each slot: compute air-to-ground gradient terms for SR and RD links (two per slot) and wiretap gradient through each eve link with shrinkage.
- **Surrogate optimization**: QP with $2M$ variables, $2M$ box constraints, $2$ equality constraints (start/end), $(M-1)$ SOC mobility constraints, $M$ SOC radius constraints, $M$ linearized collision constraints, and one SOC trust-region constraint: $O(M^3)$.
- **Dominant term**: $O(M^3 + M K)$.

### 11.3 Jammer Trajectory Block

- Identical structure to the relay block: $O(M^3 + M K)$.
- Gradient computation includes the jammer gradient through the denominator of each eve's SINR, which requires differentiating $1/(N_0 + P_j g_{\text{JE},k})^2$.

### 11.4 Alpha Block

- **Variables**: $M$
- **Gradient computation**: $O(M)$. Closed-form gradient requires no iteration over eves; it uses the precomputed raw rates.
- **Surrogate optimization**: QP with $M$ variables, box constraints, and one SOC trust-region constraint: $O(M^3)$.
- **Dominant term**: $O(M^3)$.

### 11.5 Total BCD Complexity

Each BCD outer iteration:
$$
O\bigl(M^3 + M K\bigr) \quad \text{per block, summed across 4 blocks:} \quad O(4M^3 + 3M K).
$$

With $M = 12$, $K \approx 20$ (mean of Pois($20$)), and $T_{\text{BCD}} \approx 20$ outer iterations (typical), the total complexity is:
$$
O\bigl(T_{\text{BCD}} (4M^3 + 3M K)\bigr) = O\bigl(20 \times (4 \times 1728 + 3 \times 12 \times 20)\bigr) \approx O(1.5 \times 10^5).
$$

This is modest in absolute terms, consistent with the fast convergence observed in practice.

---

## 12. Mapping Between Code and Mathematics

| Mathematical Quantity | Code Variable | File |
|---|---|---|
| $M$ (horizon) | `config.horizon` | `configs.py:11` |
| $T_{\text{slot}}$ (slot duration) | `config.slot_duration` | `configs.py:26` |
| $A_s$ (area size) | `config.area_size` | `configs.py:22` |
| $R_{\max}$ (flight radius) | `config.max_flight_radius` | `configs.py:23` |
| $v_{\max}$ (max speed) | `config.max_speed` | `configs.py:27` |
| $D_{\text{coll}}$ (collision distance) | `config.collision_distance` | `configs.py:28` |
| $B$ (bandwidth) | `config.bandwidth` | `configs.py:29` |
| $N_{\text{PSD}}$ (noise PSD) | `config.noise_psd` | `configs.py:30` |
| $N_0$ (noise power) | `config.noise_power` (property) | `configs.py:70` |
| $\beta_0$ (reference gain) | `config.beta0` | `configs.py:31` |
| $\alpha$ (path-loss exponent) | `config.alpha` | `configs.py:32` |
| $K$ (Rician factor) | `config.rician_k` | `configs.py:38` |
| $P_{s,\min}$ | `config.user_power_min` | `configs.py:39` |
| $P_{s,\max}$ | `config.user_power_max` | `configs.py:40` |
| $P_{r,\min}$ | `config.relay_power_min` | `configs.py:41` |
| $P_{r,\max}$ | `config.relay_power_max` | `configs.py:42` |
| $P_{j,\min}$ | `config.jammer_power_min` | `configs.py:43` |
| $P_{j,\max}$ | `config.jammer_power_max` | `configs.py:44` |
| $\bar{P}_s$ (avg budget) | `config.avg_user_power_budget` | `configs.py:45` |
| $\bar{P}_r$ (avg budget) | `config.avg_relay_power_budget` | `configs.py:46` |
| $\bar{P}_j$ (avg budget) | `config.avg_jammer_power_budget` | `configs.py:47` |
| $\alpha_{\min}$ | `config.alpha_min` | `configs.py:49` |
| $\alpha_{\max}$ | `config.alpha_max` | `configs.py:50` |
| $\delta$ (eve uncertainty) | `config.eve_uncertainty_radius` | `configs.py:52` |
| $\lambda$ (eve density) | `config.eve_density_lambda` | `configs.py:57` |
| $\rho$ (trust-region weight) | `config.trust_region_weight` | `configs.py:19` |
| $\Delta_{\text{power}}$ | `config.power_trust_region_radius` | `configs.py:20` |
| $\Delta_{\text{traj}}$ | `config.trajectory_trust_region_radius` | `configs.py:21` |
| $\Delta_{\alpha}$ | `config.alpha_trust_region_radius` | `configs.py:51` |
| $\varepsilon_{\text{abs}}$ (BCD) | `config.bcd_abs_tolerance` | `configs.py:16` |
| $\varepsilon_{\text{rel}}$ (BCD) | `config.bcd_rel_tolerance` | `configs.py:17` |
| $\varepsilon_{\text{SCA}}$ | `config.sca_tolerance` | `configs.py:18` |
| $M_{\text{max}}$ (BCD) | `config.max_bcd_iters` | `configs.py:12` |
| $M_{\text{min}}$ (BCD) | `config.min_bcd_iters` | `configs.py:13` |
| $P_{\text{max}}$ (patience) | `config.bcd_patience` | `configs.py:14` |
| $L_{\text{max}}$ (SCA) | `config.max_sca_iters` | `configs.py:15` |
| $P_s[m]$ | `solution.source_power[m]` | `secrecy_optimizer.py:12` |
| $P_r[m]$ | `solution.relay_power[m]` | `secrecy_optimizer.py:13` |
| $P_j[m]$ | `solution.jammer_power[m]` | `secrecy_optimizer.py:14` |
| $\mathbf{q}_r[m]$ | `solution.relay_trajectory[m]` | `secrecy_optimizer.py:10` |
| $\mathbf{q}_j[m]$ | `solution.jammer_trajectory[m]` | `secrecy_optimizer.py:11` |
| $\alpha[m]$ | `solution.alpha_trajectory[m]` | `secrecy_optimizer.py:15` |
| $\mathbf{u}[m]$ | `self.user_positions[m]` | `sca_environment.py:59` |
| $\mathbf{d}$ | `self.destination_position` | `sca_environment.py:20` |
| $\{\mathbf{e}_k\}$ | `self.eve_positions` | `sca_environment.py:58` |
| $\xi_{\text{SR}}[m]$ | `self.fading["SR"][m]` | `sca_environment.py:93` |
| $\xi_{\text{RD}}[m]$ | `self.fading["RD"][m]` | `sca_environment.py:94` |
| $\xi_{\text{SE},k}[m]$ | `self.fading["SE"][m, idx]` | `sca_environment.py:95` |
| $\xi_{\text{RE},k}[m]$ | `self.fading["RE"][m, idx]` | `sca_environment.py:96` |
| $\xi_{\text{JE},k}[m]$ | `self.fading["JE"][m, idx]` | `sca_environment.py:97` |
| $h_{\text{SR}}[m]$ | `terms["h_sr"]` | `sca_environment.py:159` |
| $h_{\text{RD}}[m]$ | `terms["h_rd"]` | `sca_environment.py:160` |
| $g_{\text{SE},k}[m]$ | `eve_terms["g_se"]` | `sca_environment.py:185` |
| $g_{\text{RE},k}[m]$ | `eve_terms["g_re"]` | `sca_environment.py:186` |
| $g_{\text{JE},k}[m]$ | `eve_terms["g_je"]` | `sca_environment.py:188` |
| $\gamma_{\text{SR}}[m]$ | `terms["gamma_sr"]` | `sca_environment.py:163` |
| $\gamma_{\text{RD}}[m]$ | `terms["gamma_rd"]` | `sca_environment.py:164` |
| $R_{\text{SR}}^{\text{(raw)}}[m]$ | `terms["r_sr_raw"]` | `sca_environment.py:165` |
| $R_{\text{RD}}^{\text{(raw)}}[m]$ | `terms["r_rd_raw"]` | `sca_environment.py:166` |
| $R_{\text{SR}}[m]$ | `terms["r_sr"]` | `sca_environment.py:167` |
| $R_{\text{RD}}[m]$ | `terms["r_rd"]` | `sca_environment.py:168` |
| $R_{\text{leg}}[m]$ | `terms["r_leg"]` | `sca_environment.py:169` |
| $\sum_k \Gamma_k[m]$ | `terms["eve_sum"]` | `sca_environment.py:182` |
| $R_{\text{wiretap}}^{\text{(raw)}}[m]$ | `terms["r_wir_raw"]` | `sca_environment.py:195` |
| $R_{\text{wiretap}}[m]$ | `terms["r_wir"]` | `sca_environment.py:196` |
| $\frac{1}{2}\alpha[m] T_{\text{slot}}$ | `terms["slot_factor"]` | `sca_environment.py:162` |
| $\bar{R}_{\text{sec}}$ (objective) | `metrics["objective"]` / `metrics["raw_objective"]` | `sca_environment.py:238-239` |
| $\bar{R}_{\text{sec}}$ (clipped) | `metrics["average_secrecy_rate"]` | `sca_environment.py:244` |
| $\nabla_{\mathbf{p}} \bar{R}_{\text{sec}}$ | `env.power_gradient(solution)` | `sca_environment.py:263` |
| $\nabla_{\mathbf{q}_r} \bar{R}_{\text{sec}}$ | `env.relay_gradient(solution)` | `sca_environment.py:287` |
| $\nabla_{\mathbf{q}_j} \bar{R}_{\text{sec}}$ | `env.jammer_gradient(solution)` | `sca_environment.py:304` |
| $\nabla_{\boldsymbol{\alpha}} \bar{R}_{\text{sec}}$ | `env.alpha_gradient(solution)` | `sca_environment.py:314` |
| SCA surrogate | `grad @ (x - xk) - 0.5 * trust_region_weight * cp.sum_squares(x - xk)` | `sca_solver.py:46` |
| Trust region constraint | `cp.norm(var - current_x, 2) <= radius` | `power_optimizer.py:73` |
| Power projector | `_project_average_bounded` | `power_optimizer.py:16` |
| Trajectory projector | Clip to area + radius normalization + fix endpoints | `trajectory_optimizer.py:73-81` |
| Collision avoidance | Linearized: `lhs >= collision_distance^2` | `trajectory_optimizer.py:38-39` |
| $\eta$ candidate set | `config.candidate_step_sizes` | `configs.py:62` |
| $\mathbf{q}_{r,\text{start}}$ | `env.relay_start` | `sca_environment.py:60` |
| $\mathbf{q}_{r,\text{end}}$ | `env.relay_end` | `sca_environment.py:62` |
| $\mathbf{q}_{j,\text{start}}$ | `env.jammer_start` | `sca_environment.py:61` |
| $\mathbf{q}_{j,\text{end}}$ | `env.jammer_end` | `sca_environment.py:63` |

---

## 13. Convergence Discussion

### 13.1 Observed Convergence Behavior

Convergence study results from the implemented solver (both Rician and Rayleigh channels):

| Metric | Rician | Rayleigh |
|---|---|---|
| Converged at BCD iteration | $20$ | $20$ |
| Initial objective (bps/Hz) | $8.28$ | $6.33$ |
| Final objective (bps/Hz) | $19.52$ | $15.86$ |
| Absolute improvement | $+11.24$ | $+9.53$ |
| Relative improvement | $135.75\%$ | $150.51\%$ |
| Final mean $\alpha$ | $0.95$ | $0.95$ |
| Dominant SCA block | Alpha ($+8.71$) | Alpha ($+6.95$) |

### 13.2 Why Objective Saturates Quickly

1. **Alpha saturation**: The time-splitting factor $\alpha[m]$ converges to the upper bound $\alpha_{\max}=0.95$ for all slots. This dominates the objective improvement because $\alpha$ multiplies every rate term linearly; the optimizer immediately drives it to the maximum feasible value. After $\alpha$ saturates, the only remaining improvements come from power and trajectory adjustments, which yield diminishing returns.

2. **Concave surrogate locality**: The SCA trust-region radius limits per-iteration changes. Initial iterations see large gains as the linear surrogate effectively points toward the alpha bound and power maxima, but subsequent iterations make finer adjustments bounded by $\Delta$.

3. **HPPP realization is fixed**: Eavesdropper positions are generated once per run. The objective therefore depends on a fixed set of $K \approx 20$ points. Once the jammer trajectory positions itself to maximally jam the closest eavesdroppers, further trajectory adjustments yield negligible wiretap rate reduction.

4. **Jammer contribution weakens**: The jammer gradient magnitude is small because $P_j g_{\text{JE},k}$ enters only in the denominator of the wiretap SINR. Once $P_j$ reaches its budget limit and the jammer is positioned near eavesdropper clusters, additional jammer trajectory changes provide marginal benefit (observed jammer update norm at convergence $\approx 0.0$).

5. **Relay bottleneck saturation**: The relay trajectory quickly places itself along a path that balances $R_{\text{SR}}$ and $R_{\text{RD}}$, after which the min-bottleneck provides limited scope for improvement (relay contribution $\approx 0.001$ bps/Hz).

### 13.3 Rician vs. Rayleigh

Rician fading yields higher absolute secrecy ($19.52$ vs. $15.86$ bps/Hz) due to the LOS component ($K=5$), which provides stronger legitimate channel gains. The relative improvement is larger for Rayleigh ($150.5\%$ vs. $135.8\%$) because the initial Rayleigh objective is lower, giving more room for relative gains. Both models converge in the same number of BCD iterations, indicating that the convergence rate is determined primarily by the BCD structure rather than the channel statistics.

---

## 14. Assumptions and Limitations

### 14.1 Assumptions

- **Fixed HPPP realization**: A single set of eavesdropper positions is drawn at environment reset and remains fixed throughout the optimization. This is appropriate for a static scenario but does not account for moving eavesdroppers.
- **Perfect channel state information (CSI)**: The optimizer has access to exact channel gains (including small-scale fading realizations) for all legitimate and eavesdropper links.
- **Finite horizon**: The optimization considers a fixed $M = 12$ slot window. No terminal value or infinite-horizon effects are modeled.
- **Half-duplex relaying**: The source and relay transmit in orthogonal time slots; full-duplex operation is not considered.
- **Fixed UAV altitudes**: Both UAVs fly at constant altitude $50$ m; altitude optimization is not performed.
- **No wind or aerodynamic constraints**: The mobility model assumes instantaneous velocity changes with a simple displacement bound, ignoring wind, turning radius, and acceleration dynamics beyond the per-slot displacement constraint.
- **Eve uncertainty is isotropic**: The uncertainty radius $\delta$ applies uniformly in all directions; elliptical or directional uncertainty is not modeled.
- **Independent fading per slot**: Fading coefficients are independent across time slots (block fading). No temporal correlation is modeled.

### 14.2 Limitations

- **Alpha upper-bound saturation**: In both Rician and Rayleigh experiments, $\alpha[m]$ reaches $0.95$ (the upper bound) for all slots. This suggests that the algorithm would benefit from allocating more time to cooperative transmission, but the bound $\alpha_{\max} = 0.95$ was selected to reserve a minimum $5\%$ slot fraction for control/guard intervals. A sensitivity analysis varying $\alpha_{\max}$ was not performed.

- **Weak jammer contribution**: The observed jammer contribution to the objective is negligible ($\approx 0.0$ bps/Hz cumulative across all BCD iterations). The jammer power and trajectory have limited impact because: (a) the jammer is constrained by the average power budget $\bar{P}_j = 0.25$ W, which is lower than the relay budget; (b) the jammer signal experiences path loss to all eavesdroppers, reducing its effectiveness; and (c) the sum-SINR aggregation in the wiretap rate means that even a high jammer power at one eve location leaves other eavesdroppers unaffected.

- **Differences from the original paper**: The implemented formulation uses a sum-SINR wiretap aggregation ($\log_2(1 + \sum_k \Gamma_k)$) rather than a max-SINR aggregation ($\max_k \log_2(1 + \Gamma_k)$). The sum-SINR form is a looser upper bound on the wiretap capacity (assuming eavesdroppers can combine their observations coherently), which makes it more conservative. This implementation choice was made for gradient tractability, as the sum-SINR form yields smoother gradients than a max operator.

- **No relay selection or role switching**: Unlike the broader codebase (which supports `role_switching` in `UAVEnvironment`), the SCA-BCD optimizer treats the relay and jammer roles as fixed. Dynamic role switching between the two UAVs is not explored.

- **Gradient-based surrogate for power**: The power gradient is computed analytically using the chain rule through $\log_2(1+\gamma)$ and the wiretap SINR. However, the min operator in $R_{\text{leg}}$ creates a non-differentiable point when $R_{\text{SR}} = R_{\text{RD}}$. The implementation handles this by computing separate gradient contributions for $R_{\text{SR}}$ and $R_{\text{RD}}$ and adding them independently, effectively using a subgradient at the non-differentiable boundary.

- **SCA solves a QCQP surrogate rather than the true objective**: The surrogate function trades exactness for convexity. The trust-region radius and step-size backtracking mitigate approximation error, but there is no convergence guarantee to a global optimum. The solver is guaranteed only to converge to a stationary point of the BCD sequence under standard SCA assumptions.

---

## References

The implementation is inspired by (but not identical to) the SCA-BCD framework described in:

> Y. Xu, "Dual-UAV Cooperative Secure Transmission," *IEEE Transactions on Vehicular Technology*, 2023.

The key differences between the implementation and the reference paper are documented in Section 14.2.
