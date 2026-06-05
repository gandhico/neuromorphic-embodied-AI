# Co-Axial Y6 Tricopter — Nengo Implementation Progress

Working log + exact equations as implemented in code. Math is written in
LaTeX (`$$ ... $$`) so it can be pasted into the Overleaf document directly.

- **Plant / parameters:** Mehndiratta & Kayacan, *Reconfigurable Fault-tolerant
  NMPC for Y6 Coaxial Tricopter*, IEEE CCTA 2018 — referred to as **[M]**.
- **Control allocation (mixer):** Czyba et al., *Development of Co-Axial Y6-Rotor
  UAV*, ICUAS 2015 — referred to as **[C]**.

Source files: `y6_model.py` (all equations, pure NumPy), `run_baseline.py`
(standalone validation), `y6_nengo.py` (Nengo Nodes).

---

## 1. Architecture

Cascaded control, following [M]'s high/low-level split, with [C]'s linear mixer
as the (Nengo-friendly) allocation layer:

```
 reference (x_r, y_r, z_r, psi_r)
        |
        v
 [ Outer / position loop ]  --(T, phi*, theta*, psi*)-->  [ Inner / attitude loop ]
                                                                  |
                                                       (tau_x, tau_y, tau_z), T
                                                                  v
                                                          [ control allocation ]
                                                                  |
                                                        omega_1..6 (rotor speeds)
                                                                  v
                                                        [ Y6 rigid-body plant ]
                                                                  |
                                            full state x in R^12 fed back to both loops
```

Implementation roadmap (current marker = ✅ done):

| Stage | File | What is neural | Status |
|---|---|---|---|
| 1 | `y6_nengo.py`         | Nothing — both loops are Nodes | ✅ |
| 2 | `y6_nef_attitude.py`  | Inner attitude loop (rate ensembles) | ✅ |
| 3 | `y6_snn_attitude.py`  | Inner attitude loop (LIF spikes) | ✅ |
| 4 | `y6_snn_integral.py`  | Stage 3 + neural integrators per axis | ✅ |
| 5 | `y6_nef_position.py`  | Stage 4 + NEF outer position loop | ✅ |
| 6 | `y6_nef_allocation.py`| Stage 5 + NEF control allocation | ✅ |
| 7 | `y6_pes_ftc.py`       | Stage 6 + PES online FTC learning | ✅ |
| 8 | `y6_full_snn.py`      | Everything in LIF ensembles | ✅ |

---

## 2. Conventions and state vector

- Frames: Earth-fixed $\mathcal{F}_E$ and body $\mathcal{F}_B$. **[M] uses a
  z-up convention**: positive thrust acts along $+z$, gravity along $-z$, and
  altitude $z$ increases upward. (This differs from [C]'s NED frame; the dynamics
  are identical up to these sign conventions.)
- State (12):
$$
\mathbf{x} = \big[\,x,\;y,\;z,\;\;u,\;v,\;w,\;\;\phi,\;\theta,\;\psi,\;\;p,\;q,\;r\,\big]^{\top}
$$
  position and attitude in $\mathcal{F}_E$; translational/rotational velocities
  in $\mathcal{F}_B$.
- Control: six rotor speeds $\boldsymbol{\omega} = [\omega_1,\dots,\omega_6]^{\top}$.
- Rotor layout (from [M], Fig. 1): front arm = rotors **3,4**; back-right =
  **1,2**; back-left = **5,6**. Odd rotors $\{1,3,5\}$ are the **top** props,
  even $\{2,4,6\}$ the **bottom** props.

---

## 3. Parameters (from [M], Table I)

| Symbol | Description | Value | Unit |
|---|---|---|---|
| $m$ | mass | $1.412$ | kg |
| $g$ | gravity | $9.81$ | m/s² |
| $l_1$ | front-arm lever (arm radius) | $0.28$ | m |
| $l_2$ | lateral lever, back arms | $0.243$ | m |
| $l_3$ | longitudinal lever, back arms | $0.14$ | m |
| $I_{xx}$ | inertia about $x_B$ | $0.014965$ | kg·m² |
| $I_{yy}$ | inertia about $y_B$ | $0.021915$ | kg·m² |
| $I_{zz}$ | inertia about $z_B$ | $0.034433$ | kg·m² |
| $I_{xz}$ | product of inertia | $0.0$ *(see §6)* | kg·m² |
| $J_P$ | propeller inertia | $6\times10^{-5}$ | kg·m² |
| $K_F$ | thrust coefficient | $6.85\times10^{-6}$ | N·s² |
| $K_\tau$ | drag-moment coefficient | $3.35\times10^{-7}$ | N·m·s² |
| $\omega_{\max}$ | rotor speed limit | $900$ | rad/s |

---

## 4. Plant — equations of motion (from [M])

### 4.1 Kinematics ([M], Eq. 1–3)
$$
\begin{bmatrix}\dot{x}\\\dot{y}\\\dot{z}\end{bmatrix} = R_{EB}\begin{bmatrix}u\\v\\w\end{bmatrix},
\qquad
\begin{bmatrix}\dot{\phi}\\\dot{\theta}\\\dot{\psi}\end{bmatrix} = T_{EB}\begin{bmatrix}p\\q\\r\end{bmatrix}
$$

$$
R_{EB}=\begin{bmatrix}
c\theta c\psi & s\phi s\theta c\psi - s\psi c\phi & c\phi s\theta c\psi + s\phi s\psi\\
c\theta s\psi & s\phi s\theta s\psi + c\psi c\phi & c\phi s\theta s\psi - s\phi c\psi\\
-s\theta & s\phi c\theta & c\phi c\theta
\end{bmatrix},
\qquad
T_{EB}=\begin{bmatrix}
1 & s\phi\, t\theta & c\phi\, t\theta\\
0 & c\phi & -s\phi\\
0 & s\phi/c\theta & c\phi/c\theta
\end{bmatrix}
$$
($c=\cos,\;s=\sin,\;t=\tan$.)

### 4.2 Force equations ([M], Eq. 4)
$$
\begin{aligned}
\dot{u} &= rv - qw + g\sin\theta + \tfrac{1}{m}F_x,\\
\dot{v} &= pw - ru - g\sin\phi\cos\theta + \tfrac{1}{m}F_y,\\
\dot{w} &= qu - pv - g\cos\phi\cos\theta + \tfrac{1}{m}F_z.
\end{aligned}
$$

### 4.3 Moment equations ([M], Eq. 5) — full $I_{xz}$-coupled form
With $D = I_{xx}I_{zz} - I_{xz}^2$:
$$
\begin{aligned}
\dot{p} &= \frac{1}{D}\Big[\{-pq\,I_{xz} + qr(I_{yy}-I_{zz})\}I_{zz}
        - \{qr\,I_{xz} + pq(I_{xx}-I_{yy})\}I_{xz} + \tau_x I_{zz} - \tau_z I_{xz}\Big],\\[2pt]
\dot{q} &= pr\,\frac{I_{zz}-I_{xx}}{I_{yy}} - (r^2 - p^2)\frac{I_{xz}}{I_{yy}} + \frac{\tau_y}{I_{yy}},\\[2pt]
\dot{r} &= \frac{1}{D}\Big[\{qr\,I_{xz} + pq(I_{xx}-I_{yy})\}I_{xx}
        - \{-pq\,I_{xz} + qr(I_{yy}-I_{zz})\}I_{xz} + \tau_z I_{xx} - \tau_x I_{xz}\Big].
\end{aligned}
$$
For $I_{xz}=0$ this reduces to the standard diagonal-inertia Euler equations used
by [C].

### 4.4 Actuator / aerodynamic model ([M], Eq. 6–7)
$$
\Omega_i = \alpha_i\,\omega_i,\qquad
F_i = K_F\,\Omega_i^{\,2},\qquad
\tau_i = K_\tau\,\Omega_i^{\,2},\qquad \alpha_i\in[0,1].
$$
$\alpha_i$ is the fault parameter ($1$ = healthy, $0$ = complete loss of rotor $i$).

### 4.5 Force / moment aggregation ([M], Eq. 8–11)
$$
F_{\text{ext}}=\begin{bmatrix}0\\0\\ \sum_{i=1}^{6}F_i\end{bmatrix},
\qquad
\tau_{\text{ext}}=\tau_{\text{prop}}+\tau_{\text{gyro}}
$$
$$
\tau_{\text{prop}}=\begin{bmatrix}
\{(F_5+F_6)-(F_1+F_2)\}\,l_2\\[2pt]
(F_3+F_4)\,l_1 - (F_1+F_2+F_5+F_6)\,l_3\\[2pt]
(\tau_1+\tau_3+\tau_5)-(\tau_2+\tau_4+\tau_6)
\end{bmatrix},
\qquad
\tau_{\text{gyro}}=J_P\begin{bmatrix}
q\{(\Omega_2+\Omega_4+\Omega_6)-(\Omega_1+\Omega_3+\Omega_5)\}\\[2pt]
p\{(\Omega_1+\Omega_3+\Omega_5)-(\Omega_2+\Omega_4+\Omega_6)\}\\[2pt]
0
\end{bmatrix}
$$

### 4.6 Integration
Fixed-step **RK4** with zero-order hold on $\boldsymbol{\omega}$ over each step;
default $\Delta t = 0.01\,$s (matches [M]'s shooting grid). In Nengo the plant
`Node` integrates with the simulator's own $\Delta t$.

---

## 6. Note on the product of inertia $I_{xz}$  ⚠

[M], Table I lists $I_{xz}=0.0261953$ kg·m². With the other table values,
$$
I_{xx}I_{zz}-I_{xz}^2 = (0.014965)(0.034433) - (0.0261953)^2 \approx 5.15\times10^{-4} - 6.86\times10^{-4} < 0,
$$
so the inertia tensor $\begin{psmallmatrix}I_{xx}&0&-I_{xz}\\0&I_{yy}&0\\-I_{xz}&0&I_{zz}\end{psmallmatrix}$
is **not positive-definite** — physically impossible (almost certainly a typo /
unit slip in the paper, e.g. an order of magnitude). Consequences if used as
printed: $D<0$ flips the sign of the roll/yaw angular acceleration and makes the
plant unstable.

**Decision:** default `Ixz = 0.0` (the diagonal assumption used by [C]). The full
$I_{xz}$-coupled equations remain in the code, so the value is a single tunable
parameter — set it to a physically valid figure (e.g. $\sim 2.6\times10^{-3}$,
keeping $I_{xz}^2 < I_{xx}I_{zz}$) to study cross-coupling later.

---

## 7. Geometry reconciliation between [C] and [M]

[C] states all three arms are equidistant ($l$) on an equilateral triangle; [M]
lists three different lever arms $l_1,l_2,l_3$. These are **consistent**: $l_2$
and $l_3$ are the projections of a single equilateral layout of radius $l_1$,
with one arm along $+x_B$ and the back arms at $\pm120^\circ$:
$$
l_2 = l_1\sin 60^\circ = 0.28\times0.866 \approx 0.243,\qquad
l_3 = l_1\cos 60^\circ = 0.28\times0.5 = 0.14.
$$
Both match the table to 3 sig. figs. So [C] and [M] describe the **same physical
frame**; only the rotor numbering differs (front arm = rotors 1,2 in [C] vs 3,4
in [M]). Our mixer uses **[M]'s numbering**.

---

## 8. Control allocation (mixer)

We use [C]'s linear-mixer **sign structure** ([C], Eq. 11) but with magnitudes
derived from [M]'s geometry, so the virtual commands are physical: total thrust
$T$ (N) and body moments $\tau_x,\tau_y,\tau_z$ (N·m). The coaxial redundancy
(6 rotors, 4 commands) is resolved by an **equal top/bottom split per arm**.

Per-arm forces (front $=F_3{+}F_4$, back-right $=F_1{+}F_2$, back-left $=F_5{+}F_6$):
$$
S_{\text{back}} = (l_1 T - \tau_y)\,k_1,\quad
F_{\text{front}} = T - S_{\text{back}},\quad
F_{BR}=\tfrac12\!\left(S_{\text{back}}-\tfrac{\tau_x}{l_2}\right),\quad
F_{BL}=\tfrac12\!\left(S_{\text{back}}+\tfrac{\tau_x}{l_2}\right),
$$
with $k_1=1/(l_1+l_3)$. Yaw is produced by a top/bottom split
$\delta = k_\psi\,\tau_z$, $\;k_\psi = K_F/(6K_\tau)$. The six rotor thrusts:
$$
\begin{aligned}
F_1 &= \tfrac12 F_{BR} + \delta, & F_2 &= \tfrac12 F_{BR} - \delta,\\
F_3 &= \tfrac12 F_{\text{front}} + \delta, & F_4 &= \tfrac12 F_{\text{front}} - \delta,\\
F_5 &= \tfrac12 F_{BL} + \delta, & F_6 &= \tfrac12 F_{BL} - \delta.
\end{aligned}
$$
Equivalently $\mathbf{F} = A\,[\,T,\;\tau_x,\;\tau_y,\;\tau_z\,]^{\top}$ with
$$
A=\begin{bmatrix}
\tfrac{l_1k_1}{4} & -\tfrac{1}{4l_2} & -\tfrac{k_1}{4} & +k_\psi\\
\tfrac{l_1k_1}{4} & -\tfrac{1}{4l_2} & -\tfrac{k_1}{4} & -k_\psi\\
\tfrac{l_3k_1}{2} & 0 & +\tfrac{k_1}{2} & +k_\psi\\
\tfrac{l_3k_1}{2} & 0 & +\tfrac{k_1}{2} & -k_\psi\\
\tfrac{l_1k_1}{4} & +\tfrac{1}{4l_2} & -\tfrac{k_1}{4} & +k_\psi\\
\tfrac{l_1k_1}{4} & +\tfrac{1}{4l_2} & -\tfrac{k_1}{4} & -k_\psi
\end{bmatrix}.
$$
The **sign pattern** of $A$ (column-wise: all $+$ for $T$; $\{-,-,0,0,+,+\}$ for
roll; $\{-,-,+,+,-,-\}$ for pitch; $\{+,-,+,-,+,-\}$ for yaw) is exactly Czyba's
mixer matrix. Rotor speed commands invert the thrust law and saturate:
$$
\omega_i = \mathrm{clip}\!\left(\sqrt{\max(F_i,0)/K_F},\;0,\;\omega_{\max}\right).
$$

---

## 9. Cascaded baseline controller (PD + gravity feed-forward)

### 9.1 Outer / position loop
World-frame velocity $[\dot{x},\dot{y},\dot{z}] = R_{EB}[u,v,w]^{\top}$.
Desired horizontal accelerations:
$$
a_x^{d} = K_p^{xy}(x_r-x) - K_d^{xy}\dot{x},\qquad
a_y^{d} = K_p^{xy}(y_r-y) - K_d^{xy}\dot{y}.
$$
Yaw-compensated attitude references (inverting $a_x\!\approx\! g(\theta c\psi+\phi s\psi)$,
$a_y\!\approx\! g(\theta s\psi-\phi c\psi)$):
$$
\theta^{*} = \frac{a_x^{d}\cos\psi + a_y^{d}\sin\psi}{g},\qquad
\phi^{*} = \frac{a_x^{d}\sin\psi - a_y^{d}\cos\psi}{g},
$$
clipped to $\pm35^\circ$ ([M], Eq. 19). Thrust with tilt compensation:
$$
T = \frac{mg + K_p^{z}(z_r-z) - K_d^{z}\dot{z}}{\cos\phi\cos\theta},
\qquad T\in[0,\,2.5\,mg].
$$

### 9.2 Inner / attitude loop
$$
\tau_x = K_p^{\phi}(\phi^{*}-\phi) - K_d^{\phi}p,\quad
\tau_y = K_p^{\theta}(\theta^{*}-\theta) - K_d^{\theta}q,\quad
\tau_z = K_p^{\psi}(\psi^{*}-\psi) - K_d^{\psi}r.
$$

### 9.3 Gains (initial, tuned in simulation)
| Loop | $K_p$ | $K_d$ |
|---|---|---|
| x, y position | $2.0$ | $2.6$ |
| z (altitude) | $9.0$ | $5.2$ |
| roll $\phi$ | $1.3$ | $0.28$ |
| pitch $\theta$ | $1.8$ | $0.36$ |
| yaw $\psi$ | $1.0$ | $0.32$ |

Steady-state altitude error is removed by the $mg$ feed-forward (no integral term
needed at this stage; an integral term is the natural next addition if modelling
constant disturbances).

---

## 10. Validation results (`run_baseline.py`)

| Test | Result |
|---|---|
| Hover @ 1.5 m | settles exactly; z-RMS error (last 2 s) $\approx 0$ mm; no attitude drift |
| Step → (1, −0.5, 2.0) m, yaw 20° | converges to target exactly |
| Square trajectory (step corners) | tracking RMSE $x/y/z = 0.305/0.249/0.183$ m |
| Rotor speeds (all runs) | hover $\approx 580$ rad/s; range $[519,\,816]$ — within $900$ limit |

Hover thrust check: $\omega_{\text{hover}} = \sqrt{mg/(6K_F)} \approx 580.6$ rad/s,
matching simulation. The Nengo `Node` wrapper (`y6_nengo.py`) reproduces these
results identically.

---

## 11. Stage 2 — NEF inner attitude loop  (`y6_nef_attitude.py`)

### Design
Three independent pairs of 1-D ensembles replace the inner attitude `Node`:

| Ensemble | Dimension | Radius | Represents |
|---|---|---|---|
| `e_err_{phi,the,psi}` | 1 | 0.70 / 0.70 / 3.20 rad | attitude error |
| `e_rate_{phi,the,psi}` | 1 | 3.0 / 3.0 / 2.0 rad/s | body rate |

Linear decoders implement the PD law:
$$
\tau_i = K_p^{(i)}\,\hat{e}_i - K_d^{(i)}\,\hat{\dot{\phi}}_i
$$
where $\hat{\cdot}$ denotes the NEF decoded value.

Neuron type: `nengo.RectifiedLinear()` (rate mode, no spike noise).
Synapse: $\tau_s = 5\,$ms.

### Key architectural note
The outer position loop and control allocation remain as plain `Node` objects
so the NEF change is fully isolated — both can be validated against the Stage-1
oracle simultaneously.

---

## 12. Stage 3 — Spiking inner attitude loop  (`y6_snn_attitude.py`)

Identical wiring to Stage 2. Changes:

| Parameter | Stage 2 | Stage 3 |
|---|---|---|
| Neuron type | `RectifiedLinear` | `LIF` |
| Neurons per ensemble | 300 | 500 |
| Output synapse $\tau_s$ | 5 ms | 10 ms |
| Recommended $\Delta t$ | 10 ms | 1 ms |

The increased neuron count and longer synapse compensate for spike-noise variance.
Probe `p_spikes_err0` gives the raw spike raster for the roll-error ensemble.

---

## 13. Stage 4 — Neural integrator  (`y6_snn_integral.py`)

Each axis gains a recurrent LIF ensemble that accumulates attitude error:
$$
\dot{I}_i = e_i \qquad \Rightarrow \qquad \tau_i = K_p e_i - K_d \dot{e}_i + K_i I_i
$$

NEF recurrent integrator:
$$
A = \alpha,\quad B = (1-\alpha),\qquad
\alpha = 0.9998 \;\text{at}\; \Delta t = 1\,\text{ms}
\;\Rightarrow\; \tau_{\text{int}} = -\Delta t/\ln\alpha \approx 5\,\text{s}.
$$

Integral gains:

| Axis | $K_i$ |
|---|---|
| roll $\phi$ | $0.08$ |
| pitch $\theta$ | $0.08$ |
| yaw $\psi$ | $0.04$ |

---

## 14. Stage 5 — NEF outer position loop  (`y6_nef_position.py`)

The nonlinear yaw-compensation is factored into three ensembles:

| Ensemble | Dim | Represents | Decodes |
|---|---|---|---|
| `e_pos_xy` | 4 | $[x_e,y_e,\dot x,\dot y]$ | $[a_x^d, a_y^d]$ (linear PD) |
| `e_yaw_mix` | 4 | $[a_x^d, a_y^d, c\psi, s\psi]$ | $[\phi^*, \theta^*]$ (bilinear) |
| `e_alt` | 2 | $[z_e, \dot z]$ | $T$ (linear PD + $mg$ feedforward) |

The bilinear decode in `e_yaw_mix`:
$$
\phi^* = (a_x^d s\psi - a_y^d c\psi)/g,\qquad
\theta^* = (a_x^d c\psi + a_y^d s\psi)/g
$$
is representable by a 4-D ensemble because it involves products of two
low-frequency signals — the NEF approximates these bilinear functions via
distributed nonlinear tuning curves.

Pre-processor Nodes compute $[\cos\psi, \sin\psi]$ and world-frame velocities
from the plant state; these trig computations remain as Nodes.

---

## 15. Stage 6 — NEF control allocation  (`y6_nef_allocation.py`)

The 6×4 linear mixer $\mathbf{F} = A\mathbf{u}$ is implemented as a single
4-D LIF ensemble `e_cmd` representing $[T, \tau_x, \tau_y, \tau_z]$, with
six separate linear decode connections:
$$
F_j = \mathbf{a}_j^{\top}\,\hat{\mathbf{u}}, \qquad j=1,\dots,6
$$
where $\mathbf{a}_j$ is row $j$ of the mixer matrix $A$.

The nonlinear inversion $F_j \to \omega_j = \sqrt{F_j/K_F}$ (and saturation)
remains as a `Node` because it is a pointwise nonlinearity that does not benefit
from population coding.

---

## 16. Stage 7 — PES fault-tolerant control  (`y6_pes_ftc.py`)

### PES learning rule
PES (Prescribed Error Sensitivity, Eliasmith & Anderson 2003) modifies the
decoders $D$ of `e_cmd` according to:
$$
\Delta D = -\eta\,E\,\mathbf{a}^{\top}
$$
where $E$ is the error signal (smoothed L2 attitude norm), $\mathbf{a}$ are the
neural activities of `e_cmd`, and $\eta$ is the learning rate.

### Fault scenario
At $t = 6\,$s rotor 0 is killed ($\alpha_0 = 0$). The tracking error spikes,
driving $E > 0$ and activating PES. The learned decoder offset compensates for
the lost rotor by redistributing thrust to the remaining rotors.

### Tuning notes
- `learning_rate = 1e-5`: conservative; increase to `1e-4` for faster
  adaptation at risk of oscillation.
- PES is switched off for $t < 1\,$s (warm-up) to avoid adapting to transient
  errors.
- Fault recovery is approximate — the vehicle may drift in yaw (one rotor pair
  controls yaw directly) but maintains altitude and position.

---

## 17. Stage 8 — Full SNN controller  (`y6_full_snn.py`)

All controller layers are LIF ensembles:

| Layer | Ensemble(s) | Dim | Neurons |
|---|---|---|---|
| XY position PD | `e_pos_xy` | 4 | 600 |
| Yaw mixing | `e_yaw_mix` | 4 | 800 |
| Altitude PD | `e_alt` | 2 | 300 |
| Attitude error (×3) | `e_err_{phi,the,psi}` | 1 | 500 ea |
| Body rate (×3) | `e_rate_{phi,the,psi}` | 1 | 500 ea |
| Neural integrator (×3) | `e_int_{phi,the,psi}` | 1 | 500 ea |
| Control allocation | `e_cmd` | 4 | 800 |

Remaining Nodes: `plant` (physics), `ref_node` (setpoint), `yaw_trig`
(sin/cos), `F_to_omega` (sqrt inversion).

Total neurons: **~8 400** (rate-coded) or **~8 400 LIF**.

PES learning is active throughout to provide online adaptation.

---

## 18. Log

- **v0.1** — Plant (full [M] EoM incl. gyroscopic + $I_{xz}$ machinery), [C]/[M]
  mixer, cascaded PD controller. Standalone validation + Nengo Node wrapper.
  Flagged & defaulted the non-physical $I_{xz}$ table value. *(stages 1–2 complete)*

- **v0.2** — Progressive SNN roadmap implemented: Stages 2–8 all code-complete.
  NEF attitude (rate), SNN attitude (LIF), neural integrators, NEF outer position
  loop with yaw-mixing bilinear ensemble, NEF allocation, PES fault-tolerant
  control, full SNN controller. Nengo GUI `.cfg` layouts added for all stages.
