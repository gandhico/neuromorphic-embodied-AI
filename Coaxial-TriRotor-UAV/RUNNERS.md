# Co-Axial Y6 Tricopter — Runners & Usage Guide

Exhaustive reference for every way to run each file.
For equations and design rationale, see `PROGRESS.md`.

---

## 0. Environment setup

```powershell
# Activate the project venv (Windows PowerShell)
& C:\02_Research\00_Code_Ecosystem\neuromorphic-robotics-control\.venv\Scripts\Activate.ps1

# Core dependencies
pip install nengo nengo-gui numpy scipy matplotlib

# Run all scripts from the code folder (so local imports resolve)
cd C:\00_Research_\Neuromorphic_Embodied_AI\02_Coaxial_TriRotor_Nengo\01_Code
```

---

## 1. File map and dependency chain

```
y6_model.py          <- single source of truth (plant, mixer, baseline PD, pure NumPy)
    ^
    |--- run_baseline.py          stage 0 : standalone math validation, no Nengo
    |--- y6_nengo.py              stage 1 : plant + controller as Nengo Nodes
             ^
             |--- y6_nef_attitude.py       stage 2 : NEF inner attitude (rate neurons)
                      ^
                      |--- y6_snn_attitude.py    stage 3 : SNN inner attitude (LIF)
                               ^
                               |--- y6_snn_integral.py   stage 4 : stage 3 + neural integrators
                                        ^
                                        |--- y6_nef_position.py   stage 5 : NEF outer loop
                                                 ^
                                                 |--- y6_nef_allocation.py  stage 6 : NEF mixer
                                                          ^
                                                          |--- y6_pes_ftc.py     stage 7 : PES FTC
                                                          |--- y6_full_snn.py   stage 8 : full SNN
```

Each stage imports helpers from its parent, so you must keep all files in the
same directory.

---

## 2. `run_baseline.py` — math validation (no Nengo)

Runs three hardcoded scenarios and saves one PNG per scenario.  No arguments;
all scenarios execute in a single call.

```powershell
python run_baseline.py
```

**What runs:**
1. Hover at (0, 0, 1.5 m), 10 s
2. Step to (1, −0.5, 2.0 m), yaw 20°, at t=1 s — 15 s total
3. Square trajectory 1 m × 1 m at 1.5 m, 5 s per leg — 28 s total

**Outputs:** `baseline_validation.png`  
**Console:** final position, yaw, z-RMS error, square RMSE x/y/z, omega range.

### Python API — custom scenario

```python
from run_baseline import simulate
import numpy as np

def ref_climb(t):
    """Steady 0.5 m/s climb from ground."""
    return (0.0, 0.0, min(0.5 * t, 3.0), 0.0)

log = simulate(ref_climb, T_end=10.0)
# log keys: 't', 'state' (N×12), 'omega' (N×6), 'ref' (N×4)

import y6_model as y6
print("final z =", log["state"][-1, y6.IZ])
```

```python
# Fault simulation (rotor 3 dead at t=5 s)
def alpha_step(t):
    a = np.ones(6)
    if t >= 5.0:
        a[3] = 0.0
    return a

log = simulate(ref_climb, T_end=15.0, alpha_fn=alpha_step)
```

---

## 3. `y6_nengo.py` — Stage 1: Node baseline

### Direct run (default mission: hover → step)
```powershell
python y6_nengo.py
# sim: 12 s, dt=0.01 s  (~2 s wall time)
# output: y6_nengo_demo.png
```

**Default mission** (`default_ref`):
- `t < 2 s` → hover at (0, 0, 1.5 m), yaw 0°
- `t ≥ 2 s` → step to (1, −0.5, 2.0 m), yaw 20°

### Nengo GUI
```powershell
nengo y6_nengo.py
# layout loaded from: y6_nengo.cfg
# opens http://localhost:8080
```

### Python API

```python
import nengo
from y6_nengo import build_network, default_ref

# Signature:
# build_network(ref_fn, x0=None, alpha_fn=None, P=Params, G=Gains)

# --- default mission ---
net = build_network(default_ref)
with nengo.Simulator(net, dt=0.01) as sim:
    sim.run(12.0)

# --- custom reference ---
def ref_hover_2m(t):
    return (0.0, 0.0, 2.0, 0.0)

net = build_network(ref_hover_2m)

# --- with fault (rotor 0 dead after t=5 s) ---
import numpy as np
def alpha_fn(t):
    a = np.ones(6)
    if t >= 5.0: a[0] = 0.0
    return a

net = build_network(default_ref, alpha_fn=alpha_fn)

# --- custom initial state (1 m above ground, slight roll) ---
import numpy as np
x0 = np.zeros(12)
x0[2] = 1.0          # z = 1 m
x0[6] = np.radians(5)  # phi = 5 deg
net = build_network(default_ref, x0=x0)

# --- access probes ---
with nengo.Simulator(net, dt=0.01) as sim:
    sim.run(12.0)
import y6_model as y6
state = sim.data[net.p_state]   # shape (N, 12)
omega = sim.data[net.p_omega]   # shape (N, 6)
print("final pos:", state[-1, [y6.IX, y6.IY, y6.IZ]])
```

---

## 4. `y6_nef_attitude.py` — Stage 2: NEF attitude (rate neurons)

### Direct run
```powershell
python y6_nef_attitude.py
# sim: 12 s, dt=0.01 s  (~5 s wall time)
# output: y6_nef_attitude_demo.png
```

### Nengo GUI
```powershell
nengo y6_nef_attitude.py
# layout: y6_nef_attitude.cfg
```

### Python API

```python
from y6_nef_attitude import build_network, default_ref
import nengo

# Signature:
# build_network(ref_fn=default_ref, x0=None, alpha_fn=None,
#               n_neurons=300, tau_syn=0.005,
#               P=Params, G=Gains)

# --- default ---
net = build_network()

# --- more neurons for higher accuracy ---
net = build_network(n_neurons=600)

# --- faster synapse (sharper response, more noise) ---
net = build_network(tau_syn=0.002)

# --- slower synapse (smoother, more lag) ---
net = build_network(tau_syn=0.02)

# --- custom mission ---
import numpy as np
def ref_circle(t):
    r, T = 1.0, 10.0
    return (r * np.cos(2*np.pi*t/T),
            r * np.sin(2*np.pi*t/T),
            1.5, 0.0)

net = build_network(ref_circle)

# --- available probes ---
# net.p_state     shape (N, 12) — full plant state
# net.p_omega     shape (N, 6)  — rotor speeds [rad/s]
# net.p_att_des   shape (N, 3)  — [phi_des, theta_des, psi_des] from outer loop
# net.p_tau       shape (N, 3)  — decoded body moments [N·m]
# net.ens_err     list of 3 Ensemble objects (attitude errors)
# net.ens_rate    list of 3 Ensemble objects (body rates)
```

---

## 5. `y6_snn_attitude.py` — Stage 3: SNN attitude (LIF spikes)

### Direct run
```powershell
python y6_snn_attitude.py
# sim: 12 s, dt=0.001 s  (~60 s wall time)
# output: y6_snn_attitude_demo.png  (position + rotor speeds + moments + spike raster)
```

### Nengo GUI
```powershell
nengo y6_snn_attitude.py
# layout: y6_snn_attitude.cfg
# Right-click e_err_phi -> add "Spike raster" for live firing activity
```

### Python API

```python
from y6_snn_attitude import build_network, default_ref
import nengo

# Signature:
# build_network(ref_fn=default_ref, x0=None, alpha_fn=None,
#               n_neurons=500, tau_syn=0.01,
#               P=Params, G=Gains)

# --- default (500 neurons, 10 ms synapse) ---
net = build_network()
with nengo.Simulator(net, dt=0.001) as sim:
    sim.run(12.0)

# --- reduce neuron count for faster iteration (lower accuracy) ---
net = build_network(n_neurons=200)

# --- increase neuron count for publication-quality accuracy ---
net = build_network(n_neurons=1000, tau_syn=0.015)

# --- available probes ---
# net.p_state          shape (N, 12)
# net.p_omega          shape (N, 6)
# net.p_att_des        shape (N, 3)
# net.p_tau            shape (N, 3)
# net.p_spikes_err0    shape (N, n_neurons)  — raw spikes, roll-error ensemble
# net.p_spikes_rate0   shape (N, n_neurons)  — raw spikes, roll-rate ensemble

# --- plot spike raster manually ---
import matplotlib.pyplot as plt
import numpy as np
with nengo.Simulator(net, dt=0.001) as sim:
    sim.run(5.0)
t   = sim.trange()
spk = sim.data[net.p_spikes_err0][:, :80]   # first 80 neurons
si, ni = np.where(spk > 0)
plt.scatter(t[si], ni, s=0.3, c='k')
plt.xlabel("t [s]"); plt.ylabel("neuron #")
plt.title("Roll-error ensemble — spike raster")
plt.show()
```

---

## 6. `y6_snn_integral.py` — Stage 4: SNN attitude + neural integrators

### Direct run
```powershell
python y6_snn_integral.py
# sim: 12 s, dt=0.001 s  (~60–90 s wall time)
# output: y6_snn_integral_demo.png  (position + moments + integrator state)
```

### Nengo GUI
```powershell
nengo y6_snn_integral.py
# layout: y6_snn_integral.cfg
# Right-click e_int_phi -> add "Value" panel to watch integrator state live
```

### Python API

```python
from y6_snn_integral import build_network, default_ref, IntegralGains
import nengo

# Signature:
# build_network(ref_fn=default_ref, x0=None, alpha_fn=None,
#               n_neurons=500, tau_syn=0.01,
#               recurrent_A=0.9998,
#               P=Params, G=Gains, Gi=IntegralGains)

# --- default ---
net = build_network()

# --- leakier integrator (faster wind-down, tau_int ≈ 0.5 s at dt=0.001) ---
net = build_network(recurrent_A=0.998)

# --- near-perfect integrator (very slow leak, tau_int ≈ 50 s) ---
net = build_network(recurrent_A=0.99998)

# --- custom integral gains ---
class MyGains(IntegralGains):
    Ki_phi = 0.15   # stronger roll integration
    Ki_the = 0.15
    Ki_psi = 0.02   # weaker yaw integration

net = build_network(Gi=MyGains)

# --- available probes ---
# net.p_state     shape (N, 12)
# net.p_omega     shape (N, 6)
# net.p_tau       shape (N, 3)
# net.p_int_phi   shape (N, 1)  — roll integrator state (decoded)
# net.ens_err     list of 3 Ensembles
# net.ens_rate    list of 3 Ensembles
# net.ens_int     list of 3 recurrent Ensembles (integrators)
```

---

## 7. `y6_nef_position.py` — Stage 5: NEF outer position loop + SNN attitude

### Direct run
```powershell
python y6_nef_position.py
# sim: 12 s, dt=0.001 s  (~2–4 min wall time; yaw-mixing ensemble is large)
# output: y6_nef_position_demo.png
```

### Nengo GUI
```powershell
nengo y6_nef_position.py
# layout: y6_nef_position.cfg
# Useful panels:
#   e_pos_xy  -> "Value"  (watch decoded [ax_des, ay_des])
#   e_yaw_mix -> "Value"  (watch decoded [phi_des, theta_des])
#   e_alt     -> "Value"  (watch decoded thrust T)
```

### Python API

```python
from y6_nef_position import build_network, default_ref
from y6_snn_integral import IntegralGains
import nengo

# Signature:
# build_network(ref_fn=default_ref, x0=None, alpha_fn=None,
#               P=Params, G=Gains, Gi=IntegralGains)
# Note: inner neuron counts are hardcoded at n_att=500; to change them
# edit the constants at the top of build_network().

# --- default ---
net = build_network()

# --- custom yaw reference only ---
import numpy as np
def ref_spin(t):
    return (0.0, 0.0, 1.5, np.radians(30.0) * min(t / 3.0, 1.0))

net = build_network(ref_spin)

# --- available probes ---
# net.p_state   shape (N, 12)
# net.p_omega   shape (N, 6)
# net.p_tau     shape (N, 3)
# net.p_outer   shape (N, 4)  — [T, phi_des, theta_des, psi_des] from NEF outer loop
```

---

## 8. `y6_nef_allocation.py` — Stage 6: NEF control allocation

### Direct run
```powershell
python y6_nef_allocation.py
# sim: 12 s, dt=0.001 s  (~3–5 min wall time)
# output: y6_nef_allocation_demo.png
```

### Nengo GUI
```powershell
nengo y6_nef_allocation.py
# layout: y6_nef_allocation.cfg
# Right-click e_cmd -> "Value" to watch the 4-D command vector live
# Right-click F_cmd -> "Value" to watch all 6 decoded thrust demands
```

### Python API

```python
from y6_nef_allocation import build_network, default_ref
from y6_snn_integral import IntegralGains
import nengo

# Signature:
# build_network(ref_fn=default_ref, x0=None, alpha_fn=None,
#               n_cmd=600, tau_syn=0.01,
#               P=Params, G=Gains, Gi=IntegralGains)

# --- default ---
net = build_network()

# --- more allocation neurons for better linearity ---
net = build_network(n_cmd=1000)

# --- inspect the mixer matrix used ---
net = build_network()
print("Mixer matrix A (6×4):")
print(net.A_mix)

# --- available probes ---
# net.p_state   shape (N, 12)
# net.p_omega   shape (N, 6)
# net.p_F       shape (N, 6)   — decoded per-rotor thrust F_1..F_6 [N]
# net.p_tau     shape (N, 3)   — body moments [N·m]
```

---

## 9. `y6_pes_ftc.py` — Stage 7: PES fault-tolerant control

### Direct run (fault on rotor 0 at t=6 s)
```powershell
python y6_pes_ftc.py
# sim: 18 s, dt=0.001 s  (~5–8 min wall time)
# output: y6_pes_ftc_demo.png  (position + rotor speeds + attitude errors + PES signal)
```

### Nengo GUI
```powershell
nengo y6_pes_ftc.py
# layout: y6_pes_ftc.cfg
# Recommended panels:
#   plant       -> "Value" (slices 0:3 for position)
#   pes_error   -> "Value" (watch learning signal activate after fault)
#   F_cmd       -> "Value" (watch thrust redistribution post-fault)
```

### Python API — all configurable parameters

```python
from y6_pes_ftc import build_network, make_fault_alpha, default_ref
import nengo

# Signature:
# build_network(ref_fn=default_ref, x0=None, alpha_fn=None,
#               fault_rotor=0, fault_t=6.0,
#               learning_rate=1e-5,
#               n_cmd=600, tau_syn=0.01,
#               P=Params, G=Gains, Gi=IntegralGains)

# --- default: kill rotor 0 (back-right top) at t=6 s ---
net = build_network()

# --- kill front-top rotor (index 2) at t=8 s ---
net = build_network(fault_rotor=2, fault_t=8.0)

# --- faster PES adaptation (may oscillate) ---
net = build_network(fault_rotor=0, fault_t=6.0, learning_rate=1e-4)

# --- conservative adaptation (slower recovery, more stable) ---
net = build_network(fault_rotor=0, fault_t=6.0, learning_rate=5e-6)

# --- no fault (pure PES regularisation, healthy vehicle) ---
net = build_network(fault_t=999.0)   # fault never triggers

# --- double fault (rotors 0 and 1 both die at t=6 s) ---
import numpy as np
def double_fault(t):
    a = np.ones(6)
    if t >= 6.0:
        a[0] = 0.0   # back-right top
        a[1] = 0.0   # back-right bottom
    return a

net = build_network(ref_fn=default_ref,
                    alpha_fn=double_fault,
                    fault_rotor=0, fault_t=6.0)

# --- custom alpha_fn only, no internal fault generator ---
net = build_network(alpha_fn=double_fault, fault_t=6.0)

# Rotor index reference:
#   0 = back-right top     1 = back-right bottom
#   2 = front top          3 = front bottom
#   4 = back-left top      5 = back-left bottom

# --- available probes ---
# net.p_state     shape (N, 12)
# net.p_omega     shape (N, 6)
# net.p_F         shape (N, 6)   — per-rotor thrust [N]
# net.p_att_err   shape (N, 3)   — smoothed attitude errors [rad]
# net.p_pes_err   shape (N, 6)   — PES learning signal broadcast
```

---

## 10. `y6_full_snn.py` — Stage 8: Full SNN controller

### Direct run
```powershell
python y6_full_snn.py
# sim: 12 s, dt=0.001 s  (~5–10 min wall time, ~8 400 LIF neurons total)
# output: y6_full_snn_demo.png  (position + rotor speeds + moments + attitude errors)
```

### Nengo GUI
```powershell
nengo y6_full_snn.py
# layout: y6_full_snn.cfg
# Recommended panels:
#   e_pos_xy  -> "Value"       (decoded horizontal acceleration commands)
#   e_yaw_mix -> "Value"       (decoded phi_des, theta_des)
#   e_alt     -> "Value"       (decoded thrust T)
#   e_cmd     -> "Spike raster" (allocation ensemble spikes)
#   e_err_phi -> "Spike raster" (roll-error ensemble)
```

### Python API — all neuron counts tunable

```python
from y6_full_snn import build_network, default_ref
from y6_snn_integral import IntegralGains
import nengo

# Signature:
# build_network(ref_fn=default_ref, x0=None, alpha_fn=None,
#               n_pos_xy=600, n_yaw_mix=800, n_alt=300,
#               n_att=500, n_cmd=800,
#               tau_syn=0.01, recurrent_A=0.9998,
#               learning_rate=5e-6,
#               P=Params, G=Gains, Gi=IntegralGains)

# --- default (full resolution, ~8 400 neurons) ---
net = build_network()

# --- fast/lightweight version for quick iteration (~2 800 neurons) ---
net = build_network(n_pos_xy=200, n_yaw_mix=300, n_alt=100,
                    n_att=200, n_cmd=300)

# --- high-accuracy version for paper results (~16 800 neurons) ---
net = build_network(n_pos_xy=1000, n_yaw_mix=1500, n_alt=500,
                    n_att=800, n_cmd=1200)

# --- disable PES (pure feedforward SNN, no online learning) ---
net = build_network(learning_rate=0.0)

# --- fault injection with PES ---
import numpy as np
def fault_fn(t):
    a = np.ones(6)
    if t >= 8.0: a[2] = 0.0   # front-top rotor
    return a

net = build_network(default_ref, alpha_fn=fault_fn, learning_rate=1e-5)

# --- custom mission (figure-8) ---
def ref_fig8(t):
    A, T_period = 1.5, 10.0
    return (A * np.sin(2*np.pi*t/T_period),
            A * np.sin(4*np.pi*t/T_period),
            1.5, 0.0)

net = build_network(ref_fig8)
with nengo.Simulator(net, dt=0.001) as sim:
    sim.run(20.0)

# --- available probes ---
# net.p_state   shape (N, 12)
# net.p_omega   shape (N, 6)
# net.p_tau     shape (N, 3)   — body moments [N·m]
# net.p_att_err shape (N, 3)   — attitude errors [rad]
# net.p_F       shape (N, 6)   — per-rotor thrust demands [N]
```

---

## 11. Cross-stage comparison

Run any two stages on the same mission and compare position RMSE:

```python
import numpy as np
import nengo
import y6_model as y6
from y6_nengo        import build_network as build_s1, default_ref
from y6_nef_attitude import build_network as build_s2
from y6_snn_integral import build_network as build_s4
from y6_full_snn     import build_network as build_s8

T_END = 12.0

results = {}
for label, builder, dt in [
    ("Stage-1 Node",         build_s1, 0.01),
    ("Stage-2 NEF-att",      build_s2, 0.01),
    ("Stage-4 SNN+integral", build_s4, 0.001),
    ("Stage-8 Full-SNN",     build_s8, 0.001),
]:
    net = builder(default_ref) if label != "Stage-1 Node" else builder(default_ref)
    with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
        sim.run(T_END)
    st = sim.data[net.p_state]
    results[label] = st[:, [y6.IX, y6.IY, y6.IZ]]

# Compare against Stage-1 (oracle)
ref = results["Stage-1 Node"]
for label, pos in results.items():
    n = min(len(pos), len(ref))
    rmse = np.sqrt(np.mean((pos[:n] - ref[:n])**2, axis=0))
    print(f"{label:30s}  RMSE x/y/z = {rmse[0]:.3f}/{rmse[1]:.3f}/{rmse[2]:.3f} m")
```

---

## 12. Custom missions (all stages)

Any `build_network` accepts a `ref_fn` callable returning
`(x_des, y_des, z_des, psi_des)` for a given time `t`.

```python
import numpy as np

# Hover at fixed point
def ref_hover(t):
    return (0.0, 0.0, 1.5, 0.0)

# Slow vertical climb (0.3 m/s)
def ref_climb(t):
    return (0.0, 0.0, min(0.3 * t, 3.0), 0.0)

# Step with yaw sweep
def ref_step_yaw(t):
    return (1.0, 0.0, 2.0, np.radians(min(t * 15.0, 90.0)))

# Square trajectory (1 m side, 5 s per leg, starts at t=3 s)
def ref_square(t):
    if t < 3.0:
        return (0.0, 0.0, 1.5, 0.0)
    seg = int((t - 3.0) // 5) % 4
    corners = [(1, 0), (1, 1), (0, 1), (0, 0)]
    cx, cy = corners[seg]
    return (cx, cy, 1.5, 0.0)

# Figure-8 (Mehndiratta benchmark)
def ref_fig8(t):
    A, T_period = 1.5, 12.0
    return (A * np.sin(2*np.pi*t/T_period),
            A * np.sin(4*np.pi*t/T_period),
            1.5, 0.0)

# Helical climb
def ref_helix(t):
    r, v_z = 1.0, 0.1
    T_period = 8.0
    return (r * np.cos(2*np.pi*t/T_period),
            r * np.sin(2*np.pi*t/T_period),
            1.0 + v_z * t, 0.0)
```

Pass to any stage:

```python
from y6_full_snn import build_network
net = build_network(ref_fig8)
```

---

## 13. Nengo GUI tips

| Action | How |
|---|---|
| Add value panel | Right-click ensemble → "Value" |
| Add spike raster | Right-click ensemble → "Spike raster" (LIF stages only) |
| Add slider | Right-click node → "Slider" (useful on `reference` node) |
| Pause/resume | Click the play/pause button in the SimControl panel |
| Reset simulation | Click the reset (↺) button in SimControl |
| Save layout | Automatic — changes to `.cfg` on every drag |
| Change sim speed | Drag the speed slider in SimControl |

Launch any stage with its `.cfg` layout:

```powershell
nengo y6_nengo.py          # Stage 1  ->  y6_nengo.cfg
nengo y6_nef_attitude.py   # Stage 2  ->  y6_nef_attitude.cfg
nengo y6_snn_attitude.py   # Stage 3  ->  y6_snn_attitude.cfg
nengo y6_snn_integral.py   # Stage 4  ->  y6_snn_integral.cfg
nengo y6_nef_position.py   # Stage 5  ->  y6_nef_position.cfg
nengo y6_nef_allocation.py # Stage 6  ->  y6_nef_allocation.cfg
nengo y6_pes_ftc.py        # Stage 7  ->  y6_pes_ftc.cfg
nengo y6_full_snn.py       # Stage 8  ->  y6_full_snn.cfg
```

---

## 14. Performance guide

| Stage | Recommended dt | ~Wall time (12 s sim) | Notes |
|---|---|---|---|
| 1 Node baseline | 10 ms | < 5 s | No neurons, instant |
| 2 NEF attitude | 10 ms | ~10 s | Rate neurons, fast |
| 3 SNN attitude | 1 ms | ~60 s | LIF, needs small dt |
| 4 SNN + integrators | 1 ms | ~90 s | Extra recurrent ensembles |
| 5 NEF position | 1 ms | ~3 min | Large yaw-mix ensemble |
| 6 NEF allocation | 1 ms | ~4 min | 4-D allocation ensemble |
| 7 PES FTC | 1 ms | ~6 min | 18 s sim default |
| 8 Full SNN | 1 ms | ~8 min | ~8 400 LIF neurons |

To speed up stages 3–8 for development, reduce neuron counts:
```python
# Quick iteration preset
net = build_network(n_neurons=150)          # stages 3–4
net = build_network(n_cmd=200)              # stage 6
net = build_network(n_pos_xy=150, n_yaw_mix=200, n_alt=80,
                    n_att=150, n_cmd=200)   # stage 8
```

---

## 15. Changelog

- **v0.1** — Baseline runners: `run_baseline.py` (3 hardcoded scenarios) and
  `y6_nengo.py` (Nengo Node baseline).
- **v0.2** — Full progressive SNN roadmap Stages 2–8; Nengo GUI `.cfg` files;
  comprehensive Python API documentation for every `build_network` signature;
  cross-stage comparison recipe; custom mission library; performance guide.
