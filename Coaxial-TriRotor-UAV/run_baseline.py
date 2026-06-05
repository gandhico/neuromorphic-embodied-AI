"""
run_baseline.py
===============
Standalone closed-loop simulation (no Nengo) used to VALIDATE the equations in
y6_model.py before they are wrapped in Nengo Nodes. Runs three scenarios:
  1. pure hover
  2. step to a 3-D waypoint
  3. a square trajectory + yaw
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from y6_model import (Params, Gains, rk4_step, baseline_control,
                      IX, IY, IZ, IPHI, ITHE, IPSI)

P, G = Params, Gains
DT = 0.01


def simulate(ref_fn, T_end, alpha_fn=None, x0=None):
    n = int(T_end / DT)
    state = np.zeros(12) if x0 is None else np.array(x0, float)
    log = dict(t=np.zeros(n), state=np.zeros((n, 12)),
               omega=np.zeros((n, 6)), ref=np.zeros((n, 4)))
    for k in range(n):
        t = k * DT
        ref = ref_fn(t)
        alpha = np.ones(6) if alpha_fn is None else alpha_fn(t)
        omega, _ = baseline_control(state, ref, P, G)
        state = rk4_step(state, omega, alpha, DT, P)
        log["t"][k] = t
        log["state"][k] = state
        log["omega"][k] = omega
        log["ref"][k] = ref
    return log


def rms(e):
    return float(np.sqrt(np.mean(e**2)))


# --- scenario 1: hover at 1.5 m ---
log_h = simulate(lambda t: (0, 0, 1.5, 0), T_end=10.0)

# --- scenario 2: step to (1, -0.5, 2.0), yaw 20 deg ---
def ref_step(t):
    return (1.0, -0.5, 2.0, np.radians(20)) if t > 1.0 else (0, 0, 1.5, 0)
log_s = simulate(ref_step, T_end=15.0)

# --- scenario 3: square trajectory at 1.5 m ---
def ref_square(t):
    if t < 3:   return (0, 0, 1.5, 0)
    seg = int((t - 3) // 5) % 4
    corners = [(1, 0), (1, 1), (0, 1), (0, 0)]
    cx, cy = corners[seg]
    return (cx, cy, 1.5, 0)
log_q = simulate(ref_square, T_end=28.0)

# ---------------------------------------------------------------- report
print("=== VALIDATION SUMMARY ===")
hov = log_h["state"][-200:]
print(f"[hover]  final pos = ({hov[-1,IX]:+.4f}, {hov[-1,IY]:+.4f}, "
      f"{hov[-1,IZ]:+.4f}) m   target (0,0,1.5)")
print(f"[hover]  z RMS error (last 2 s) = {rms(hov[:,IZ]-1.5)*1000:.2f} mm")
print(f"[hover]  attitude drift |phi,theta,psi| max = "
      f"{np.max(np.abs(hov[:, [IPHI,ITHE,IPSI]]))*1e6:.2f} urad")

ss = log_s["state"][-200:]
print(f"[step]   final pos = ({ss[-1,IX]:+.4f}, {ss[-1,IY]:+.4f}, "
      f"{ss[-1,IZ]:+.4f}) m   target (1,-0.5,2.0)")
print(f"[step]   final yaw = {np.degrees(ss[-1,IPSI]):+.3f} deg  target 20")

err = log_q["state"][:, [IX,IY,IZ]] - log_q["ref"][:, :3]
print(f"[square] tracking RMSE x/y/z = "
      f"{rms(err[:,0]):.3f}/{rms(err[:,1]):.3f}/{rms(err[:,2]):.3f} m")
print(f"[all] omega range over runs = "
      f"[{log_q['omega'].min():.1f}, {log_q['omega'].max():.1f}] rad/s "
      f"(limit {P.omega_max})")

# ---------------------------------------------------------------- plots
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0,0].plot(log_h["t"], log_h["state"][:,IZ], label="z")
ax[0,0].axhline(1.5, ls="--", c="k", lw=.8); ax[0,0].set_title("Hover: altitude")
ax[0,0].set_xlabel("t [s]"); ax[0,0].set_ylabel("z [m]"); ax[0,0].legend()

for i,lbl in zip([IX,IY,IZ], ["x","y","z"]):
    ax[0,1].plot(log_s["t"], log_s["state"][:,i], label=lbl)
ax[0,1].plot(log_s["t"], log_s["ref"][:,0], "k--", lw=.6)
ax[0,1].plot(log_s["t"], log_s["ref"][:,1], "k--", lw=.6)
ax[0,1].plot(log_s["t"], log_s["ref"][:,2], "k--", lw=.6)
ax[0,1].set_title("Step response"); ax[0,1].set_xlabel("t [s]"); ax[0,1].legend()

ax[1,0].plot(log_q["state"][:,IX], log_q["state"][:,IY], label="path")
ax[1,0].plot(log_q["ref"][:,0], log_q["ref"][:,1], "k--", lw=.8, label="ref")
ax[1,0].set_title("Square trajectory (x-y)"); ax[1,0].axis("equal")
ax[1,0].set_xlabel("x [m]"); ax[1,0].set_ylabel("y [m]"); ax[1,0].legend()

ax[1,1].plot(log_q["t"], log_q["omega"])
ax[1,1].axhline(P.omega_max, ls="--", c="r", lw=.8)
ax[1,1].set_title("Rotor speeds (square run)")
ax[1,1].set_xlabel("t [s]"); ax[1,1].set_ylabel("omega [rad/s]")

plt.tight_layout()
plt.savefig("baseline_validation.png", dpi=110)
print("\nsaved baseline_validation.png")
