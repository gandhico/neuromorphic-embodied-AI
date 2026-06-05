"""
y6_model.py
===========
Single source of truth for the Co-Axial Y6 Tricopter implementation.

Everything here is plain NumPy so it can be (a) driven by a standalone RK4 loop
for validation and (b) wrapped inside Nengo ``Node`` objects without changing the
equations.

References
----------
[M] Mehndiratta & Kayacan, "Reconfigurable Fault-tolerant NMPC for Y6 Coaxial
    Tricopter with Complete Loss of One Rotor", IEEE CCTA 2018.   -> plant + params
[C] Czyba et al., "Development of Co-Axial Y6-Rotor UAV", ICUAS 2015. -> mixer
"""

import numpy as np

# ---------------------------------------------------------------------------
# Parameters  (values from [M], Table I)
# ---------------------------------------------------------------------------
class Params:
    # mass / gravity
    m   = 1.412          # kg          mass of the tricopter
    g   = 9.81           # m/s^2       gravitational acceleration

    # geometry (see PROGRESS.md: l1,l2,l3 are projections of ONE equilateral
    # triangle of radius l1, with one arm pointing along +x_B):
    #   l1 = arm radius (front arm), l2 = l1*sin60 = 0.243, l3 = l1*cos60 = 0.14
    l1  = 0.28           # m           front-arm lever (full arm radius)
    l2  = 0.243          # m           lateral lever of the two back arms
    l3  = 0.14           # m           longitudinal lever of the two back arms

    # inertia tensor
    Ixx = 0.014965       # kg m^2
    Iyy = 0.021915       # kg m^2
    Izz = 0.034433       # kg m^2
    # NOTE: [M] lists Ixz = 0.0261953, but Ixx*Izz - Ixz^2 < 0 with that value,
    # i.e. the inertia tensor would NOT be positive-definite (non-physical).
    # We default to 0.0 (matches Czyba's diagonal assumption). See PROGRESS.md.
    Ixz = 0.0            # kg m^2

    Jp  = 6e-5           # kg m^2      propeller moment of inertia (gyroscopic)

    # aerodynamic coefficients (steady-state momentum-theory lumped model)
    KF   = 6.85e-6       # N s^2       thrust  coefficient   F_i  = KF  * Omega_i^2
    Ktau = 3.35e-7       # N m s^2     drag-moment coeff      tau_i = Ktau* Omega_i^2

    # actuator limit  ([M], Eq. 25)
    omega_max = 900.0    # rad/s


# state layout (12 states), used everywhere:
#   0:X  1:Y  2:Z   3:u  4:v  5:w   6:phi 7:theta 8:psi   9:p 10:q 11:r
IX, IY, IZ, IU, IV, IW, IPHI, ITHE, IPSI, IP, IQ, IR = range(12)


# ---------------------------------------------------------------------------
# Kinematic transforms  ([M], Eq. 2 and Eq. 3)
# ---------------------------------------------------------------------------
def R_EB(phi, theta, psi):
    """Rotation matrix body -> Earth (z-up convention used by [M])."""
    cph, sph = np.cos(phi),   np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi),   np.sin(psi)
    return np.array([
        [cth*cps, sph*sth*cps - sps*cph, cph*sth*cps + sph*sps],
        [cth*sps, sph*sth*sps + cps*cph, cph*sth*sps - sph*cps],
        [-sth,    sph*cth,               cph*cth],
    ])


def T_EB(phi, theta):
    """Euler-rate transform: [phidot,thetadot,psidot] = T_EB * [p,q,r]."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, tth = np.cos(theta), np.tan(theta)
    return np.array([
        [1.0, sph*tth,      cph*tth],
        [0.0, cph,         -sph],
        [0.0, sph/cth,      cph/cth],
    ])


# ---------------------------------------------------------------------------
# Actuator model  ([M], Eq. 6 and Eq. 7)
# ---------------------------------------------------------------------------
def rotor_forces(omega, alpha, P=Params):
    """Per-rotor thrust F_i and drag torque tau_i from commanded speeds.

    omega : (6,) commanded rotor speeds [rad/s]
    alpha : (6,) fault parameters in [0,1] (1 = healthy, 0 = dead rotor)
    returns F (6,), tau_drag (6,), Omega_eff (6,)
    """
    omega = np.asarray(omega, float)
    alpha = np.asarray(alpha, float)
    Omega_eff = alpha * omega            # Eq. 7
    F        = P.KF   * Omega_eff**2     # Eq. 6 (thrust)
    tau_drag = P.Ktau * Omega_eff**2     # Eq. 6 (drag moment)
    return F, tau_drag, Omega_eff


# ---------------------------------------------------------------------------
# Plant: continuous-time state derivative  ([M], Eq. 4, 5, 8, 9, 10, 11)
# ---------------------------------------------------------------------------
def state_derivative(state, omega, alpha, P=Params):
    """xdot = f(x, omega, alpha) for the Y6 coaxial tricopter."""
    u, v, w        = state[IU],   state[IV],   state[IW]
    phi, theta, psi = state[IPHI], state[ITHE], state[IPSI]
    p, q, r        = state[IP],   state[IQ],   state[IR]

    F, tau_drag, Om = rotor_forces(omega, alpha, P)
    F1, F2, F3, F4, F5, F6 = F

    # --- total external force in body frame (Eq. 8) ---
    Fx, Fy = 0.0, 0.0
    Fz = F1 + F2 + F3 + F4 + F5 + F6

    # --- propeller reaction moments (Eq. 10) ---
    tpx = ((F5 + F6) - (F1 + F2)) * P.l2
    tpy = (F3 + F4) * P.l1 - (F1 + F2 + F5 + F6) * P.l3
    tpz = (tau_drag[0] + tau_drag[2] + tau_drag[4]) \
        - (tau_drag[1] + tau_drag[3] + tau_drag[5])

    # --- gyroscopic moments (Eq. 11) ---
    sum_odd  = Om[0] + Om[2] + Om[4]    # top rotors 1,3,5
    sum_even = Om[1] + Om[3] + Om[5]    # bottom rotors 2,4,6
    tgx = P.Jp * q * (sum_even - sum_odd)
    tgy = P.Jp * p * (sum_odd - sum_even)
    tgz = 0.0

    tau_x = tpx + tgx
    tau_y = tpy + tgy
    tau_z = tpz + tgz

    # --- force equations (Eq. 4) ---
    udot = r*v - q*w + P.g*np.sin(theta)                 + Fx/P.m
    vdot = p*w - r*u - P.g*np.sin(phi)*np.cos(theta)     + Fy/P.m
    wdot = q*u - p*v - P.g*np.cos(phi)*np.cos(theta)     + Fz/P.m

    # --- moment equations (Eq. 5), full Ixz-coupled form ---
    D = P.Ixx*P.Izz - P.Ixz**2
    pdot = (1.0/D) * (
        (-p*q*P.Ixz + q*r*(P.Iyy - P.Izz)) * P.Izz
        - (q*r*P.Ixz + p*q*(P.Ixx - P.Iyy)) * P.Ixz
        + tau_x*P.Izz - tau_z*P.Ixz
    )
    qdot = p*r*(P.Izz - P.Ixx)/P.Iyy - (r**2 - p**2)*(P.Ixz/P.Iyy) + tau_y/P.Iyy
    rdot = (1.0/D) * (
        (q*r*P.Ixz + p*q*(P.Ixx - P.Iyy)) * P.Ixx
        - (-p*q*P.Ixz + q*r*(P.Iyy - P.Izz)) * P.Ixz
        + tau_z*P.Ixx - tau_x*P.Ixz
    )

    # --- kinematics (Eq. 1) ---
    pos_dot   = R_EB(phi, theta, psi) @ np.array([u, v, w])
    euler_dot = T_EB(phi, theta)      @ np.array([p, q, r])

    xdot = np.empty(12)
    xdot[IX:IZ+1]    = pos_dot
    xdot[IU:IW+1]    = [udot, vdot, wdot]
    xdot[IPHI:IPSI+1] = euler_dot
    xdot[IP:IR+1]    = [pdot, qdot, rdot]
    return xdot


def rk4_step(state, omega, alpha, dt, P=Params):
    """One fixed-step RK4 integration with zero-order-hold on omega."""
    k1 = state_derivative(state,            omega, alpha, P)
    k2 = state_derivative(state + 0.5*dt*k1, omega, alpha, P)
    k3 = state_derivative(state + 0.5*dt*k2, omega, alpha, P)
    k4 = state_derivative(state +     dt*k3, omega, alpha, P)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# ---------------------------------------------------------------------------
# Control allocation  (Czyba mixer [C, Eq.11] re-mapped to Mehndiratta geometry)
# ---------------------------------------------------------------------------
#  Virtual commands  ->  6 rotor thrusts.  Sign structure is exactly Czyba's
#  mixer; magnitudes use Mehndiratta's lever arms so the commands are physical
#  (T in N, tau in N*m).  Rotor numbering follows [M]: front arm = rotors 3,4;
#  back-right = 1,2; back-left = 5,6; odd = top, even = bottom.
def allocate(T, tau_x, tau_y, tau_z, P=Params):
    """Map (total thrust, body moments) to the 6 per-rotor thrust demands."""
    k1   = 1.0 / (P.l1 + P.l3)
    kpsi = P.KF / (6.0 * P.Ktau)          # converts a yaw moment to a thrust split

    S_back  = (P.l1 * T - tau_y) * k1     # F_BR + F_BL
    F_front = T - S_back                  # F3 + F4
    F_BR    = 0.5 * (S_back - tau_x / P.l2)
    F_BL    = 0.5 * (S_back + tau_x / P.l2)
    delta   = kpsi * tau_z                # per-rotor top/bottom split for yaw

    F = np.array([
        0.5*F_BR    + delta,   # rotor 1 (back-right, top)
        0.5*F_BR    - delta,   # rotor 2 (back-right, bottom)
        0.5*F_front + delta,   # rotor 3 (front, top)
        0.5*F_front - delta,   # rotor 4 (front, bottom)
        0.5*F_BL    + delta,   # rotor 5 (back-left, top)
        0.5*F_BL    - delta,   # rotor 6 (back-left, bottom)
    ])
    return F


def thrust_to_omega(F_cmd, P=Params):
    """Invert F = KF*Omega^2 (assuming healthy rotors), then clip to limits."""
    omega = np.sqrt(np.clip(F_cmd, 0.0, None) / P.KF)
    return np.clip(omega, 0.0, P.omega_max)


# ---------------------------------------------------------------------------
# Cascaded baseline controller (math reference: PD with gravity feedforward)
# ---------------------------------------------------------------------------
class Gains:
    # outer (position) loop
    Kp_xy, Kd_xy = 2.0, 2.6
    Kp_z,  Kd_z  = 9.0, 5.2
    # inner (attitude) loop
    Kp_phi, Kd_phi = 1.3, 0.28
    Kp_the, Kd_the = 1.8, 0.36
    Kp_psi, Kd_psi = 1.0, 0.32
    # safety limits
    tilt_max = np.radians(35.0)           # [M], Eq. 19b/c
    T_max    = 2.5                         # * m g


def position_controller(state, ref, P=Params, G=Gains):
    """Outer loop: position error -> (total thrust T, phi*, theta*, psi*)."""
    x, y, z = state[IX], state[IY], state[IZ]
    phi, theta, psi = state[IPHI], state[ITHE], state[IPSI]
    # world-frame velocity
    vx, vy, vz = R_EB(phi, theta, psi) @ state[IU:IW+1]

    x_des, y_des, z_des, psi_des = ref

    ax_des = G.Kp_xy*(x_des - x) - G.Kd_xy*vx
    ay_des = G.Kp_xy*(y_des - y) - G.Kd_xy*vy

    cps, sps = np.cos(psi), np.sin(psi)
    theta_des =  (ax_des*cps + ay_des*sps) / P.g
    phi_des   =  (ax_des*sps - ay_des*cps) / P.g
    theta_des = np.clip(theta_des, -G.tilt_max, G.tilt_max)
    phi_des   = np.clip(phi_des,   -G.tilt_max, G.tilt_max)

    ez = z_des - z
    T = (P.m*P.g + G.Kp_z*ez - G.Kd_z*vz) / max(np.cos(phi)*np.cos(theta), 0.5)
    T = np.clip(T, 0.0, G.T_max*P.m*P.g)
    return T, phi_des, theta_des, psi_des


def attitude_controller(state, att_des, P=Params, G=Gains):
    """Inner loop: attitude error -> body moments (tau_x, tau_y, tau_z)."""
    phi, theta, psi = state[IPHI], state[ITHE], state[IPSI]
    p, q, r = state[IP], state[IQ], state[IR]
    phi_des, theta_des, psi_des = att_des

    tau_x = G.Kp_phi*(phi_des - phi) - G.Kd_phi*p
    tau_y = G.Kp_the*(theta_des - theta) - G.Kd_the*q
    tau_z = G.Kp_psi*(psi_des - psi) - G.Kd_psi*r
    return tau_x, tau_y, tau_z


def baseline_control(state, ref, P=Params, G=Gains):
    """Full cascade -> 6 rotor speed commands [rad/s]."""
    T, phi_des, theta_des, psi_des = position_controller(state, ref, P, G)
    tau_x, tau_y, tau_z = attitude_controller(
        state, (phi_des, theta_des, psi_des), P, G)
    F_cmd = allocate(T, tau_x, tau_y, tau_z, P)
    omega = thrust_to_omega(F_cmd, P)
    return omega, dict(T=T, phi_des=phi_des, theta_des=theta_des,
                       psi_des=psi_des, tau=(tau_x, tau_y, tau_z))
