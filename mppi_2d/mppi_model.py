import numpy as np
import CONSTANTS as C

## Diagonal inertia matrix
Ix = C.IX
Iy = C.IY
Iz = C.IZ

## PD tuning
kp = C.KP_ATT
kd = C.KD_ATT

## Reference angular velocities
p_ref = C.P_REF
q_ref = C.Q_REF
r_ref = C.R_REF
g = C.G


def double_integrator_dynamics(x, u, dt):
    """
    Flat double-integrator:
    p_{t+1} = p_t + v_t dt
    v_{t+1} = v_t + a_t dt

    State: x = [px,py,pz, vx,vy,vz]
    Input: u = [ax,ay,az]
    """
    p = x[0:3]
    v = x[3:6]
    a = u[0:3]

    p_next = p + v * dt
    v_next = v + a * dt

    x_next = np.zeros(x.shape[0])
    x_next[0:3] = p_next
    x_next[3:6] = v_next

    return x_next


def accel_to_attitude(a_des, psi_des):
    """
    Differential-flatness mapping (small-angle approx):
    phi_c, theta_c from desired acceleration and yaw.

    a_des = [ddot{x}, ddot{y}, ddot{z}] (world)
    psi_des = [psi_des]
    """
    ddx, ddy, ddz = a_des
    phi_c   = (ddx * np.sin(psi_des) - ddy * np.cos(psi_des)) / g
    theta_c = (ddx * np.cos(psi_des) + ddy * np.sin(psi_des)) / g

    # clip angles (rad) to avoind warnings from enormous angular acceleration
    phi_c   = np.clip(phi_c, -0.4, 0.4)     # ±23 deg
    theta_c = np.clip(theta_c, -0.4, 0.4)

    return phi_c, theta_c


def attitude_PD(phi, theta, psi, p, q, r, phi_des, theta_des, psi_des):
    """
    Simple PD controller in angles + rates (rates_des = 0)
    tau = Kp*(angle_err) + Kd*(rate_err) = u2 (inner loop)
    """
    ang_err = np.array([phi_des - phi, theta_des - theta, psi_des - psi])
    rate_err = np.array([p_ref - p, q_ref - q, r_ref - r])
    tau = kp * ang_err + kd * rate_err
    return tau  # [tau_phi, tau_theta, tau_psi]


def quad_12d_model(x, a_des, psi_des, dt):
    """
    State: x = [px,py,pz,  vx,vy,vz,  phi,theta,psi,  p,q,r]
    a_des: desired translational accel from MPPI
    psi_des: desired yaw (path-aligned)

    Implements:
    1. position: double integrator with accel input
    2. attitude: a_des,psi_des -> phi_c,theta_c -> PD -> torques -> angular accel -> integration
    """
    pos = x[0:3]
    vel = x[3:6]
    phi, theta, psi = x[6:9]
    p, q, r = x[9:12]

    # 1) translational update ("flat" accel input)
    pos_next = pos + vel * dt
    vel_next = vel + a_des * dt

    # 2) desired angles from accel & yaw
    phi_des, theta_des = accel_to_attitude(a_des, psi_des)

    # 3) attitude controller -> torques
    tau = attitude_PD(phi, theta, psi, p, q, r, phi_des, theta_des, psi_des)

    # 4) angular accelerations 
    p_dot = tau[0] / Ix; p_dot = np.clip(p_dot, -50.0, 50.0)
    q_dot = tau[1] / Iy; q_dot = np.clip(q_dot, -50.0, 50.0)
    r_dot = tau[2] / Iz; r_dot = np.clip(r_dot, -50.0, 50.0)

    # 5) integrate angular velocities
    p_next = p + p_dot * dt
    q_next = q + q_dot * dt
    r_next = r + r_dot * dt

    # 6) integrate angles
    phi_next   = phi + p_next * dt
    theta_next = theta + q_next * dt
    psi_next   = psi + r_next * dt

    x_next = np.zeros_like(x)
    x_next[0:3] = pos_next
    x_next[3:6] = vel_next
    x_next[6:9] = np.array([phi_next, theta_next, psi_next])
    x_next[9:12] = np.array([p_next, q_next, r_next])

    return x_next, (phi_des, theta_des)

