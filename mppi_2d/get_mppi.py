import numpy as np
import CONSTANTS as C
from mppi_2d.mppi import MPPI, mppi_cost, count_wall_hits
from mppi_2d.mppi_model import double_integrator_dynamics, quad_12d_model
from mppi_2d.mppi_reference_path import nodes_to_waypoints, resample_polyline, make_ref_sequence, yaw_from_waypoints
from environmentBuilder.is_free import is_free, is_free_xyz


# # Convert node objects to an array of 3D waypoints
# global_ref = resample_polyline(nodes_to_waypoints(path_nodes), ds=C.MPPI_DS)


def get_mppi(global_ref, walls, N=C.MPPI_HORIZON, K=C.MPPI_ROLLOUTS):

    print("\nStarting MPPI...")

    def collision_fn(p_xy):
        x, y = p_xy
        return is_free(x, y, walls)


    ## MPPI params 
    N = N
    K = K
    lamda_ = C.MPPI_LAMBDA
    sigma = C.MPPI_SIGMA
    u_min = C.MPPI_U_MIN
    u_max = C.MPPI_U_MAX
    dt = C.MPPI_DT
    seed = C.SEED
    # ------------------------------------------------------
    # STATE INITIALIZATION
    # MPPI state:
    #   6D flat state used internally by MPPI for planning
    #   [x, y, z, vx, vy, vz]
    x_mppi = np.zeros(6)
    x_mppi[0:3] = global_ref[0]

    # Real system state:
    #   12D state including position, velocity, attitude, and rates
    #   [x, y, z, vx, vy, vz, φ, θ, ψ, p, q, r]
    x_real = np.zeros(12)
    x_real[0:3] = global_ref[0]



    # ------------------------------------------------------
    # MPPI INITIALIZATION
    # MPPI optimizes a sequence of accelerations over a finite horizon using stochastic rollouts and importance sampling
    mppi = MPPI(horizon = N, rollouts = K, lamda_= lamda_,
                sigma = sigma, u_min = u_min, u_max = u_max, dt = dt,
                cost_fn=lambda X, U, ref: mppi_cost(X, U, ref, collision_fn),
                dynamics_fn = double_integrator_dynamics, seed = seed)
    
    # ------------------------------------------------------
    # CLOSED-LOOP SIMULATION
    T = C.MPPI_SIMULATION_STEPS
    local_traj = np.zeros((T+1, 3))
    local_traj[0] = x_real[0:3]

    ref_index = 0
    
    yaw_hist = np.zeros(T+1)  # for plot
    yaw_hist[0] = x_real[8]   # initial yaw, for plot


    for t in range(T):

        # Build a local reference sequence for the MPPI horizon
        # This is the receding-horizon reference
        ref_seq = make_ref_sequence(global_ref, ref_index, N)

        # Compute the optimal acceleration command using MPPI
        # based on the current flat state
        u0, _, _ = mppi.command(x_mppi, ref_seq)

        # Compute desired yaw angle from the direction of the
        # reference path (path-aligned yaw)
        psi_des = yaw_from_waypoints(ref_seq)

        # Propagate the full 12D quadrotor dynamics:
        #  - translational motion driven by acceleration command
        #  - attitude commands from differential flatness
        #  - PD attitude controller producing torques
        x_real, _ = quad_12d_model(x_real, u0, psi_des, dt)
        yaw_hist[t+1] = x_real[8] # for plot

        # Feed back only position and velocity to MPPI
        # MPPI does not reason about attitude
        x_mppi[:] = x_real[0:6]

        # Store executed trajectory for visualization
        local_traj[t+1] = x_real[0:3]

        # Stop simulation if the vehicle reaches the goal region
        if np.linalg.norm(x_real[0:2] - global_ref[-1, 0:2]) < 0.2:
            local_traj = local_traj[:t+2]
            break
            
        # Advance along the reference path
        ref_index += 1
        if t % 100 == 0:
            print(f"Horizon={N}, rollouts={K}, step: {t}/{T}")

    # #--------------- RESULT----------------------------
    hits = count_wall_hits(local_traj, collision_fn)

    return local_traj, yaw_hist, hits

