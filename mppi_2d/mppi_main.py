import numpy as np
import time
import CONSTANTS as C

from mppi_2d.mppi import MPPI, mppi_cost, count_wall_hits
from mppi_2d.mppi_model import double_integrator_dynamics, quad_12d_model
from mppi_2d.mppi_reference_path import nodes_to_waypoints, resample_polyline, make_ref_sequence, yaw_from_waypoints
from mppi_2d.mppi_visualization import plot_mppi_position, plot_mppi_orientation, plot_mppi_tracking_error, plot_mppi_position_3d, plot_vel_and_acc_and_clearance_along_path
from environmentBuilder.is_free import is_free
from environmentBuilder.getWalls import getLayout, distance_to_obstacle, clearance_along_path



def run_mppi(path_nodes, walls, seed=C.SEED, visualize=True, N=C.MPPI_HORIZON, K=C.MPPI_ROLLOUTS):

    print("\nStarting MPPI...")

    ## MPPI params 
    T = C.MPPI_SIMULATION_STEPS
    lamda_ = C.MPPI_LAMBDA
    sigma = C.MPPI_SIGMA
    u_min = C.MPPI_U_MIN
    u_max = C.MPPI_U_MAX
    dt = C.MPPI_DT

    def collision_fn(p_xy):
        x, y = p_xy
        return is_free(x, y, walls)
    
    # Convert node objects to an array of 3D waypoints
    waypoints = nodes_to_waypoints(path_nodes)

    # ------------------------------------------------------
    # REFERENCE PATH PROCESSING
    # Resample the global path so that waypoints are approximately equally spaced in arc-length.
    # This makes the reference suitable for receding-horizon control (MPPI).
    global_ref = resample_polyline(waypoints, ds=C.MPPI_DS)


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
    
    positions  = np.zeros((T+1, 3)); positions[0]  = x_real[0:3]
    velocities = np.zeros((T+1, 3)); velocities[0] = x_real[3:6]
    angles     = np.zeros((T+1, 3)); angles[0]     = x_real[6:9]
    angular_vel= np.zeros((T+1, 3)); angular_vel[0]= x_real[9:12]
    distance   = np.zeros(T+1)
    velocity_xyz= np.zeros(T+1); velocity_xyz[0] = np.linalg.norm(x_real[3:6])
    acceleration_xyz= np.zeros(T+1)
    ref_index = 0
    mppi_times = []   # store per-step MPPI computation time
    error_global_local_path = []   # tracking error (global - local)
    clock_time = None

    t_start_total = time.perf_counter()
    # ------------------------------------------------------
    # CLOSED-LOOP SIMULATION

    for t in range(T):

        # Build a local reference sequence for the MPPI horizon
        # # This is the receding-horizon reference
        ref_seq = make_ref_sequence(global_ref, ref_index, N)

        t0 = time.perf_counter()
        # Compute the optimal acceleration command using MPPI
        # based on the current flat state
        u0, _, _ = mppi.command(x_mppi, ref_seq)

        t1 = time.perf_counter()
        mppi_times.append(t1 - t0)

        # Compute desired yaw angle from the direction of the
        # reference path (path-aligned yaw)
        psi_des = yaw_from_waypoints(ref_seq)

        # Propagate the full 12D quadrotor dynamics:
        #  - translational motion driven by acceleration command
        #  - attitude commands from differential flatness
        #  - PD attitude controller producing torques
        x_real, _ = quad_12d_model(x_real, u0, psi_des, dt)

        # Feed back only position and velocity to MPPI # MPPI does not reason about attitude
        x_mppi[:] = x_real[0:6]

    
        #---------------PLOT & RESULT----------------------------
        positions[t+1]       = x_real[0:3]
        velocities[t+1]      = x_real[3:6]
        angles[t+1]          = x_real[6:9]
        angular_vel[t+1]     = x_real[9:12]

        distance[t+1]= np.round(distance[t] + np.linalg.norm(positions[t+1] - positions[t]), decimals=2)
        velocity_xyz[t+1]    = np.linalg.norm(x_real[3:6])
        acceleration_xyz[t+1]= np.linalg.norm(u0)

        # tracking error    
        e = positions[t+1] - ref_seq[1]   # tracking error at this timestep
        error_global_local_path.append(e)
                  
        if t % 100 == 0:
            print(f"Horizon={N}, rollouts={K}, step: {t}/{T}")



        # Stop simulation if the vehicle reaches the goal region
        if np.linalg.norm(x_real[0:3] - global_ref[-1, 0:3]) < 0.2:
            end = t + 2
            positions = positions[:end]
            velocities = velocities[:end]
            angles = angles[:end]
            angular_vel = angular_vel[:end]
            distance = distance[:end]
            velocity_xyz = velocity_xyz[:end]
            acceleration_xyz = acceleration_xyz[:end]
            clock_time = np.round((t + 1) * dt, decimals=2)
            break

        ref_index += 1 
        # find closest reference index to current position
        # dists = np.linalg.norm(global_ref[:,0:3] - x_real[0:3], axis=1)
        # closest = np.argmin(dists)
        # # ensure monotonic forward progress
        # ref_index = max(ref_index, closest)

    #---------------PLOT & RESULT----------------------------

    # ------------------------------------------------------
    # TIME CALCULATION
    t_end_total = time.perf_counter()
    computational_time = t_end_total - t_start_total
    mppi_times = np.array(mppi_times)
    
    # ------------------------------------------------------
    # OBSTACLES HITS
    hits = count_wall_hits(positions, collision_fn)
    
    clearance = clearance_along_path(positions, walls)
    # ------------------------------------------------------
    # TRAJECTORY VISUALIZATION  # positions (T,3) positions, yaw(T,1) yaw
    error_global_local_path = np.array(error_global_local_path)
    MPPI_freq_cap = 1.0 / mppi_times.mean()
    # if visualize:
    print(f"\nTotal computational time (planning + control + simulation): {computational_time:.3f} s")

    print(f"Number of wall hits: {hits}")
    print(f"Reaching the total path length of {distance[-1]} [m] in {clock_time} [sec]")

    print("\nMPPI timing statistics:")
    print(f"  Mean  per-step MPPI time: {mppi_times.mean()*1000:.2f} ms")
    print(f"  MPPI frequency capability: {MPPI_freq_cap:.1f} Hz")

        # plot_mppi_position(global_ref, positions, N, K, walls)
        # plot_mppi_position_3d(global_ref, positions, walls, N, K)
        # plot_mppi_orientation(global_ref, positions, angles[:,2], N, K, walls, step=10)
        
        # plot_mppi_tracking_error(error_global_local_path, dt)
        # plot_vel_and_acc_and_clearance_along_path(distance, velocity_xyz, acceleration_xyz, clearance)

    return (positions, velocities, angles, angular_vel, 
            distance, velocity_xyz, acceleration_xyz, 
            hits, clearance, error_global_local_path,
            clock_time, computational_time, MPPI_freq_cap)
