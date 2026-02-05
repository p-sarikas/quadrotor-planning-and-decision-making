import numpy as np
import time
import CONSTANTS as C

from aStar.aStar import aStarAlgo
from mppi_3d.mppi import MPPI, mppi_cost, count_wall_hits
from mppi_3d.mppi_model import double_integrator_dynamics, quad_12d_model
from mppi_3d.mppi_reference_path import nodes_to_waypoints, resample_polyline, make_ref_sequence, yaw_from_waypoints
from mppi_3d.mppi_visualization import plot_mppi_position, plot_mppi_orientation, plot_mppi_tracking_error, plot_mppi_position_3d, plot_mppi_orientation_3d
from prm.prm_algorithm_v2 import get_prm_output
from prm.grid_graph import get_grid_output
from environmentBuilder.is_free import is_free, is_free_xyz
from environmentBuilder.getWalls import getLayout



# ------------------INPUTS------------------------------
# Fix random seed for reproducibility of MPPI rollouts
seed = C.SEED
# ------------------------------------------------------
# GLOBAL PLANNING (PRM + A*)
# Generate a graph using PRM and solve for a collision-free global path using A* search
graph_search_type='aStar'
# graph_search_type = 'Dijkstra'
nodes_list, start_id, goal_id = get_prm_output()
# nodes_list, start_id, goal_id = get_grid_output()

# walls = getLayout(["base","bounds","randomObstacles"])
walls = getLayout(layout_id="difficult",  # or "medium" / "difficult"
            random_obstacles="crowded",     # "none" or "sparse" or "crowded"
            bounding_walls=False)
# ------------------INPUTS------------------------------



def run_global_planner(graph_search_type, nodes_list, start_id, goal_id, planner_type="prm"):
    """
    Run global planning (PRM/Grid + A*/Dijkstra).

    Returns
        path_nodes: List of nodes representing the planned path
    """
    if goal_id is None:
        raise ValueError(f"[{planner_type.upper()}] Goal node is None. Planning failed.")
    path_nodes, total_cost = aStarAlgo(start_id, goal_id, nodes_list, graph_search_type=graph_search_type)

    return path_nodes

path_nodes = run_global_planner(graph_search_type=graph_search_type, nodes_list=nodes_list,
                                start_id=start_id, goal_id=goal_id, planner_type="grid")




def run_mppi(path_nodes, walls):


    def collision_fn(p_xyz):
        x, y, z = p_xyz
        return is_free_xyz(x, y, z, walls)
    
    # Convert node objects to an array of 3D waypoints
    waypoints = nodes_to_waypoints(path_nodes)

    # ------------------------------------------------------
    # REFERENCE PATH PROCESSING
    # Resample the global path so that waypoints are approximately equally spaced in arc-length.
    # This makes the reference suitable for receding-horizon control (MPPI).
    dt = C.MPPI_DT
    global_ref = resample_polyline(waypoints, ds=C.MPPI_DS)


    ## MPPI params 
    N = C.MPPI_HORIZON
    K = C.MPPI_ROLLOUTS
    lamda_ = C.MPPI_LAMBDA
    sigma = C.MPPI_SIGMA
    u_min = C.MPPI_U_MIN
    u_max = C.MPPI_U_MAX


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


    t_start_total = time.perf_counter()
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

    mppi_times = []   # store per-step MPPI computation time
    error_global_local_path = []   # tracking error (global - local)

    for t in range(T):

        # Build a local reference sequence for the MPPI horizon
        # This is the receding-horizon reference
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
        yaw_hist[t+1] = x_real[8] # for plot

        # Feed back only position and velocity to MPPI
        # MPPI does not reason about attitude
        x_mppi[:] = x_real[0:6]

        # Store executed trajectory for visualization
        local_traj[t+1] = x_real[0:3]

        # Stop simulation if the vehicle reaches the goal region
        if np.linalg.norm(x_real[0:2] - global_ref[-1, 0:2]) < 0.3:
            local_traj = local_traj[:t+2]
            break
            
        # tracking error    
        e = local_traj[t+1] - ref_seq[1]   # tracking error at this timestep
        error_global_local_path.append(e)
                  
        # Advance along the reference path
        ref_index += 1
        print(f"Horizon={N}, rollouts={K}, step: {t}/{T}")


    #---------------PLOT & RESULT----------------------------

    # ------------------------------------------------------
    # TIME CALCULATION
    t_end_total = time.perf_counter()
    total_runtime = t_end_total - t_start_total
    print(f"\nTOTAL runtime (planning + control + simulation): {total_runtime:.3f} s")

    mppi_times = np.array(mppi_times)
    print("\nMPPI timing statistics:")
    print(f"  Mean  per-step MPPI time: {mppi_times.mean()*1000:.2f} ms")
    print(f"  MPPI frequency capability: {1.0 / mppi_times.mean():.1f} Hz")
    
    # ------------------------------------------------------
    # OBSTACLES HITS
    hits = count_wall_hits(local_traj, collision_fn)
    print(f"Number of wall hits: {hits}")

    # ------------------------------------------------------
    # TRAJECTORY VISUALIZATION  # local_traj (T,3) positions, yaw_hist (T,1) yaw
    plot_mppi_position(global_ref, local_traj, N, K, walls)
    plot_mppi_position_3d(global_ref, local_traj, N, K, walls)
    plot_mppi_orientation(global_ref, local_traj, yaw_hist, N, K, walls, step=10)
    plot_mppi_orientation_3d(global_ref, local_traj, yaw_hist, N, K, walls, step=10)
    error_global_local_path = np.array(error_global_local_path)
    plot_mppi_tracking_error(error_global_local_path, dt)

    '''
    Output variables form run_mppi:
        local_traj:              local planner trajectory
        yaw_hist:                yaw 
        hits:                    number of obstacles hits
        error_global_local_path: global traj - local traj
        total_runtime:
    '''

if __name__ == "__main__":
    run_mppi(path_nodes=path_nodes, walls=walls)
