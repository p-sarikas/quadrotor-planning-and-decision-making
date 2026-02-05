from gym_pybullet_drones.prm_pid_control.pid_v2 import compute_target
import numpy as np
from environmentBuilder.is_free import is_free
from mppi_2d.mppi_reference_path import nodes_to_waypoints, resample_polyline
from mppi_2d.get_mppi import get_mppi
from CONSTANTS import MPPI_DS
import time


def run_pid_2d(path_nodes, start_position, walls, speed=1, dt=0.1, verbose=False):

    waypoints = []
    for node in path_nodes:
        waypoints.append(node.position)

    x = [] #keeping track of positions
    #pos = waypoints[0] #start position = current position x
    pos = start_position
    vel = np.array([0, 0, 0])
    goal_position = waypoints[-1] #goal = last waypoint
    seg_idx = 0
    lookahead = 0.1 #distance to look ahead as target input for drone
    goal_reached = False
    collision = False
    t_travel = 0
    Kp = 100
    Kd = 10
    max_acc = 10
    while not goal_reached:

        # check for collision
        if not is_free(pos[0], pos[1], walls):
            collision = True
            return x, t_travel, collision



        p1 = waypoints[seg_idx + 1]
        x.append(pos)
        ## check if waypoint or goal reached:
        if np.linalg.norm(pos - p1) < 0.13:
            if verbose:
                print(f"Reached Waypoint {seg_idx}")
            if seg_idx + 2 == len(waypoints):
                if verbose:
                    print("Reached Goal")
                goal_reached = True


        p1 = waypoints[seg_idx + 1] #end of current segment (= start of next)
        
            

    
        #### Compute control for the current way point #############
        seg_idx, target_pos = compute_target(pos, waypoints, seg_idx, lookahead)

        d  = target_pos - pos
        L  = np.linalg.norm(d) + 1e-9
        u  = d / L
        target_unit_vector = u
        acc = Kp * d - Kd * vel
        acc_norm = np.linalg.norm(acc)
        if acc_norm > max_acc:
            acc = acc / (acc_norm + 1e-9) * max_acc
            
        vel = vel + dt*acc
        pos = pos + dt * vel
        # pos = pos + speed * dt * target_unit_vector
        t_travel += dt
    return x, t_travel, collision




def run_mppi_2d(path_nodes, walls_local, horizon, rollouts):

    global_ref = resample_polyline(nodes_to_waypoints(path_nodes), ds=MPPI_DS)

    t5 = time.perf_counter()
    local_traj, yaw_hist, hits = get_mppi(global_ref, walls_local, N=horizon, K=rollouts)
    t6 = time.perf_counter()

    positions = [row.reshape(3, ) for row in local_traj]
    t_travel = t6 - t5
    collision = hits > 0

    return positions, t_travel, collision
    

