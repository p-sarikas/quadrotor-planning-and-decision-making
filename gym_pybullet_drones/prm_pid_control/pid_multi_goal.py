"""
Simplified pid.py to one drone, linear trajectory with constant speed

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from prm.prm_algorithm_v2 import get_prm_graph, add_checkpoint
from aStar.aStar import aStarAlgo

class DefaultValues:
    def __init__(self):
        self.DEFAULT_DRONES = DroneModel("cf2x")
        self.DEFAULT_NUM_DRONES = 1
        self.DEFAULT_PHYSICS = Physics("pyb")
        self.DEFAULT_GUI = True
        self.DEFAULT_RECORD_VISION = False
        self.DEFAULT_PLOT = True
        self.DEFAULT_USER_DEBUG_GUI = False
        self.DEFAULT_OBSTACLES = True
        self.DEFAULT_SIMULATION_FREQ_HZ = 240
        self.DEFAULT_CONTROL_FREQ_HZ = 48
        self.DEFAULT_DURATION_SEC = 12
        self.DEFAULT_OUTPUT_FOLDER = 'results'
        self.DEFAULT_COLAB = False

def compute_target(x, waypoints, seg_idx, lookahead):

    # project onto current segment
    p0 = waypoints[seg_idx]
    p1 = waypoints[seg_idx + 1]
    d  = p1 - p0
    L  = np.linalg.norm(d) + 1e-9
    u  = d / L

    s_act = float(np.dot(x - p0, u))
    s_act = float(np.clip(s_act, 0.0, L))

    s = s_act + lookahead

    overshoot = s - L
    
    if overshoot <= 0:
        target_pos = p0 + u * s
        return seg_idx, target_pos
    
    if seg_idx == len(waypoints) - 2: #last segment already
        return seg_idx, p1
    
    p2 = waypoints[seg_idx + 2]
    d_new = p2 - p1
    L_new = np.linalg.norm(d_new) + 1e-9
    u_new = d_new / L_new
    seg_idx += 1
    # move into next segment by overshoot
    target_pos = p1 + u_new * overshoot

    return seg_idx, target_pos



def run(nodes_list, task, walls):
    
    ## Default parameters:
    defaultValues = DefaultValues()
    drone = defaultValues.DEFAULT_DRONES
    physics = defaultValues.DEFAULT_PHYSICS
    gui = defaultValues.DEFAULT_GUI
    record_video = defaultValues.DEFAULT_RECORD_VISION
    plot = defaultValues.DEFAULT_PLOT
    user_debug_gui = defaultValues.DEFAULT_USER_DEBUG_GUI
    obstacles = defaultValues.DEFAULT_OBSTACLES
    simulation_freq_hz = defaultValues.DEFAULT_SIMULATION_FREQ_HZ
    control_freq_hz = defaultValues.DEFAULT_CONTROL_FREQ_HZ
    #duration_sec = defaultValues.DEFAULT_DURATION_SEC
    duration_sec = 200
    output_folder = defaultValues.DEFAULT_OUTPUT_FOLDER
    colab = defaultValues.DEFAULT_COLAB
    r = 1.5
    #### Initialize the simulation #############################
    checkpoints = task.list_of_checkpoints
    num_checkpoints = len(checkpoints)
    start_position = checkpoints[0]

    nodes_list, connected, start_id = add_checkpoint(
        nodes_list, 
        checkpoint_position=start_position, walls=walls, r=r)

    #goal_position = waypoints[-1]

    INIT_XYZS = start_position
    INIT_RPYS = np.array([0, 0,  0]) # start orientation

  
    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=1,
                        initial_xyzs=INIT_XYZS.reshape(1,3),
                        initial_rpys=INIT_RPYS.reshape(1,3),
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    p.resetDebugVisualizerCamera(
        cameraDistance=6.0,              # zoom out/in
        cameraYaw=0.0,                   # rotation around vertical axis (doesn’t matter much top-down)
        cameraPitch=-89.9,               # -90 is straight down
        cameraTargetPosition=[0, 0, 0],  # what you look at
        physicsClientId=PYB_CLIENT
    )
    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=1,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = DSLPIDControl(drone_model=drone)

    #### Run the simulation ####################################
    action = np.zeros((1,4))
    seg_idx = 0
    dt = env.CTRL_TIMESTEP
    lookahead = 0.1 #distance to look ahead as target input for drone
    START = time.time()
    goal_position = checkpoints[1]
    waypoints, goal_id, nodes_list = get_waypoints(nodes_list, goal_position, start_id, walls, r=1.5)
    chpt_idx = 1

    for i in range(0, int(duration_sec*env.CTRL_FREQ)):


        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        x = obs[0][0:3] # actual position of drone
        p0 = waypoints[seg_idx] #start of current segment
        p1 = waypoints[seg_idx + 1] #end of current segment (= start of next)
        if seg_idx + 2 < len(waypoints):
            p2 = waypoints[seg_idx + 2] #end of next segment
        else:
            p2 = goal_position
        #### Compute control for the current way point #############
        seg_idx, target_pos = compute_target(x, waypoints, seg_idx, lookahead)

        if np.linalg.norm(x - p1) < 0.13:
            print(f"Reached Waypoint {seg_idx}")
            if seg_idx + 2 == len(waypoints):
                print("Reached Goal")
                
                if chpt_idx + 1 < num_checkpoints:
                    chpt_idx += 1
                    start_id = goal_id
                    goal_position = checkpoints[chpt_idx]
                    waypoints, goal_id, nodes_list = get_waypoints(nodes_list, checkpoints[chpt_idx], start_id, walls, r=1.5)
                    seg_idx = 0
                else:
                    print("No further goals specified")
                    break


        action[0,:], _, _ = ctrl.computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                state=obs[0],
                                                                target_pos=target_pos,
                                                                target_rpy=INIT_RPYS
                                                                )

        
        
        
        # Log the simulation
        
        logger.log(drone=0,
                    timestamp=i/env.CTRL_FREQ,
                    state=obs[0],
                    control=np.hstack([target_pos, INIT_RPYS, np.zeros(6)])
                    )

        # Sync the simulation
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)
        
        if gui and i % 5 == 0:  # update every 5 control steps to reduce overhead
    
            p.resetDebugVisualizerCamera(
                cameraDistance=4.0,
                cameraYaw=0.0,
                cameraPitch=-89.9,
                cameraTargetPosition=x.tolist(),
                physicsClientId=PYB_CLIENT
            )

    # Close the environment
    env.close()

    # Save the simulation results
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    # Plot the simulation results
    if plot:
        logger.plot()


def get_waypoints(nodes_list, goal_checkpoint, start_id, walls, r=1.5):
    goal_position = goal_checkpoint
    nodes_list, connected, goal_id = add_checkpoint(
                nodes_list, 
                checkpoint_position=goal_position, walls=walls, r=r)

    
    new_path_nodes, new_total_cost = aStarAlgo(start_id, goal_id, nodes_list, graph_search_type="Dijkstra")
    
    waypoints = []
    for node in new_path_nodes:
        waypoints.append(node.position)

    return waypoints, goal_id, nodes_list