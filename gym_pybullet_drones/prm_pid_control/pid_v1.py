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

def run():
    
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
    duration_sec = defaultValues.DEFAULT_DURATION_SEC
    output_folder = defaultValues.DEFAULT_OUTPUT_FOLDER
    colab = defaultValues.DEFAULT_COLAB

    #### Initialize the simulation #############################
    H = 0.5 #z value of init - cruising altitude

    start_position = np.array([0.5, 0.5, H])
    goal_position = np.array([1.0, 1.0, H])

    INIT_XYZS = start_position
    INIT_RPYS = np.array([0, 0,  0]) # start orientation?

    #### Initialize a circular trajectory ######################
    PERIOD = 10
    NUM_WP = control_freq_hz*PERIOD + 1
    TARGET_POS = np.linspace(start_position, goal_position, num=NUM_WP)
 
    wp_idx = 0

    

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
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        
        action[0,:], _, _ = ctrl.computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                state=obs[0],
                                                                target_pos=TARGET_POS[wp_idx, :],
                                                                target_rpy=INIT_RPYS
                                                                )

        #### Go to the next way point and loop #####################
        
        wp_idx = min(wp_idx + 1, NUM_WP - 1)
        # Log the simulation
        
        logger.log(drone=0,
                    timestamp=i/env.CTRL_FREQ,
                    state=obs[0],
                    control=np.hstack([TARGET_POS[wp_idx, 0:2], INIT_XYZS[2], INIT_RPYS[:], np.zeros(6)])
                    )

        # Sync the simulation
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    # Close the environment
    env.close()

    # Save the simulation results
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    # Plot the simulation results
    if plot:
        logger.plot()

if __name__ == "__main__":

    run()
