import numpy as np


##################################################
# RANDOM SEED
##################################################
SEED = 10


##################################################
# MPPI PARAMETERS
##################################################
MPPI_HORIZON = 25         # N
MPPI_ROLLOUTS = 200       # K
MPPI_LAMBDA = 5.0
MPPI_SIGMA = (0.8**2) * np.eye(3) # control noise covariance Σ (3x3)
MPPI_U_MIN = -3.0 * np.ones(3)  # acceleration upper bounds, hard constraint
MPPI_U_MAX =  3.0 * np.ones(3)  # acceleration upper bounds, hard constraint
MPPI_SIMULATION_STEPS = 1000     # MAXIMUM SIMULATION STEPS 

# PHYSICAL CONSTANTS
G = 9.81                          # gravity [m/s^2]
IX, IY, IZ = 0.02, 0.02, 0.04     # inertia [kg·m^2]
DRONE_RADIUS = 0.20               # collision radius [m]

# GLOBAL PATH / RESAMPLING
V_REF = 1.0               # [m/s] reference speed along path
MPPI_DT = 0.1             # [s] used only for path resampling
MPPI_DS = V_REF * MPPI_DT      # [m] spacing of resampled global_ref

# ATTITUDE PD CONTROLLER
KP_ATT = np.array([6.0, 6.0, 4.0])   # roll, pitch, yaw
KD_ATT = np.array([3.0, 3.0, 2.0])
P_REF = 0.0
Q_REF = 0.0
R_REF = 0.0

# OBSTACLES (MATCH MPPI COST)
# RECT_W = 0.10 #   half_w = rect_w/2 + drone_radius
# RECT_H = 0.10 #   half_h = rect_h/2 + drone_radius
# MARGIN = 0.05

# inflated half-extents used BOTH by MPPI penalty and PyBullet geometry
# HALF_W = RECT_W / 2 + DRONE_RADIUS + MARGIN
# HALF_H = RECT_H / 2 + DRONE_RADIUS + MARGIN


##################################################
# ENVIRONMENT PARAMETERS
##################################################
# Map size
MAP_WIDTH = 16
MAP_HEIGHT = 16
MAP_LENGTH = 16
DEFAULT_WALL_THICKNESS = 0.25
PILLAR_L = 1.0

EASY_LOCATIONS = [np.array([1,1,1]), np.array([5,13,1]),np.array([11,10,1])]
MEDIUM_LOCATIONS = [np.array([1,1,1]), np.array([6,12,1]),np.array([12,12,1]),np.array([10,4,1])]
HARD_LOCATIONS = [np.array([1,1,1]), np.array([14,2,1]), np.array([2,14,1])]

PATH_LENGTH_EASY = 24.72
PATH_LENGTH_MEDIUM = 30.99
PATH_LENGTH_DIFFICULT = 46.45