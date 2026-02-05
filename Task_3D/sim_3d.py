from Task_2D.task_wrapper import Task, get_dx_from_max_nodes
from CONSTANTS import HARD_LOCATIONS, MEDIUM_LOCATIONS, EASY_LOCATIONS, SEED
import numpy as np
from environmentBuilder.getWalls import getLayout
from prm.prm_algorithm_v2 import add_checkpoint
from aStar.aStar import aStarAlgo
#from gym_pybullet_drones.prm_pid_control.pid_v2 import run as run_pidv2
from gym_pybullet_drones.prm_pid_control.pid_multi_goal import run as run_multi


environment = "easy" # choose out of "easy", "medium", "difficult"
obstacle_density = "none" # choose from "none", "sparse", "crowded"
graph_creator = "PRM" # choose from "Grid", "PRM"
search_algorithm = "aStar" # choose from "aStar", "Dijkstra"
local_planner = "PID" # choose from "MPPI", "PID"
max_nodes = 2000 # select the number of nodes the graph consists of 



np.random.seed(0)
locations = None

if environment == "easy":
    locations = EASY_LOCATIONS
elif environment == "medium":
    locations = MEDIUM_LOCATIONS
elif environment == "difficult":
    locations = HARD_LOCATIONS
else:
    raise ValueError("Unknown environment")


task = Task(locations)
task.set_environment(environment=environment, obstacles_density=obstacle_density)
task.set_graph_createor(graph_creator)
task.set_path_finder(search_algorithm)
task.set_local_planner(local_planner)

walls_global = getLayout(task.global_layout_id)
walls_local = getLayout(
        layout_id=task.global_layout_id, 
        random_obstacles=task.local_layout_id, 
        bounding_walls=True,
        blockingPositions=task.list_of_checkpoints
        )
nodes_list = task.get_graph(walls=walls_global, max_nodes=max_nodes)
start_position = task.list_of_checkpoints[0]

# r = 1.5
# if task.graph_creator == "Grid":
#     r = get_dx_from_max_nodes(max_nodes)
# nodes_list, connected, start_id = add_checkpoint(
#     nodes_list, 
#     checkpoint_position=start_position, walls=walls_global, r=r)

# if not connected:
#     print("Couldn't connect next checkpoint")
    
# else:
#     goal_position = task.list_of_checkpoints[1]
#     nodes_list, connected, goal_id = add_checkpoint(
#                 nodes_list, 
#                 checkpoint_position=goal_position, walls=walls_global, r=r)

#     graph_search_type = task.path_finder
#     new_path_nodes, new_total_cost = aStarAlgo(start_id, goal_id, nodes_list, graph_search_type=graph_search_type)
    
#     waypoints = []
#     for node in new_path_nodes:
#         waypoints.append(node.position)

#     print(len(waypoints))
#     run_pidv2(waypoints)

run_multi(nodes_list, task, walls_local)