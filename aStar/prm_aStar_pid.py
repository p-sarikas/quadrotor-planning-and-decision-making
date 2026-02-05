from prm.prm_algorithm_v2 import get_prm_output
from prm.grid_graph import get_grid_output
from aStar import aStarAlgo
from gym_pybullet_drones.prm_pid_control.pid_v2 import run as run_pidv2
import numpy as np

## Start Node, Goal Node, nodes list
#np.random.seed(5)
nodes_list,start_id, goal_id = get_prm_output()

if goal_id:
    graph_search_type='aStar'
    path_nodes, total_cost = aStarAlgo(start_id, goal_id, nodes_list, graph_search_type=graph_search_type)
    print(path_nodes)
    waypoints = []
    for node in path_nodes:
        waypoints.append(node.position)

    print(len(waypoints))
    run_pidv2(waypoints)

