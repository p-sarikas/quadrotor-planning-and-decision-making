from task_wrapper import Task, execute_task, Results
from CONSTANTS import HARD_LOCATIONS, EASY_LOCATIONS, MEDIUM_LOCATIONS, SEED
import numpy as np
import os
from Task_2D.data_handling import save_time_data_npz


def eval_time():
    """
    get compute time for: grid creation (PID), prm creation (PID), dijkstra (PRM, PID), A* (PRM, PID)
    - run grid, a*, pid
    - run prm, a*, pid
    - run prm, dijkstra, pid
    """
    t_prm_raw, t_grid_raw, t_aStar_raw, t_dijkstra_raw = [], [], [], []

    # easy, max_nodes = 1000
    max_nodes = 1000
    env = "easy"

    t_graph, t_search = eval_time_helper(env=env, max_nodes=max_nodes, graph_creator="Grid", graph_search="aStar")
    t_grid_raw.append(t_graph)

    t_graph, t_search = eval_time_helper(env=env, max_nodes=max_nodes, graph_creator="PRM", graph_search="aStar")
    t_prm_raw.append(t_graph)
    t_aStar_raw.append(t_search)

    t_graph, t_search = eval_time_helper(env=env, max_nodes=max_nodes, graph_creator="PRM", graph_search="Dijkstra")
    t_dijkstra_raw.append(t_search)

    # medium, max_nodes = 2000
    max_nodes = 2000
    env = "medium"

    t_graph, t_search = eval_time_helper(env=env, max_nodes=max_nodes, graph_creator="Grid", graph_search="aStar")
    t_grid_raw.append(t_graph)

    t_graph, t_search = eval_time_helper(env=env, max_nodes=max_nodes, graph_creator="PRM", graph_search="aStar")
    t_prm_raw.append(t_graph)
    t_aStar_raw.append(t_search)

    t_graph, t_search = eval_time_helper(env=env, max_nodes=max_nodes, graph_creator="PRM", graph_search="Dijkstra")
    t_dijkstra_raw.append(t_search)

    # difficult, max_nodes = 5000
    max_nodes = 5000
    env = "difficult"

    t_graph, t_search = eval_time_helper(env=env, max_nodes=max_nodes, graph_creator="Grid", graph_search="aStar", local_planner="MPPI")
    t_grid_raw.append(t_graph)

    t_graph, t_search = eval_time_helper(env=env, max_nodes=max_nodes, graph_creator="PRM", graph_search="aStar", local_planner="MPPI")
    t_prm_raw.append(t_graph)
    t_aStar_raw.append(t_search)

    t_graph, t_search = eval_time_helper(env=env, max_nodes=max_nodes, graph_creator="PRM", graph_search="Dijkstra", local_planner="MPPI")
    t_dijkstra_raw.append(t_search)


    return t_prm_raw, t_grid_raw, t_aStar_raw, t_dijkstra_raw


def eval_time_helper(env="easy", max_nodes=1000, graph_creator="PRM", graph_search="aStar", local_planner="PID"):
    locations = None
    tot_checkpoints = 0
    if env == "easy":
        locations = EASY_LOCATIONS
    elif env == "medium":
        locations = MEDIUM_LOCATIONS
    elif env == "difficult":
        locations = HARD_LOCATIONS
    else:
        raise ValueError(f"Unknown environment {env}")
    
    tot_checkpoints = len(locations)

    t_graph = []
    
    t_search = []
    

    for i in range(3):
        # Creat Task
        np.random.seed(SEED + i)
        task0 = Task(locations)
        task0.set_environment(environment=env, obstacles_density="none")
        task0.set_graph_createor(graph_creator)
        task0.set_path_finder(graph_search)
        task0.set_local_planner(local_planner)
        results0 = execute_task(task0, max_nodes=max_nodes)

        if results0.checkpoints_reached == tot_checkpoints-1:
                t_graph.append(results0.t_graph_creation)
                t_search.append(results0.t_path_find)

    print(f"Evaluated env: {env}, for {graph_creator}, {graph_search}")
    return t_graph, t_search


# ## log data:
# save_path = os.path.join("Task_2D", "exp_data", "compute_time")
# os.makedirs(save_path, exist_ok=True)
# # run experiment
# t_prm_raw, t_grid_raw, t_aStar_raw, t_dijkstra_raw = eval_time()
# file_name = "time_data.npz"
# file_path = os.path.join(save_path, file_name)
# save_time_data_npz(file_path, t_prm_raw, t_grid_raw, t_aStar_raw, t_dijkstra_raw)

