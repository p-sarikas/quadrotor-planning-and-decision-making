from Task_2D.task_wrapper import Task, execute_task, Results
from CONSTANTS import HARD_LOCATIONS, EASY_LOCATIONS, MEDIUM_LOCATIONS, SEED
import numpy as np
import os
from Task_2D.data_handling import save_data_npz



def evaluate_graph_creation(environment="easy", max_nodes = [500, 1000]):
    # running PRM and Grid several times for different num_nodes and comparing success rate and path length
    locations = None
    tot_checkpoints = 0
    if environment == "easy":
        locations = EASY_LOCATIONS
    elif environment == "medium":
        locations = MEDIUM_LOCATIONS
    elif environment == "difficult":
        locations = HARD_LOCATIONS
    else:
        raise ValueError(f"Unknown environment {environment}")
    
    tot_checkpoints = len(locations)
    #max_nodes = [500, 1000 ,2000, 4000]
    
    chpts_prm_avgs = [] # checkpoints reached on average for each num_nodes
    chpts_prm_raw = []
    path_prm_avgs = [] # average pathlength of successfull runs for each num_nodes
    path_prm_raw = []
    t_graph_prm_avgs = []

    chpts_grid_avgs = [] #same but grid
    
    path_grid_avgs = [] # same but grid
    
    t_graph_grid_avgs = []

    for num_nodes in max_nodes:

        ### PRM:
        checkpoints_reached_prm = []
        path_length_of_successful_prm = []
        t_graph_prm = []

        for i in range(5):
            # Creat Task
            task0 = Task(locations)
            task0.set_environment(environment=environment, obstacles_density="none")
            task0.set_graph_createor("PRM")
            task0.set_local_planner("PID")
            results0 = execute_task(task0, max_nodes=num_nodes)

            t_graph_prm.append(results0.t_graph_creation)
            checkpoints_reached_prm.append(results0.checkpoints_reached)
            if results0.checkpoints_reached == tot_checkpoints-1:
                path_length_of_successful_prm.append(results0.path_length_planner)

            print(f"Finished: {i}/5 for {num_nodes} in env: {env}")

        chpts_prm_avgs.append(float(np.mean(checkpoints_reached_prm)/(tot_checkpoints-1))*100) #devided by total number of checkpoints, *100 to make %
        chpts_prm_raw.append([(x / (tot_checkpoints-1))*100 for x in checkpoints_reached_prm])
        path_prm_avgs.append(float(np.mean(path_length_of_successful_prm)))
        path_prm_raw.append(path_length_of_successful_prm)
        t_graph_prm_avgs.append(float(np.mean(t_graph_prm))*1000) #ms


        ### Grid (since it is deterministic not multiple runs per num_nodes)

        task0 = Task(locations)
        task0.set_environment(environment=environment, obstacles_density="none")
        task0.set_graph_createor("Grid")
        task0.set_local_planner("PID")
        results0 = execute_task(task0, max_nodes=num_nodes)

        chpts_grid_avgs.append((results0.checkpoints_reached/(tot_checkpoints-1)) * 100)
        t_graph_grid_avgs.append(results0.t_graph_creation)
        if results0.checkpoints_reached == tot_checkpoints-1:
            path_grid_avgs.append(results0.path_length_planner)

        else:
           path_grid_avgs.append(0.0)


    return chpts_prm_avgs, chpts_prm_raw, path_prm_avgs, path_prm_raw, t_graph_prm_avgs, chpts_grid_avgs, path_grid_avgs, t_graph_grid_avgs




np.random.seed(SEED)

# Running experiments and saving data
# Creating directory:
base_path = os.path.join("Task_2D", "exp_data", "graph_creation")
os.makedirs(base_path, exist_ok=True)

#' Easy environment
env = "easy"

save_path = os.path.join(base_path, env)
os.makedirs(save_path, exist_ok=True)

# Commented out to avoid rerunning (time consuming), results already saved
#max_nodes = [250, 500, 1000]
#chpts_prm_avgs, chpts_prm_raw, path_prm_avgs, path_prm_raw, t_graph_prm_avgs, chpts_grid_avgs, path_grid_avgs, t_graph_grid_avgs = evaluate_graph_creation(environment="easy", max_nodes=max_nodes)
#file_name = "data_graph_easy.npz"
#file_path = os.path.join(save_path, file_name)
#save_data_npz(file_path, chpts_prm_avgs, chpts_prm_raw, path_prm_avgs, path_prm_raw, t_graph_prm_avgs, chpts_grid_avgs, path_grid_avgs, t_graph_grid_avgs, max_nodes)

# Medium env:
env = "medium"
save_path = os.path.join(base_path, env)
os.makedirs(save_path, exist_ok=True)
max_nodes = [250, 500, 1000, 2000]
chpts_prm_avgs, chpts_prm_raw, path_prm_avgs, path_prm_raw, t_graph_prm_avgs, chpts_grid_avgs, path_grid_avgs, t_graph_grid_avgs = evaluate_graph_creation(environment=env, max_nodes=max_nodes)
file_name = "data_graph_medium_v2.npz"
file_path = os.path.join(save_path, file_name)
save_data_npz(file_path, chpts_prm_avgs, chpts_prm_raw, path_prm_avgs, path_prm_raw, t_graph_prm_avgs, chpts_grid_avgs, path_grid_avgs, t_graph_grid_avgs, max_nodes)


# # Difficult env:
# env = "difficult"
# save_path = os.path.join(base_path, env)
# os.makedirs(save_path, exist_ok=True)
# max_nodes = [500, 1000, 2500, 5000]
# chpts_prm_avgs, chpts_prm_raw, path_prm_avgs, path_prm_raw, t_graph_prm_avgs, chpts_grid_avgs, path_grid_avgs, t_graph_grid_avgs = evaluate_graph_creation(environment=env, max_nodes=max_nodes)
# file_name = "data_graph_difficult_v2.npz"
# file_path = os.path.join(save_path, file_name)
# save_data_npz(file_path, chpts_prm_avgs, chpts_prm_raw, path_prm_avgs, path_prm_raw, t_graph_prm_avgs, chpts_grid_avgs, path_grid_avgs, t_graph_grid_avgs, max_nodes)




