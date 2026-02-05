import numpy as np
from prm.prm_algorithm_v2 import get_prm_graph, add_checkpoint
from prm.grid_graph import get_grid_graph
from prm.node import Node
from aStar.aStar import aStarAlgo
from environmentBuilder.getWalls import getLayout
import time
from Task_2D.pid_2d import run_pid_2d, run_mppi_2d
import matplotlib.pyplot as plt
from CONSTANTS import MAP_HEIGHT, MAP_WIDTH, EASY_LOCATIONS, MEDIUM_LOCATIONS, HARD_LOCATIONS
from visualization.visualization import draw_tool
from visualization.visualization3d import draw_tool_3d


def get_dx_from_max_nodes(max_nodes):

    dx = np.sqrt((MAP_HEIGHT * MAP_WIDTH)/max_nodes)
    return dx


class Task:
    def __init__(self, list_of_checkpoints):

        self.list_of_checkpoints = list_of_checkpoints #list of position with position as np.array(3,)
        self.graph_creator = "PRM"
        self.path_finder = "aStar"
        self.local_planner = "PID"
        self.global_layout_id = "easy"
        self.local_layout_id = "sparse" #determine random obstacles only visible to local planner
        self.mppi_horizon = 25
        self.mppi_rollouts = 200
    
    def set_graph_createor(self, graph_creator):
        if graph_creator in ["PRM", "Grid"]:
            self.graph_creator = graph_creator
        else:
            raise ValueError(f"Unknown graph creator {graph_creator}")
    
    def set_path_finder(self, path_finder):
        if path_finder == "aStar":
            self.path_finder = path_finder
        elif path_finder == "Dijkstra":
            self.path_finder = path_finder
        else:
            raise ValueError("Unknown path finder")

    def set_path_finder(self, path_finder):
        if path_finder == "aStar":
            self.path_finder = path_finder
        elif path_finder == "Dijkstra":
            self.path_finder = path_finder
        else:
            raise ValueError("Unknown path finder")

    def set_local_planner(self, local_planner):
        if local_planner == "PID":
            self.local_planner = "PID"
        elif local_planner == "MPPI":
            self.local_planner = "MPPI"
        else:
            raise ValueError(f"Unknown local planner {local_planner}")
        
    def set_environment(self, environment, obstacles_density):
        if environment not in ["easy", "medium", "difficult"]:
            raise ValueError(f"Unknown environment type: {environment}")
        if obstacles_density not in ["none", "sparse", "crowded"]:
            raise ValueError(f"Unknown obstacles_density type: {obstacles_density}")
        
        self.global_layout_id = environment # will be "easy" for example
        
        self.local_layout_id = obstacles_density # will be "none" for PRM and anything else for "MPPI"

    def set_mppi_horizon(self, horizon):
        self.mppi_horizon = horizon
    
    def set_mppi_rollouts(self, rollouts):
        self.mppi_rollouts = rollouts
                  
    def get_graph(self, walls, max_nodes=4000):
        if self.graph_creator == "PRM":
            
            nodes_list = get_prm_graph(walls, max_nodes=max_nodes)
            return nodes_list
        
        elif self.graph_creator == "Grid":
            dx = get_dx_from_max_nodes(max_nodes)
            
            return get_grid_graph(walls, dx=dx)

        else:
            raise ValueError("Unknown graph type")
        
    def follow_path_nodes(self, path_nodes, start_position, walls_local, speed=1, dt=0.1, verbose=False):
        if self.local_planner == "PID":
            
            x, t_travel, collision = run_pid_2d(path_nodes, start_position, walls_local, speed=speed, dt=dt, verbose=verbose)
            return x, t_travel, collision
        
        elif self.local_planner == "MPPI": 
            
            x, t_travel, collision = run_mppi_2d(path_nodes, walls_local, self.mppi_horizon, self.mppi_rollouts) # TO DO: integrate start position
            return x, t_travel, collision
        else:
            raise NotImplementedError()


class Results:
    def __init__(self):
        self.t_graph_creation = 0.0
        self.t_path_find = 0.0
        self.t_travel = 0.0
        self.checkpoints_reached = 0
        self.path_length_planner = 0.0
        self.path_length = 0.0
        self.collision = False
        self.connection_failed = False
        

    def print_results(self):
        if self.connection_failed:
            print("Connection failed! Aborting")
            print(f"Checkpoints connected: {self.checkpoints_reached}")
        else:
            print(f"Computation time:")
            print(f"Graph Creation: {self.t_graph_creation * 1000 :.2f}ms")
            print(f"Path finding: {self.t_path_find * 1000 :.2f}ms")
            print(f"Path length of planner: {self.path_length_planner :.2f}m")
            print(f"Path length followed: {self.path_length :.2f}m")
            print(f"Checkpoints reached by global planner: {self.checkpoints_reached}")
            print(f"Total travel time: {self.t_travel :.2f}s")
            if self.collision:
                print("Encountered collision!")
            else:
                print("Collision Free!")

              

def execute_task(task: Task, max_nodes=2000, visualize=False, verbose=False):
    results = Results()

    walls_global = getLayout(task.global_layout_id)
    walls_local = getLayout(
        layout_id=task.global_layout_id, 
        random_obstacles=task.local_layout_id, 
        bounding_walls=True,
        blockingPositions=task.list_of_checkpoints
        )

    if visualize:
        draw_tool(
            goal_list=task.list_of_checkpoints,
            walls=walls_global
        )

    ### Create graph:

    t1 = time.perf_counter() #Timer

    nodes_list = task.get_graph(walls=walls_global, max_nodes=max_nodes)
    
    t2 = time.perf_counter() #Timer
    results.t_graph_creation = t2 - t1

    if visualize:
        draw_tool(
            nodes_list=nodes_list, 
            goal_list=task.list_of_checkpoints,
            walls=walls_global
        )


    ### Create Path
    num_checkpoints = len(task.list_of_checkpoints) # Number of checkpoints

    # first checkpoint = start position
    start_position = task.list_of_checkpoints[0]
    r = 1.5
    if task.graph_creator == "Grid":
        r = get_dx_from_max_nodes(max_nodes)
    nodes_list, connected, start_id = add_checkpoint(
        nodes_list, 
        checkpoint_position=start_position, walls=walls_global, r=r)
    
    if not connected:
        print("Couldn't connect start node")
        return results
    
    #start_node = Node(id=start_id, position=start_position)
    path_nodes = []
    positions_visited = []
    
    for i in range(num_checkpoints-1):
        
        goal_position = task.list_of_checkpoints[i+1]

        # Add checkpoint: next goal
        nodes_list, connected, goal_id = add_checkpoint(
            nodes_list, 
            checkpoint_position=goal_position, walls=walls_global, r=r)
        
        if not connected:
            print("Couldn't connect next checkpoint")
            break
        
        ## Get path for next segment (segment meaning between two checkpoints)
        ## Search graph with A* or Dijkstra:
        graph_search_type = task.path_finder
        t3 = time.perf_counter() #Timer
        new_path_nodes, new_total_cost = aStarAlgo(start_id, goal_id, nodes_list, graph_search_type=graph_search_type)
        t4 = time.perf_counter() #Timer
        if new_path_nodes is None:
            print("Graph not connected, impossible to find path")
            break

        if i > 0:
            new_path_nodes = new_path_nodes[1:]
        
        path_nodes.extend(new_path_nodes)

        ## Follow path with local planner        
        new_positions, t_travel, collision = task.follow_path_nodes(new_path_nodes, start_position=start_position, walls_local=walls_local, verbose=verbose)
        results.collision = collision
        positions_visited.extend(new_positions) # add positions of current segment
        
        if np.linalg.norm(new_positions[-1] - goal_position) > 0.5:
            print("Goal not reached")
            if visualize:
                draw_tool(
                    nodes_list=None, 
                    goal_list=task.list_of_checkpoints,
                    path_nodes=path_nodes,
                    walls=walls_local,
                    positions=positions_visited
                )
            break


        if collision:
            if visualize:
                draw_tool(
                    nodes_list=None, 
                    goal_list=task.list_of_checkpoints,
                    path_nodes=path_nodes,
                    walls=walls_local,
                    positions=positions_visited
                )
            break
        
        # Result calculations and updates
        results.checkpoints_reached += 1 
        results.path_length_planner += new_total_cost
        t_path_find_new = t4 - t3
        results.t_path_find += t_path_find_new
        results.t_travel += t_travel
        for i in range(len(new_positions)-1):
            results.path_length += np.linalg.norm(new_positions[i] - new_positions[i+1])

        # update: prev goal becomes new start
        
        start_position = new_positions[-1]
        start_id = goal_id


        if visualize:
            draw_tool(
                nodes_list=nodes_list, 
                goal_list=task.list_of_checkpoints,
                path_nodes=path_nodes,
                walls=walls_global
            )

    

        if visualize:
            draw_tool(
                nodes_list=None, 
                goal_list=task.list_of_checkpoints,
                path_nodes=path_nodes,
                walls=walls_local,
                positions=positions_visited)
            
        # draw_tool_3d(
        #     nodes_list=nodes_list, 
        #     goal_list=task.list_of_checkpoints,
        #     path_nodes=path_nodes,
        #     walls=walls_local,
        #     positions=positions)

    
    return results


            

        
