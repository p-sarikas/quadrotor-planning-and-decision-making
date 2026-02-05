from Task_2D.task_wrapper import Task, execute_task
from CONSTANTS import HARD_LOCATIONS, MEDIUM_LOCATIONS, EASY_LOCATIONS, SEED
import numpy as np
from Task_2D.interactive_arg_parse import parse_args
import warnings

args = parse_args()

np.random.seed(args.seed)

locations = None

if args.environment == "easy":
    locations = EASY_LOCATIONS
    environment = args.environment
elif args.environment == "medium":
    locations = MEDIUM_LOCATIONS
    environment = args.environment
elif args.environment == "difficult":
    locations = HARD_LOCATIONS
    environment = args.environment
else:
    raise ValueError("Unknown environment")

if (args.obstacle_density == "none") or (args.obstacle_density == "sparse") or (args.obstacle_density == "crowded"):
    obstacle_density = args.obstacle_density
else:
    raise ValueError("Unknown obstacle_density")

if (args.graph_creator == "Grid") or (args.graph_creator == "PRM"):
    graph_creator = args.graph_creator
else:
    raise ValueError("Unknown graph_creator")

if (args.search_algorithm == "aStar") or (args.search_algorithm == "Dijkstra"):
    search_algorithm = args.search_algorithm
else:
    raise ValueError("Unknown search_algorithm")

if (args.local_planner == "PID") or (args.local_planner == "MPPI"):
    local_planner = args.local_planner
else:
    raise ValueError("Unknown local_planner")

if (args.max_nodes >= 10000):
    warnings.warn("That is a lot of nodes! Perfomance might be slow...")
elif (args.max_nodes < 999):
    warnings.warn("That is not a lot of nodes, goals might not be reaached in harder environments")

max_nodes = args.max_nodes

if (args.horizon >= 50):
    warnings.warn("That is a long horizon! Perfomance might be slow...")
elif (args.horizon < 10):
    warnings.warn("That is a short horizon, MPPI might get stuck")

horizon = args.horizon

if (args.rollouts >= 500):
    warnings.warn("That is a lot of rollouts! Perfomance might be slow...")
elif (args.rollouts < 100):
    warnings.warn("That is a not a lot of rollouts, MPPI might get stuck")

rollouts = args.rollouts

task0 = Task(locations)
task0.set_environment(environment=environment, obstacles_density=obstacle_density)
task0.set_graph_createor(graph_creator)
task0.set_path_finder(search_algorithm)
task0.set_local_planner(local_planner)
task0.set_mppi_horizon(horizon)
task0.set_mppi_rollouts(rollouts)
results = execute_task(task0,visualize=True, max_nodes=max_nodes)
results.print_results()