import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="2D Drone Navigation Task")

    parser.add_argument("--environment", type=str, default="medium",
                        choices=["easy", "medium", "difficult"],
                        help="Environment complexity")

    parser.add_argument("--obstacle_density", type=str, default="sparse",
                        choices=["none", "sparse", "crowded"],
                        help="Obstacle density")

    parser.add_argument("--graph_creator", type=str, default="PRM",
                        choices=["Grid", "PRM"],
                        help="Graph generation method")

    parser.add_argument("--search_algorithm", type=str, default="aStar",
                        choices=["aStar", "Dijkstra"],
                        help="Search algorithm")

    parser.add_argument("--local_planner", type=str, default="MPPI",
                        choices=["PID", "MPPI"],
                        help="Local planner")

    parser.add_argument("--max_nodes", type=int, default=1500,
                        help="Maximum number of graph nodes")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    parser.add_argument("--horizon", type=int, default=25,
                            help="MPPI horizon")
    
    parser.add_argument("--rollouts", type=int, default=200,
                            help="MPPI rollouts")
 
    return parser.parse_args()
