import numpy as np
import matplotlib.pyplot as plt

from prm.grid_graph import get_grid_output
from matplotlib.patches import Rectangle
from prm.node import Node, Edge, EdgeDraw


def aStarAlgo(start_id, goal_id, nodes_list, graph_search_type):
    '''
    Finds the shortest path between start_id and goal_id using the nodes_list graph
    
    :param start_id:   id of start node (object)
    :param goal_id:    id of goal node (object)
    :param nodes_list: graph, a list of total nodes ids
    '''
    # Fast lookup: id -> Node |  dictionary that maps: node_id -> Node object
    nodes_id_dict = {node.id: node for node in nodes_list}

    open_set = set([start_id])
    closed_set = set()

    g = {start_id: 0.0} # Store the cost of getting to each node from the start node
    neighbors = {start_id: None} # Store the neighbor nodes of each node in the path


    while open_set:
        current_id = get_node_with_lowest_f(open_set, g, goal_id, nodes_id_dict, graph_search_type)

        # Goal reached
        if current_id == goal_id:
            path_ids = reconstruct_path(neighbors, current_id)
            path_nodes = [nodes_id_dict[i] for i in path_ids]
            total_cost = g[goal_id]
            return path_nodes, total_cost

        open_set.remove(current_id)
        closed_set.add(current_id)

        current_node = nodes_id_dict[current_id]

        # Explore neighbors via connections (undirected OK)
        for edge in current_node.connections:
            neighbor_id = edge.second_node
            cost = edge.distance

            if neighbor_id in closed_set:
                continue

            tentative_g = g[current_id] + cost

            if tentative_g < g.get(neighbor_id, float("inf")):
                neighbors[neighbor_id] = current_id
                g[neighbor_id] = tentative_g
                open_set.add(neighbor_id)

    # open_set is empty but goal was never reached
    return None, float("inf")


def reconstruct_path(parents, current_id):
    '''
    Rebuilds the final path after the goal is reached.
    '''
    path = []
    while current_id is not None:
        path.insert(0, current_id)
        current_id = parents.get(current_id) # Move to parent
    return path


def heuristic(node_id, goal_id, nodes_id_dict, graph_search_type='aStar'):
    '''
    Heuristic function:
    - A* = Euclidean distance
    - Dijkstra = 0 
    '''
    if graph_search_type == 'aStar':
        pos = nodes_id_dict[node_id].position
        goal_pos = nodes_id_dict[goal_id].position
        h = np.linalg.norm(pos - goal_pos)

    elif graph_search_type == 'Dijkstra':
        h = 0

    else:
        raise ValueError("graph_search_type must be 'aStar' or 'Dijkstra'")

    return h


def get_node_with_lowest_f(open_set, g, goal_id, nodes_id_dict, graph_search_type):
    '''
    Selects the best node to expand next. Best node has the smallest f=g+h
    '''
    best_id = None
    lowest_f_score = float("inf") # infitive initial value

    for node_id in open_set:
        f_score = heuristic(node_id, goal_id, nodes_id_dict, graph_search_type) + g[node_id]
        if f_score < lowest_f_score:
            lowest_f_score = f_score
            best_id = node_id

    return best_id


edges_draw_list = []

def draw_map_aStar(nodes_list, path_nodes, total_cost, graph_search_type):
    fig, ax = plt.subplots()

    # Map dimensions
    map_width, map_height = 10, 10
    ax.set_xlim(0, map_width)
    ax.set_ylim(0, map_height)
    ax.set_aspect('equal')

    # Grid
    ax.set_xticks(range(map_width + 1))
    ax.set_yticks(range(map_height + 1))

    # Obstacle (box)
    box_x, box_y = 3, 3
    box_width, box_height = 5, 2

    box = Rectangle(
        (box_x, box_y),
        box_width,
        box_height,
        edgecolor='black',
        facecolor='gray',
        alpha=0.7
    )
    ax.add_patch(box)

    # --- Draw PRM Nodes ---
    x_scatter = [node.position[0] for node in nodes_list]
    y_scatter = [node.position[1] for node in nodes_list]
    ax.scatter(x_scatter, y_scatter)

    # --- Draw PRM Edges ---
    if edges_draw_list is not None:
        for edge in edges_draw_list:
            x_vals = [edge.start[0], edge.end[0]]
            y_vals = [edge.start[1], edge.end[1]]
            ax.plot(x_vals, y_vals, linewidth=1)

    # --- Draw A* Path ---
    path_x = [node.position[0] for node in path_nodes]
    path_y = [node.position[1] for node in path_nodes]

    if graph_search_type == 'aStar':
        ax.plot(path_x, path_y, linewidth=3, color='blue', marker='o', label="A* Path", zorder=2)
    if graph_search_type == 'Dijkstra':
        ax.plot(path_x, path_y, linewidth=3, color='blue', marker='o', label="Dijkstra Path", zorder=2)

    # --- Highlight Start & Goal ---
    ax.scatter(path_x[0], path_y[0], s=150, color='red', marker='X', label='Start', zorder=5)
    ax.scatter(path_x[-1], path_y[-1], s=100, color='green', marker='X', label='Goal', zorder=5)

    # Labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if graph_search_type == 'aStar':
        ax.set_title(f"A* Shortest Path (Cost = {total_cost:.2f})")
    if graph_search_type == 'Dijkstra':
        ax.set_title(f"Dijkstra Shortest Path (Cost = {total_cost:.2f})")
    ax.legend()

    plt.show()