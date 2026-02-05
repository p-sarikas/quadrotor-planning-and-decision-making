import numpy as np
import matplotlib.pyplot as plt
from aStar import aStarAlgo, draw_map_aStar, get_graph
from prm.prm_algorithm_v2 import get_prm_output
from matplotlib.patches import Rectangle
from prm.node import EdgeDraw
import imageio.v2 as imageio
#np.random.seed(1)



## Run PRM:
from prm.node import Node, Edge, EdgeDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import numpy as np
import os
import imageio.v2 as imageio
import glob


#for visualization: save frames in
os.makedirs("visualization/frames", exist_ok=True)
# Map size
MAP_WIDTH = 10
MAP_HEIGHT = 10

# Rectangle obstacle: bottom-left at (3,3), width=5, height=2
RECT_X = 3
RECT_Y = 3
RECT_W = 5
RECT_H = 2

# Circles: centers and radius
CIRCLE1_CENTER = (3, 3)
CIRCLE2_CENTER = (4, 6)
CIRCLE_RADIUS = 2.0

def is_free(x, y):
    """Return True if (x, y) is outside all obstacles and inside the map."""
    # 1) Check map bounds (optional, but usually useful)
    if not (0 <= x <= MAP_WIDTH and 0 <= y <= MAP_HEIGHT):
        return False  # treat outside map as invalid

    # 2) Check rectangle
    in_rect = (RECT_X <= x <= RECT_X + RECT_W) and (RECT_Y <= y <= RECT_Y + RECT_H)
    if in_rect:
        return False

   

    # If none of the obstacles contain the point, it's free
    return True


def draw_map_frame(frame_path):
    fig, ax = plt.subplots()

    # Map dimensions
    map_width, map_height = 10, 10

    # Draw map boundary
    ax.set_xlim(0, map_width)
    ax.set_ylim(0, map_height)
    ax.set_aspect('equal')

    # Add grid lines
    ax.set_xticks(range(map_width + 1))
    ax.set_yticks(range(map_height + 1))
    

    # Box parameters
    box_x, box_y = 3, 3   # bottom-left corner
    box_width, box_height = 5, 2

    # Draw the box
    box = Rectangle(
        (box_x, box_y),
        box_width,
        box_height,
        edgecolor='black',
        facecolor='gray',
        alpha=0.7,
        zorder=0
    )
    ax.add_patch(box)

    # Nodes:
    x_scatter = []
    y_scatter = []

    for node in nodes_list:
        x_scatter.append(node.position[0])
        y_scatter.append(node.position[1])

    # --- Edges (paths) ---
    if edges_draw_list is not None:
        for edge in edges_draw_list:
            x_vals = [edge.start[0], edge.end[0]]
            y_vals = [edge.start[1], edge.end[1]]
            ax.plot(x_vals, y_vals, linewidth=1, color='blue', alpha=0.3, zorder=1)

    
    # Labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("2D Map 10x10 with Box 5x2 at (3,3)")

    # Nodes
    ax.scatter(x_scatter, y_scatter, zorder=1)

    # Start and goal:
    ax.scatter(
        start_position[0], start_position[1], 
        color="yellow", 
        label="Start", 
        s=50, 
        edgecolors="black", 
        alpha=1, 
        clip_on=False,
        zorder=3
    )
    ax.scatter(goal_position[0], goal_position[1],
               color="green", 
               label="Goal", 
               s=50,
               zorder=3)

    for node in nodes_list:
        if goal_id == node.id:
            ax.scatter(
                node.position[0], node.position[1], 
                color="green",  
                label="Found Goal", 
                s=50, 
                edgecolors="black", 
                alpha=1,
                zorder=4
            )
    ax.legend(loc="upper right")
    #plt.show()

    fig.savefig(frame_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def draw_map_frame_aStar(path_nodes, nodes_list, edges_draw_list, start_position, goal_position, goal_id, frame_path=None, show=False):
    fig, ax = plt.subplots()

    # Map dimensions
    map_width, map_height = 10, 10

    # Draw map boundary
    ax.set_xlim(0, map_width)
    ax.set_ylim(0, map_height)
    ax.set_aspect('equal')

    # Add grid lines
    ax.set_xticks(range(map_width + 1))
    ax.set_yticks(range(map_height + 1))
    

    # Box parameters
    box_x, box_y = 3, 3   # bottom-left corner
    box_width, box_height = 5, 2

    # Draw the box
    box = Rectangle(
        (box_x, box_y),
        box_width,
        box_height,
        edgecolor='black',
        facecolor='gray',
        alpha=0.7,
        zorder=0
    )
    ax.add_patch(box)

    # Nodes:
    x_scatter = []
    y_scatter = []

    for node in nodes_list:
        x_scatter.append(node.position[0])
        y_scatter.append(node.position[1])

    # --- Edges (paths) ---
    if edges_draw_list is not None:
        for edge in edges_draw_list:
            x_vals = [edge.start[0], edge.end[0]]
            y_vals = [edge.start[1], edge.end[1]]
            ax.plot(x_vals, y_vals, linewidth=1, color='blue', alpha=0.3, zorder=1)

    
    # Labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("2D Map 10x10 with Box 5x2 at (3,3)")

    # Nodes
    ax.scatter(x_scatter, y_scatter, zorder=1)

    # Start and goal:
    ax.scatter(
        start_position[0], start_position[1], 
        color="yellow", 
        label="Start", 
        s=50, 
        edgecolors="black", 
        alpha=1, 
        clip_on=False,
        zorder=6
    )
    ax.scatter(goal_position[0], goal_position[1],
               color="green", 
               label="Goal", 
               s=50,
               zorder=6)

    for node in nodes_list:
        if goal_id == node.id:
            ax.scatter(
                node.position[0], node.position[1], 
                color="green",  
                label="Found Goal", 
                s=50, 
                edgecolors="black", 
                alpha=1,
                zorder=4
            )

    ## A Star path:
    x_path = []
    y_path = []

    path_edges_draw_list = []
    prev_node = nodes_list[0]
    for node in path_nodes:
        x_path.append(node.position[0])
        y_path.append(node.position[1])
        
        path_edges_draw_list.append(EdgeDraw(prev_node, node))
        
        prev_node = node

    ax.scatter(x_path, y_path, color='black', zorder=5)

    for edge in path_edges_draw_list:
            x_vals = [edge.start[0], edge.end[0]]
            y_vals = [edge.start[1], edge.end[1]]
            ax.plot(x_vals, y_vals, linewidth=1, color='black', alpha=1, zorder=5)
    
    ax.legend(loc="upper right")

    if show:
        plt.show()

    else:
        fig.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)



def check_edge(node1, node2):
    """
    Check if intermediate points between two nodes are free to determine if they can be connected
    """
    pos1 = node1.position
    pos2 = node2.position

    edge_discretized = np.linspace(pos1, pos2, num=100)
    
    for e in edge_discretized:
        if is_free(e[0], e[1]) == False:
            return False
        
    return True

def check_collision(node):
    return is_free(node.position[0], node.position[1])


def sample_position():
    x = np.random.uniform(low=0.0, high=MAP_WIDTH)
    y = np.random.uniform(low=0.0, high=MAP_HEIGHT)
    
    return np.array([x, y])


nodes_list = []
edges_draw_list = []
max_nodes = 300
r = 1.0


start_position = np.array([0.5, 0.5, 0.0])
goal_position = np.array([6.0, 6.0, 0.0])

nodes_list.append(Node(0, start_position))
#nodes_list.append(Node(1, goal_position))
goal_node = Node(1, goal_position)
id = 2
found_goal = False
goal_threshold = 0.5
goal_id = None
goal_dist = np.inf

visualize = True
k=0
while len(nodes_list) < max_nodes+1 or found_goal==False:
    # sample new position for new node
    position = sample_position()
    x = position[0]
    y = position[1]

    # check if node in free space
    if is_free(x, y):
            
        new_node = Node(id, np.array([x, y, 0]))
        add = False
        if len(nodes_list)==0:
            add = True

        # Check if any nodes within radius
        for node in nodes_list:
            distance = node.get_distance(new_node)
            
            if distance < r:
                # check if there is obstacle between the two nodes
                if check_edge(node, new_node):
                    node.add_connection(new_node)
                    new_node.add_connection(node)
                    edges_draw_list.append(EdgeDraw(node, new_node))
                    add = True
        
        if add:
            nodes_list.append(new_node)
            if new_node.get_distance(goal_node) < goal_threshold:
                found_goal = True
                if new_node.get_distance(goal_node) < goal_dist:
                    goal_id = id
                    goal_dist = new_node.get_distance(goal_node)

            id += 1
            k+=1
            if visualize:
                draw_map_frame(f"visualization/frames/frame_{k:04d}.png")
                    

if found_goal:
    print(f"Found goal, node id: {goal_id}")
else:
    print("Goal not found")


    


## A*
graph_search_type='aStar'
start_id = 0
path_nodes, total_cost = aStarAlgo(start_id, goal_id, nodes_list, graph_search_type=graph_search_type)

start_position = np.array([0.5, 0.5, 0.0])
goal_position = np.array([6.0, 6.0, 0.0])


frame_path = f"visualization/frames/frame_{k:04d}.png"
draw_map_frame_aStar(path_nodes, nodes_list, edges_draw_list, start_position, goal_position, goal_id, frame_path=frame_path, show=False)

frames = sorted(glob.glob("visualization/frames/frame_*.png"))
imgs = [imageio.imread(f) for f in frames]
imageio.mimsave("visualization/animation.gif", imgs, fps=20)