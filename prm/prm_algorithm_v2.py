"""
Now better consistent with the theory: graph is created before knowing where start and goal are

"""


from prm.node import Node, Edge, EdgeDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import numpy as np
import os
import imageio.v2 as imageio
import glob
from environmentBuilder.getWalls import getLayout
from environmentBuilder.is_free import is_free
from CONSTANTS import MAP_HEIGHT, MAP_WIDTH


def check_edge(node1, node2, walls):
    """
    Check if intermediate points between two nodes are free to determine if they can be connected
    """
    pos1 = node1.position
    pos2 = node2.position

    edge_discretized = np.linspace(pos1, pos2, num=50)
    
    for e in edge_discretized:
        if is_free(e[0], e[1], walls) == False:
            return False
        
    return True


def sample_position():
    """
    This function will be changed to 3d, with map limits given by the env
    """
    x = np.random.uniform(low=0.0, high=MAP_WIDTH)
    y = np.random.uniform(low=0.0, high=MAP_HEIGHT)
    
    return np.array([x, y])


def get_prm_graph(walls, max_nodes=1000, r=1.5):
    "Function to generate the graph, represented as a list of nodes"

    print("Generating graph...")
    nodes_list = []
    
    k = 0
    id = 0

    while len(nodes_list) < max_nodes:
        # sample new position for new node
        position = sample_position()
        x = position[0]
        y = position[1]

        # check if node in free space
        if is_free(x, y, walls):
                
            new_node = Node(id, np.array([x, y, 1.0]))
            add = False
            if len(nodes_list)==0:
                add = True

            # Check if any nodes within radius
            for node in nodes_list:
                distance = node.get_distance(new_node)
                
                if distance < r:
                    # check if there is obstacle between the two nodes
                    if check_edge(node, new_node, walls):
                        node.add_connection(new_node)
                        new_node.add_connection(node)
                        #edges_draw_list.append(EdgeDraw(node, new_node))
                        add = True
            
            if add:
                nodes_list.append(new_node)
                
                id += 1
                k+=1

                #if k%100 == 0:
                #    print(f"{k} / {max_nodes}")

                # if visualize:
                #     frame_path = f"prm/frames/frame_{k:04d}.png"
                #     # draw function for pdm?

    return nodes_list


def add_checkpoint(nodes_list, checkpoint_position=np.array([1.5, 1.5, 1.0]), walls=None, r=1.5):
    """
    Returns: 
    nodes_list
    connected - True if checkpoint was connected
    checkpoint_id
    """


    # Append start position and end position and connect to nodes
    centers = walls # Refactor such that centers get passed as input?

    connected = False
    r_start = r
    # Add Start node:
    id = len(nodes_list)
    
    checkpoint_id = id
    checkpoint_node = Node(checkpoint_id, checkpoint_position)

    while not connected and r < 8*r_start:
        # Connect Start node:
        for node in nodes_list:
            distance = node.get_distance(checkpoint_node)
            
            if distance < r:
                # check if there is obstacle between the two nodes
                if check_edge(node, checkpoint_node, centers):
                    node.add_connection(checkpoint_node)
                    checkpoint_node.add_connection(node)
                    connected = True
            
        if connected:
            nodes_list.append(checkpoint_node)
            return nodes_list, connected, checkpoint_id

        else:
            print("Checkpoint was not connected to graph initially, searching with larger radius")
            r = 2*r


    connected = False
    print(f"Could not connect checkpoints to any nodes")

    return nodes_list, connected, None