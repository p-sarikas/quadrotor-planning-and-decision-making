from prm.node import Node, Edge, EdgeDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import numpy as np
from environmentBuilder.is_free import is_free
from environmentBuilder.getWalls import getLayout
from CONSTANTS import MAP_HEIGHT, MAP_WIDTH

def check_edge(node1, node2, walls):
    """
    Check if intermediate points between two nodes are free to determine if they can be connected
    """
    pos1 = node1.position
    pos2 = node2.position

    edge_discretized = np.linspace(pos1, pos2, num=4)
    
    for e in edge_discretized:
        if is_free(e[0], e[1], walls) == False:
            return False
        
    return True


def get_grid_output(start_position=np.array([0.5, 0.5, 1.0]), goal_position=np.array([9.0, 9.0, 1.0])):
    
    MAP_WIDTH, MAP_HEIGHT = [10.0, 10.0]

    dx = 1.0 #distance between gridpoints in both x and y direction
    nodes_list = []
    edges_draw_list = []


    nodes_list.append(Node(0, start_position))
    nodes_list.append(Node(1, goal_position))
    goal_id = 1
    id = 2

    x_dir = np.arange(0.0, MAP_WIDTH, dx)
    y_dir = np.arange(0.0, MAP_HEIGHT, dx)

    walls = getLayout()

    for x in x_dir:
        for y in y_dir:
            if is_free(x, y, walls):
                
                new_node = Node(id, np.array([x, y, 1]))
                add = False
                if len(nodes_list)==0:
                    add = True
                for node in nodes_list:
                    distance = node.get_distance(new_node)
                    
                    if distance < 1.2*dx:
                        if check_edge(node, new_node, walls):
                            node.add_connection(new_node)
                            new_node.add_connection(node)
                            edges_draw_list.append(EdgeDraw(node, new_node))
                            add = True
                
                if add:
                    nodes_list.append(new_node)
                    id += 1
                        
    #for node in nodes_list:
    #   node.print_node()

    return nodes_list, goal_id


def get_grid_graph(walls, dx=0.5):
    nodes_list = []
    x_dir = np.arange(dx, MAP_WIDTH, dx)
    y_dir = np.arange(dx, MAP_HEIGHT, dx)

    id = 0

    for x in x_dir:
        for y in y_dir:
            if is_free(x, y, walls):
                
                new_node = Node(id, np.array([x, y, 1]))
                
                for node in nodes_list:
                    distance = node.get_distance(new_node)
                    
                    if distance < 1.2*dx:
                        if check_edge(node, new_node, walls):
                            node.add_connection(new_node)
                            new_node.add_connection(node)
                            
                        
                
                
                nodes_list.append(new_node)
                id += 1
                        
    #for node in nodes_list:
    #   node.print_node()

    return nodes_list
