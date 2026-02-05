import numpy as np

class Edge:
    """
    Edge contains the id of the connecting node and the distance
    """
    def __init__(self, second_node_id, distance):
        self.second_node = second_node_id
        self.distance = distance

class EdgeDraw:
    """
    Class for visualization purposes
    """
    def __init__(self, node1, node2):
        self.start = node1.position
        self.end = node2.position

class Node:

    def __init__(self, id, position):
        self.id = id
        self.position = position #np.array()
        self.connections = [] #list of edges
        

    def add_connection(self, connecting_node):
        connecting_node_id = connecting_node.id
        distance = self.get_distance(connecting_node)
        self.connections.append(
            Edge(connecting_node_id, distance)
        )

    def get_distance(self, second_node):
        distance = np.linalg.norm(self.position - second_node.position)
        return distance

    def print_node(self):
        print(f"Node: {self.id}")
        print(f"Connections: ")
        for edge in self.connections:
            print(f"Edge to {edge.second_node} with cost {edge.distance}") 

    
