import networkx as nx
import random
import math
from itertools import combinations
import copy

# def distance(node1, node2, net):
#     a = nx.get_node_attributes(net,"pos")
#     x1, y1 = node1
#     x2, y2 = node2
#     return math.sqrt((x2 - x1)*2 + (y2 - y1)*2)

# def rel(proximity, coef, min=0, max=1):
#     value = proximity * coef
#     if value < min:
#         value = min
#     elif value > max:
#         value = max
#     return value

# def relu6(x, minim = 0.05, maxim = 0.2):
#     return min(minim, max(x, maxim))


def reajustaRed(G: nx.Graph, 
                iterationsTot: int, 
                cooperations: dict, 
                proximity_factor: float, 
                reputation: dict
                ) -> nx.Graph: # base_probability: float = 0.1
    
    """
    Adjusts the network graph G based on the history of cooperation and defection between nodes.
    
    :param G: The input network graph
    :param iterationsTot: Total number of iterations in the simulation
    :param cooperations: Dictionary tracking historical cooperations between node pairs
    :param proximity_factor: Factor controlling the likelihood of interaction based on proximity
    :return: The adjusted network graph
    """
    
    # Helper function to get node distance (or proximity)
    def node_distance(node1, node2):
        pos = copy.deepcopy(nx.get_node_attributes(G, 'pos'))
        pos1 = pos[node1]
        pos2 = pos[node2]
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

    # Set of permanently removed edges
    permanently_removed_edges = set()

    for node1 in G.nodes:
        for node2 in G.nodes:
            if node1 != node2:
                
                defections1 = G.graph["defections"].get((node2, node1), 0)
                defections2 = G.graph["defections"].get((node1, node2), 0)
                # coops1 = G.graph["coops"].get((node2, node1), 0)
                # coops2 = G.graph["coops"].get((node1, node2), 0)
                previous_cooperations = cooperations.get((node1, node2), False)

                # probability of interaction based on distance
                # we add a small value to the denominator (epsilon) as nodes can't be physically occupaying other nodes' space because distancess became null
                # interaction_prob = min(proximity_factor / (node_distance(node1, node2) + 1e-9), 1.0) 

                  # Fetch dynamic reputation
                rep_node1 = reputation.get(node1, 0)
                rep_node2 = reputation.get(node2, 0)

                if G.has_edge(node1, node2):

                    edge = (node1, node2) if G.has_edge(node1, node2) else (node2, node1)

                    if 9.4 <= G[edge[0]][edge[1]]["weight"] < 9.7:
                        G.remove_edge(edge[0], edge[1])

                    elif G[edge[0]][edge[1]]["weight"] < 9.4:
                        permanently_removed_edges.add(edge)
                        G.remove_edge(edge[0], edge[1])
                                                
                elif (defections1 < 3 and defections2 < 3): #and random.random() <= interaction_prob):

                    if (node1, node2) not in permanently_removed_edges:

                        if rep_node1 >= 0.8 or rep_node2 >= 0.8:
                            edge_creation_prob = 1.0
                        elif rep_node1 >= 0.5 or rep_node2 >= 0.5:
                            edge_creation_prob = 0.75
                        else:
                            edge_creation_prob = 0.2

                        # probability = 0.5 + (0.9 * previous_cooperations / iterationsTot)
                        coop_effect = 0.5 + (0.5 * (previous_cooperations / iterationsTot)) # Cooperation effect based on previous cooperations
                        rep_attraction_factor = (rep_node1 + rep_node2) / 2.0  # Hub effect based on the average reputation of both nodes
                        final_prob = edge_creation_prob * coop_effect *rep_attraction_factor
                        
                        final_prob = min(final_prob, 1.0)

                        if random.random() < final_prob:
                            G.add_edge(node1, node2, weight=10)
                

                # elif (defections1 < 3 and defections2 < 3 and random.random() <= interaction_prob):
                #     # if (node1, node2) not in permanently_removed_edges and (node2, node1) not in permanently_removed_edges:
                #     probability = 0.5  # Default probability

                #     if previous_cooperations > 0:
                #         probability += 0.9

                #     if random.random() < probability:
                #         G.add_edge(node1, node2, weight=10)

                    # if previous_cooperations:
                    #     coop_count = sum(cooperations.get((node1, node2), 0) for _ in range(iterationsTot))
                    #     if coop_count >= 1: 
                    #         probability = 1   
                    
    return G
    


def reajustaRe(G: nx.Graph, 
                iterationsTot: int, 
                cooperations: dict, 
                proximity_factor: float, 
                reputation: dict
                ) -> nx.Graph: # base_probability: float = 0.1

 # Helper function to calculate the distance between two nodes
    def node_distance(node1, node2):
        pos = nx.get_node_attributes(G, 'pos')
        pos1, pos2 = pos[node1], pos[node2]
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

    for node1 in G.nodes:
        for node2 in G.nodes:
            if node1 != node2:
                
                # Calculate interaction probability purely based on distance
                interaction_prob = min(proximity_factor / (node_distance(node1, node2) + 1e-9), 1.0)
                
                # Handle edge creation/removal based on proximity
                if G.has_edge(node1, node2):
                    # Potentially remove an existing edge
                    if random.random() < 0.5:  # Fixed removal probability
                        G.remove_edge(node1, node2)
                else:
                    # Potentially add a new edge based purely on proximity
                    if random.random() <= interaction_prob:
                        G.add_edge(node1, node2, weight=10)

    return G



def reajustaRed_Reputation(G: nx.Graph, 
                iterationsTot: int, 
                cooperations: dict, 
                proximity_factor: float, 
                reputation: dict
                ) -> nx.Graph:
    """
    Adjusts the network graph G based on the history of cooperation and defection between nodes.
    
    :param G: The input network graph
    :param iterationsTot: Total number of iterations in the simulation
    :param cooperations: Dictionary tracking historical cooperations between node pairs
    :param proximity_factor: Factor controlling the likelihood of interaction based on proximity
    :param reputation: Dictionary containing the reputation score for each node
    :return: The adjusted network graph
    """
    
    # Helper function to get node distance (or proximity)
    def node_distance(node1, node2):
        pos = copy.deepcopy(nx.get_node_attributes(G, 'pos'))
        pos1 = pos[node1]
        pos2 = pos[node2]
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

    # Set of permanently removed edges
    permanently_removed_edges = set()

    for node1 in G.nodes:
        for node2 in G.nodes:
            if node1 != node2:
                
                defections1 = G.graph["defections"].get((node2, node1), 0)
                defections2 = G.graph["defections"].get((node1, node2), 0)
                previous_cooperations = cooperations.get((node1, node2), 0)
                
                # Get reputations of node1 and node2
                rep_node1 = reputation.get(node1, 0)
                rep_node2 = reputation.get(node2, 0)

                # Calculate the probability of interaction based on distance
                interaction_prob = min(proximity_factor / node_distance(node1, node2), 1.0)

                # Determine the edge creation probability based on reputation
                if rep_node1 >= 10 or rep_node2 >= 10:
                    edge_creation_prob = 1.0
                elif rep_node1 >= 5 or rep_node2 >= 5:
                    edge_creation_prob = 0.75
                else:
                    edge_creation_prob = 0.2

                if G.has_edge(node1, node2):
                    edge = (node1, node2) if G[node1][node2] else (node2, node1)
                    if G[edge[0]][edge[1]]["weight"] < 9.4:
                        G.remove_edge(edge[0], edge[1])
                        permanently_removed_edges.add(edge)
                        
                elif (defections1 < 3 and defections2 < 3 and random.random() <= interaction_prob):
                    if (node1, node2) not in permanently_removed_edges:
                        if random.random() < edge_creation_prob:
                            G.add_edge(node1, node2, weight=10)
    
    return G