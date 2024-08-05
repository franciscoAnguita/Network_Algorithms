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


def reajustaRed(G: nx.Graph, iterationsTot: int, cooperations: dict, proximity_factor: float) -> nx.Graph: # base_probability: float = 0.1
    
    
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
                coops1 = G.graph["coops"].get((node2, node1), 0)
                coops2 = G.graph["coops"].get((node1, node2), 0)
                previous_cooperations = cooperations.get((node1, node2), False)

                # try catch
                # Calculate the probability of interaction based on distance
                interaction_prob = min(proximity_factor / node_distance(node1, node2), 1.0)
                # print('REAJUSTARED - interaction_prob - node_distance', interaction_prob, node_distance(node1, node2))

                if G.has_edge(node1, node2):
                    edge = (node1, node2) if G[node1][node2] else (node2, node1)
                    # print('REAJUSTARED - has edge', edge)
                    if G[edge[0]][edge[1]]["weight"] < 9.4:
                        G.remove_edge(edge[0], edge[1])
                        if defections1 >= 3 or defections2 >= 3:
                            permanently_removed_edges.add(edge)

                elif (defections1 < 3 and defections2 < 3 and random.random() <= interaction_prob):
                    # if (node1, node2) not in permanently_removed_edges and (node2, node1) not in permanently_removed_edges:
                    probability = 0.5  # Default probability

                    if previous_cooperations > 0:
                        probability += 0.9

                    if random.random() < probability:
                        G.add_edge(node1, node2, weight=10)

                    # if previous_cooperations:
                    #     coop_count = sum(cooperations.get((node1, node2), 0) for _ in range(iterationsTot))
                    #     if coop_count >= 1: 
                    #         probability = 1
                    

    return G
    


