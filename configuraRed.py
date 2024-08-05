import random
import networkx as nx

def configuraRed(config: dict) -> nx.Graph:
    ''' This function receives a dictionary of the nodes with which to create the network, 
    and we will initially generate it with edges chosen randomly'''

    G = nx.Graph()
    # Create nodes
    for key, value in config["nodes"].items():
        for i in range(1,value +1):
            G.add_node(f"{key}_{i}")
    
 
     # We fix nodes positions with Fruchterman-Reingold layout.
    # nx.set_node_attributes(G, nx.circular_layout(G), 'pos')
    pos = nx.spring_layout(G)
    nx.set_node_attributes(G, pos, 'pos') 
   
    # Create edges
    edges_to_create = random.sample([(i, j) for i in G.nodes() for j in G.nodes() if i != j], k = config["initialEdges"]) # FOR HETERO/HOMO EDGES
    edges_to_create = [(edge[0], edge[1], {'weight': 10}) for edge in edges_to_create ]
    G.add_edges_from(edges_to_create)

    # Initially there are no defections or cooperations 
    G.graph["defections"] = {(i,j): 0 for i in G.nodes() for j in G.nodes() if i != j}
    G.graph["coops"] = {(i,j): 0 for i in G.nodes() for j in G.nodes() if i != j}
   

    return G

