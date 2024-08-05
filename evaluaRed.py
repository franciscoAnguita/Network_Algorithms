import networkx as nx
import subprocess
import os
from utils import remove_files_from_folder
import numpy as np

def adjust_node_positions(G: nx.Graph, node1: str, node2: str, factor: float, action:str):
    
    positions = nx.get_node_attributes(G, 'pos')
    
    # Compute the direction vector of the edge
    edge_vector = positions[node1] - positions[node2]
    
    new_distance = np.linalg.norm(edge_vector) * factor
    displacement = (np.linalg.norm(edge_vector) - new_distance ) / 2  # Divide by 2 to split displacement evenly between the two nodes
       
    # Normalize the edge vector to get the direction --> RuntimeWarning: invalid value encountered in divide
    direction = edge_vector / np.linalg.norm(edge_vector)
    
    # Update node positions
    if action =='expand':
        positions[node1] -= direction * displacement
        positions[node2] += direction * displacement
    if action =='shrink':
        positions[node1] -= direction * displacement
        positions[node2] += direction * displacement

    nx.set_node_attributes(G, positions, 'pos')
    
       
    # positions = nx.get_node_attributes(G, 'pos')
    # p1, p2 = positions[node1], positions[node2]
    # # Compute the displacement
    # displacement = (p2 - p1) * factor / 2
    # # Update node positions
    # new_p1 = p1 + displacement
    # new_p2 = p2 - displacement
    # # Update node positions in the graph
    # positions[node1] = new_p1
    # positions[node2] = new_p2
    # nx.set_node_attributes(G, positions, 'pos')



# Custom function to adjust edge widths based on edge weights
def adjust_edge_widths(G, positions, ax):
    for u, v in G.edges():
        weight = G[u][v].get('weight', 1.0)
        nx.draw_networkx_edges(G, pos=positions, edgelist=[(u, v)], width=weight, ax=ax)



def readLogFile(G: nx.Graph, node1: str, node2: str, weight: int, defections: dict, coops: dict, cooperations: dict, type_of_game: str) -> float: #, interaction: dict,  
    
    file_name = f"{type_of_game}_{node1}_{node2}.txt"
    logPath = "../log"
    log_file = os.path.join(logPath, file_name)

    if type_of_game not in file_name:
        print("Error, file not found")
        return
    
    last_action_player1 = None
    last_action_player2 = None
    
    with open(log_file, 'r') as file:
        for line in file:
            columns = line.strip().split()
            last_action_player1 = int(columns[0])
            last_action_player2 = int(columns[1])

    # # Update interaction table
    # if node1 not in interaction:
    #     interaction[node1] = {}
    # if node2 not in interaction:
    #     interaction[node2] = {}
        
    # interaction[node1][node2] = last_action_player1
    # interaction[node2][node1] = last_action_player2

    if last_action_player1 == 1 or last_action_player2 == 1:
        weight -= 0.3  # Adjust as needed   
        adjust_node_positions(G, node1, node2, factor=1, action='expand')     
        
        if last_action_player1 == 1:
            defections[(node1, node2)] += 1
           
        if last_action_player2 == 1:
            defections[(node2, node1)] += 1
                
    if last_action_player1 == 0 and last_action_player2 == 0:
        weight += 0.3  # Adjust as needed
        adjust_node_positions(G, node1, node2, factor=0.90, action= 'shrink') 

    if last_action_player1 == 0:
            coops[(node1, node2)] += 1

    if last_action_player2 == 0:
        coops[(node2, node1)] += 1

    # Record cooperation
    cooperations[(node1, node2)] = (last_action_player1 == 0 and last_action_player2 == 0)
    # print('EVALUARED - cooperations', cooperations)

    return round(weight,1)



def evaluaRed(G: nx.Graph, type_of_game: str = "prisoners", iters: int = 5, numSamp: int = 1) -> dict:
    '''  '''
    remove_files_from_folder("../log")
    cooperations = {}
    for edge in G.edges:
        node1, node2 = edge
        # print('EVALUARED 1, coops', G.graph["coops"])
        command = f'./game {type_of_game} {node1} {node2} {iters} {numSamp}' # Placeholder for game type
        subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # stdout y err están silenciados
        weight = readLogFile(G, node1, node2, G[node1][node2]["weight"], G.graph["defections"], G.graph["coops"], cooperations, type_of_game) #, G.graph["interactions"], 
        G[node1][node2]["weight"] = weight
        
        # print('')
        # print('EVALUARED 2, cooperations', cooperations)
        # print('EVALUARED, weight', weight)
        # print('')
    
    return nx.get_edge_attributes(G, "weight"), G.graph["defections"], G.graph["coops"], cooperations #, G.graph["interactions"], 


