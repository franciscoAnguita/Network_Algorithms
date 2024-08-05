import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy
import time
from configuraRed import *
from evaluaRed import *
from reajustaRed import *
from utils import *
import warnings



def run_network_analysis(num_agents, skill_type, num_networks=5, iterationsTot=20):
       
    start_time = time.time()
    proximity_factor = 0.1  # proximity factor defined here
    all_defections = []
    all_coops = []
    all_tipping_points = []
    pad_value = 0 # pad value to fill in absent values in degree histogram to make the lists homogeneous
    all_edge_data = []  # To collect edge data

    configuration = {
        "nodes": {skill_type: num_agents},
        "initialEdges": 2
        }
        
    all_metrics = {
            "clustering_coefficient": [],
            "average_path_length": [],
            "degree_distribution": [],
            "degree_histogram": [],
            "cooperation_proportion": [],
            "graphs": [],
            "density": [],
            "assortativity": [],
            "modularity": [],
            "edge_weight": []
            }
    

    for run in range(num_networks):
        
        affinities = []
        defections = []
        coops = []
        tipping_points = []
        interactions = []
        metrics = {
            "clustering_coefficient": [],
            "average_path_length": [],
            "degree_distribution": [],
            "degree_histogram": [],
            "cooperation_proportion": [],
            "density": [],
            "assortativity": [],
            "modularity": [],
            "edge_weight": []
        }

        print(f'NETWORK {run + 1}/{num_networks}')
        G = configuraRed(configuration)
        # pos = nx.get_node_attributes(G, 'pos')    
        # generateImage(G, 0, pos)

        for iter in range(0, iterationsTot):

            print('NETWORKLOOP, iteration', iter)
            
            # prev_pos = copy.deepcopy(nx.get_node_attributes(G, 'pos'))
            aff, defect, cooperaciones,cooperations = evaluaRed(G)  #inter, 
            defections.append(defect)
            coops.append(cooperaciones)
            affinities.append(aff)
            # interactions.append(inter)
            
            G  = reajustaRed(G, iterationsTot, cooperations, proximity_factor) #
            # pos = nx.get_node_attributes(G, 'pos')
            # generateImage(G, iter + 1, pos, prev_pos)
        
            # if (iter + 1) % 5 == 0:
            #     elapsed_time = time.time() - start_time
            #     print(f"Elapsed time after {iter + 1} iterations: {elapsed_time:.2f} seconds")
            
            with warnings.catch_warnings():
                
                warnings.filterwarnings('error', category=RuntimeWarning)

                # Density
                try:
                    density = nx.density(G)
                except ZeroDivisionError:
                    density = float('nan')
                    print("ZeroDivisionError: Cannot calculate density on a graph with no edges.")
                except RuntimeWarning as e:
                    density = float('nan')
                    print(f"RuntimeWarning during density calculation: {e}")

                # Assortativity
                try:
                    assortativity = nx.degree_assortativity_coefficient(G)
                    if np.isnan(assortativity):
                        print("Warning: Assortativity calculation resulted in NaN value.")
                except ZeroDivisionError:
                    assortativity = float('nan')
                    print("ZeroDivisionError: Cannot calculate assortativity on a graph with no edges.")
                except RuntimeWarning as e:
                    assortativity = float('nan')
                    print(f"RuntimeWarning during assortativity calculation: {e}")

                # Edge Weights
                try:
                    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
                except KeyError:
                    edge_weights = []
                    print("KeyError: Edge weights are not defined.")

                # Community Detection and Modularity
                try:
                    communities = list(nx_comm.greedy_modularity_communities(G))
                    # Modularity
                    try:
                        modularity = nx_comm.modularity(G, communities)
                        if np.isnan(modularity):
                            print("Warning: Modularity calculation resulted in NaN value.")
                    except ZeroDivisionError:
                        modularity = float('nan')
                        print("ZeroDivisionError: Cannot calculate modularity on a graph with no edges.")
                    except RuntimeWarning as e:
                        modularity = float('nan')
                        print(f"RuntimeWarning during modularity calculation: {e}")
                except ZeroDivisionError:
                    communities = []
                    modularity = float('nan')
                    print("ZeroDivisionError: Cannot detect communities on a graph with no edges.")
                except RuntimeWarning as e:
                    communities = []
                    modularity = float('nan')
                    print(f"RuntimeWarning during community detection: {e}")

            if nx.is_connected(G):
                metrics["average_path_length"].append(nx.average_shortest_path_length(G))
            else:
                metrics["average_path_length"].append(float('inf'))  # or another measure for disconnected graphs
            metrics["degree_distribution"].append([d for n, d in G.degree()])
            metrics["degree_histogram"].append(nx.degree_histogram(G))
            metrics["cooperation_proportion"].append(measure_cooperation(cooperations))
            metrics["clustering_coefficient"].append(nx.average_clustering(G))
            metrics["assortativity"].append(assortativity)
            metrics["modularity"].append(modularity)
            metrics["density"].append(density)
            metrics["edge_weight"].append(edge_weights)

            coop_proportion = metrics["cooperation_proportion"]
            network_tipping_points = detect_tipping_points(coop_proportion)
            tipping_points.extend([(num_agents, skill_type, run, tp) for tp in network_tipping_points])
            
            # Collect edge data for the current iteration
            for u, v, data in G.edges(data=True):
                all_edge_data.append({
                    "network_id": run+1,
                    "iteration": iter+1,
                    "num_agents": num_agents,
                    "skill_type": skill_type,
                    "source": u,
                    "target": v,
                    "weight": data["weight"]
                })
        
        for key in all_metrics:
            if key != "graphs":
                all_metrics[key].append(metrics[key])

        all_defections.append(defections)
        all_coops.append(coops)
        all_metrics["graphs"].append(G)  # Collect the final graph of each run
        
        # elapsed_time = time.time() - start_time
        # print(f"Elapsed time after {run + 1} iterations: {elapsed_time:.2f} seconds")
        
            
    all_tipping_points.extend(tipping_points)  
    all_metrics["degree_histogram"] = pad_to_same_length(all_metrics["degree_histogram"],pad_value)
    stats = compute_statistics(all_metrics)
           
    # plot_metrics_mean_median(all_metrics, stats, iterationsTot, num_networks)
   
    # coop_proportion = all_metrics["cooperation_proportion"]
    # tipping_points_cooperation = detect_tipping_points(coop_proportion)
    # # plot_tipping_points(coop_proportion, tipping_points_cooperation, "Cooperation Proportion")
    # # plot_all_metrics(all_metrics, stats, iterationsTot)
    
    # generateGif(iterationsTot)
    # summary(all_metrics, num_networks, iterationsTot) #G

    # connected_nodes_summary = summarize_connected_nodes(G)
    # save_summary_to_file(connected_nodes_summary, 'summary.txt')
    # save_deffections_to_file(defections, 'defections.txt')
    # # save_interactions_to_file(interactions, 'interactions.txt')
 
        
    return all_metrics, stats, all_tipping_points, all_edge_data
