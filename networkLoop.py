from configuraRed import *
from evaluaRed import *
from reajustaRed import *
from utils import *
import networkx as nx
import time
import copy

def main():

    start_time = time.time()
    num_networks = 1
    iterationsTot = 8
    proximity_factor = 0.1  # proximity factor defined here
    all_defections = []
    all_coops = []
    all_tipping_points = []
    pad_value = 0 # pad value to fill in absent values in degeree histogram to make the lists homogeneous
    configuration = {
                        "nodes" : {'manipulator-bully': 100}, #, 'br1':50,  'mqubed'
                        "initialEdges": 2
                    } #,'br1':15
        
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
        interactions = []
        tipping_points = []
        

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
        pos = nx.get_node_attributes(G, 'pos')    
        generateImage(G, 0, pos)

        for iter in range(0, iterationsTot):
            print('NETWORKLOPP, iteration', iter)
            
            prev_pos = copy.deepcopy(nx.get_node_attributes(G, 'pos'))
            aff, defect, cooperaciones,cooperations = evaluaRed(G)  #inter, 
            defections.append(defect)
            coops.append(cooperaciones)
            affinities.append(aff)
            # interactions.append(inter)
            
            G  = reajustaRed(G, iterationsTot, cooperations, proximity_factor) #
            pos = nx.get_node_attributes(G, 'pos')
            generateImage(G, iter + 1, pos, prev_pos)
        
            # if (iter + 1) % 5 == 0:
            #     elapsed_time = time.time() - start_time
            #     print(f"Elapsed time after {iter + 1} iterations: {elapsed_time:.2f} seconds")
            
            density = nx.density(G)
            assortativity = nx.degree_assortativity_coefficient(G)
            communities = list(nx_comm.greedy_modularity_communities(G))
            modularity = nx_comm.modularity(G, communities)
            edge_weights = [G[u][v]["weight"] for u, v in G.edges()]


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
            tipping_points.extend([(run, iter, tp) for tp in network_tipping_points])

           
            # print('NETWORKLOOP 0 - degree_histogram', metrics["degree_histogram"])
            # print('NETWORKLOOP 0 - assortativity', metrics["assortativity"])
            # print('NETWORKLOOP 0 - modularity', metrics["modularity"])
            # print('NETWORKLOOP 0 - edge_weights', metrics["edge_weight"])
        
        for key in all_metrics:
            if key != "graphs":
                all_metrics[key].append(metrics[key])
          
        all_defections.append(defections)
        all_coops.append(coops)
        all_metrics["graphs"].append(G)  # Collect the final graph of each run
        all_tipping_points.extend(tipping_points) 
        # print('NETWORKLOOP fuera - coops', coops)
    
    
    all_metrics["degree_histogram"] = pad_to_same_length(all_metrics["degree_histogram"],pad_value)
    stats = compute_statistics(all_metrics)
    coop_proportion = all_metrics["cooperation_proportion"]
    tipping_points = detect_tipping_points(coop_proportion)
    # coop_proportion = metrics["cooperation_proportion"]
    # tipping_points = detect_tipping_points(coop_proportion)
    

    plot_metrics_mean_median(all_metrics, stats, iterationsTot, num_networks)
    plot_all_tipping_points(all_tipping_points)
    # plot_tipping_points(coop_proportion,tipping_points)
    # plot_all_metrics(all_metrics, stats, iterationsTot)
    

    generateGif(iterationsTot)
    summary(all_metrics, num_networks, iterationsTot) #G

    connected_nodes_summary = summarize_connected_nodes(G)
    save_summary_to_file(connected_nodes_summary, 'summary.txt')
    save_deffections_to_file(defections, 'defections.txt')
    # save_interactions_to_file(interactions, 'interactions.txt')
  
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    return   
    # return metrics, defections

if __name__ == "__main__":
    main()