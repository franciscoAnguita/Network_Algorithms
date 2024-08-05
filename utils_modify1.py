import os
import matplotlib.pyplot as plt
import networkx as nx
import json
import imageio
import networkx.algorithms.community as nx_comm
import numpy as np
import random
import pandas as pd
import ruptures as rpt
from fa2 import ForceAtlas2
from matplotlib.cm import get_cmap
from matplotlib.ticker import MaxNLocator
from platform import node
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv

networkNum = -1


def initialize_graph(nodes):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    pos = {node: (random.random(), random.random()) for node in nodes}
    nx.set_node_attributes(G, pos, "pos")
    return G

def update_graph(G):
    for edge in G.edges():
        node1, node2 = edge
        weight_change = random.uniform(-1, 1)  # Change in edge weight (replace this with your logic)
        G[node1][node2]["weight"] += weight_change

    for edge in G.edges():
        node1, node2 = edge
        if G[node1][node2]["weight"] > 10:
            G.nodes[node1]["pos"] = tuple(x - 1 for x in G.nodes[node1]["pos"])
            G.nodes[node2]["pos"] = tuple(x - 1 for x in G.nodes[node2]["pos"])
        elif G[node1][node2]["weight"] < 10:
            G.nodes[node1]["pos"] = tuple(x + 0.5 for x in G.nodes[node1]["pos"])
            G.nodes[node2]["pos"] = tuple(x + 0.5 for x in G.nodes[node2]["pos"])


# def adjust_node_positions(G, positions, iterations=2000):
#     for _ in range(iterations):
#         for edge in G.edges():
#             u, v = edge
#             weight = G[u][v].get('weight', 1.0)
#             delta = positions[v] - positions[u]
#             displacement = delta / np.linalg.norm(delta) * weight
#             positions[u] += displacement
#             positions[v] -= displacement
            


def draw_graph_to_file(G: nx.Graph, iteration: int, positions=None, prev_pos=None):
    
    # Construct current and previous image paths
    current_image_path = f"generatedImages/Network{networkNum}/Random Network - {iteration}.png"
    prev_iteration = iteration - 1
    prev_image_path = f"generatedImages/Network{networkNum}/Random Network - {prev_iteration}.png"

    if prev_image_path:
        
        num_intermediate_frames=15

        cleaned_labels = {node: node.split('_')[1] for node in G.nodes()}
        node_colors = ['LightSkyBlue' if 'S++' in node else 'red' for node in G.nodes()]
        # node.set_edgecolor('black')
        # node_colors = ['orange' if 'S++' in node else 'red' for node in G.nodes()]

        # Get edge weights and colors
        edge_weights = nx.get_edge_attributes(G, 'weight')
        # edge_labels = {(node1, node2): weight for (node1, node2), weight in edge_weights.items()}
        edge_widths = [1 + (G[node1][node2]['weight'] - 10) for node1, node2 in G.edges()]
        edge_colors = ['grey' for weight in edge_weights.values()]
        # edge_colors = ['red' if weight < 10 else 'black' if weight > 10 else 'green' for weight in edge_weights.values()]

        # # Calculate node sizes based on the maximum weight of edges connected to each node
        # node_weights = {node: max([edge_weights.get((node, neighbor), edge_weights.get((neighbor, node), 0)) for neighbor in G.neighbors(node)], default=1) for node in G.nodes()}
        # node_sizes = {node: 300 + 50 * weight for node, weight in node_weights.items()}  # Base size of 300 plus additional size based on weight
 
        # Calculate node sizes based on degree (or another measure of centrality)
        degrees = dict(G.degree())
        if iteration == 0:
            node_sizes = 100
        else:
            node_sizes = [100 + 50 * degrees[node] for node in G.nodes()]  # Base size of 300 plus additional size based on degree

        
        # Interpolate positions if previous positions are available
        if prev_pos:
            # Calculate difference vectors between current and previous positions
            differences = {node: positions[node] - prev_pos[node] for node in positions}
            # differences = {node: positions[node] - prev_positions.get(node, (0, 0)) for node in positions}
            # print(differences)

            # Interpolate positions for intermediate frames
            intermediate_positions = {}
            for i in range(1, num_intermediate_frames + 1):
                alpha = i / (num_intermediate_frames + 1)  # Linear interpolation factor
                intermediate_positions[i] = {node: prev_pos[node] + alpha * differences[node] for node in positions}
            #print("UTILS - GIF\n",intermediate_positions)
            
            
            # Draw and save intermediate frames
            for i, intermediate_pos in intermediate_positions.items():
                
                fig, ax = plt.subplots(figsize=(10, 8))
                nx.draw_networkx_nodes(G, pos=intermediate_pos, node_color=node_colors, node_size=node_sizes, ax=ax,edgecolors='black')
                nx.draw_networkx_edges(G, pos=intermediate_pos, edge_color=edge_colors, width=edge_widths, ax=ax, alpha=0.5) #,connectionstyle='arc3,rad=0.1'
                # nx.draw_networkx_labels(G, pos=intermediate_pos, labels=cleaned_labels, font_size=10, font_color='black', ax=ax)
                # nx.draw_networkx_edge_labels(G, pos=intermediate_pos, edge_labels=edge_labels, font_size=8, ax=ax)  # Add edge labels
                plt.title(f"Random Network - {iteration}")
                plt.axis('off')
                path = f"generatedImages/Network{networkNum}/Random Network - Intermediate Frame {iteration}-{i}.png"
                plt.savefig(f"generatedImages/Network{networkNum}/Random Network - Intermediate Frame {iteration}-{i}.png")
                plt.close(fig)
                # print('UTILS Generate - iteration', iteration)
                # print('UTILS Generate - i', i)
                # print('UTILS Generate iter - i', iteration,'-', i)

    # Draw and save current frame
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos=positions, node_color=node_colors, node_size=node_sizes, ax=ax,edgecolors='black')
    nx.draw_networkx_edges(G, pos=positions, edge_color=edge_colors, width=edge_widths, ax=ax, alpha=0.5) #, connectionstyle='arc3,rad=0.1'
    # nx.draw_networkx_labels(G, pos=positions, labels=cleaned_labels, font_size=10, font_color='black', ax=ax)
    # nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_labels, font_size=8, ax=ax)  # Add edge labels
    plt.title(f"Random Network - {iteration}")
    create_folder_if_not_exists(f"generatedImages/Network{networkNum}")
    plt.axis('off')
    plt.savefig(current_image_path)
    plt.close(fig)
    plt.clf()
   
    return current_image_path


def save_to_txt(diction: dict, file_path: str):
    with open(file_path, 'w') as file:
        json.dump(diction, file)


def save_to_txt(lista: list, file_path: str):
    with open(file_path, 'w') as file:
        json.dump(lista, file)


def summarize_connected_nodes(G):
    connected_nodes_summary = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        connected_nodes_summary[node] = {
            'neighbors': neighbors,
            'num_neighbors': len(neighbors)
        }
    return connected_nodes_summary

def save_deffections_to_file(defections, filename):
    directory = os.getcwd() + "/generatedDeffections"

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as file:
        for ronda, defe in enumerate(defections):
            file.write(f"Ronda {ronda + 1}\n")
            for nodes, nDefe in defe.items():
                file.write(f"Nodes: {nodes}, Defections: {nDefe}\n")


# Copy of Defections to create Interactions
def save_interactions_to_file(interactions, filename):
    directory = os.getcwd() + "/generatedInteractions"

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as file:
        for ronda, inter in enumerate(interactions):
            file.write(f"Ronda {ronda + 1}\n")
            for nodes, nInter in inter.items():
                file.write(f"Nodes: {nodes}, Interactions: {nInter}\n")


def save_summary_to_file(summary, filename):
    # Determine the directory path
    directory = os.getcwd() + "/generatedSummary"

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as file:
        for node, data in summary.items():
            neighbors = ', '.join(data['neighbors'])
            num_neighbors = data['num_neighbors']
            file.write(f"Node: {node}, Neighbors: {neighbors}, Num Neighbors: {num_neighbors}\n")


def remove_files_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            # print(f"File '{filename}' removed.")


def generateImage(G: nx.Graph, iter: int, positions, prev_pos=None): # prev_pos=None
    create_folder_if_not_exists("./generatedImages")
    global networkNum
    if(networkNum == -1):
        networkNum = len(os.listdir("./generatedImages"))
        # print("UTILS networkNum", networkNum)
    
    if not os.path.exists(f"./generatedImages/Network{networkNum}"):
        os.mkdir(f"./generatedImages/Network{networkNum}")

    # draw_graph_to_file(G, iter,positions)
    draw_graph_to_file(G, iter, positions, prev_pos)


def generateGif(iter: int):
    images = []
    counter = 0
    nImages = iter
    # nImages = len(os.listdir(f"./generatedImages/Network{networkNum}"))
    # print('UTILS - nImages', nImages)
    for iter in range(1, nImages):
        for i in range(1,15):
            fileName = f"./generatedImages/Network{networkNum}/Random Network - Intermediate Frame {iter}-{i}.png"
            images.append(imageio.imread(fileName))
                       
        fileName = f"./generatedImages/Network{networkNum}/Random Network - {iter}.png"
        images.append(imageio.imread(fileName))
           
    create_folder_if_not_exists("./generatedGif")
    nGifs = len(os.listdir("./generatedGif"))
    imageio.mimsave(f'./generatedGif/network{nGifs}.gif', images, fps=10)
   


# # Performs alpha blending between two frames to create a smooth transition effect.
# def alpha_blend(image1, image2, alpha):
#     blended_image = (1 - alpha) * image1 + alpha * image2
#     blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
#     return blended_image



def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def measure_cooperation(cooperations: dict) -> float:
    total_interactions = len(cooperations)
    cooperative_interactions = sum(1 for coop in cooperations.values() if coop)
    return cooperative_interactions / total_interactions if total_interactions > 0 else 0



def detect_tipping_points(coop_proportion): #coop_proportion

    # This solution is for all_coop_proportion (when we work running several networks, not just one)
    # The solution of one network we were using coop_proportion

    # # Flatten the list of lists into a single list
    # combined_coop_proportion = [item for sublist in all_coop_proportion for item in sublist] #coop_proportion

    # Convert the list to a numpy array
    coop_proportion_array = np.array(coop_proportion) #coop_proportion
    # print('UTILS - coop_proportion_array', coop_proportion_array)
   
    # Reshape the array if it's 1D to make it a 2D array with one feature
    if coop_proportion_array.ndim == 1:
        coop_proportion_array = coop_proportion_array.reshape(-1, 1)
    # print('UTILS - coop_proportion_array', coop_proportion_array)

    # Check if the array has enough data and variability
    if coop_proportion_array.shape[0] < 2 or np.all(coop_proportion_array == coop_proportion_array[0]):
        print("Not enough data or no variability for change point detection.")
        return []

    model = "normal"  # model to use for change point detection
    algo = rpt.Pelt(model=model).fit(coop_proportion_array)
    try:
        result = algo.predict(pen=1)
    except rpt.exceptions.BadSegmentationParameters:
        print("BadSegmentationParameters: insufficient data for change point detection.")
        result = []   

    return result

# The ruptures library provides several models for change point detection, 
# each based on different statistical properties:

# "l2": This model uses the least-squares loss function, which is suitable for detecting changes in the mean of a signal.
# "l1": This model uses the l1-norm, which is more robust to outliers and can be used to detect changes in the median.
# "rbf": This model uses a radial basis function kernel and is suitable for detecting changes in more complex structures of the data.
# "linear": This model can be used to detect changes in the linear trend of the data.
# "normal": This model assumes normally distributed data and detects changes in the mean and variance.

# def detect_tipping_points(metric_list, threshold=0.1):
#     tipping_points = []
#     for i in range(1, len(metric_list)):
#         if abs(metric_list[i] - metric_list[i-1]) > threshold:
#             tipping_points.append(i)
#     return tipping_points
    


def compute_statistics(all_metrics):
    stats = {
        "mean": {},
        "median": {}
    }
    
    for metric_name, values in all_metrics.items():
        if metric_name == "graphs":
            continue  # Skip the 'graphs' key
        if metric_name == "edge_weight":
            continue
        # print('UTILS - compute_statistics - metrics names and values', metric_name, values)
        values_array = np.array(values)
        # print('ILS - values_array', values_array)
        stats["mean"][metric_name] = np.mean(values_array, axis=0)
        stats["median"][metric_name] = np.median(values_array, axis=0)
    # print('UTILS - compute_statistics - values_array ->', metric_name, values_array)
    return stats


# Function to pad lists to the same length
def pad_to_same_length(list_of_lists, pad_value):
    max_length = max(len(lst) for sublist in list_of_lists for lst in sublist)
    return [[lst + [pad_value] * (max_length - len(lst)) for lst in sublist] for sublist in list_of_lists]



def plot_metrics_mean_median (all_metrics, stats, iterationsTot, num_Networks):

    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(wspace=0.3, hspace=1, right=0.85) 
    colours = get_cmap('tab10', num_Networks)

    # Plotting Cooperation Proportion
    plt.subplot(3, 2, 1)
    plt.subplots_adjust(hspace=0.5)
    aggregated_coop_proportion = stats["mean"]["cooperation_proportion"]
    aggregated_tipping_points = detect_tipping_points(aggregated_coop_proportion)
    
    for i, coop_per_network in enumerate(all_metrics["cooperation_proportion"]):
        plt.plot(range(iterationsTot), coop_per_network, alpha=0.6, color=colours(i), marker='o', label=f'Network {i+1}')
    plt.plot(range(iterationsTot), aggregated_coop_proportion, label='Mean Cooperation Proportion', color='blue', linewidth=2)
    plt.plot(range(iterationsTot), stats["median"]["cooperation_proportion"], label='Median Cooperation Proportion', color='red', linestyle='--', linewidth=2)
    
    for tp in aggregated_tipping_points:
        plt.axvline(x=tp, color='black', linestyle='--', label='Aggregated Tipping Point' if tp == aggregated_tipping_points[0] else "")
    
    plt.xlabel('Iterations')
    plt.ylabel('Proportion of Cooperation')
    plt.title('Aggregated Cooperation Proportion with Tipping Points')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.1), fontsize='small')


    # Plotting Clustering Coefficient
    plt.subplot(3, 2, 2)
    plt.subplots_adjust(hspace=0.5)
    aggregated_clustering_coefficient = stats["mean"]["clustering_coefficient"]
    
    for i, clustering_coefficient in enumerate(all_metrics["clustering_coefficient"]):
        plt.plot(range(iterationsTot), clustering_coefficient, alpha=0.6, color=colours(i), marker='o', label=f'Network {i+1}')
    plt.plot(range(iterationsTot), aggregated_clustering_coefficient, label='Mean Clustering Coefficient', color='blue', linewidth=2)
    plt.plot(range(iterationsTot), stats["median"]["clustering_coefficient"], label='Median Clustering Coefficient', color='red', linestyle='--', linewidth=2)
    
    plt.xlabel('Iterations')
    plt.ylabel('Clustering Coefficient')
    plt.title('Aggregated Clustering Coefficient')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.1), fontsize='small')

    
        # Plotting Average Path Length
    plt.subplot(3, 2, 3)
    plt.subplots_adjust(hspace=0.5)
    aggregated_avg_path_length = stats["mean"]["average_path_length"]
    
    for i, avg_path_length in enumerate(all_metrics["average_path_length"]):
        plt.plot(range(iterationsTot), avg_path_length, alpha=0.6, color=colours(i), marker='o', label=f'Network {i+1}')
    plt.plot(range(iterationsTot), aggregated_avg_path_length, label='Mean Average Path Length', color='blue', linewidth=2)
    plt.plot(range(iterationsTot), stats["median"]["average_path_length"], label='Median Average Path Length', color='red', linestyle='--', linewidth=2)
    
    plt.xlabel('Iterations')
    plt.ylabel('Average Path Length')
    plt.title('Aggregated Average Path Length')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.1), fontsize='small')

    
    # Plotting Degree Distribution
    
    plt.subplot(3, 2, 4)
    plt.subplots_adjust(hspace=0.5)

    colormap_names = plt.colormaps()  # Get a list of all available colormaps
    colormap_names = colormap_names[::max(1, len(colormap_names)//num_Networks)]  # Select a subset
            
    for net_index in range(num_Networks):
        # cmap = get_cmap(cmap_list[net_index % len(cmap_list)], iterationsTot)
        cmap_name = colormap_names[net_index % len(colormap_names)]
        cmap = get_cmap(cmap_name, len(all_metrics["degree_distribution"][net_index]))
       
        for i, degree_distribution in enumerate(all_metrics["degree_distribution"][net_index]):
            degrees, counts = np.unique(degree_distribution, return_counts=True)
            plt.plot(degrees, counts, alpha=0.3, color=cmap(i), marker='o', label=f'Network {net_index+1}, Iteration {i+1}' if i == 0 else "")
    
    # Plot mean and median degree distribution
    # plt.plot(range(iterationsTot), stats["mean"]["degree_distribution"], label='Mean Degree Distribution', color='blue', linewidth=2)
    # plt.plot(range(iterationsTot), stats["median"]["degree_distribution"], label='Median Degree Distribution', color='red', linestyle='--', linewidth=2)

    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Frequency of Degree Against Degree')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.1),fontsize='small')

    # Plot density evolution
    
    plt.subplot(3, 2, 5)
    plt.subplots_adjust(hspace=0.5)
    for i, density in enumerate(all_metrics["density"]):
        plt.plot(range(iterationsTot), density, label=f'Network {i+1}', alpha=0.5)

    mean_density = stats["mean"]["density"]
    plt.plot(range(iterationsTot), mean_density, label='Mean Density', color='black', linewidth=2)

    plt.title('Density Evolution Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Density')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.1),fontsize='small')
    plt.show()
    
    plt.tight_layout()
    plt.show()



def summary(all_metrics, num_Networks, iterationsTot):

    colours = get_cmap('tab10', num_Networks)
    num_iterations = len(all_metrics["graphs"]) // num_Networks
    print('SUMMARY - num_iterations',num_iterations)

    # Aggregate degree histograms
    degree_histograms = np.array(all_metrics["degree_histogram"])
    num_networks = degree_histograms.shape[0]
    num_iterations2 = degree_histograms.shape[1]
    print('SUMMARY - num_iterations',num_iterations2)
    aggregated_degree_histogram = np.sum(degree_histograms, axis=(0, 1)) / (num_networks * num_iterations2)

   # Aggregate other metrics
    modularity_values = [nx_comm.modularity(G, nx_comm.greedy_modularity_communities(G)) for G in all_metrics["graphs"]]
    mean_modularity = np.mean(modularity_values)
    assortativity_values = [nx.degree_assortativity_coefficient(G) for G in all_metrics["graphs"]]
    mean_assortativity = np.mean(assortativity_values)
    edge_weights = []
    for G in all_metrics["graphs"]:
        edge_weights.extend([G[node1][node2]["weight"] for node1, node2 in G.edges()])
    
    # Plotting
    plt.figure(figsize=(20, 12))
    plt.subplots_adjust(wspace=0.3, hspace=1, right=0.85) 

    plt.subplot(4, 2, 1)
    plt.bar(range(len(aggregated_degree_histogram)), aggregated_degree_histogram, width=0.8, color="skyblue")
    plt.title("Average Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Average Frequency")

    plt.subplot(4, 2, 2)
    plt.hist(modularity_values, bins=10, color="green", edgecolor="black")
    plt.title("Modularity Distribution\nMean: {:.2f}".format(mean_modularity))
    plt.xlabel("Modularity")
    plt.ylabel("Frequency")

    plt.subplot(4, 2, 5)
    plt.hist(assortativity_values, bins=10, color="salmon", edgecolor="black")
    plt.title("Assortativity Distribution\nMean: {:.2f}".format(mean_assortativity))
    plt.xlabel("Assortativity")
    plt.ylabel("Frequency")

    plt.subplot(4, 2, 6)
    plt.hist(edge_weights, bins=20, color="purple", edgecolor="black")
    plt.title("Edge Weight Distribution")
    plt.xlabel("Edge Weight")
    plt.ylabel("Frequency")


    modularity_values = np.array(all_metrics["modularity"])
    assortativity_values = np.array(all_metrics["assortativity"])
    # edge_weights_values = np.array(all_metrics["edge_weight"])
    # mean_edge_weights = np.array([np.mean([np.mean(weights) for weights in iteration_weights]) for iteration_weights in edge_weights_values.T])
    # median_edge_weights = np.array([np.median([np.median(weights) for weights in iteration_weights]) for iteration_weights in edge_weights_values.T])

    # Get a list of all available colormaps
    colormap_names = plt.colormaps() 
    colormap_names = colormap_names[::max(1, len(colormap_names)//num_Networks)]  # Select a subset
   
    print('UTILS 0 - assortativity', assortativity_values)
    print('UTILS 0 - modularity', modularity_values)
    # print('UTILS 0 - edge_weights', edge_weights_values)

    plt.subplot(4, 2, 3)
    for i, degree in enumerate(all_metrics["degree_histogram"]):
        plt.plot(range(iterationsTot), degree, alpha=0.3, color=colours(i), marker='o', label=f'Network {i+1}' )
    plt.title("Average Degree Distribution")
    plt.xlabel("Iterations")
    plt.ylabel("Degree")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.1),fontsize='small')

    plt.subplot(4, 2, 4)    
    for i, modularity in enumerate(all_metrics["modularity"]):
        plt.plot(range(iterationsTot), modularity, alpha=0.3, color=colours(i), marker='o', label=f'Network {i+1}' )
    plt.title("Modularity Distribution\nMean: {:.2f}".format(mean_modularity))
    plt.xlabel("Iterations")
    plt.ylabel("Modularity")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.1),fontsize='small')

    plt.subplot(4, 2, 7)
    for i, assort in enumerate(all_metrics["assortativity"]):
        plt.plot(range(iterationsTot), assort, alpha=0.3, color=colours(i), marker='o', label=f'Network {i+1}' )
    plt.title("Assortativity Distribution\nMean: {:.2f}".format(mean_assortativity))
    plt.xlabel("Iterations")
    plt.ylabel("Assortativity")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.1),fontsize='small')

    plt.subplot(4, 2, 8)
    plt.title("Edge Weight Distribution")
    plt.xlabel("Edge Weight")
    plt.ylabel("Frequency")

    for net in range(num_Networks):
        # cmap = get_cmap(cmap_list[net_index % len(cmap_list)], iterationsTot)
        cmap_name = colormap_names[net % len(colormap_names)]
        cmap = get_cmap(cmap_name, len(all_metrics["edge_weight"][net]))
        
        print('len(colormap_names)', len(colormap_names))
        print('cmap_name]', cmap_name)
        print('cmap]', cmap)

        for i, edge_weight in enumerate(all_metrics["edge_weight"][net]):
            edge, counts = np.unique(edge_weight, return_counts=True)
            plt.plot(edge, counts, alpha=0.3, color=cmap(i), marker='o', label=f'Network {net+1}, Iteration {i+1}' if i == 0 else "")
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.1),fontsize='small')
    plt.tight_layout()
    plt.show()


# def plot_tipping_points(metrics, tipping_points, metric_name):
#     plt.figure(figsize=(10, 6))
#     plt.plot(metrics, label=metric_name)
#     for tp in tipping_points:
#         plt.axvline(x=tp, color='r', linestyle='--', label='Tipping Point' if tp == tipping_points[0] else "")
#     plt.xlabel('Iteration')
#     plt.ylabel(metric_name)
#     plt.title(f'{metric_name} Over Iterations with Tipping Points')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def plot_all_tipping_points(all_tipping_points):
#     fig, ax = plt.subplots(figsize=(10, 6))
#     for num_agents, skill_type, tp in all_tipping_points:
#         ax.scatter(num_agents, tp, label=f'{skill_type}', alpha=0.6, edgecolors='w', s=100)
#     ax.set_xlabel('Number of Agents')
#     ax.set_ylabel('Iteration of Tipping Point')
#     ax.set_title('Tipping Points for Cooperation Proportion')
#     ax.legend(title='Skill Type')
#     plt.grid(True)
#     plt.show()

def check_constant_segments(all_metrics):
    for key, values in all_metrics.items():
        for metric_list in values:
            if isinstance(metric_list, list) and len(metric_list) > 1:
                if all(x == metric_list[0] for x in metric_list):
                    print(f"Constant segment found in {key}")


# Plot for main.py

def plot_all_results(results, iterationsTot):
    
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(wspace=0.3, hspace=1, right=0.85)

    # Collect all unique number of agents and skill types
    num_agents_list = sorted(set(num_agents for (num_agents, skill_type) in results.keys()))
    skill_types = sorted(set(skill_type for (num_agents, skill_type) in results.keys()))

    skills_colours = cm.get_cmap('viridis', len(skill_types))
    skills_colours_dict = {skill_types[i]: skills_colours(i / len(skill_types)) for i in range(len(skill_types))}
    
    # Plotting the tipping points for skill_type vs num_agents
    fig1, ax1 = plt.subplots(figsize=(15, 10))
    added_skill_types = set()
    for skill_type in skill_types:
        for num_agents in num_agents_list:
            if (num_agents, skill_type) in results:
                tipping_points = results[(num_agents, skill_type)]["tipping_points"]
                
                for tp in tipping_points:
                    if skill_type not in added_skill_types:
                        ax1.scatter(skill_type, num_agents, color=skills_colours_dict[skill_type], 
                                    marker='o', s=100, label=f"{skill_type} ({num_agents} agents)")
                        added_skill_types.add(skill_type)
                    else:
                        ax1.scatter(skill_type, num_agents, color=skills_colours_dict[skill_type], 
                                    marker='o', s=100)

    ax1.set_xlabel('Skill Type')
    ax1.set_ylabel('Number of Agents')
    ax1.set_title('Tipping Points by Skill Type and Number of Agents')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.grid(True)
    
     
    # Plotting disaggregated data
    fig1, (ax2, ax3, ax4) = plt.subplots(3, 1, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5)
    added_skill_types = set()

    for num_agents in num_agents_list:
        for skill_type in skill_types:
            # Get base color for the number of agents
            base_colour = skills_colours_dict[skill_type]

            for (num_agents_key, skill_type_key), data in results.items():
                
                all_metrics = data["metrics"]
                tipping_points = data["tipping_points"]

                for i, coop_per_network in enumerate(all_metrics["cooperation_proportion"]):
                    
                    if skill_type_key not in added_skill_types:
                        ax2.scatter(np.arange(1, iterationsTot + 1), coop_per_network,
                                    color=skills_colours_dict[skill_type_key], marker='o',
                                    label=skill_type_key)  # Add label only once
                        added_skill_types.add(skill_type_key)
                    else:
                        ax2.scatter(np.arange(1, iterationsTot + 1), coop_per_network, 
                                    color=skills_colours_dict[skill_type_key], marker='o')

                    
                    # Adding tipping points for the network level--> run = tp[2], tipping_point = tp[3]
                    network_tipping_points = [tp[3] for tp in tipping_points if tp[2] == i]
                    for tp in network_tipping_points:
                        if 0 <= tp < iterationsTot:
                            ax3.plot(tp, coop_per_network[tp], 'o', color='black', markersize=6)
                    
                    for tp in network_tipping_points:
                        if 0 <= tp < iterationsTot:
                            ax3.axvline(tp, coop_per_network[tp], linestyle='--', color='black', markersize=6)

                    # for i, coop_per_network in enumerate(all_metrics["cooperation_proportion"]):
                    #     plt.plot(range(iterationsTot), coop_per_network, alpha=0.6, color=colours(i), marker='o', label=f'Network {i+1}')
                    # plt.plot(range(iterationsTot), aggregated_coop_proportion, label='Mean Cooperation Proportion', color='blue', linewidth=2)
                    # plt.plot(range(iterationsTot), stats["median"]["cooperation_proportion"], label='Median Cooperation Proportion', color='red', linestyle='--', linewidth=2)
                    
                    # for tp in aggregated_tipping_points:
                    #     plt.axvline(x=tp, color='black', linestyle='--', label='Aggregated Tipping Point' if tp == aggregated_tipping_points[0] else "")
                

    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Cooperation Proportion')
    ax2.set_title('Cooperation Proportion per Network')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Cooperation Proportion at Tipping Points')
    ax3.set_title('Tipping Points in Cooperation Proportion')
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
                
    # Aggregated data
    plt.subplot(3, 1, 3)
    aggregated_colours = get_cmap('tab20', len(results))
    aggregated_colours_list = [aggregated_colours(i) for i in range(len(results))]

    
    for i, ((num_agents, skill_type), data) in enumerate(results.items()): 

        stats = data["stats"]
        aggregated_coop_proportion = stats["mean"]["cooperation_proportion"]
        aggregated_tipping_points = detect_tipping_points(aggregated_coop_proportion)

        ax4.plot(range(iterationsTot), aggregated_coop_proportion,
                 label=skill_type, 
                 color=aggregated_colours_list[i], linewidth=2)
        for tp in aggregated_tipping_points:
            if 0 <= tp < iterationsTot:
                ax4.plot(tp, aggregated_coop_proportion[tp], 'o', color='black', markersize=6)

        ax4.set_xlabel('Iterations')
        ax4.set_ylabel('Mean Cooperation Proportion')
        ax4.set_title('Mean Cooperation Proportion with Tipping Points')
        ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Adjusted legend position

    # Set x-axis tick formatter to remove decimals
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()
        

# def plot_all_results(results, iterationsTot):
    
#     # Set up for the first type of plot: Tipping points as a function of skill_type and num_agents
#     plt.figure(figsize=(15, 10))
#     plt.subplots_adjust(wspace=0.3, hspace=1, right=0.85)
#     num_agents_list = sorted(set(num_agents for (num_agents, skill_type) in results.keys()))
#     skill_types = sorted(set(skill_type for (num_agents, skill_type) in results.keys()))
#     skills_colours = get_cmap('viridis', len(skill_types))
#     skills_colours_dict = {skill_types[i]: skills_colours(i / len(skill_types)) for i in range(len(skill_types))}

#     # Plotting the tipping points for skill_type vs num_agents
#     fig1, ax1 = plt.subplots(figsize=(15, 10))
#     for skill_type in skill_types:
#         for num_agents in num_agents_list:
#             if (num_agents, skill_type) in results:
#                 tipping_points = results[(num_agents, skill_type)]["tipping_points"]
#                 for tp in tipping_points:
#                     run = tp[2]
#                     tipping_point = tp[3]
#                     ax1.scatter(skill_type, num_agents, color=skills_colours_dict[skill_type], marker='o', s=100, label=f"{skill_type} ({num_agents} agents)")

#     ax1.set_xlabel('Skill Type')
#     ax1.set_ylabel('Number of Agents')
#     ax1.set_title('Tipping Points by Skill Type and Number of Agents')
#     ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     ax1.grid(True)

#     # Set up for the second type of plot: Cooperation proportions and other metrics
#     fig2, (ax2, ax3, ax4, ax5) = plt.subplots(4, 1, figsize=(15, 15))
#     plt.subplots_adjust(hspace=0.5)

#     added_skill_types = set()
#     for num_agents in num_agents_list:
#         coop_proportions_per_skill = []
#         for skill_type in skill_types:
#             if (num_agents, skill_type) in results:
#                 all_metrics = results[(num_agents, skill_type)]["metrics"]
#                 tipping_points = results[(num_agents, skill_type)]["tipping_points"]

#                 # Plot cooperation proportion per network over iterations (ax2)
#                 for i, coop_per_network in enumerate(all_metrics["cooperation_proportion"]):
#                     if skill_type not in added_skill_types:
#                         ax2.plot(np.arange(1, iterationsTot + 1), coop_per_network,
#                                  color=skills_colours_dict[skill_type], marker='o', label=skill_type)
#                         added_skill_types.add(skill_type)
#                     else:
#                         ax2.plot(np.arange(1, iterationsTot + 1), coop_per_network,
#                                  color=skills_colours_dict[skill_type], marker='o')

#                 # Adding cooperation proportion per skill_type over num_agents (ax3)
#                 mean_coop_proportion = np.mean(all_metrics["cooperation_proportion"], axis=0)
#                 ax3.plot(num_agents, mean_coop_proportion, 'o', color=skills_colours_dict[skill_type], label=skill_type)

#                 # Aggregate cooperation proportion with tipping points for each (skill_type, num_agents) (ax4)
#                 stats = results[(num_agents, skill_type)]["stats"]
#                 aggregated_coop_proportion = stats["mean"]["cooperation_proportion"]
#                 aggregated_tipping_points = detect_tipping_points(aggregated_coop_proportion)
#                 ax4.plot(range(iterationsTot), aggregated_coop_proportion,
#                          label=f"{skill_type} ({num_agents} agents)",
#                          color=skills_colours_dict[skill_type], linewidth=2)
#                 for tp in aggregated_tipping_points:
#                     if 0 <= tp < iterationsTot:
#                         ax4.plot(tp, aggregated_coop_proportion[tp], 'o', color='black', markersize=6)

#                 # Aggregate cooperation proportion with tipping points for each skill_type over num_agents (ax5)
#                 coop_proportions_per_skill.append(aggregated_coop_proportion)

#         # Plot ax5 for aggregated cooperation proportion across skill_types for current num_agents
#         if coop_proportions_per_skill:
#             mean_coop_proportion = np.mean(coop_proportions_per_skill, axis=0)
#             ax5.plot(num_agents, mean_coop_proportion, 'o', label=f"{num_agents} agents",
#                      color='blue')

#     ax2.set_xlabel('Iterations')
#     ax2.set_ylabel('Cooperation Proportion')
#     ax2.set_title('Cooperation Proportion per Network')
#     ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

#     ax3.set_xlabel('Skill Type')
#     ax3.set_ylabel('Cooperation Proportion')
#     ax3.set_title('Cooperation Proportion per Skill Type over Number of Agents')
#     ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

#     ax4.set_xlabel('Iterations')
#     ax4.set_ylabel('Mean Cooperation Proportion')
#     ax4.set_title('Mean Cooperation Proportion with Tipping Points')
#     ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     ax4.xaxis.set_major_locator(MaxNLocator(integer=True))

#     ax5.set_xlabel('Number of Agents')
#     ax5.set_ylabel('Mean Cooperation Proportion')
#     ax5.set_title('Mean Cooperation Proportion by Skill Type over Number of Agents')
#     ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     ax5.xaxis.set_major_locator(MaxNLocator(integer=True))

#     plt.show()  


# Plot for dummy_data.py
def plot_all_tipping_points(all_tipping_points):

    fig, ax = plt.subplots(figsize=(10, 6))
    unique_skill_types = list(set([skill_type for _, skill_type, _ in all_tipping_points]))
    color_map = plt.get_cmap('tab10')
    color_dict = {skill_type: color_map(i) for i, skill_type in enumerate(unique_skill_types)}
    
    for num_agents, skill_type, tp in all_tipping_points:
        ax.scatter(num_agents, tp, label=skill_type, color=color_dict[skill_type], alpha=0.6, edgecolors='w', s=100)
    
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Iteration of Tipping Point')
    ax.set_title('Tipping Points for Cooperation Proportion')
    # Create a custom legend to avoid duplicate entries
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=skill_type, markersize=10, markerfacecolor=color_dict[skill_type]) for skill_type in unique_skill_types]
    # ax.legend(handles=handles, title='Skill Type')
    
    # Create a legend outside the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=handles,title='Skill Type', loc='center left', bbox_to_anchor=(1, 0.5))

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.grid(True)
    plt.show()


def save_metrics_to_csv(results):
    # Define the fieldnames for the CSV
    fieldnames = [
        'num_agents', 'skill_type', 'clustering_coefficient', 'average_path_length',
        'degree_distribution', 'degree_histogram', 'cooperation_proportion', 'density',
        'assortativity', 'modularity', 'edge_weight'
    ]

    # Open the CSV file for writing
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over the results and write each entry to the CSV
        for (num_agents, skill_type), data in results.items():
            metrics = data['metrics']

            # Prepare the row dictionary
            row = {
                'num_agents': num_agents,
                'skill_type': skill_type,
                'clustering_coefficient': metrics['clustering_coefficient'],
                'average_path_length': metrics['average_path_length'],
                'degree_distribution': metrics['degree_distribution'],
                'degree_histogram': metrics['degree_histogram'],
                'cooperation_proportion': metrics['cooperation_proportion'],
                'density': metrics['density'],
                'assortativity': metrics['assortativity'],
                'modularity': metrics['modularity'],
                'edge_weight': metrics['edge_weight']
            }

            writer.writerow(row)

                
# def plot_tipping_points(coop_proportion,tipping_points):
#     # Plot cooperation proportion and tipping points
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(len(coop_proportion)), coop_proportion, label='Cooperation Proportion')
#     # for tp in tipping_points:
#     #     plt.axvline(x=tp, color='red', linestyle='--', label='Tipping Point' if tp == tipping_points[0] else "")
#     plt.xlabel('Iterations')
#     plt.ylabel('Proportion of Cooperation')
#     plt.title('Cooperation Proportion and Tipping Points')
#     plt.legend()
#     plt.show()


# def plot_tipping_points(coop_proportion,tipping_points):
#     # Plot cooperation proportion and tipping points
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(len(coop_proportion)), coop_proportion, label='Cooperation Proportion')
#     # for tp in tipping_points:
#     #     plt.axvline(x=tp, color='red', linestyle='--', label='Tipping Point' if tp == tipping_points[0] else "")
#     plt.xlabel('Iterations')
#     plt.ylabel('Proportion of Cooperation')
#     plt.title('Cooperation Proportion and Tipping Points')
#     plt.legend()
#     plt.show()



def summary_OLD(G: nx.Graph):
    
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    hist = [degree_sequence.count(i) for i in range(max(degree_sequence) + 1)]
    print('degree_sequence', degree_sequence)
    print('hist', hist)

    communities = nx_comm.greedy_modularity_communities(G)
    modularity = nx_comm.modularity(G, communities)

    # print('communities', communities)
    # print('modularity', modularity)

    assortativity = nx.degree_assortativity_coefficient(G)
    # print('assortativity', assortativity)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = (2 * num_edges) / (num_nodes * (num_nodes - 1))
    # print('density', density)

    # Extract edge weights
    edge_weights = [G[node1][node2]["weight"] for node1, node2 in G.edges()]

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 4, 1)
    plt.bar(range(len(hist)), hist, width=0.8, color="skyblue")
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")

    plt.subplot(1, 4, 2)
    pos = nx.spring_layout(G)
    update_graph(G)  # Update graph before drawing
    nx.draw(G, pos, node_color="lightgray", with_labels=False)
    plt.title("Modularity: {:.2f}".format(modularity))

    plt.subplot(1, 4, 3)
    plt.hist([d for n, d in G.degree()], bins=np.arange(min(degree_sequence), max(degree_sequence) + 2) - 0.5,
             color="salmon", edgecolor="black")
    plt.title("Assortativity: {:.2f}".format(assortativity))
    plt.xlabel("Degree")
    plt.ylabel("Frequency")

    plt.subplot(1, 4, 4)
    plt.hist(edge_weights, bins=20, color="purple", edgecolor="black")
    plt.title("Edge Weight Distribution")
    plt.xlabel("Edge Weight")
    plt.ylabel("Frequency")

    plt.show(block=True)