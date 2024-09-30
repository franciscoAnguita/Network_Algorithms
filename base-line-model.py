import numpy as np
import networkx as nx
import pandas as pd
import random
import itertools
import warnings
import matplotlib.pyplot as plt
import networkx.algorithms.community as nx_comm


def simple_random_model(G: nx.Graph, edge_creation_prob: float, edge_removal_prob: float) -> nx.Graph:
    """
    Simple random model where edges are created and removed based on random probabilities.
    """
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1 = nodes[i]
            node2 = nodes[j]
            if G.has_edge(node1, node2):
                # Randomly remove an existing edge
                if random.random() < edge_removal_prob:
                    G.remove_edge(node1, node2)
            else:
                # Randomly create a new edge
                if random.random() < edge_creation_prob:
                    G.add_edge(node1, node2, weight=10)
    return G


def run_multiple_networks(agent_sizes, iterationsTot, edge_creation_prob, edge_removal_prob, edge_data_paths, num_repetitions):
   
    all_metrics = {size: [] for size in agent_sizes}

    for agents in agent_sizes:
        print(f"Running simulation for {agents} agents across {num_repetitions} repetitions.")
        edge_data_path = edge_data_paths[agents]

        # Store each repetition's results
        all_runs_metrics = []

        for repetition in range(num_repetitions):
            print(f"Repetition {repetition + 1}/{num_repetitions} for {agents} agents.")
            metrics = run_simple_random_network(agents, iterationsTot, edge_creation_prob, edge_removal_prob, edge_data_path)
            all_runs_metrics.append(metrics)

        all_metrics[agents] = all_runs_metrics  # Store metrics for all repetitions for this agent size

    return all_metrics



def run_simple_random_network(num_agents: int, iterationsTot: int, edge_creation_prob: float, edge_removal_prob: float, edge_data_path: str):
    # Initialize the graph
    G = nx.erdos_renyi_graph(num_agents, p=0.1)  # Initial random graph
    df = pd.read_csv(edge_data_path)
    edges_per_iteration = df.groupby(['network_id', 'iteration']).size().reset_index(name='num_edges')

    # Initialize metrics storage
    metrics = {
        "clustering_coefficient": [],
        "average_path_length": [],
        "degree_distribution": [],
        "degree_histogram": [],
        "density": [],
        "assortativity": [],
        "modularity": [],
        "edge_weight": [],
        "connected_nodes": [],
        "graphs": []
    }

    for iteration in range(iterationsTot):
        print(f"Iteration {iteration + 1}/{iterationsTot}")

        # Get the target L from the edge data
        try:
            target_L = edges_per_iteration[(edges_per_iteration['iteration'] == iteration)]['num_edges'].values[0]
        except IndexError:
            target_L = G.number_of_edges()  # Fallback to the current L if missing

        # Adjust the number of edges to match the target L
        current_L = G.number_of_edges()

        if current_L > target_L:
            # Remove random edges to match L
            edges_to_remove = random.sample(list(G.edges()), current_L - target_L)
            G.remove_edges_from(edges_to_remove)
        elif current_L < target_L:
            # Add random edges to match L
            possible_edges = list(itertools.combinations(G.nodes(), 2))
            existing_edges = set(G.edges())
            available_edges = [e for e in possible_edges if e not in existing_edges]

            edges_needed = target_L - current_L

            if len(available_edges) >= edges_needed:
                edges_to_add = random.sample(available_edges, edges_needed)
            else:
                # If not enough edges are available, add all remaining possible edges
                print(f"Warning: Only {len(available_edges)} edges available, "
                    f"but {edges_needed} are needed. Adding all available edges.")
                edges_to_add = available_edges

            G.add_edges_from(edges_to_add)

        # Run the simple random interaction model
        G = simple_random_model(G, edge_creation_prob, edge_removal_prob)

        # Collect metrics after the graph has been updated
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

        # Collect metrics
        metrics["density"].append(density)
        metrics["assortativity"].append(assortativity)
        metrics["modularity"].append(modularity)
        metrics["edge_weight"].append(edge_weights)
        metrics["graphs"].append(G.copy())  # Store a copy of the graph at this iteration

        if nx.is_connected(G):
            try:
                avg_path_length = nx.average_shortest_path_length(G)
            except nx.NetworkXError:
                avg_path_length = float('inf')
                print("NetworkXError: Cannot calculate average path length on a disconnected graph.")
        else:
            avg_path_length = float('inf')  # Or another measure for disconnected graphs

        metrics["average_path_length"].append(avg_path_length)
        metrics["clustering_coefficient"].append(nx.average_clustering(G))
        metrics["degree_distribution"].append([d for n, d in G.degree()])
        metrics["degree_histogram"].append(nx.degree_histogram(G))
        metrics["connected_nodes"].append(len(max(nx.connected_components(G), key=len)))

        
    return metrics


def plot_comparison_metrics(all_metrics, iterationsTot, agent_sizes):
  
    iterations = range(1, iterationsTot + 1)
    
    # Generate a gradient of blue colors for the lines
    blue_colors = plt.cm.Blues(np.linspace(0.45, 0.85, len(agent_sizes)))  # Adjust the range for desired color intensity

    plt.figure(figsize=(10, 6))

    for idx, agents in enumerate(agent_sizes):
        metrics_list = all_metrics[agents]  # List of metrics from multiple runs
        
        for metrics in metrics_list:
            plt.plot(iterations, metrics["clustering_coefficient"], label=f'{agents} Agents', 
                     color=blue_colors[idx], marker='o', markersize=5, markeredgecolor='black', alpha=0.5)
            
        plt.plot(iterations, metrics_list[0]["clustering_coefficient"], 
                 label=f'{agents} Agents', color=blue_colors[idx], marker='o', markersize=5, markeredgecolor='black')


    plt.xlabel('Iteration')
    plt.ylabel('Clustering Coefficient')
    plt.title('Clustering Coefficient Comparison Over Iterations')   
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot Average Path Length
    plt.figure(figsize=(10, 6))
    for idx, agents in enumerate(agent_sizes):
        metrics_list = all_metrics[agents]
        for metrics in metrics_list:
            plt.plot(iterations, metrics["average_path_length"], label=f'{agents} Agents', 
                     color=blue_colors[idx], marker='o', markersize=5, markeredgecolor='black', alpha=0.5)

    plt.xlabel('Iteration')
    plt.ylabel('Average Path Length')
    plt.title('Average Path Length Comparison Over Iterations')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot Density
    plt.figure(figsize=(10, 6))
    for idx, agents in enumerate(agent_sizes):
        metrics_list = all_metrics[agents]
        for metrics in metrics_list:
            plt.plot(iterations, metrics["density"], label=f'{agents} Agents', 
                     color=blue_colors[idx], marker='o', markersize=5, markeredgecolor='black', alpha=0.5)
            
        plt.plot(iterations, metrics_list[0]["average_path_length"], label=f'{agents} Agents', color=blue_colors[idx], marker='o', markersize=5, markeredgecolor='black')

    plt.xlabel('Iteration')
    plt.ylabel('Density')
    plt.title('Density Comparison Over Iterations')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot Assortativity
    plt.figure(figsize=(10, 6))
    for idx, agents in enumerate(agent_sizes):
        metrics_list = all_metrics[agents]
        for metrics in metrics_list:
            plt.plot(iterations, metrics["assortativity"], label=f'{agents} Agents', 
                     color=blue_colors[idx], marker='o', markersize=5, markeredgecolor='black', alpha=0.5)

    plt.xlabel('Iteration')
    plt.ylabel('Assortativity Coefficient')
    plt.title('Assortativity Comparison Over Iterations')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot Modularity
    plt.figure(figsize=(10, 6))
    for idx, agents in enumerate(agent_sizes):
        metrics_list = all_metrics[agents]
        for metrics in metrics_list:
            plt.plot(iterations, metrics["modularity"], label=f'{agents} Agents', 
                     color=blue_colors[idx], marker='o', markersize=5, markeredgecolor='black', alpha=0.5)

    plt.xlabel('Iteration')
    plt.ylabel('Modularity')
    plt.title('Modularity Comparison Over Iterations')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot Connected Nodes (size of the largest connected component)
    plt.figure(figsize=(10, 6))
    for idx, agents in enumerate(agent_sizes):
        metrics_list = all_metrics[agents]
        for metrics in metrics_list:
            plt.plot(iterations, metrics["connected_nodes"], label=f'{agents} Agents', 
                     color=blue_colors[idx], marker='o', markersize=5, markeredgecolor='black', alpha=0.5)

    plt.xlabel('Iteration')
    plt.ylabel('Largest Connected Component Size')
    plt.title('Largest Connected Component Over Iterations')
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    # Parameters for the model
    agent_sizes = [25, 50, 75, 100]  # Different agent sizes
    iterationsTot = 30  # Number of iterations
    edge_creation_prob = 0.7 # Probability to create a new edge
    edge_removal_prob = 0.7  # Probability to remove an existing edge
    num_repetitions = 10  # Number of repetitions for each agent size

    # File paths for different agent sizes
    edge_data_paths = {
        25: 'edge_data_10-exp3w++_25_30iter.csv',
        50: 'edge_data_10-exp3w++_50_30iter.csv',
        75: 'edge_data_10-exp3w++_75_30iter.csv',
        100: 'edge_data_10-exp3w++_100_30iter.csv'
    }

    # 25: 'edge_data_exp3w++_25_ASP.csv',
    #     50: 'edge_data_exp3w++_50_ASP.csv',
    #     75: 'edge_data_exp3w++_75_ASP.csv',
    #     100: 'edge_data_exp3w++_100_ASP.csv'

    # Run the baseline model for multiple networks and average the metrics
    all_metrics = run_multiple_networks(agent_sizes, iterationsTot, edge_creation_prob, edge_removal_prob, edge_data_paths, num_repetitions)

    # Plot the comparison of metrics for the different agent sizes
    plot_comparison_metrics(all_metrics, iterationsTot, agent_sizes)


if __name__ == "__main__":
    main()
