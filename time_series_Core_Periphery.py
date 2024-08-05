import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import seaborn as sns


# Function to calculate Core-Periphery Coefficient (CPC)
def core_periphery_coefficient(G):
    try:
        core_numbers = nx.core_number(G)
        max_core_number = max(core_numbers.values())
        core_nodes = [node for node, core_num in core_numbers.items() if core_num == max_core_number]
        periphery_nodes = [node for node in G.nodes() if node not in core_nodes]
        
        core_subgraph = G.subgraph(core_nodes)
        periphery_subgraph = G.subgraph(periphery_nodes)
        
        core_edges = core_subgraph.number_of_edges()
        periphery_edges = periphery_subgraph.number_of_edges()
        total_edges = G.number_of_edges()
        
        cpc = (core_edges + periphery_edges) / total_edges if total_edges > 0 else 0
        return cpc
    except Exception as e:
        print(f"Error calculating core-periphery coefficient: {e}")
        return np.nan
    

# Function to calculate Rich-Club Coefficient
def rich_club_coefficient(G):
    try:
        rc_coeff = nx.rich_club_coefficient(G, normalized=False)
        return np.mean(list(rc_coeff.values()))
    except:
        return np.nan
    

# Function to calculate Nestedness Index (NODF)
def nestedness_index(G):
    try:
        adjacency_matrix = nx.to_numpy_array(G)
        row_sums = adjacency_matrix.sum(axis=1)
        col_sums = adjacency_matrix.sum(axis=0)
        
        row_overlap = 0
        for i in range(len(row_sums)):
            for j in range(i + 1, len(row_sums)):
                row_overlap += (adjacency_matrix[i] * adjacency_matrix[j]).sum() / min(row_sums[i], row_sums[j])
        
        col_overlap = 0
        for i in range(len(col_sums)):
            for j in range(i + 1, len(col_sums)):
                col_overlap += (adjacency_matrix[:, i] * adjacency_matrix[:, j]).sum() / min(col_sums[i], col_sums[j])
        
        nestedness = (row_overlap + col_overlap) / (len(row_sums) + len(col_sums) - 2)
        return nestedness
    except:
        return np.nan


# Function to calculate Bow-Tie Structures
def bow_tie_structure(G):
    try:
        # Find all connected components
        components = list(nx.connected_components(G))
        largest_component = max(components, key=len)
        
        # Calculate sizes
        scc_size = len(largest_component)  # Largest component size
        in_size = 0  # Not applicable for undirected graphs
        out_size = 0  # Not applicable for undirected graphs
        other_size = sum(len(comp) for comp in components if comp != largest_component)

        return {
            "scc_size": scc_size,
            "in_size": in_size,
            "out_size": out_size,
            "other_size": other_size
        }
    except Exception as e:
        print(f"Error calculating bow-tie structure: {e}")
        return {"scc_size": np.nan, "in_size": np.nan, "out_size": np.nan, "other_size": np.nan}


# Function to calculate Onion Network structure
def onion_structure(G):
    try:
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        
        tau, _ = kendalltau(list(degree_centrality.values()), list(betweenness_centrality.values()))
        return tau
    except:
        return np.nan


# Plot the variation of metrics over iterations
def plot_metric(metric_name, title, metrics_df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=metrics_df, x='iteration', y=metric_name, hue='skill_type', style='num_agents', markers=True, dashes=False)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(metric_name)
    plt.legend(title='Skill Type & Number of Agents')
    plt.grid(True)
    plt.show()
    

def plot_metric_scatter(metric_name, title, metrics_df):
    
    plt.figure(figsize=(10, 6))
    skill_types = metrics_df['skill_type'].unique()
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', '+', 'x']

    for idx, skill_type in enumerate(skill_types):
        skill_df = metrics_df[metrics_df['skill_type'] == skill_type]
        for num_agents in skill_df['num_agents'].unique():
            subset = skill_df[skill_df['num_agents'] == num_agents]
            plt.scatter(subset['iteration'], subset[metric_name], label=f'{skill_type}, {num_agents} agents', marker=markers[idx % len(markers)])

    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(metric_name)
    plt.legend(title='Skill Type & Number of Agents')
    plt.grid(True)
    plt.show()


def plot_metric_line(metric_name, title, metrics_df):
    plt.figure(figsize=(8, 6))
    skill_types = metrics_df['skill_type'].unique()
    line_styles = ['-', '--', '-.', ':']

    for idx, skill_type in enumerate(skill_types):
        skill_df = metrics_df[metrics_df['skill_type'] == skill_type]
        for num_agents in skill_df['num_agents'].unique():
            subset = skill_df[skill_df['num_agents'] == num_agents]
            plt.plot(subset['iteration'], subset[metric_name], label=f'{skill_type}, {num_agents} agents', linestyle=line_styles[idx % len(line_styles)], marker='o')

    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(metric_name)
    plt.legend(title='Skill Type & Number of Agents')
    plt.grid(True)
    plt.show()

def main():

    edge_df = pd.read_csv('edge_data1.csv')
    all_metrics = []    
    grouped = edge_df.groupby(['network_id', 'skill_type', 'num_agents'])
    
    # Process each group
    for (network_id, skill_type, num_agents), group in grouped:
        
        iterations = group['iteration'].unique()
    
        for iteration in iterations:
            sub_group = group[group['iteration'] == iteration]
            
            G = nx.Graph()
            for _, row in sub_group.iterrows():
                G.add_edge(row['source'], row['target'], weight=row['weight'])
            
            # Calculate network metrics
            cpc = core_periphery_coefficient(G)
            rcc = rich_club_coefficient(G)
            nestedness = nestedness_index(G)
            bow_tie = bow_tie_structure(G)
            onion = onion_structure(G)
            
            metrics = {
                "network_id": network_id,
                "skill_type": skill_type,
                "num_agents": num_agents,
                "iteration": iteration,
                "core_periphery_coefficient": cpc,
                "rich_club_coefficient": rcc,
                "nestedness_index": nestedness,
                "scc_size": bow_tie["scc_size"],
                "in_size": bow_tie["in_size"],
                "out_size": bow_tie["out_size"],
                "other_size": bow_tie["other_size"],
                "onion_structure": onion
            }
            
            all_metrics.append(metrics)

   
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('advanced_network_metrics.csv', index=False)

   
    # plot_metric_scatter('rich_club_coefficient', 'Rich-Club Coefficient over Iterations',metrics_df)
    # plot_metric_scatter('nestedness_index', 'Nestedness Index over Iterations',metrics_df)
    # plot_metric('onion_structure', 'Onion Structure (Kendall Tau) over Iterations',metrics_df)
    # plot_metric_line('core_periphery_coefficient', 'Core-Periphery Coefficient over Iterations',metrics_df)
    
    plot_metric_line('scc_size', 'SCC Size over Iterations',metrics_df) # NOT WORKING
    plot_metric_line('in_size', 'In Component Size over Iterations',metrics_df)
    plot_metric_line('out_size', 'Out Component Size over Iterations',metrics_df)
    plot_metric_line('other_size', 'Other Component Size over Iterations', metrics_df)
   


if __name__ == "__main__":
    main()