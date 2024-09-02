import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import seaborn as sns
from networkx.algorithms import community
import cpnet
from matplotlib.ticker import MaxNLocator


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
        other_size = sum(len(comp) for comp in components if comp != largest_component)
        # in_size = 0  # Not applicable for undirected graphs

        # out_size = 0  # Not applicable for undirected graphs

        return {
            "scc_size": scc_size, #(Strongly Connected Component Size):
            # "in_size": in_size,
            # "out_size": out_size,
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


# -------

def holme_metric(G):
    communities = community.greedy_modularity_communities(G, weight='weight')
    modularity = community.modularity(G, communities, weight='weight')
    return modularity




# Plot the variation of metrics over iterations
# def plot_metric(metric_name, title, metrics_df):

#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=metrics_df, x='iteration', y=metric_name, hue='skill_type', style='num_agents', markers=True, dashes=False)
#     plt.title(title)
#     plt.xlabel('Iteration')
#     plt.ylabel(metric_name)
#     plt.legend(title='Skill Type & Number of Agents')
#     plt.grid(True)
#     plt.show()
    
def plot_metric(metric_name, title, metrics_df):
    plt.figure(figsize=(10, 6))
    
    # Group data by skill type and number of agents
    grouped = metrics_df.groupby(['skill_type', 'num_agents'])
    
    for (skill_type, num_agents), group in grouped:
        plt.plot(group['iteration'], group[metric_name], marker='o', label=f'{skill_type}, {num_agents} agents')
    
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
            # subset = subset.sort_values(by='iteration')  # Sort by iteration to avoid looping lines
            plt.plot(subset['iteration'], subset[metric_name], label=f'{skill_type}, {num_agents} agents', linestyle=line_styles[idx % len(line_styles)], marker='o')

            # # Fit a trend line
            # z = np.polyfit(subset['iteration'], subset[metric_name], 1)
            # p = np.poly1d(z)
            # plt.plot(subset['iteration'], p(subset['iteration']), linestyle=line_styles[idx % len(line_styles)], color='gray', alpha=0.5)

    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(metric_name)
    plt.legend(title='Skill Type & Number of Agents')
    plt.grid(True)
    plt.show()


#  CPNET
# https://github.com/skojaku/core-periphery-detection/blob/master/README.md
# Function to detect core-periphery, perform statistical test, and extract significant coreness

def get_significant_coreness(group):
    
    G = nx.from_pandas_edgelist(group, 'source', 'target', ['weight'])
    algorithm = cpnet.KM_config()
    algorithm.detect(G)
    c = algorithm.get_pair_id()
    x = algorithm.get_coreness()
    
    # Perform the statistical test
    sig_c, sig_x, significant, p_values = cpnet.qstest(c, x, G, algorithm, significance_level=0.05)
    
    # Filter out None values from sig_x
    significant_coreness_values = [value for value in sig_x.values() if value is not None]
    
    # Check if there are any significant coreness values
    if significant_coreness_values:
        return sum(significant_coreness_values) / len(significant_coreness_values)  # Average significant coreness
    else:
        return None
    

# PLOT CORENESS

def plot_coreness(results_df):

    fig, ax = plt.subplots(figsize=(12, 6))
    for key, grp in results_df.groupby(['network_id', 'num_agents', 'skill_type']):
        ax.plot(grp['iteration'], grp['average_coreness'], marker='o', label=f'Network: {key[0]}, Agents: {key[1]}, Skill: {key[2]}')

    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Significant Coreness')
    ax.set_title('Evolution of Significant Core-Periphery Structure')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) 
    plt.show()

def plot_coreness_snapshots(edge_df, iterations):
    
    fig, axs = plt.subplots(1, len(iterations), figsize=(18, 6))
    
    for i, iteration in enumerate(iterations):
        ax = axs[i]
        
        for (network_id, skill_type, num_agents), group in edge_df.groupby(['network_id', 'skill_type', 'num_agents']):
            sub_group = group[group['iteration'] == iteration]
            
            G = nx.Graph()
            for _, row in sub_group.iterrows():
                G.add_edge(row['source'], row['target'], weight=row['weight'])
            
            # Detect core-periphery structure
            algorithm = cpnet.KM_config()
            algorithm.detect(G)
            coreness = algorithm.get_coreness()
            
            # Draw the graph
            pos = nx.spring_layout(G)  # Positioning for visualization
            core_nodes = [node for node, score in coreness.items() if score >= 0.5]
            periphery_nodes = [node for node, score in coreness.items() if score < 0.5]

            edge_width = 1
            edge_color ='grey'            
            
            nx.draw_networkx_nodes(G, pos, nodelist=core_nodes, node_color='LightSkyBlue',ax=ax, label='Core')
            nx.draw_networkx_nodes(G, pos, nodelist=periphery_nodes, node_color='red', ax=ax, label='Periphery')
            nx.draw_networkx_edges(G, pos, ax=ax, width=edge_width,edge_color=edge_color,alpha=0.5)
              
            # nx.draw_networkx_labels(G, pos, ax=ax)
            
            ax.set_title(f'Network {network_id}, Iteration {iteration}')
            ax.legend()
    
    plt.suptitle('Core-Periphery Structure at Different Iterations')
    plt.show()


def main():

    edge_df = pd.read_csv('edge_data75Spp.csv')
    iterationsTot = edge_df['iteration'].nunique()
    all_metrics = []   
    coreness = [] 
    grouped = edge_df.groupby(['network_id', 'skill_type', 'num_agents'])
    
    stage1 = iterationsTot // 5
    stage2 = iterationsTot // 2
    stage3 = iterationsTot - stage1
    iterations_to_plot = [stage1, stage2, stage3]

     
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
                # "in_size": bow_tie["in_size"],
                # "out_size": bow_tie["out_size"],
                "other_size": bow_tie["other_size"],
                "onion_structure": onion
            }
            
            all_metrics.append(metrics)

   
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('advanced_network_metrics.csv', index=False)


    # for CPNET package 
    
    for (network_id, iteration, num_agents, skill_type), group in edge_df.groupby(['network_id', 'iteration', 'num_agents', 'skill_type']):
        avg_coreness = get_significant_coreness(group)
        if avg_coreness is not None:
            coreness.append({'network_id': network_id, 'iteration': iteration, 'num_agents': num_agents, 'skill_type': skill_type, 'average_coreness': avg_coreness})

    coreness_df = pd.DataFrame(coreness)    


    # PLOTS 1.  

    # plot_metric_line('core_periphery_coefficient', 'Core-Periphery Coefficient over Iterations',metrics_df)
    # plot_metric_line('rich_club_coefficient', 'Rich-Club Coefficient over Iterations',metrics_df)
    # plot_metric_line('nestedness_index', 'Nestedness Index over Iterations',metrics_df)
    # plot_metric_line('onion_structure', 'Onion Structure (Kendall Tau) over Iterations',metrics_df)
    # plot_metric_line('scc_size', 'SCC Size over Iterations',metrics_df) # NOT WORKING
    # plot_metric_line('other_size', 'Other Component Size over Iterations', metrics_df)

    # plot_metric_line('in_size', 'In Component Size over Iterations',metrics_df)
    # plot_metric_line('out_size', 'Out Component Size over Iterations',metrics_df)

   
    # PLOTS 2.
    plot_coreness(coreness_df)
    plot_coreness_snapshots(coreness_df, iterations_to_plot)



if __name__ == "__main__":
    main()