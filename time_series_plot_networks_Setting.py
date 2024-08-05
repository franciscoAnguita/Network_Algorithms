
from run_network_analysis import *
import pickle
from utils import save_metrics_to_csv
from time_series_analysis import split_multistage_values
from time_series_analysis import plot_tipping_points
from time_series_analysis import identify_tipping_points
# from dummy_data import *
# from dummy_data import generate_dummy_data
# from dummy_data import save_data_to_csv
# from dummy_data import load_data_from_csv
# from dummy_data import plot_cooperation_and_tipping_points

def main():

    expanded_df = pd.read_csv('records3.csv')

    iterationsTot = expanded_df['iteration'].nunique()
    skill_types = expanded_df['skill_type'].unique()
    num_networks = expanded_df['network'].nunique()
    num_agents_list = expanded_df['num_agents'].unique().tolist()
    
    # Plot specific stages (e.g., first 5 stages)
    stage1_length = iterationsTot // 5
    middle_start = iterationsTot // 2 - stage1_length // 2
    stage3_start = iterationsTot - stage1_length

    stage1 = list(range(stage1_length))
    stage2 = list(range(middle_start, middle_start + stage1_length))
    stage3 = list(range(stage3_start, iterationsTot))


    # Plot all combinations
    plot_all_combinations(expanded_df, 'assortativity', stage1, stage2, stage3,skill_types, num_agents_list)
    
    # # Plot for each number of agents
    # for num_agent in num_agents_list:
    #     df = expanded_df[expanded_df['num_agents'] == num_agent]
    #     plot_by_num_agents(df, 'cooperation_proportion', num_agent, stage1, stage2, stage3)
  
    # # Plot for each skill type
    # for skill_type in skill_types:
    #     df = expanded_df[expanded_df['skill_type'] == skill_type]
    #     plot_by_skill_type(df, 'cooperation_proportion', skill_type, stage1, stage2, stage3)


    
    # plot_metrics_mean_median(expanded_df,'cooperation_proportion', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)
    # plot_metrics_mean_median(expanded_df,'clustering_coefficient', num_networks, stage1, stage2, stage3,skill_type, num_agents_list)
    # plot_metrics_mean_median(expanded_df,'average_path_length', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)
    # plot_metrics_mean_median(expanded_df,'density', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)
    # plot_metrics_mean_median(expanded_df,'density', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)
    # plot_metrics_mean_median(expanded_df,'assortativity', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)
    # # For a sample of networks per skill type
    
    # for num_agent in num_agents_list:
    #     df = expanded_df[(expanded_df['num_agents'] == num_agent)]
    #     plot_metrics_mean_median(df,'cooperation_proportion', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)
    #     plot_metrics_mean_median(df,'clustering_coefficient', num_networks, stage1, stage2, stage3,skill_type, num_agents_list)
    #     plot_metrics_mean_median(df,'average_path_length', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)
    #     plot_metrics_mean_median(df,'density', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)
    #     plot_metrics_mean_median(df,'density', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)
    #     plot_metrics_mean_median(df,'assortativity', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)
        
    # for skill_type in skill_types:
    #     df = expanded_df[(expanded_df['skill_type'] == skill_type)]
    #     plot_metrics_mean_median(df,'cooperation_proportion', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)
    #     plot_metrics_mean_median(df,'clustering_coefficient', num_networks, stage1, stage2, stage3,skill_type, num_agents_list)
    #     plot_metrics_mean_median(df,'average_path_length', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)
    #     plot_metrics_mean_median(df,'density', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)
    #     plot_metrics_mean_median(df,'density', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)
    #     plot_metrics_mean_median(df,'assortativity', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)

         
    # # plot_all_tipping_points(all_tipping_points)
    # plot_all_results(results, iterationsTot,num_networks)
    
    # Tipping points
    # df_melt, tipping_points_df = identify_tipping_points(df,iterationsTot)
    # plot_skill_vs_agents(expanded_df)
    plot_skill_vs_agents_scatter(expanded_df) 
    plot_skill_vs_agents_heatmap(expanded_df)
    

    
if __name__ == "__main__":
    main()

