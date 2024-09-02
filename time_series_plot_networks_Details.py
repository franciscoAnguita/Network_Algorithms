
from run_network_analysis import *
import pickle
from utils import save_metrics_to_csv
from time_series_analysis import split_multistage_values
from time_series_analysis import plot_tipping_points
from time_series_analysis import identify_tipping_points
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf
import statsmodels.api as sm

# from dummy_data import *
# from dummy_data import generate_dummy_data
# from dummy_data import save_data_to_csv
# from dummy_data import load_data_from_csv
# from dummy_data import plot_cooperation_and_tipping_points

def main():

    # ANOVA 1

    # # ANOVA between the constrained model and the null modell
    # df_constrained = pd.read_csv('records_test.csv')
    # df_unconstrained = pd.read_csv('records_test2.csv')
    # df_constrained['type'] = 1
    # df_unconstrained['type'] = 0
    # df_combined = pd.concat([df_constrained, df_unconstrained], ignore_index=True)

    # model = ols('cooperation_proportion ~ C(type)', data=df_combined).fit()
    # anova_table = anova_lm(model, typ=2)
    # print(anova_table)


    # sns.set_theme(style="whitegrid")
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(data=df_combined, x='iteration', y='cooperation_proportion', hue='type', marker='o', style='type', palette="tab10")
    # plt.title('Cooperation Proportion vs Iteration by Type')
    # plt.xlabel('Iteration')
    # plt.ylabel('Cooperation Proportion')
    # plt.show()

    # ANOVA 2

    df_unconstrained = pd.read_csv('results_test2.csv')
    df_constrained = pd.read_csv('results_test.csv')
    df_constrained['type'] = 1
    df_unconstrained['type'] = 0
    df_combined = pd.concat([df_constrained, df_unconstrained], ignore_index=True)
    
    # Determine the number of stages
    stage_count = df_combined['cooperation_proportion'].str.split(',').str.len().max()
    
    # Split variables data to create stages of time series (iterations)
    # for column in ['clustering_coefficient', 'average_path_length', 'cooperation_proportion', 'density', 'assortativity', 'modularity']:
    for column in ['clustering_coefficient', 'average_path_length', 'degree_distribution','degree_histogram','cooperation_proportion', 'density', 'assortativity', 'modularity','edge_weight']:
        df = split_multistage_values(df_combined, column, stage_count)

    df_melt = df_combined.melt(id_vars=['type'], 
                               value_vars=[f'cooperation_proportion_stage_{i+1}' for i in range(stage_count)],
                               var_name='stage', value_name='cooperation_proportion')



    # Fit the two-way ANOVA model
    model = smf.ols('cooperation_proportion ~ C(type) + C(stage) + C(type):C(stage)', data=df_melt).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)


"""
    # expanded_df = pd.read_csv('records3.csv')
    expanded_df = pd.read_csv('records_500_50_several.csv')

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
    plot_all_combinations(expanded_df, 'cooperation_proportion', stage1, stage2, stage3,skill_types, num_agents_list)
    plot_all_combinations(expanded_df, 'clustering_coefficient', stage1, stage2, stage3,skill_types, num_agents_list)
    plot_all_combinations(expanded_df, 'average_path_length', stage1, stage2, stage3,skill_types, num_agents_list)
    plot_all_combinations(expanded_df, 'density', stage1, stage2, stage3,skill_types, num_agents_list)
    
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
    # plot_metrics_mean_median(expanded_df,'assortativity', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)
    # # For a sample of networks per skill type
    
    # for num_agent in num_agents_list:
    #     df = expanded_df[(expanded_df['num_agents'] == num_agent)]
    #     plot_metrics_mean_median(df,'cooperation_proportion', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)
    #     plot_metrics_mean_median(df,'clustering_coefficient', num_networks, stage1, stage2, stage3,skill_type, num_agents_list)
    #     plot_metrics_mean_median(df,'average_path_length', num_networks, stage1, stage2, stage3, skill_type, num_agents_list)
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
    plot_skill_vs_agents_heatmap(expanded_df) """
    

    
if __name__ == "__main__":
    main()
