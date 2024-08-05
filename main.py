
from run_network_analysis import *
import pickle
from utils import save_metrics_to_csv
from time_series_analysis import split_multistage_values


#  dummy_data import *
# from dummy_data import generate_dummy_data
# from dummy_data import save_data_to_csv
# from dummy_data import load_data_from_csv
# from dummy_data import plot_cooperation_and_tipping_points

def main():

    overall_start_time = time.time()
    num_networks = 2
    iterationsTot = 2
    num_agents_list = [2,3] #[400,300,200,100,50] #40,50,60,70,80,90,100,150,200,
    skill_types = ["S++", "S"]
    
    # ["S++", "S", "br2", "manipulator-gf", "eeew_simp", "gigawolf", "random"]#,"manipulator-bully","manipulator-gf" "S++_simp", "S", "fp","br1"]
    
    # Ranking
    # "S++","manipulator-bully", "S++_simp", "S", "fp","br1","eeew","br2","memory1","mqubed","memory2",
    # "manipulator-gf", "wolf","qlearn","godfather", "eeew_simp", "exp3w", "cjal","pavlov","gigawolf", "wma","sfp","exp3w_simp", "random"

    #  Available algorithms
    # "S","S_simp","S++","S++_simp","exp3w","exp3w_simp","exp3w++","eeew","eeew_simp","eeew++","ucbw","ucbw++","br1","br2","wolf",
    # "qlearn","mqubed","manipulator-bully","manipulator-gf","bully","godfather","random","a0","a1","expert","cjal","fp","exp3_a",
    # "gigawolf","wma","salg","sfp","pavlov","memory1","memory2","deepq","fPl_",

    
    all_tipping_points = []
    all_edge_data = []
    results = {}
    time_records = []
    all_metrics = {
            # "clustering_coefficient": [],
            # "average_path_length": [],
            # "degree_distribution": [],
            # "degree_histogram": [],
            "cooperation_proportion": []
            # "graphs": [],
            # "density": [],
            # "assortativity": [],
            # "modularity": [],
            # "edge_weight": []
            }

    for num_agents in num_agents_list:

        for skill_type in skill_types:
         
            start_time = time.time()
            print(f'Running analysis for {num_agents} agents with skill {skill_type}')
            all_metrics, stats, tipping_points, edge_data = run_network_analysis(num_agents, skill_type, num_networks, iterationsTot)
            all_tipping_points.extend(tipping_points)
            elapsed_time = time.time() - start_time
            time_records.append((num_agents, skill_type, elapsed_time))
            all_edge_data.extend(edge_data)  # Accumulate edge data

            results[(num_agents, skill_type)] = {
                "metrics": all_metrics,
                "stats": stats,
                "tipping_points": all_tipping_points
            }
    
    check_constant_segments(all_metrics)       
    overall_elapsed_time = time.time() - overall_start_time
    print(elapsed_time)

    # Save the elapsed time and parameters to a text file
    with open('elapsed_time.txt', 'w') as f:
        for record in time_records:
            f.write(f"Elapsed time for {record[1]} with {record[0]} agents: {record[2]:.2f} seconds\n")
        f.write(f"Total elapsed time: {overall_elapsed_time:.2f} seconds")

    save_metrics_to_csv(results)
    records = flatten_results_to_records(results)
    df = pd.DataFrame(records)
    for col in df.columns[3:]:
      df[col] = split_string_column(df[col]) # Apply the split function to each metric column (in order to create as many rows as iterations)
   
    expanded_df = expand_dataframe(df,iterationsTot)
    expanded_df.to_csv('records.csv', index=False)

    # Save the accumulated edge data to a CSV file
    edge_df = pd.DataFrame(all_edge_data)
    edge_df.to_csv('edge_data.csv', index=False)

    # # plot_all_tipping_points(all_tipping_points)
    # plot_all_results(results, iterationsTot,num_networks)

    # df1 = pd.read_csv('results3.csv')
    # df2 = pd.read_csv('results4.csv')
    # df = pd.concat([df1, df2], ignore_index=True)
    # df.to_csv('combined_results.csv', index=False)
    
    
if __name__ == "__main__":
    main()




            
# example_data = [
# ([0.05, 0.05, 0.03, 0.06, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], "S++", 50),
# ([0.001, 0.001, 0.001, 0.001, 0.2, 0.2, 0.2, 0.6, 0.6, 0.7, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], "S", 60),
# ([0.001, 0.001, 0.001, 0.001,0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.3, 0.2, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], "S_simp", 70)
# ]
# all_tipping_points = []
# for data, skill_type, num_agents in example_data:
#     tipping_points = detect_tipping_points(data)
#     all_tipping_points.extend([(num_agents, skill_type, tp) for tp in tipping_points])
# plot_all_tipping_points(all_tipping_points)



# def main():
#     skill_types = ["S++", "S", "S_simp", "S++_simp", "exp3w", "exp3w_simp", "exp3w++", "eeew", "eeew_simp", "eeew++", "ucbw"]
#     # ,"ucbw++", "br2", "wolf", "qlearn", "br1", "mqubed", "manipulator-bully", "manipulator-gf", "bully", "godfather", "random", "a0"]
#     #  ,"a1", "expert (6)", "expertExists", "cjal", "fp", "exp3_a", "gigawolf", "wma", "salg", "sfp", "pavlov", "memory1", "memory2", 
#     #  "deepq", "fPl_"]
#     num_agents_list = range(10, 2010, 10)
#     iterations = 2000
    
#     # Uncomment the following lines to generate and save the data (run only once)
#     dummy_data = generate_dummy_data(skill_types, num_agents_list, iterations)
#     save_data_to_csv(dummy_data, 'dummy_data.csv')
    
#     # Load the pre-generated data
#     data = load_data_from_csv('dummy_data.csv')
    
#     # Plot the metrics
#     plot_cooperation_and_tipping_points(data, iterations)

# if __name__ == "__main__":
#     main()