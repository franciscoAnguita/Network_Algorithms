import random
import csv
import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt

def generate_dummy_data(skill_types, num_agents_list, iterations):
    data = []
    for skill_type in skill_types:
        for num_agent in num_agents_list:
            coop_proportion = [random.random() for _ in range(iterations)]
            for i in range(5, iterations, 100):
                coop_proportion[i] = coop_proportion[i] + 0.5 if coop_proportion[i] + 0.5 <= 1 else 1
            data.append((num_agent, skill_type, coop_proportion))
    return data

def save_data_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for num_agents, skill_type, coop_proportion in data:
            writer.writerow([num_agents, skill_type] + coop_proportion)


def load_data_from_csv(filename):
    data = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            num_agent = int(row[0])
            skill_type = row[1]
            coop_proportion = list(map(float, row[2:]))
            data.append((num_agent, skill_type, coop_proportion))
    return data



def detect_tipping_points(coop_proportion):
    coop_proportion_array = np.array(coop_proportion)
    if coop_proportion_array.ndim == 1:
        coop_proportion_array = coop_proportion_array.reshape(-1, 1)
    if coop_proportion_array.shape[0] < 2 or np.all(coop_proportion_array == coop_proportion_array[0]):
        return []
    model = "l2 "
    algo = rpt.Pelt(model=model).fit(coop_proportion_array)
    try:
        result = algo.predict(pen=1)
    except rpt.exceptions.BadSegmentationParameters:
        result = []
    return result



def plot_cooperation_and_tipping_points(data, iterations):
    
    plt.figure(figsize=(15, 10))
    colours = plt.get_cmap('tab10', len(data))
    plotted_labels = set()  # To keep track of plotted labels


    for i, (num_agents, skill_type, coop_proportion) in enumerate(data):
        # plt.plot(range(iterations), coop_proportion, alpha=0.6, color=colours(i), label=f'{skill_type} - {num_agents} agents')

        tipping_points = detect_tipping_points(coop_proportion)
        for tp in tipping_points:
            # Ensure the tipping point is within the valid range
            if 0 <= tp < iterations:
                label = f'{skill_type}'
                if label not in plotted_labels: 
                    plt.plot(tp, coop_proportion[tp], 'o', color=colours(i), markersize=3, label=label)
                    plotted_labels.add(label)
                else:
                    plt.plot(tp, coop_proportion[tp], 'o', color=colours(i), markersize=3)

    plt.xlabel('Iterations')
    plt.ylabel('Proportion of Cooperation')
    plt.title('Cooperation Proportions and Tipping Points for Different Networks')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.grid(True)
    plt.show()

def plot_metrics_mean_median(data, iterations):
    
    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(wspace=0.3, hspace=1, right=0.85) 
    colours = plt.get_cmap('tab10', len(data))

    plt.subplot(3, 2, 1)
    plt.subplots_adjust(hspace=0.5)
    
    all_coop_proportions = [item[2] for item in data]
    aggregated_coop_proportion = np.mean(all_coop_proportions, axis=0)
    aggregated_tipping_points = detect_tipping_points(aggregated_coop_proportion)
    
    # for i, (_, skill_type, coop_proportion) in enumerate(data):
    #     plt.plot(range(iterations), coop_proportion, alpha=0.6, color=colours(i), marker='o', label=f'{skill_type} - {i+1}')
    
    plt.plot(range(iterations), aggregated_coop_proportion, label='Mean Cooperation Proportion', color='blue', linewidth=2)
    for tp in aggregated_tipping_points:
        plt.axvline(x=tp, color='black', linestyle='--', label='Aggregated Tipping Point' if tp == aggregated_tipping_points[0] else "")
    
    plt.xlabel('Iterations')
    plt.ylabel('Proportion of Cooperation')
    plt.title('Aggregated Cooperation Proportion with Tipping Points')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.1), fontsize='small')

    # Ensure x and y axis show integer values
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.show()


    # def plot_all_tipping_points(all_tipping_points):
    # fig, ax = plt.subplots(figsize=(10, 6))
    # unique_skill_types = list(set([skill_type for _, skill_type, _ in all_tipping_points]))
    # color_map = plt.get_cmap('tab10')
    # color_dict = {skill_type: color_map(i) for i, skill_type in enumerate(unique_skill_types)}
    
    # for num_agents, skill_type, tp in all_tipping_points:
    #     ax.scatter(num_agents, tp, label=skill_type, color=color_dict[skill_type], alpha=0.6, edgecolors='w', s=100)
    
    # ax.set_xlabel('Number of Agents')
    # ax.set_ylabel('Iteration of Tipping Point')
    # ax.set_title('Tipping Points for Cooperation Proportion')
    # # Create a custom legend to avoid duplicate entries
    # handles = [plt.Line2D([0], [0], marker='o', color='w', label=skill_type, markersize=10, markerfacecolor=color_dict[skill_type]) for skill_type in unique_skill_types]
    # # ax.legend(handles=handles, title='Skill Type')
    
    # # Create a legend outside the plot
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(handles=handles,title='Skill Type', loc='center left', bbox_to_anchor=(1, 0.5))

    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # plt.grid(True)
    # plt.show()

