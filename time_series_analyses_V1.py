from httpx import head
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def split_multistage_values(df, column_name, stage_count):
    
    split_data = df[column_name].str.split(', ', expand=True)
    new_columns = {i: f"{column_name}_stage_{i+1}" for i in range(stage_count)}  # Create new column names
    split_data.rename(columns=new_columns, inplace=True)  # Rename the split data columns
    split_data = split_data.apply(pd.to_numeric, errors='coerce')
    df = pd.concat([df, split_data], axis=1)  # Concatenate the new columns with the original dataframe
    df.drop(columns=[column_name], inplace=True)   # Drop the original column
    
    return df

# def perform_autocorrelation_analysis(df, column_prefix, stage_count):
    

#     # Initialize an empty list to store series
#     combined_series_list = []

#     # Loop through possible stage columns
#     for i in range(len(df.columns)):
#         column_name = f'{column_prefix}_stage_{i+1}'
#         if column_name in df.columns:
#             combined_series_list.append(df[column_name])

#     # Concatenate all series into one and drop NaN values
#     combined_series = pd.concat(combined_series_list).dropna()

#     # Plot Autocorrelation
#     plt.figure(figsize=(12, 6))

#     plt.subplot(2, 1, 1)
#     plot_acf(combined_series, lags=20,ax=plt.gca())
#     plt.title('Autocorrelation of Entire Series')

#     plt.subplot(2, 1, 2)
#     plot_pacf(combined_series, lags=20,ax=plt.gca())
#     plt.title('Partial Autocorrelation of Entire Series')

#     plt.tight_layout()
#     plt.show()



# def perform_seasonal_decomposition(df, column_prefix, stage_count):
#     for i in range(stage_count):
#         column_name = f'{column_prefix}_stage_{i+1}'
#         if df[column_name].dropna().empty:
#             print(f"Stage {i+1}: Not enough data for seasonal decomposition.")
#             continue
#         decomposition = seasonal_decompose(df[column_name].dropna(), model='additive', period=1)
#         trend = decomposition.trend
#         seasonal = decomposition.seasonal
#         residual = decomposition.resid

#         plt.figure(figsize=(12, 8))
#         plt.subplot(411)
#         plt.plot(df.index, df[column_name], label='Original')
#         plt.legend(loc='best')
#         plt.subplot(412)
#         plt.plot(df.index, trend, label='Trend')
#         plt.legend(loc='best')
#         plt.subplot(413)
#         plt.plot(df.index, seasonal, label='Seasonality')
#         plt.legend(loc='best')
#         plt.subplot(414)
#         plt.plot(df.index, residual, label='Residuals')
#         plt.legend(loc='best')
#         plt.tight_layout()
#         plt.show()

def perform_stationarity_test(df, column_prefix, stage_count):
    
    results = []

    for i in range(stage_count):
        column_name = f'{column_prefix}_stage_{i+1}'
        if df[column_name].dropna().empty:
            print(f"Stage {i+1}: Not enough data for stationarity test.")
            continue
        result = adfuller(df[column_name].dropna())
        results.append({
            'Stage': i+1,
            'ADF Statistic': result[0],
            'p-value': result[1],
            '1% Critical Value': result[4]['1%'],
            '5% Critical Value': result[4]['5%'],
            '10% Critical Value': result[4]['10%']
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    print(results_df)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Stage')
    ax1.set_ylabel('ADF Statistic', color=color)
    ax1.plot(results_df['Stage'], results_df['ADF Statistic'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=results_df['1% Critical Value'].mean(), color='r', linestyle='--', label='1% Critical Value')
    ax1.axhline(y=results_df['5% Critical Value'].mean(), color='g', linestyle='--', label='5% Critical Value')
    ax1.axhline(y=results_df['10% Critical Value'].mean(), color='b', linestyle='--', label='10% Critical Value')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('p-value', color=color)
    ax2.plot(results_df['Stage'], results_df['p-value'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    fig.legend(loc='upper right',bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.title('Stationarity Test Results Across Stages', pad=0.1)
    plt.show()


# def perform_arima_forecasting(df, column_prefix, stage_count, forecast_steps=10):
#     for i in range(stage_count):
#         column_name = f'{column_prefix}_stage_{i+1}'
#         if df[column_name].dropna().empty:
#             print(f"Stage {i+1}: Not enough data for ARIMA forecasting.")
#             continue
#         model = ARIMA(df[column_name].dropna(), order=(1, 0, 1))  # Example order parameters (p, d, q)
#         results = model.fit()
#         forecast = results.get_forecast(steps=forecast_steps)

#         plt.figure(figsize=(12, 6))
#         plt.plot(df.index, df[column_name], label='Observed')
#         plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, color='red', label='Forecast')
#         plt.fill_between(forecast.predicted_mean.index,
#                          forecast.conf_int()[:, 0],
#                          forecast.conf_int()[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
#         plt.title(f'ARIMA Forecasting Stage {i+1}')
#         plt.xlabel('Time')
#         plt.ylabel(column_prefix.replace('_', ' ').title())
#         plt.legend()
#         plt.show()

# def perform_granger_causality_test(df, column_prefix_1, column_prefix_2, stage_count, max_lag=3):
#     for i in range(stage_count):
#         column_name_1 = f'{column_prefix_1}_stage_{i+1}'
#         column_name_2 = f'{column_prefix_2}_stage_{i+1}'
#         if df[column_name_1].dropna().empty or df[column_name_2].dropna().empty:
#             print(f"Stage {i+1}: Not enough data for Granger causality test.")
#             continue
#         data = pd.DataFrame({
#             column_prefix_1: df[column_name_1], 
#             column_prefix_2: df[column_name_2]
#         }).dropna()
#         print(f"Granger Causality Test for Stage {i+1}")
#         grangercausalitytests(data, max_lag, verbose=True)


def plot_cooperation_proportion(df, column_prefix, stage_count):
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    skill_types = df['skill_type'].unique()
    skill_types = [skill for skill in skill_types if skill]  # Remove any empty skill types
    cmap = plt.get_cmap("Greens")
    colors = [cmap(i / len(skill_types)) for i in range(len(skill_types))]
    colors.reverse()  # Reverse the order of colors
    color_map = dict(zip(skill_types, colors))

    unique_group_sizes = sorted(df['num_agents'].unique())
    marker_shapes = ['o', 'o', 'o', 'o', '*', '*', '*', '*', '*']
    group_size_to_shape = {size: marker_shapes[i] for i, size in enumerate(unique_group_sizes)}

    
    for idx in range(len(df)):

        skill = df.at[idx, 'skill_type']
        color = color_map.get(skill, 'black')  # Default to black if skill type is not found
        
        group_size = df.at[idx, 'num_agents']
        marker_shape = group_size_to_shape[group_size]
        
        cooperation_proportions = [df.at[idx, f'{column_prefix}_stage_{i+1}'] for i in range(stage_count)]
        ax.scatter(range(1, stage_count + 1), cooperation_proportions,  marker=marker_shape, label=f'{skill} Row {idx + 1}', color=color, edgecolors='black',linewidth=1)
          
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cooperation Proportion')
    ax.set_title('Cooperation Proportion Across Stages')

    # legend for skill types
    skill_handles = [plt.Line2D([0], [0], color=color_map[skill], marker='o', linestyle='None', markersize=10, markeredgecolor='black') for skill in skill_types]
    skill_labels = skill_types
    
    # Legend for marker shapes indicating group sizes
    shape_handles = [
        plt.Line2D([0], [0], color='grey', marker='o', linestyle='None', markersize=10, markeredgecolor='black', label='<= 100 agents'),
        plt.Line2D([0], [0], color='grey', marker='*', linestyle='None', markersize=10, markeredgecolor='black', label='> 100 agents')
      ]
    shape_labels = ['<= 100 agents', '> 100 agents']

    blank_handles = [plt.Line2D([0], [0], color='white', marker='None', linestyle='None', markersize=10)] #* 2
    blank_labels = [''] #* 2
    
    handles = skill_handles + blank_handles + shape_handles 
    labels = skill_labels + blank_labels + shape_labels 
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.75)
    plt.show()


def preprocess_data(df):
    
    df['intelligence'] = df['skill_type'].apply(lambda x: 'high' if x in ['S++', 'manipulator-bully', "manipulator-gf"] else 'low')
    df['group_size'] = df['num_agents'].apply(lambda x: 'large' if x > 100 else 'small')

    return df


def perform_two_way_anova(df, stage_count):
    # Reshape the dataframe for the ANOVA test
    df_melt = df.melt(id_vars=['intelligence', 'group_size'], 
                      value_vars=[f'cooperation_proportion_stage_{i+1}' for i in range(stage_count)],
                      var_name='stage', value_name='cooperation_proportion')

    # Fit the two-way ANOVA model
    model = smf.ols('cooperation_proportion ~ C(intelligence) * C(group_size)', data=df_melt).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    return anova_table



def identify_tipping_points(df,stage_count):
    
    tipping_points = []

    df_melt = df.melt(id_vars=['intelligence', 'group_size', 'num_agents', 'skill_type', 'network'], 
                      value_vars=[f'cooperation_proportion_stage_{i+1}' for i in range(stage_count)],
                      var_name='stage', value_name='cooperation_proportion')

    df_melt['stage'] = df_melt['stage'].str.extract('(\d+)').astype(int)

    # print("Column names:")
    # print(df_melt.columns.tolist())
    # print("\nNumber of columns:")
    # print(len(df_melt.columns))

    # Assuming df_melt has columns: 'stage', 'cooperation_proportion', 'intelligence', 'group_size'
    for (intelligence, group_size, num_agents, skill_type, network), group in df_melt.groupby(['intelligence', 'group_size', 'num_agents', 'skill_type', 'network']):
        group = group.sort_values(by='stage')
        cooperation_props = group['cooperation_proportion'].tolist()
        stages = group['stage'].tolist()

        for i in range(1, len(cooperation_props)):
            if abs(cooperation_props[i] - cooperation_props[i - 1]) > 0.5:
                tipping_points.append({
                    'intelligence': intelligence,
                    'group_size': group_size,
                    'num_agents': num_agents,
                    'skill_type': skill_type,
                    'network': network,
                    'stage': stages[i],
                    'cooperation_proportion': cooperation_props[i],
                    'tipping_point_detected': True
                })
    
    return df_melt, pd.DataFrame(tipping_points)


def plot_tipping_points(df,tipping_points_df):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Extract additional information for annotations
    num_agents_range = f"{df['num_agents'].min()} - {df['num_agents'].max()}"
    skill_types = ", ".join(df['skill_type'].unique())
    num_networks = df['network'].nunique()    
    group_size_counts = df.groupby('group_size')['num_agents'].sum()
    intelligence_counts = df.groupby('intelligence')['num_agents'].sum()    
    group_size_info = '\n'.join([f"  {group_size}: {count}" for group_size, count in group_size_counts.items()])
    intelligence_counts_info = '\n'.join([f"  {intelligence}: {count}" for intelligence, count in intelligence_counts.items()])

    
    # Plot by Intelligence
    for intelligence, group in df.groupby('intelligence'):
        ax1.scatter(group['stage'], group['cooperation_proportion'], marker='o', linestyle='-', label=intelligence)
    
    ax1.set_xlabel('Stage')
    ax1.set_ylabel('Cooperation Proportion')
    ax1.set_title('Cooperation Proportion by Intelligence')
    ax1.legend(title='Intelligence', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    
    # Highlight tipping points
    for index, row in tipping_points_df.iterrows():
        ax1.scatter(row['stage'], row['cooperation_proportion'], edgecolors='black', facecolors='none', zorder=5,linewidth=1)
    
   
    # Plot by Group Size
    for group_size, group in df.groupby('group_size'):
        ax2.scatter(group['stage'], group['cooperation_proportion'], marker='o', linestyle='-', label=group_size)
    
    ax2.set_xlabel('Stage')
    ax2.set_ylabel('Cooperation Proportion')
    ax2.set_title('Cooperation Proportion by Group Size')
    ax2.legend(title='Group Size', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    
    # Highlight tipping points
    for index, row in tipping_points_df.iterrows():
        ax2.scatter(row['stage'], row['cooperation_proportion'], edgecolors='black', facecolors='none', zorder=5,linewidth=1)
    

    # Add annotations with plot characteristics
    fig.text(0.02, 0.5, f'Number of Agents:\n {num_agents_range}\n'
                        f'Skill Types:\n  {skill_types}\n'
                        f'Number of Networks:\n {num_networks}\n'
                        f'Group Sizes:\n{group_size_info}\n'
                        f'Intelligence:\n{intelligence_counts_info}',                        
             ha='left', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
   
    # Adjust layout to make room for the legends
    plt.subplots_adjust(wspace=0.5,left=0.3, right=0.85) #wspace=0.5 increases widht
    plt.show()




def main(file_path):
    # Load data
    df = load_data(file_path)
    
    # Determine the number of stages
    stage_count = df['cooperation_proportion'].str.split(',').str.len().max()
    
    # Split variables data to create stages of time series (iterations)
    for column in ['clustering_coefficient', 'average_path_length', 'cooperation_proportion', 'density', 'assortativity', 'modularity']:
        df = split_multistage_values(df, column, stage_count)

    df.to_csv('resultado.csv', index=False)
    
    # # Perform analyses
    # perform_autocorrelation_analysis(df, 'cooperation_proportion', stage_count)
    # perform_seasonal_decomposition(df, 'cooperation_proportion', stage_count)
    # perform_stationarity_test(df, 'cooperation_proportion', stage_count)
    # perform_arima_forecasting(df, 'cooperation_proportion', stage_count)
    # perform_granger_causality_test(df, 'cooperation_proportion', 'clustering_coefficient', stage_count)
    plot_cooperation_proportion(df, 'cooperation_proportion', stage_count)

    # for anova and tipping points we create two new variables (intelligence -low,high- and group size -low, high-)
    df = preprocess_data(df) 
       
    # # ANOVA
    # anova_results = perform_two_way_anova(df, stage_count)
    # print(anova_results)

    # Tipping points
    df_melt, tipping_points_df = identify_tipping_points(df,stage_count)
    plot_tipping_points(df_melt,tipping_points_df)



# Example usage
if __name__ == "__main__":
    file_path = 'results_COpy.csv'  # Update this path to your actual CSV file path
    main(file_path)