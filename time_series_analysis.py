from httpx import head
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import os
import seaborn as sns

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

def perform_autocorrelation_analysis(df, column_prefix, stage_count):
    

    # Initialize an empty list to store series
    combined_series_list = []

    # Loop through possible stage columns
    for i in range(len(df.columns)):
        column_name = f'{column_prefix}_stage_{i+1}'
        if column_name in df.columns:
            combined_series_list.append(df[column_name])

    # Concatenate all series into one and drop NaN values
    combined_series = pd.concat(combined_series_list).dropna()

    # Plot Autocorrelation
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plot_acf(combined_series, lags=20,ax=plt.gca())
    plt.title('Autocorrelation of Entire Series')

    plt.subplot(2, 1, 2)
    plot_pacf(combined_series, lags=20,ax=plt.gca())
    plt.title('Partial Autocorrelation of Entire Series')

    plt.tight_layout()
    plt.show()



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


# def rgba_to_hex(rgba):
#     """ Convert an RGBA tuple to a HEX color code. """
#     r, g, b, a = [int(x * 255) for x in rgba]
#     return f'rgba({r},{g},{b},{a})'

# def plot_measurement(df, column_prefix, stage_count, measurement_name):
    
#     fig = make_subplots(rows=1, cols=2, subplot_titles=(f'{measurement_name} Across Stages', f'{measurement_name} Across Stages'))

#     # Step 1: Get skill types and colors
#     skill_types = df['skill_type'].unique()
#     skill_types = [skill for skill in skill_types if skill]  # Remove any empty skill types
#     cmap = plt.get_cmap("Greens")
#     colors = [cmap(i / len(skill_types)) for i in range(len(skill_types))]
#     colors.reverse()  # Reverse the order of colors
#     color_map = dict(zip(skill_types, colors))
    
#     # Step 2: Add lines to the plot with a base faded color
#     for idx in range(len(df)):
#         skill = df.at[idx, 'skill_type']
#         color = rgba_to_hex(color_map.get(skill, (0, 0, 0, 0.3)))  # Faded color
#         magnitude = [df.at[idx, f'{column_prefix}_stage_{i+1}'] for i in range(stage_count)]
#         fig.add_trace(
#             go.Scatter(
#                 x=list(range(1, stage_count + 1)),
#                 y=magnitude,
#                 mode='lines+markers',
#                 name=f'{skill} Row {idx + 1}',
#                 line=dict(color=color, width=2),  # Faded color
#                 marker=dict(color=color, size=6, line=dict(color='black', width=1)),
#                 hoverinfo='none',  # No hover info for the main lines
#                 legendgroup=skill
#             ),
#             row=1, col=1
#         )
    
#     # Step 3: Add hover highlight traces
#     for idx in range(len(df)):
#         skill = df.at[idx, 'skill_type']
#         color = rgba_to_hex(color_map.get(skill, (0, 0, 0, 1)))  # Brighter color on hover
#         magnitude = [df.at[idx, f'{column_prefix}_stage_{i+1}'] for i in range(stage_count)]
#         fig.add_trace(
#             go.Scatter(
#                 x=list(range(1, stage_count + 1)),
#                 y=magnitude,
#                 mode='lines+markers',
#                 name=f'{skill} Hover {idx + 1}',
#                 line=dict(color=color, width=3),  # Brighter color on hover
#                 marker=dict(color=color, size=6, line=dict(color='black', width=1)),
#                 hoverinfo='none',  # No hover info for the highlight traces
#                 legendgroup=skill,
#                 visible='legendonly'  # Initially not visible
#             ),
#             row=1, col=1
#         )
    
#     # Step 4: Add hover effect using transparent scatter traces
#     for skill in skill_types:
#         fig.add_trace(
#             go.Scatter(
#                 x=[None],
#                 y=[None],
#                 mode='markers',
#                 marker=dict(size=10, color=color_map[skill], line=dict(color='black', width=1)),
#                 showlegend=False,
#                 hoverinfo='none'
#             ),
#             row=1, col=1
#         )
    
#     # Adding group size traces and legends (unchanged)
#     for idx in range(len(df)):
#         group_size = df.at[idx, 'num_agents']
#         color = 'black' if group_size >= 100 else 'white'
#         magnitude = [df.at[idx, f'{column_prefix}_stage_{i+1}'] for i in range(stage_count)]
#         fig.add_trace(
#             go.Scatter(
#                 x=list(range(1, stage_count + 1)),
#                 y=magnitude,
#                 mode='lines+markers',
#                 line=dict(color=color, width=1),
#                 marker=dict(color=color, size=6, line=dict(color='black', width=1)),
#                 showlegend=False
#             ),
#             row=1, col=2
#         )

#     # Legend for marker shapes indicating group sizes
#     fig.add_trace(
#         go.Scatter(
#             x=[None],
#             y=[None],
#             mode='markers',
#             marker=dict(size=10, color='white', line=dict(color='black', width=1)),
#             showlegend=True,
#             name='<= 100 agents'
#         ),
#         row=1, col=2
#     )

#     fig.add_trace(
#         go.Scatter(
#             x=[None],
#             y=[None],
#             mode='markers',
#             marker=dict(size=10, color='black', line=dict(color='black', width=1)),
#             showlegend=True,
#             name='> 100 agents'
#         ),
#         row=1, col=2
#     )

#     fig.update_layout(
#         title_text=f'{measurement_name} Across Stages',
#         height=600,
#         width=1200,
#         showlegend=True,
#         legend=dict(
#             x=1.05,
#             y=0.5,
#             traceorder='normal',
#             font=dict(size=12),
#             bgcolor='rgba(0,0,0,0)',
#             bordercolor='rgba(0,0,0,0)'
#         )
#     )

#     fig.update_xaxes(title_text='Iteration', row=1, col=1)
#     fig.update_yaxes(title_text=measurement_name, row=1, col=1)
#     fig.update_xaxes(title_text='Iteration', row=1, col=2)
#     fig.update_yaxes(title_text=measurement_name, row=1, col=2)

#     fig.show()


def plot_measurement(df, column_prefix, stage_count, measurement_name):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    skill_types = df['skill_type'].unique()
    skill_types = [skill for skill in skill_types if skill]  # Remove any empty skill types
    cmap = plt.get_cmap("Greens")
    colors = [cmap(i / len(skill_types)) for i in range(len(skill_types))]
    colors.reverse()  # Reverse the order of colors
    color_map = dict(zip(skill_types, colors))

    for idx in range(len(df)):

        skill = df.at[idx, 'skill_type']
        color = color_map.get(skill, 'black')  # Default to black if skill type is not found
        magnitude = [df.at[idx, f'{column_prefix}_stage_{i+1}'] for i in range(stage_count)]
        # ax1.plot(range(1, stage_count + 1), magnitude,  marker='o', label=f'{skill} Row {idx + 1}', color=color, markeredgecolor='black') # linewidth=1,markeredgecolor='black'
        ax1.scatter(range(1, stage_count + 1), magnitude,  marker='o', label=f'{skill} Row {idx + 1}', color=color, edgecolor='black') # linewidth=1,markeredgecolor='black'
   
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel(measurement_name)
    ax1.set_title(f'{measurement_name} Across Stages')
    # legend for skill types
    skill_handles = [plt.Line2D([0], [0], color=color_map[skill], marker='o', linestyle='None', markersize=10, markeredgecolor='black') for skill in skill_types]
    skill_labels = skill_types    
    ax1.legend(skill_handles, skill_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    
    for idx in range(len(df)):
        
        group_size = df.at[idx, 'num_agents']
        color = 'black' if group_size > 200 else 'white'
        magnitude = [df.at[idx, f'{column_prefix}_stage_{i+1}'] for i in range(stage_count)]
        ax2.plot(range(1, stage_count + 1), magnitude,  marker='o', color=color, markeredgecolor='black')
        # ax2.scatter(range(1, stage_count + 1), magnitude,  marker='o', color=color,linewidth=1, edgecolor='black') #linewidth=1,markeredgecolor='black'
           
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel(measurement_name)
    ax2.set_title(f'{measurement_name} Across Stages')
    
    # Legend for marker shapes indicating group sizes
    shape_handles = [plt.Line2D([0], [0], color='white', marker='o', linestyle='None', markersize=10, markeredgecolor='black', label='<= 200 agents'),
                     plt.Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=10, markeredgecolor='black', label='> 200 agents')
                     ]
    shape_labels = ['<= 200 agents', '> 200 agents']
    ax2.legend(shape_handles, shape_labels, loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.subplots_adjust(right=0.75)
    plt.subplots_adjust(wspace=1,left=0.2, right=0.9) #wspace=0.5 increases widht
    plt.show()



# CLUSTERS (skill - number)
def preprocess_data(df):
    
    # resuls_COpy.csv
    # df['intelligence'] = df['skill_type'].apply(lambda x: 'high' if x in ['S++', 'manipulator-bully', "manipulator-gf"] else 'low')
    # df['group_size'] = df['num_agents'].apply(lambda x: 'large' if x > 100 else 'small')
    
    # for results3.csv
    df['intelligence'] = df['skill_type'].apply(lambda x: 'high' if x in ["S++","manipulator-bully", "S++_simp", "S", "fp","br1","eeew","br2"] 
                                                else 'medium' if x in ["memory1","mqubed","memory2","manipulator-gf", "wolf","qlearn","godfather", "eeew_simp"]
                                                else 'low') 
    df['group_size'] = df['num_agents'].apply(lambda x: 'large' if x > 200 else 'small')

    return df

# "S++","manipulator-bully", "S++_simp", "S", "fp","br1","eeew","br2",
# "memory1","mqubed","memory2","manipulator-gf", "wolf","qlearn","godfather", "eeew_simp", 
# "exp3w", "cjal","pavlov","gigawolf", "wma","sfp","exp3w_simp", "random"

# ANOVA
def perform_two_way_anova(df, stage_count):
    # Reshape the dataframe for the ANOVA test
    df_melt = df.melt(id_vars=['intelligence', 'group_size'], 
                      value_vars=[f'cooperation_proportion_stage_{i+1}' for i in range(stage_count)],
                      var_name='stage', value_name='cooperation_proportion')

    # Fit the two-way ANOVA model
    model = smf.ols('cooperation_proportion ~ C(intelligence)', data=df_melt).fit()  #* C(group_size)
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # Calculate partial eta squared for each factor and interaction
    ss_effects = anova_table['sum_sq'].iloc[:-1]  # Exclude the error term
    ss_error = anova_table['sum_sq'].iloc[-1]  # The error term is the last one
    partial_eta_squared = ss_effects / (ss_effects + ss_error)
    
    # Add partial eta squared to the ANOVA table
    anova_table['partial_eta_squared'] = partial_eta_squared
    
    return anova_table


# Conduct Tukey's HSD post-hoc test
def perform_post_hoc_test(df_melt):
    tukey = pairwise_tukeyhsd(endog=df_melt['cooperation_proportion'],
                              groups=df_melt['intelligence'],
                              alpha=0.05)
    return tukey.summary()



# TIPPING POINTS
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



# A) PLOT TIPPING POINTS

# def plot_tipping_points_Skill_Num(df, tipping_points_df):
#     fig, ax = plt.subplots(figsize=(18, 10))

#     # Scatter plot of skill_type vs. num_agents, colored by cooperation_proportion
#     scatter = ax.scatter(df['skill_type'], df['num_agents'], c=df['cooperation_proportion'], cmap='viridis', s=100, edgecolor='k', alpha=0.7)

#     # Highlight tipping points
#     for index, row in tipping_points_df.iterrows():
#         ax.scatter(row['skill_type'], row['num_agents'], edgecolors='red', facecolors='none', s=150, linewidth=2)

#     # Add color bar
#     cbar = plt.colorbar(scatter, ax=ax)
#     cbar.set_label('Cooperation Proportion')

#     ax.set_xlabel('Skill Type')
#     ax.set_ylabel('Number of Agents')
#     ax.set_title('Cooperation Proportion by Skill Type and Number of Agents')

#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()


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

    # Define color maps
    intelligence_colors = {
        'high': 'red',
        'low': 'blue',
        'medium': 'yellow' # Add more levels if necessary
    }
    
    group_size_colors = {
        'small': 'blue',
        'medium': 'yellow',
        'large': 'red' # Add more levels if necessary
    }
    
    # Plot by Intelligence
    for intelligence, group in df.groupby('intelligence'):
        color = intelligence_colors.get(intelligence, 'grey') 
        ax1.scatter(group['stage'], group['cooperation_proportion'], marker='o', linestyle='-', label=intelligence, color=color, edgecolors='black')
    
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
        color = group_size_colors.get(group_size, 'grey')  # Default to grey if not found
        ax2.scatter(group['stage'], group['cooperation_proportion'], marker='o', linestyle='-', label=group_size, color=color)
    
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

# B) PLOTTING TIPPING POINTS skill vs number

def plot_skill_vs_agents(df):

   
    plt.figure(figsize=(12, 8))

    # Scatter plot for cooperation proportion
    scatter = sns.scatterplot(data=df, x='num_agents', y='skill_type', hue='cooperation_proportion', size='cooperation_proportion', sizes=(20, 200), legend='full', palette='coolwarm')

    # # Highlight tipping points
    # tipping_points = df[df['tipping_point_detected'] == True]
    # plt.scatter(tipping_points['num_agents'], tipping_points['skill_type'], edgecolor='black', facecolor='none', s=200, linewidth=1.5)

    plt.xlabel('Number of Agents')
    plt.ylabel('Skill Type')
    plt.title('Skill Type vs. Number of Agents with Cooperation Proportion and Tipping Points')
    plt.legend(title='Cooperation Proportion', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()

# C) PLOTTING TIPPING POINTS HEAT MAP

def plot_skill_vs_agents_heatmap(df):
    pivot_table = df.pivot_table(index='skill_type', columns='num_agents', values='cooperation_proportion', aggfunc='mean')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Cooperation Proportion'})
    plt.title('Heatmap of Cooperation Proportion by Skill Type and Number of Agents')
    plt.xlabel('Number of Agents')
    plt.ylabel('Skill Type')
    plt.show()




def main(file_path):
    # Load data
    df = load_data(file_path)

    # General classification order
    general_classification = ["S++","manipulator-bully", "S++_simp", "S", "fp","br1","eeew","br2",
                          "memory1","mqubed","memory2", "manipulator-gf", "wolf","qlearn",
                          "godfather", "eeew_simp", "exp3w", "cjal","pavlov","gigawolf", 
                          "wma","sfp","exp3w_simp", "random"]

    # Convert skill_types to a categorical type with the specified order
    df['skill_type'] = pd.Categorical(df['skill_type'], categories=general_classification, ordered=True)

    # Determine the number of stages
    stage_count = df['cooperation_proportion'].str.split(',').str.len().max()
    
    # Split variables data to create stages of time series (iterations)
    for column in ['clustering_coefficient', 'average_path_length', 'cooperation_proportion', 'density', 'assortativity', 'modularity']:
    # for column in ['clustering_coefficient', 'average_path_length', 'degree_distribution','degree_histogram','cooperation_proportion', 'density', 'assortativity', 'modularity','edge_weight']:
        df = split_multistage_values(df, column, stage_count)

    # df.to_csv('resultado.csv', index=False)
    
    # # Perform analyses
    # perform_autocorrelation_analysis(df, 'cooperation_proportion', stage_count)
    # perform_seasonal_decomposition(df, 'cooperation_proportion', stage_count)
    # perform_stationarity_test(df, 'cooperation_proportion', stage_count)
    # perform_arima_forecasting(df, 'cooperation_proportion', stage_count)
    # perform_granger_causality_test(df, 'cooperation_proportion', 'clustering_coefficient', stage_count)
    
    plot_measurement(df, 'cooperation_proportion', stage_count, 'Cooperation Proportion' )
    plot_measurement(df, 'clustering_coefficient', stage_count,'Clustering Coefficient' )
    plot_measurement(df, 'average_path_length', stage_count, 'Average Path Length')
    plot_measurement(df, 'density', stage_count, 'Density')
    plot_measurement(df, 'assortativity', stage_count, 'Assortativity')
    plot_measurement(df, 'modularity', stage_count, 'Modularity')

    # for anova and tipping points we create two new variables (intelligence -low,high- and group size -low, high-)
    df = preprocess_data(df) 
       
 
    # Tipping points
    df_melt, tipping_points_df = identify_tipping_points(df,stage_count)
    plot_tipping_points(df_melt,tipping_points_df)
    # plot_tipping_points_Skill_Num(df_melt, tipping_points_df)

   # # ANOVA
    anova_results = perform_two_way_anova(df, stage_count)
    tukey_results = perform_post_hoc_test(df_melt)
    print(anova_results)
    print(tukey_results)


# Example usage
if __name__ == "__main__":
    # file_path = 'combined_results500.csv'  # Update this path to your actual CSV file path
    file_path = 'results3.csv'  # Update this path to your actual CSV file path - this one is the original csv file.
    # file_path = 'results_COpy.csv'  # Update this path to your actual CSV file path - this one is the original csv file.
    main(file_path)