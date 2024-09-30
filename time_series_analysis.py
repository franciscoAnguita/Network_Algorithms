from httpx import head
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
from matplotlib import colormaps
from matplotlib import colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import seaborn as sns
import ruptures as rpt

def load_data(file_path):
    return pd.read_csv(file_path)

def split_multistage_values(df, column_name, stage_count):
    
    if column_name in ['degree_distribution', 'edge_weight']:
        # Split by semicolon to separate the stages
        split_data = df[column_name].str.split(';', expand=True)
        
        # For each stage, split by commas to handle individual values within each stage
        for i in range(stage_count):
            if i in split_data.columns:
                split_data[i] = split_data[i].str.split(',').apply(lambda x: [float(val.strip()) for val in x if val.strip()])

    else:
        # For other variables, simply split by commas
        split_data = df[column_name].str.split(', ', expand=True)
        split_data = split_data.apply(pd.to_numeric, errors='coerce')

    # Create new column names for each stage
    new_columns = {i: f"{column_name}_stage_{i+1}" for i in range(stage_count)}
    split_data.rename(columns=new_columns, inplace=True)

    # Concatenate the new columns with the original dataframe
    df = pd.concat([df, split_data], axis=1)
    df.drop(columns=[column_name], inplace=True)

    # split_data = df[column_name].str.split(', ', expand=True)
    # new_columns = {i: f"{column_name}_stage_{i+1}" for i in range(stage_count)}  # Create new column names
    # split_data.rename(columns=new_columns, inplace=True)  # Rename the split data columns
    # split_data = split_data.apply(pd.to_numeric, errors='coerce')
    # df = pd.concat([df, split_data], axis=1)  # Concatenate the new columns with the original dataframe
    # df.drop(columns=[column_name], inplace=True)   # Drop the original column
    
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
#         decomposition = seasonal_decompose(df[column_name].dropna(), model='additive', period=50)
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

def perform_rolling_analysis(df, column_prefix, stage_count, window_size=10):
    # Initialize lists to store aggregated data
    aggregated_data = []
    rolling_mean_list = []
    rolling_std_list = []
    time_indices = []
    continuous_time_index = 0

    # Loop through each stage and calculate rolling statistics
    for i in range(stage_count):
        column_name = f'{column_prefix}_stage_{i+1}'
        data = df[column_name].dropna()  # Drop missing data
        
        if data.empty:
            print(f"Stage {i+1}: Not enough data for rolling analysis.")
            continue

        # Prepare the time index
        y = data.values
        time_indices.extend(np.arange(continuous_time_index, continuous_time_index + len(data)))

        # Calculate rolling statistics
        rolling_mean = data.rolling(window=window_size, min_periods=1).mean()
        rolling_std = data.rolling(window=window_size, min_periods=1).std()

        # Append to the lists for later plotting
        aggregated_data.extend(y)
        rolling_mean_list.extend(rolling_mean)
        rolling_std_list.extend(rolling_std)

        # Update the continuous time index
        continuous_time_index += len(data)

    # Create plots for original data, rolling mean, and rolling standard deviation
    plt.figure(figsize=(15, 12))
    
    # Plot original data
    plt.subplot(311)
    plt.plot(time_indices, aggregated_data, label='Original Data')
    plt.title('Original Data')
    plt.legend(loc='best')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Plot rolling mean
    plt.subplot(312)
    plt.plot(time_indices, rolling_mean_list, label='Rolling Mean', color='orange')
    plt.title('Rolling Mean')
    plt.legend(loc='best')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Plot rolling standard deviation
    plt.subplot(313)
    plt.plot(time_indices, rolling_std_list, label='Rolling Std Dev', color='red')
    plt.title('Rolling Standard Deviation')
    plt.legend(loc='best')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def perform_moving_average_analysis(df, column_prefix, stage_count, window_size=3):
    # Initialize lists to store aggregated data
    aggregated_data = []
    trend_list = []
    residual_list = []
    time_indices = []
    continuous_time_index = 0

    # Aggregate data from each stage
    for i in range(stage_count):
        column_name = f'{column_prefix}_stage_{i+1}'
        data = df[column_name].dropna()
        
        if data.empty:
            print(f"Stage {i+1}: Not enough data for trend analysis.")
            continue

        # Prepare the time index
        y = data.values
        time_indices.extend(np.arange(continuous_time_index, continuous_time_index + len(data)))

        # Moving average for trend detection
        trend = pd.Series(y).rolling(window=window_size, min_periods=1).mean()

        # Calculate residuals (difference between actual and trend)
        residual = y - trend

        # Append values to the lists
        aggregated_data.extend(y)
        trend_list.extend(trend)
        residual_list.extend(residual)

        continuous_time_index += len(data)  # Continue the time index

    # Create plots for the aggregated data, trend, and residuals
    plt.figure(figsize=(15, 12))
    
    # Plot original data
    plt.subplot(311)
    plt.plot(time_indices, aggregated_data, label='Datos Originales')
    plt.title('Datos Originales')
    plt.legend(loc='best')
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    
    # Plot trend (moving average)
    plt.subplot(312)
    plt.plot(time_indices, trend_list, label='Tendencia (Promedio Móvil)', color='orange')
    plt.title('Tendencia (Promedio Móvil)')
    plt.legend(loc='best')
    plt.xlabel('Índice')
    plt.ylabel('Valor')

    # Plot residuals
    plt.subplot(313)
    plt.plot(time_indices, residual_list, label='Residuos', color='red')
    plt.title('Residuos')
    plt.legend(loc='best')
    plt.xlabel('Índice')
    plt.ylabel('Valor')

    plt.tight_layout()
    plt.show()





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


def perform_arima_forecasting(df, column_prefix, stage_count, forecast_steps=10):
    for i in range(stage_count):
        column_name = f'{column_prefix}_stage_{i+1}'
        if df[column_name].dropna().empty:
            print(f"Stage {i+1}: Not enough data for ARIMA forecasting.")
            continue
        model = ARIMA(df[column_name].dropna(), order=(1, 0, 1))  # Example order parameters (p, d, q)
        results = model.fit()
        forecast = results.get_forecast(steps=forecast_steps)

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[column_name], label='Observed')
        plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, color='red', label='Forecast')
        plt.fill_between(forecast.predicted_mean.index,
                         forecast.conf_int()[:, 0],
                         forecast.conf_int()[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
        plt.title(f'ARIMA Forecasting Stage {i+1}')
        plt.xlabel('Time')
        plt.ylabel(column_prefix.replace('_', ' ').title())
        plt.legend()
        plt.show()

def perform_granger_causality_test(df, column_prefix_1, column_prefix_2, stage_count, max_lag=3):
    for i in range(stage_count):
        column_name_1 = f'{column_prefix_1}_stage_{i+1}'
        column_name_2 = f'{column_prefix_2}_stage_{i+1}'
        if df[column_name_1].dropna().empty or df[column_name_2].dropna().empty:
            print(f"Stage {i+1}: Not enough data for Granger causality test.")
            continue
        data = pd.DataFrame({
            column_prefix_1: df[column_name_1], 
            column_prefix_2: df[column_name_2]
        }).dropna()
        print(f"Granger Causality Test for Stage {i+1}")
        grangercausalitytests(data, max_lag, verbose=True)


##### MORE TIME SERIES.

# CUMULATIVE SUM
# def perform_cusum_analysis(df, measurement_prefix, stage_count, threshold=0.5):

#     # Collect data across stages
#     data = np.concatenate([df[f'{measurement_prefix}_stage_{i+1}'].values for i in range(stage_count)])
#     mean = np.mean(data)
#     cusum_pos = np.maximum(0, np.cumsum(data - mean - threshold))
#     cusum_neg = np.maximum(0, np.cumsum(mean - data - threshold))

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.plot(cusum_pos, label='CUSUM Positive', color='blue')
#     plt.plot(cusum_neg, label='CUSUM Negative', color='red')
#     plt.legend()
#     plt.title(f'CUSUM Analysis for {measurement_prefix}')
#     plt.xlabel('Iteration')
#     plt.ylabel('CUSUM Value')
#     plt.show()


def perform_cusum_analysis(df, measurement_prefix, stage_count, color_dict, color_mode, threshold=0.5):
        
    # Get unique combinations of skill_type, num_agents, and network
    combinations = df[['skill_type', 'num_agents', 'network']].drop_duplicates()

    plt.figure(figsize=(10, 6))

    multiple_networks = len(df['network'].unique()) > 1

    for _, row in combinations.iterrows():
        skill_type = row['skill_type']
        num_agents = row['num_agents']
        network = row['network']
        
        # Filter the data for the current combination
        subset_df = df[(df['skill_type'] == skill_type) & (df['num_agents'] == num_agents) & (df['network'] == network)]
        if subset_df.empty:
            continue
        
        # Collect data across stages
        data = np.concatenate([subset_df[f'{measurement_prefix}_stage_{i+1}'].values for i in range(stage_count)])
        mean = np.mean(data)
        cusum_pos = np.maximum(0, np.cumsum(data - mean - threshold))
        cusum_neg = np.maximum(0, np.cumsum(mean - data - threshold))

        # color = color_dict.get((num_agents, skill_type), 'black')  # Fallback color is black if not found
        if multiple_networks:
            color_key = (skill_type, num_agents)  # Use both skill_type and num_agents for multi-network plots
        else:
            color_key = num_agents  # Use only num_agents for single network plots

        color = color_dict.get(color_key, 'black')  # Fallback to 'black' if not found

        plt.plot(cusum_pos, color=color)  # Plot positive CUSUM
        plt.plot(cusum_neg, linestyle='--', color=color)  # Plot negative CUSUM

        # Add a single legend entry per combination (ignore separating positive/negative)
        plt.plot([], [], label=f'{skill_type}, {num_agents}, {network}', color=color)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(f'CUSUM Analysis for {measurement_prefix}')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))    
    plt.ylabel('CUSUM Value')
    plt.show()




# BAYESIAN CHANGE POINT DETECTION
# def perform_change_point_detection(df, measurement_prefix, stage_count):

#     # Collect data across stages
#     data = np.concatenate([df[f'{measurement_prefix}_stage_{i+1}'].values for i in range(stage_count)])
#     model = "l2"
#     algo = rpt.Pelt(model=model).fit(data)
#     result = algo.predict(pen=10)

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.plot(data, label=measurement_prefix)
#     for cp in result:
#         plt.axvline(x=cp, color='red', linestyle='--')
#     plt.legend()
#     plt.title(f'Change Point Detection for {measurement_prefix}')
#     plt.xlabel('Iteration')
#     plt.ylabel(measurement_prefix)
#     plt.show()

def perform_change_point_detection(df, measurement_prefix, stage_count, color_dict, color_mode):
    
    combinations = df[['skill_type', 'num_agents', 'network']].drop_duplicates()    
    multiple_networks = len(df['network'].unique()) > 1

    plt.figure(figsize=(10, 6))
    
    for _, row in combinations.iterrows():
        skill_type = row['skill_type']
        num_agents = row['num_agents']
        network = row['network']
        
        subset_df = df[(df['skill_type'] == skill_type) & (df['num_agents'] == num_agents) & (df['network'] == network)]
        if subset_df.empty:
            continue

        data = np.concatenate([subset_df[f'{measurement_prefix}_stage_{i+1}'].values for i in range(stage_count)])
        model = "l2"
        algo = rpt.Pelt(model=model).fit(data)
        result = algo.predict(pen=10)
        
        if multiple_networks:
            color_key = (skill_type, num_agents)  # Use both skill_type and num_agents for multi-network plots
        else:
            color_key = num_agents  # Use only num_agents for single network plots

        color = color_dict.get(color_key, 'black')  # Fallback to 'black' if not found

        plt.plot(data, label=f'{skill_type}, {num_agents}, {network}', color=color)

        for cp in result:
            plt.axvline(x=cp, color='red', linestyle='--')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(f'Change Point Detection for {measurement_prefix}')
    plt.xlabel('Iteration')
    plt.ylabel(measurement_prefix)
    plt.show()


# PIECEWISE REGRESSION
# def perform_piecewise_regression(df, measurement_prefix, stage_count, change_points):
#     # Collect data across stages
#     data = np.concatenate([df[f'{measurement_prefix}_stage_{i+1}'].values for i in range(stage_count)])
#     iterations = np.arange(len(data))
#     segments = np.split(data, change_points)
#     segment_models = []

#     for i, segment in enumerate(segments):
#         if len(segment) > 0:  # Check if the segment is non-empty
#             segment_iterations = iterations[:len(segment)]
#             model = LinearRegression().fit(segment_iterations.reshape(-1, 1), segment)
#             segment_models.append(model)
#         else:
#             segment_models.append(None)  # Placeholder for empty segments

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.plot(iterations, data, label=measurement_prefix)

#     for i, model in enumerate(segment_models):
#         if model is not None:  # Only plot for non-empty segments
#             start = sum(len(seg) for seg in segments[:i])
#             end = start + len(segments[i])
#             plt.plot(iterations[start:end], model.predict(iterations[start:end].reshape(-1, 1)), linestyle='--')

#     plt.legend()
#     plt.title(f'Piecewise Regression for {measurement_prefix}')
#     plt.xlabel('Iteration')
#     plt.ylabel(measurement_prefix)
#     plt.show()

def perform_piecewise_regression(df, measurement_prefix, stage_count, color_dict, color_mode, change_points):
    
    combinations = df[['skill_type', 'num_agents', 'network']].drop_duplicates()
    multiple_networks = len(df['network'].unique()) > 1
    
    plt.figure(figsize=(10, 6))
    
    for _, row in combinations.iterrows():
        skill_type = row['skill_type']
        num_agents = row['num_agents']
        network = row['network']
        
        subset_df = df[(df['skill_type'] == skill_type) & (df['num_agents'] == num_agents) & (df['network'] == network)]
        if subset_df.empty:
            continue

        data = np.concatenate([subset_df[f'{measurement_prefix}_stage_{i+1}'].values for i in range(stage_count)])
        iterations = np.arange(len(data))
        segments = np.split(data, change_points)
        segment_models = []

        for i, segment in enumerate(segments):
            if len(segment) > 0:
                segment_iterations = iterations[:len(segment)]
                model = LinearRegression().fit(segment_iterations.reshape(-1, 1), segment)
                segment_models.append(model)
            else:
                segment_models.append(None)
        
        
        if multiple_networks:
            color_key = (skill_type, num_agents)  # Use both skill_type and num_agents for multi-network plots
        else:
            color_key = num_agents  # Use only num_agents for single network plots

        color = color_dict.get(color_key, 'black')  # Fallback to 'black' if not found

        plt.plot(iterations, data, label=f'{skill_type}, {num_agents}, {network}', color=color)
        
        for i, model in enumerate(segment_models):
            if model is not None:
                start = sum(len(seg) for seg in segments[:i])
                end = start + len(segments[i])
                plt.plot(iterations[start:end], model.predict(iterations[start:end].reshape(-1, 1)), linestyle='--', color=color)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(f'Piecewise Regression for {measurement_prefix}')
    plt.xlabel('Iteration')
    plt.ylabel(measurement_prefix)
    plt.show()


# AUTOCRRELATION ANALYSIS
# def perform_autocorrelation_analysis(df, measurement_prefix, stage_count):
    
#     # Collect data across stages
#     data = np.concatenate([df[f'{measurement_prefix}_stage_{i+1}'].values for i in range(stage_count)])
#     plt.figure(figsize=(10, 6))
#     plot_acf(data, lags=len(data)//2, ax=plt.gca())  # Use half the length of data for lags
#     plt.title(f'Autocorrelation for {measurement_prefix}')
#     plt.xlabel('Lag')
#     plt.ylabel('Autocorrelation')
#     plt.show()

def perform_autocorrelation_analysis(df, measurement_prefix, stage_count, color_dict, color_mode):
    
    combinations = df[['skill_type', 'num_agents', 'network']].drop_duplicates()
    multiple_networks = len(df['network'].unique()) > 1
    
    plt.figure(figsize=(10, 6))
    
    for _, row in combinations.iterrows():
        skill_type = row['skill_type']
        num_agents = row['num_agents']
        network = row['network']
        
        subset_df = df[(df['skill_type'] == skill_type) & (df['num_agents'] == num_agents) & (df['network'] == network)]
        if subset_df.empty:
            continue

        data = np.concatenate([subset_df[f'{measurement_prefix}_stage_{i+1}'].values for i in range(stage_count)])
        
        if multiple_networks:
            color_key = (skill_type, num_agents)  # Use both skill_type and num_agents for multi-network plots
        else:
            color_key = num_agents  # Use only num_agents for single network plots
        
        color = color_dict.get(color_key, 'black')  # Fallback to 'black' if not found
        
        plot_acf(data, lags=len(data)//2, ax=plt.gca(), label=f'{skill_type}, {num_agents}, {network}', color=color)

    plt.title(f'Autocorrelation for {measurement_prefix}')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    # plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()


def rgba_to_hex(rgba):
    """ Convert an RGBA tuple to a HEX color code. """
    r, g, b, a = [int(x * 255) for x in rgba]
    return f'rgba({r},{g},{b},{a})'

def plot_measurement_(df, column_prefix, stage_count, measurement_name):
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'{measurement_name} Across Stages', f'{measurement_name} Across Stages'))

    # Step 1: Get skill types and colors
    skill_types = df['skill_type'].unique()
    skill_types = [skill for skill in skill_types if skill]  # Remove any empty skill types
    cmap = plt.get_cmap("Greens")
    colors = [cmap(i / len(skill_types)) for i in range(len(skill_types))]
    colors.reverse()  # Reverse the order of colors
    color_map = dict(zip(skill_types, colors))
    
    # Step 2: Add lines to the plot with a base faded color
    for idx in range(len(df)):
        skill = df.at[idx, 'skill_type']
        color = rgba_to_hex(color_map.get(skill, (0, 0, 0, 0.3)))  # Faded color
        magnitude = [df.at[idx, f'{column_prefix}_stage_{i+1}'] for i in range(stage_count)]
        fig.add_trace(
            go.Scatter(
                x=list(range(1, stage_count + 1)),
                y=magnitude,
                mode='lines+markers',
                name=f'{skill} Row {idx + 1}',
                line=dict(color=color, width=2),  # Faded color
                marker=dict(color=color, size=6, line=dict(color='black', width=1)),
                hoverinfo='none',  # No hover info for the main lines
                legendgroup=skill
            ),
            row=1, col=1
        )
    
    # Step 3: Add hover highlight traces
    for idx in range(len(df)):
        skill = df.at[idx, 'skill_type']
        color = rgba_to_hex(color_map.get(skill, (0, 0, 0, 1)))  # Brighter color on hover
        magnitude = [df.at[idx, f'{column_prefix}_stage_{i+1}'] for i in range(stage_count)]
        fig.add_trace(
            go.Scatter(
                x=list(range(1, stage_count + 1)),
                y=magnitude,
                mode='lines+markers',
                name=f'{skill} Hover {idx + 1}',
                line=dict(color=color, width=3),  # Brighter color on hover
                marker=dict(color=color, size=6, line=dict(color='black', width=1)),
                hoverinfo='none',  # No hover info for the highlight traces
                legendgroup=skill,
                visible='legendonly'  # Initially not visible
            ),
            row=1, col=1
        )
    
    # Step 4: Add hover effect using transparent scatter traces
    for skill in skill_types:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color_map[skill], line=dict(color='black', width=1)),
                showlegend=False,
                hoverinfo='none'
            ),
            row=1, col=1
        )
    
    # Adding group size traces and legends (unchanged)
    for idx in range(len(df)):
        group_size = df.at[idx, 'num_agents']
        color = 'black' if group_size >= 100 else 'white'
        magnitude = [df.at[idx, f'{column_prefix}_stage_{i+1}'] for i in range(stage_count)]
        fig.add_trace(
            go.Scatter(
                x=list(range(1, stage_count + 1)),
                y=magnitude,
                mode='lines+markers',
                line=dict(color=color, width=1),
                marker=dict(color=color, size=6, line=dict(color='black', width=1)),
                showlegend=False
            ),
            row=1, col=2
        )

    # Legend for marker shapes indicating group sizes
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color='white', line=dict(color='black', width=1)),
            showlegend=True,
            name='<= 100 agents'
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color='black', line=dict(color='black', width=1)),
            showlegend=True,
            name='> 100 agents'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text=f'{measurement_name} Across Stages',
        height=600,
        width=1200,
        showlegend=True,
        legend=dict(
            x=1.05,
            y=0.5,
            traceorder='normal',
            font=dict(size=12),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        )
    )

    fig.update_xaxes(title_text='Iteration', row=1, col=1)
    fig.update_yaxes(title_text=measurement_name, row=1, col=1)
    fig.update_xaxes(title_text='Iteration', row=1, col=2)
    fig.update_yaxes(title_text=measurement_name, row=1, col=2)

    fig.show()


def plot_measurement_(df, column_prefix, stage_count, measurement_name):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique networks, skill types, and group sizes
    networks = df['network'].unique()
    skill_types = df['skill_type'].unique()
    skill_types = [skill for skill in skill_types if skill]  # Remove any empty skill types
    group_sizes = df['num_agents'].unique()

    # Check if there are multiple networks
    if len(networks) == 1:
        # Single network case: Different colors for each number of agents
        color_palette = colormaps['tab10'](np.linspace(0, 1, len(group_sizes)))
        color_dict = {agent: color_palette[i] for i, agent in enumerate(group_sizes)}
    else:
        # Multiple networks case: Base color per skill_type, gradient for num_agents
        base_color_palette = colormaps['tab10'](np.linspace(0, 1, len(skill_types)))
        skill_color_map = {skill: base_color_palette[i] for i, skill in enumerate(skill_types)}
        
        # Create gradient for each skill type's base color
        norm = plt.Normalize(vmin=min(group_sizes), vmax=max(group_sizes))
        color_dict = {}
        
        for skill in skill_types:
            base_color = np.array(skill_color_map[skill])
            for agent in group_sizes:
                # Adjust the base color by blending with the normalized group size
                color_intensity = norm(agent)
                color = mcolors.to_rgba(base_color, alpha=1)[:3]  # Extract RGB from base color
                color = mcolors.to_rgba((color_intensity * base_color)[:3])  # Adjust intensity
                color_dict[(skill, agent)] = color

    # For each network, skill type, and group size, plot the data
    for network in networks:
        network_df = df[df['network'] == network]  # Filter for the current network
        
        for skill in skill_types:
            skill_df = network_df[network_df['skill_type'] == skill]  # Filter for the current skill type
            
            for idx in range(len(skill_df)):
                group_size = skill_df.iloc[idx]['num_agents']
                
                # Select color based on number of networks
                if len(networks) == 1:
                    # Single network: Use different color for each num_agents
                    color = color_dict[group_size]
                else:
                    # Multiple networks: Use skill_type color with a gradient for num_agents
                    color = color_dict[(skill, group_size)]
                
                # Collect magnitude for each stage
                magnitude = [skill_df.iloc[idx][f'{column_prefix}_stage_{i+1}'] for i in range(stage_count)]
                
                # Plot the data for each skill type, network, and group size
                ax.plot(range(1, stage_count + 1), magnitude, label=f'Network {network} - {skill} - {group_size} agents', 
                        marker='o', linestyle='-', markeredgecolor='black', linewidth=1, color=color)

    ax.set_xlabel('Iteration')
    ax.set_ylabel(measurement_name)
    ax.set_title(f'{measurement_name} Across Stages for Networks')

    # Legend to distinguish skill types and group sizes
    legend_handles = []
    legend_labels = []

    if len(networks) == 1:
        # Single network: Legend by number of agents
        for size in sorted(group_sizes):
            legend_handles.append(plt.Line2D([0], [0], color=color_dict[size], 
                                             linestyle='-', marker='o', markersize=6, markeredgecolor='black'))
            legend_labels.append(f'{size} agents')
    else:
        # Multiple networks: Legend by skill type and number of agents
        for skill in skill_types:
            for size in sorted(group_sizes):
                legend_handles.append(plt.Line2D([0], [0], color=color_dict[(skill, size)], 
                                                 linestyle='-', marker='o', markersize=6, markeredgecolor='black'))
                legend_labels.append(f'{skill} - {size} agents')

    ax.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()



def plot_measurement(df, column_prefix, stage_count, measurement_name):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique networks, skill types, and group sizes
    networks = df['network'].unique()
    skill_types = df['skill_type'].unique()
    skill_types = [skill for skill in skill_types if skill]  # Remove any empty skill types
    group_sizes = df['num_agents'].unique()

    # Dynamically choose a specific base color for each skill type using tab10
    base_color_palette = colormaps['tab10'](np.linspace(0, 1, len(skill_types)))
    skill_color_map = {skill: base_color_palette[i] for i, skill in enumerate(skill_types)}

    # Normalize group sizes to create a gradient within each skill type's color
    norm = plt.Normalize(vmin=min(group_sizes), vmax=max(group_sizes))

    # Adjust the intensity bounds to avoid overly light colors
    min_intensity = 0.3  # Avoid very light colors
    max_intensity = 0.9  # Keep the range constrained
    
    # Function to interpolate between two colors
    def interpolate_color(base_color, intensity, darker=True):

        intensity = min_intensity + (max_intensity - min_intensity) * intensity  # Scale intensity within bounds
        base_rgb = np.array(base_color[:3])  # Extract only RGB from base_color (ignore alpha)
        if darker:
            return mcolors.to_rgba(mcolors.to_hex((1 - intensity) * np.array([1, 1, 1]) + intensity * base_rgb))
        else:
            return mcolors.to_rgba(mcolors.to_hex(intensity * base_rgb + (1 - intensity) * np.array([1, 1, 1])))


    # For each network, skill type, and group size, plot the data
    for network in networks:
        network_df = df[df['network'] == network]  # Filter for the current network
        
        for skill in skill_types:
            skill_df = network_df[network_df['skill_type'] == skill]  # Filter for the current skill type
            
            for idx in range(len(skill_df)):
                group_size = skill_df.iloc[idx]['num_agents']
                
                # Get the base color for the skill type
                base_color = np.array(skill_color_map[skill])
                
                # Create interpolated color for the current group size
                color_intensity = norm(group_size)
                color = interpolate_color(base_color, color_intensity, darker=True)
                
                # Collect magnitude for each stage
                # magnitude = [skill_df.iloc[idx][f'{column_prefix}_stage_{i+1}'] for i in range(stage_count)]
               
                
                if column_prefix in ['edge_weight', 'degree_distribution']:
                    # Handle both string (comma-separated or semicolon-separated) and float cases
                    magnitude = []
                    
                    for i in range(stage_count):

                        stage_value = skill_df.iloc[idx][f'{column_prefix}_stage_{i+1}']  
                        stage_value_str = str(stage_value)   # Convert to string to handle both numeric and string values
                        
                        if ';' in stage_value_str:  # Check if the value contains multiple stages separated by semicolons
                            
                            stage_values = stage_value_str.split(';')  # Split by semicolons to separate stages and handle commas for values within a stage
                            all_stage_values = []
                            
                            for sub_stage in stage_values:
                                # Clean up any extra brackets or characters, and split by commas
                                sub_stage_clean = sub_stage.replace('[', '').replace(']', '').strip()

                                if sub_stage_clean:
                                    values = [float(x) for x in sub_stage_clean.split(',')]
                                    all_stage_values.extend(values)  # Collect all values
                            
                            # Take the mean of all values within this stage, if any values exist
                            if all_stage_values:
                                magnitude.append(np.mean(all_stage_values))
                            else:
                                magnitude.append(np.nan)  # Append NaN if no valid values exist
                        
                        elif ',' in stage_value_str:  # Handle cases where there are only commas (without semicolons)
                            stage_value_str_clean = stage_value_str.replace('[', '').replace(']', '').strip()  # Clean up brackets
                            values = [float(x) for x in stage_value_str_clean.split(',')]  # Split and convert to float
                            if values:
                                magnitude.append(np.mean(values))
                            else:
                                magnitude.append(np.nan)  # Append NaN if no valid values exist
                        else:
                            # Only append float if value is not empty
                            clean_value = stage_value_str.replace('[', '').replace(']', '').strip()
                            if clean_value:
                                magnitude.append(float(clean_value))
                            else:
                                magnitude.append(np.nan)  # Append NaN if value is empty


                else:
                    # Plot just one value for each stage
                    magnitude = [skill_df.iloc[idx][f'{column_prefix}_stage_{i+1}'] for i in range(stage_count)]   

                # Ensure the magnitude list is complete
                if len(magnitude) != stage_count:
                    magnitude.extend([np.nan] * (stage_count - len(magnitude)))             
                
                # ax.plot(range(1, stage_count + 1), magnitude, label=f'Network {network} - {skill} - {group_size} agents', 
                #         marker='o', linestyle='-', markeredgecolor='black', linewidth=1, color=color)
                
                ax.scatter(range(1, stage_count + 1), magnitude, label=f'Network {network} - {skill} - {group_size} agents', 
                           edgecolor='black', color=color, s=50)
                
            
                # Now plot all individual values (dots) for each stage
                # if column_prefix in ['edge_weight', 'degree_distribution']:

                #     for i in range(stage_count):

                #         stage_value = skill_df.iloc[idx][f'{column_prefix}_stage_{i+1}']
                #         stage_value_str = str(stage_value)
                        
                #         if ';' in stage_value_str:
                #             stage_values = stage_value_str.split(';')
                #             all_individual_values = []

                #             for sub_stage in stage_values:
                #                 # Clean up and split by commas to get the individual values within the stage
                #                 sub_stage_clean = sub_stage.replace('[', '').replace(']', '').strip()
                #                 individual_values = [float(x) for x in sub_stage_clean.split(',')]
                #                 all_individual_values.extend(individual_values)  # Collect all individual values

                #         elif ',' in stage_value_str:
                #             # Handle the case where the stage only contains comma-separated values (no semicolons)
                #             stage_value_str_clean = stage_value_str.replace('[', '').replace(']', '').strip()  # Clean up brackets
                #             all_individual_values = [float(x) for x in stage_value_str_clean.split(',')]

                #         else:
                #             # If it's a single value (no semicolons or commas), handle as a single value list
                #             all_individual_values = [float(stage_value_str.replace('[', '').replace(']', '').strip())]
                         
                #         # Scatter all values for each stage
                #         ax.scatter([i + 1] * len(all_individual_values), all_individual_values, color=color, s=20, alpha=0.6, edgecolor='black')  # Changed to use all_individual_values


   
    ax.set_xlabel('Iteration')
    ax.set_ylabel(measurement_name)
    ax.set_title(f'{measurement_name} Across Stages for Multiple Networks')
    ax.grid(True) 

    legend_handles = []
    legend_labels = []

    for skill in skill_types:
        for size in sorted(group_sizes):
            # Generate color for legend based on gradient
            base_color = np.array(skill_color_map[skill])
            gradient_color = interpolate_color(base_color, norm(size), darker=True)
            legend_handles.append(plt.Line2D([0], [0], color=gradient_color, 
                                             linestyle='-', marker='o', markersize=6, markeredgecolor='black'))
            legend_labels.append(f'{skill} - {size} agents')

    ax.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
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
   
    df = load_data(file_path)
    color_mode = 'skill_type'  # Choose skill_type first, then apply gradient for num_agents


    # # General classification order
    # general_classification = ["S++","manipulator-bully", "S++_simp", "S", "fp","br1","eeew","br2",
    #                       "memory1","mqubed","memory2", "manipulator-gf", "wolf","qlearn",
    #                       "godfather", "eeew_simp", "exp3w", "cjal","pavlov","gigawolf", 
    #                       "wma","sfp","exp3w_simp", "random"]

    # # Convert skill_types to a categorical type with the specified order
    # df['skill_type'] = pd.Categorical(df['skill_type'], categories=general_classification, ordered=True)

    # Determine the number of stages
    stage_count = df['cooperation_proportion'].str.split(',').str.len().max()

    # Split variables data to create stages of time series (iterations)
    for column in ['clustering_coefficient', 'average_path_length','degree_distribution', 'degree_histogram','cooperation_proportion', 'density', 'assortativity', 'modularity','edge_weight']:
        df = split_multistage_values(df, column, stage_count)

  
    unique_networks = df['network'].unique()
    unique_skills = df['skill_type'].unique()
    unique_agents = df['num_agents'].unique()

    
    if len(unique_networks) == 1:
       
        color_palette = colormaps['tab10'](np.linspace(0, 1, len(unique_agents)))
        color_dict = {agent: color_palette[i] for i, agent in enumerate(unique_agents)}

    else:
              
        base_color_palette = colormaps['tab10'](np.linspace(0, 1, len(unique_skills)))
        skill_color_map = {skill: base_color_palette[i] for i, skill in enumerate(unique_skills)}
        
        norm = plt.Normalize(vmin=min(unique_agents), vmax=max(unique_agents)) # Normalize group sizes for the gradient within each skill type

        # Set intensity bounds to avoid too light or too dark colors
        min_intensity = 0.3  # Minimum intensity to avoid very light colors
        max_intensity = 0.9  # Maximum intensity to avoid very dark colors

        # Function to adjust color intensity with bounds
        def adjust_color_intensity(base_color, intensity):
            intensity = min_intensity + (max_intensity - min_intensity) * intensity  # Scale intensity within bounds
            base_rgb = np.array(base_color[:3])  # Extract only RGB from base_color (ignore alpha)
            return mcolors.to_rgba(mcolors.to_hex((1 - intensity) * np.array([1, 1, 1]) + intensity * base_rgb))

        # Prepare to apply gradient for num_agents based on the base skill color
        color_dict = {}

        for skill in unique_skills:
            for agent in unique_agents:
                # Adjust the skill base color by blending with the normalized agent size
                base_color = np.array(skill_color_map[skill])
                color_intensity = norm(agent)  # Gradient intensity based on group size
                
                # Apply the intensity adjustment
                color = adjust_color_intensity(base_color, color_intensity)
                
                # Save the color for this combination of skill and num_agents
                color_dict[(skill, agent)] = color

    # for num_agents, color_map in color_maps.items():        
    #         color_dict[(num_agents)] = color_map[num_agents]
    
    # # for color_map in color_maps.items():
    # #     for i, skill in enumerate(skill_order):
    # #         color_dict[(skill)] = color_map[i]

    
    # MORE TIME SERIES ANALYSES
    # perform_autocorrelation_analysis(df, 'degree_distribution', stage_count)
    # perform_seasonal_decomposition(df, 'degree_distribution', stage_count)   # doesn't work with data without periodic repetition: the concept of seasonality might be different compared to time series data with regular periodic patterns
    # perform_moving_average_analysis(df, 'degree_distribution', stage_count)
    # perform_rolling_analysis(df, 'degree_distribution', stage_count, window_size=10)
    # # perform_stationarity_test(df, 'cooperation_proportion', stage_count)
    # perform_arima_forecasting(df, 'cooperation_proportion', stage_count)
    # perform_granger_causality_test(df, 'cooperation_proportion', 'clustering_coefficient', stage_count)

    # TIME SERIES ANALYSES
    # perform_cusum_analysis(df, 'cooperation_proportion', stage_count, color_dict, color_mode, threshold=0.5)
    # perform_change_point_detection(df, 'cooperation_proportion', stage_count, color_dict, color_mode)
    # perform_piecewise_regression(df, 'cooperation_proportion', stage_count, color_dict, color_mode, change_points=[50, 100])
    # perform_autocorrelation_analysis(df, 'cooperation_proportion', stage_count, color_dict, color_mode)

    # print(df['degree_distribution'].head)

    # FREQUENT ANALYSES
    plot_measurement(df, 'cooperation_proportion', stage_count, 'Cooperation Proportion' )
    # plot_measurement(df, 'degree_distribution', stage_count, 'Degree Distribution' )
    # plot_measurement(df, 'edge_weight', stage_count, 'Edge Weight' )
    # plot_measurement(df, 'clustering_coefficient', stage_count,'Clustering Coefficient' )
    # plot_measurement(df, 'average_path_length', stage_count, 'Average Path Length')
    # plot_measurement(df, 'density', stage_count, 'Density')
    # plot_measurement(df, 'assortativity', stage_count, 'Assortativity')
    # plot_measurement(df, 'modularity', stage_count, 'Modularity')

    # for anova and tipping points we create two new variables (intelligence -low,high- and group size -low, high-)
    df = preprocess_data(df) 
    df.to_csv('results_time_series.csv', index=False)
       
     # Tipping points
    df_melt, tipping_points_df = identify_tipping_points(df,stage_count)
    # plot_tipping_points(df_melt,tipping_points_df)
    # plot_tipping_points_Skill_Num(df_melt, tipping_points_df)

#    # # ANOVA
#     anova_results = perform_two_way_anova(df, stage_count)
#     tukey_results = perform_post_hoc_test(df_melt)
#     print(anova_results)
#     print(tukey_results)




if __name__ == "__main__":

    file_path = 'results_10-exp3w++_25_30iter.csv'  # Update this path to your actual CSV file path
    # file_path = 'results_COpy.csv'  # Update this path to your actual CSV file path - this one is the original csv file.
    # file_path = 'CSVs/results_40-60_20Net_ASP.csv'  # Update this path to your actual CSV file path - this one is the original csv file.
   
    main(file_path)