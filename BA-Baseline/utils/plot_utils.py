import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import re
import matplotlib.dates as mdates
import math

from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error

import config


# 1 Data Exploration

def plot_data_availability(data):
    """
    Plots the availability of data for each sensor in a given DataFrame.

    The function assumes the DataFrame's index contains Unix timestamps and each column represents a sensor. 
    The values in the DataFrame should be 1 (data available) or 0 (data missing). 
    The plot illustrates the availability and missing data over time for each dataset.

    Parameters:
    - data (pandas.DataFrame): The DataFrame containing time series data for various datasets.

    The function converts the Unix timestamp index to datetime, sorts the columns based on their numeric part 
    (assuming column names start with 'SFH'), calculates the percentage of available data for each sensor, 
    and plots the data availability over time with color-coded bars.
    """

    df = data.copy()
    df.index = pd.to_datetime(df.index, unit='s')

    # Sort columns based on the numeric part in their names (assuming 'SFH' prefix)
    sorted_columns = sorted(df.columns, key=lambda x: int(x.replace("SFH", "")))

    # Calculate the percentage of available data for each column
    percentages = (df.sum() / len(df) * 100).round(2)
    percentages = percentages[sorted_columns]

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(7, 8))

    # Plot data availability for each sensor
    for i, column in enumerate(sorted_columns):
        ax.fill_between(df.index, i, i + 1, where=(df[column] == 1), color='#66D37A', step='mid', label='Available' if i == 0 else "")
        ax.fill_between(df.index, i, i + 1, where=(df[column] == 0), color='#FF5252', step='mid', label='Missing' if i == 0 else "")
        # Add percentage next to each bar
        ax.text(df.index[-1] + pd.Timedelta(days=1), i + 0.5, f" {percentages[column]}%", verticalalignment='center')

    # Adjust plot settings
    ax.set_xlim([df.index.min(), df.index.max() + pd.Timedelta(days=1)])
    ax.set_ylim([0, len(sorted_columns)])
    ax.set_yticks(np.arange(len(sorted_columns)) + 0.5)
    ax.set_yticklabels(sorted_columns)
    ax.set_title("Data Availability")
    ax.set_ylabel("Sensor")

    # Add color legend
    ax.legend(loc='upper right')

    # Format the x-axis with a 3-month interval
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


def plot_comsumption(data, name):
    """
    Creates a time series plot with dual y-axes to display the consumption metrics P_TOT, Q_TOT, S_TOT, and PF_TOT.

    Parameters:
    - data (pandas.DataFrame): The DataFrame containing the time series data with columns 'P_TOT', 'Q_TOT', 'S_TOT', and 'PF_TOT'.
    - name (str): A descriptive name for the dataset, used in the plot title.

    This function resets the index of the DataFrame to use it as the x-axis, then adds traces for each metric to the plot.
    P_TOT, Q_TOT, and S_TOT are plotted on the primary y-axis, while PF_TOT is plotted on the secondary y-axis with a dashed line style.
    """

    # Reset the index of the DataFrame to use it as the x-axis
    df = data.reset_index()

    # Create a figure with a secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces for P_TOT, Q_TOT, and S_TOT to the primary y-axis
    fig.add_trace(go.Scatter(x=df['index'], y=df['P_TOT'], name='P_TOT'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['index'], y=df['Q_TOT'], name='Q_TOT'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['index'], y=df['S_TOT'], name='S_TOT'), secondary_y=False)

    # Add a trace for PF_TOT to the secondary y-axis with a dashed line style
    fig.add_trace(go.Scatter(x=df['index'], y=df['PF_TOT'], name='PF_TOT', line=dict(dash='dot')), secondary_y=True)

    # Set axis titles
    fig.update_xaxes(title_text='Time')
    fig.update_yaxes(title_text='P_TOT, Q_TOT, S_TOT', secondary_y=False)
    fig.update_yaxes(title_text='PF_TOT', secondary_y=True)

    # Add a title and adjust the layout
    fig.update_layout(
        title_text='Temporal Representation of Metrics - {}'.format(name),
        xaxis=dict(
            tickmode='auto',
            nticks=20,
            ticks='outside',
            tickson='boundaries',
            ticklen=20
        )
    )

    # Display the figure
    fig.show()

def plot_resampled_consumption(data, name):
    """
    Creates a time series plot with dual y-axes to display resampled consumption metrics P_TOT, Q_TOT, S_TOT, and PF_TOT over 24-hour intervals.

    Parameters:
    - data (pandas.DataFrame): The DataFrame containing the time series data with columns 'P_TOT', 'Q_TOT', 'S_TOT', and 'PF_TOT'. The DataFrame should have an 'index' column with datetime values.
    - name (str): A descriptive name for the dataset, used in the plot title.

    This function first resamples the data into 24-hour intervals, calculating the mean for each interval. It then creates a plot with P_TOT, Q_TOT, and S_TOT on the primary y-axis and PF_TOT on the secondary y-axis, using a dashed line for PF_TOT.
    """

    # Ensure 'index' is converted to datetime and set as the DataFrame index
    df = data.copy()
    df['index'] = pd.to_datetime(df['index'])
    df.set_index('index', inplace=True)

    # Resample data into 24-hour intervals and calculate the mean
    df_resampled = df.resample('24H').mean().reset_index()

    # Create a figure with a secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces for P_TOT, Q_TOT, and S_TOT to the primary y-axis
    fig.add_trace(go.Scatter(x=df_resampled['index'], y=df_resampled['P_TOT'], name='P_TOT'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_resampled['index'], y=df_resampled['Q_TOT'], name='Q_TOT'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_resampled['index'], y=df_resampled['S_TOT'], name='S_TOT'), secondary_y=False)

    # Add a trace for PF_TOT to the secondary y-axis with a dashed line style
    fig.add_trace(go.Scatter(x=df_resampled['index'], y=df_resampled['PF_TOT'], name='PF_TOT', line=dict(dash='dot')), secondary_y=True)

    # Set axis titles
    fig.update_xaxes(title_text='Time')
    fig.update_yaxes(title_text='P_TOT, Q_TOT, S_TOT', secondary_y=False)
    fig.update_yaxes(title_text='PF_TOT', secondary_y=True)

    # Add a title and adjust the layout
    fig.update_layout(
        title_text='Temporal Representation of Metrics with 24-Hour Intervals - {}'.format(name),
        xaxis=dict(
            tickmode='auto',
            nticks=20,
            ticks='outside',
            tickson='boundaries',
            ticklen=20
        )
    )

    # Display the figure
    fig.show()

def plot_yearly_resampled_consumption(data, year, name):
    """
    Creates a time series plot for a specific year, showing resampled daily average values of P_TOT, Q_TOT, S_TOT, and PF_TOT.

    Parameters:
    - data (pandas.DataFrame): The DataFrame containing time series data with columns 'P_TOT', 'Q_TOT', 'S_TOT', and 'PF_TOT'. The DataFrame should have an 'index' column with datetime values.
    - year (int): The year for which the data is to be plotted.
    - name (str): A descriptive name for the dataset, used in the plot title.

    The function first resets the DataFrame index to ensure it's datetime, then resamples the data into 24-hour intervals to calculate the daily mean. It then filters the data for the specified year and plots P_TOT, Q_TOT, and S_TOT on the primary y-axis and PF_TOT on the secondary y-axis.
    """

    # Ensure 'index' is converted to datetime and set as the DataFrame index
    df = data.reset_index()
    df['index'] = pd.to_datetime(df['index'])
    df.set_index('index', inplace=True)

    # Resample data into 24-hour intervals and calculate the mean
    df_resampled = df.resample('24H').mean()
    
    # Filter data for the specified year
    df_year = df_resampled[df_resampled.index.year == year]

    # Create a figure with a secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces for P_TOT, Q_TOT, and S_TOT to the primary y-axis
    fig.add_trace(go.Scatter(x=df_year.index, y=df_year['P_TOT'], name='P_TOT'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_year.index, y=df_year['Q_TOT'], name='Q_TOT'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_year.index, y=df_year['S_TOT'], name='S_TOT'), secondary_y=False)

    # Add a trace for PF_TOT to the secondary y-axis with a dashed line style
    fig.add_trace(go.Scatter(x=df_year.index, y=df_year['PF_TOT'], name='PF_TOT', line=dict(dash='dot')), secondary_y=True)

    # Update layout settings
    fig.update_layout(
        title_text=f"Average Values in 24-Hour Intervals for the Year {year} - {name}",
        height=600,
        width=1400
    )
    fig.update_yaxes(title_text='P_TOT, Q_TOT, S_TOT', secondary_y=False)
    fig.update_yaxes(title_text='PF_TOT', secondary_y=True)

    # Display the figure
    fig.show()

def plot_consumption_type_histo(df_consumptions, years):
    """
    Plots a grouped bar chart representing the power consumption in different modes for each household over the given years.

    Parameters:
    - df_consumptions (pandas.DataFrame): DataFrame containing consumption data with columns 'Standby', 'Compression Mode', and 'Heating Rod Mode'.
    - years (str or int): The years for which the data is being plotted, used in the title.

    The function resets the DataFrame index and uses it as the x-axis to represent households. Each consumption mode is represented by a different color bar.
    """

    # Reset the index of the DataFrame to use it for the x-axis labels
    df_consumptions.reset_index(inplace=True)

    # Create the bar chart
    fig = go.Figure()

    # Add bars for each consumption category
    fig.add_trace(go.Bar(
        x=df_consumptions['index'],
        y=df_consumptions['Standby'],
        name='Standby Mode',
        marker_color='green'
    ))

    fig.add_trace(go.Bar(
        x=df_consumptions['index'],
        y=df_consumptions['Compression Mode'],
        name='Compression Mode',
        marker_color='blue'
    ))

    fig.add_trace(go.Bar(
        x=df_consumptions['index'],
        y=df_consumptions['Heating Rod Mode'],
        name='Heating Rod Mode',
        marker_color='red'
    ))

    # Update the layout
    fig.update_layout(
        title=f'Active Power Consumption in kWh/year - {years}',
        title_x=0.5,
        xaxis_tickangle=-45,
        xaxis_title='Household',
        yaxis_title='Active Power in kWh/year',
        barmode='group',
        legend_title='Legend',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Display the figure
    fig.show()


def plot_consumption_with_band(df, start_date, end_date, dataset_name, column_name='P_TOT'):
    """
    Creates a band plot showing the daily average and standard deviation for a specified column in a DataFrame over a given date range.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data, indexed by datetime.
    - start_date (str): The start date of the period to plot, in a format that can be parsed by pandas, e.g., 'YYYY-MM-DD'.
    - end_date (str): The end date of the period to plot, in the same format.
    - dataset_name (str): A descriptive name for the dataset, used in the plot title.
    - column_name (str, optional): The name of the column to plot. Defaults to 'P_TOT'.

    The function filters the DataFrame for the specified date range, resamples the data to daily intervals, calculates the mean and standard deviation for each day, and creates a band plot with these values.
    """

    # Filter the DataFrame for the desired period
    filtered_df = df.loc[start_date:end_date]

    # Daily resampling and calculation of mean and standard deviation
    daily_mean = filtered_df[column_name].resample('D').mean()
    daily_std = filtered_df[column_name].resample('D').std()

    # Create the band plot
    fig = go.Figure()

    # Line for the mean
    fig.add_trace(go.Scatter(x=daily_mean.index, y=daily_mean, mode='lines', name='Mean', line=dict(color='blue')))

    # Band for the standard deviation, adjusted for values less than zero
    upper_band = daily_mean + daily_std
    lower_band = np.maximum(daily_mean - daily_std, 0)  # Compare with zero to avoid negative values

    fig.add_trace(go.Scatter(x=daily_mean.index, y=upper_band, mode='lines', name='+1 Standard Deviation', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=daily_mean.index, y=lower_band, mode='lines', name='-1 Standard Deviation', line=dict(width=0), fill='tonexty', fillcolor='rgba(135, 206, 235, 0.4)', showlegend=False))

    # Customize the layout
    fig.update_layout(title=f'Band Plot for {dataset_name}', title_y=0.85, title_x=0.5, legend_x=0.85, legend_y=0.9, xaxis_title='Time', yaxis_title=f"Power {column_name} [W]", template='plotly_white')

    # Show the plot
    fig.show()


def plot_quantile_comparison(data, threshold=7900):
    """
    Creates a stacked bar chart comparing maximum values before and after outlier filtering based on a 99.5th percentile threshold.

    Parameters:
    - data (pandas.DataFrame): The DataFrame containing the data, indexed by a unique identifier with a 'P_TOT' column for power values.
    - threshold (float, optional): A threshold value to add as a horizontal dashed line for reference.

    The function iterates over each unique index in the DataFrame, filters the data considering the 99.5th percentile for the 'P_TOT' column, calculates the maximum values before and after filtering, and plots these values in a stacked bar chart.
    """

    quantile_values = []  # Maximum values considering the 99.5th percentile
    total_values = []  # Maximum values without considering the quantile
    indices = []

    for index in data.index.unique():
        sub_df = data[data.index == index]
        
        # Filtering considering the 99.5th percentile
        quantile_sub_df = sub_df[sub_df["P_TOT"] < sub_df["P_TOT"].quantile(0.995)]
        max_quantile = quantile_sub_df["P_TOT"].max()
        quantile_values.append(max_quantile)
        
        # Maximum value without quantile
        max_total = sub_df["P_TOT"].max()
        total_values.append(max_total - max_quantile)  # Difference for the top part of the bar
        
        indices.append(index)

    # Create the stacked bar chart
    fig = go.Figure()

    # Lower part of the bar: Maximum values considering the 99.5th percentile
    fig.add_trace(go.Bar(
        x=indices,
        y=quantile_values,
        name='Maximalwert unter Beachtung des 99,5% Perzentil',
        #marker_color='rgba(135, 206, 235, 0.5)'  # Soft blue with some transparency
    ))

    # Upper part of the bar: Difference to the maximum value without quantile
    fig.add_trace(go.Bar(
        x=indices,
        y=total_values,
        name='Maximalwert des Originaldatensatz',
        marker_color='rgba(135, 206, 235, 0.5)'  # A different color for the top part
    ))

    # Update layout for stacked bars
    fig.update_layout(
        title='Vergleich der Maximalwerte vor und nach Ausreißerfilterung',
        title_x=0.5,
        legend_x=0.6,
        legend_y=1.05,
        xaxis_title='Datensatz',
        yaxis_title='Maximaler Leistungswert P [W]',
        template='plotly_white',
        barmode='stack',  # Important for stacked bars
        width=1000
    )

    # Add a horizontal dashed red line at the threshold
    fig.add_trace(go.Scatter(
        x=indices,  # Use the same x-values as for the bars
        y=[threshold] * len(indices),  # A constant y-value list at the threshold
        mode='lines',
        name='Grenzwert {}'.format(threshold),
        showlegend=False,
        line=dict(color='red', width=2, dash='dash')  # Red dashed line
    ))

    # Display the chart
    fig.show()

#######################################
# 2 Data Preparation
def plot_metrics_lr(df_metrics):
    """
    Creates a plot to visualize the RMSE and R2 error metrics for different datasets.

    Parameters:
    - df_metrics (pd.DataFrame): A DataFrame containing 'RMSE' and 'R2' columns for various datasets.
    """
    df = df_metrics.reset_index().rename(columns={'index': 'SFH'})
    # Creating the plot
    fig = go.Figure(data=[
        go.Bar(name='RMSE', x=df['SFH'], y=df['RMSE'], yaxis='y', offsetgroup=1, marker_color='rgb(49,130,189)'),
        go.Scatter(name='R2', x=df['SFH'], y=df['R2'], yaxis='y2', mode='lines+markers', line=dict(color='rgb(214,39,40)'))
    ])

    # Updating the layout with two different Y-axes
    fig.update_layout(
        title='Vergleich der Fehlermetriken für die interpolierten Verbrauchswerte',
        title_x=0.5,
        xaxis=dict(title='Datensatz'),
        yaxis=dict(title='RMSE in W', side='left'),
        yaxis2=dict(title='R2', overlaying='y', side='right', 
                    #position=0.85, 
                    range=[0, 1]),
        legend=dict(x=0.01, y=0.99, bordercolor='Black', borderwidth=1),
        template="plotly_white",
        width=900
    )

    # Displaying the plot
    fig.show()

from plotly.subplots import make_subplots
import plotly.graph_objs as go

def plot_consumption_filled(df, columns, name, intervals):
    """
    Plots the resampled consumption data with vertical lines indicating specified intervals.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the consumption data.
    - columns (list): List of column names in the DataFrame to be plotted.
    - name (str): Name of the dataset for the plot title.
    - intervals (list): List of tuples, each containing the start and end of an interval to be highlighted in the plot.
    """

    df_copy = df.copy()
    df_copy.index = pd.to_datetime(df_copy.index, unit='s')

    # Resample data to 24-hour intervals and calculate the mean
    df_resampled = df_copy.resample('24H').mean().reset_index()

    # Create a figure with a secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot the specified columns
    for column in columns:
        if column == 'PF_TOT':
            fig.add_trace(
                go.Scatter(x=df_resampled['index'], y=df_resampled[column], name="cos(φ)", line=dict(dash='dot')),
                secondary_y=True,
            )
            fig.update_yaxes(title_text='Leistungsfaktor cos(φ)', secondary_y=True)
        else:
            fig.add_trace(
                go.Scatter(x=df_resampled['index'], y=df_resampled[column], name=column),
                secondary_y=False,
            )

    # Add vertical lines for the specified intervals
    for interval in intervals:
        start = pd.to_datetime(interval[0], unit='s')
        end = pd.to_datetime(interval[1], unit='s')
        fig.add_vline(x=start, line_dash='dash')
        fig.add_vline(x=end, line_dash='dash')

    # Set axis titles
    fig.update_xaxes(title_text='Zeit')
    fig.update_yaxes(title_text="Wirkleistung P [W]", secondary_y=False)

    # Customize layout
    fig.update_layout(
        title_text='Zeitliche Darstellung der Werte mit 24-Stunden-Intervallen - {}'.format(name),
        title_x=0.5,
        title_y=0.9,
        xaxis=dict(
            tickmode='auto',
            nticks=20,
            ticks='outside',
            tickson='boundaries',
            ticklen=20,
            tickangle=-45,
        ),
        template="plotly_white",
        width=900
    )

    # Display the figure
    fig.show()


def plot_scaling(scaled_data, data, scaler):
    """
    Displays original, scaled, and inverse-scaled data for comparison.

    Parameters:
    - scaled_data (np.array): The scaled data resulting from a scaling process.
    - data (pd.DataFrame): The original data before scaling.
    - scaler (object): The scaler object used for scaling and inverse-scaling the data.
    """
    # Display only for SFH23 as an example
    start = 0 * (4 * 24 * 90)  # Adjust the start index as needed
    end = 1 * (4 * 24 * 90)    # Adjust the end index as needed

    df_scaled_23 = pd.DataFrame(scaled_data, columns=data.columns).iloc[start:end]
    data_23 = data.iloc[start:end]

    # Inverse scaling
    df_inverse_scaled = pd.DataFrame(scaler.inverse_transform(scaled_data), columns=data.columns)
    df_inverse_scaled_23 = df_inverse_scaled.iloc[start:end]
    df_inverse_scaled_23["index"] = df_inverse_scaled_23["index"].astype(int)

    # Create subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Original Daten", "Skalierte Daten", "Zurückskalierte Daten"))

    # Add original data plot
    fig.add_trace(
        go.Scatter(x=pd.to_datetime(data_23["index"], unit="s"), y=data_23["P_TOT"], name="Original"),
        row=1, col=1
    )

    # Add scaled data plot
    fig.add_trace(
        go.Scatter(x=df_scaled_23.index, y=df_scaled_23["P_TOT"], name="Skaliert"),
        row=1, col=2
    )

    # Add inverse-scaled data plot
    fig.add_trace(
        go.Scatter(x=pd.to_datetime(df_inverse_scaled_23["index"], unit="s"), y=df_inverse_scaled_23["P_TOT"], name="Zurück-Skaliert"),
        row=1, col=3
    )

    # Update layout
    fig.update_layout(title_text="Vergleich von Original-, Skalierten- und Zurück-Skalierten-Daten", title_x=0.5)

    # Display the plot
    fig.show()


def plot_sequences(X, y, scaler, load_dict, input_sequence_length, prediction_length, column_nr=0, column_name='P_TOT', index=0):
    """
    Function to test and display a created sequence and its target variables.

    Parameters:
    - X: Input sequences.
    - y: Output sequences (target variables).
    - scaler: The scaler object used for scaling and inverse scaling the data.
    - load_dict: Dictionary containing load data for each house.
    - input_sequence_length: Length of the input sequences.
    - prediction_length: Length of the prediction sequences.
    - column_nr: The column number in the sequence to be plotted.
    - column_name: Name of the column to be plotted.
    - index: Index of the sequence to be displayed.
    """
    
    actual_X_data = pd.DataFrame(scaler.inverse_transform(X[index]), columns=config.columns_P)

    # Retrieve the timestamp from the first column of the X data and convert it to datetime
    start_timestamp = pd.to_datetime(actual_X_data["index"].iloc[0], unit='s')
    date_str = (start_timestamp + pd.Timedelta(days=3, minutes=30)).strftime('%d-%m-%Y')
    
    # Create time series for input and target sequence
    time_series_input = [start_timestamp + pd.Timedelta(minutes=15*i) for i in range(input_sequence_length)]
    time_series_output = [time_series_input[-1] + pd.Timedelta(minutes=15*i) for i in range(1, prediction_length + 1)]
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Plot the input sequence
    fig.add_trace(go.Scatter(x=time_series_input, y=X[index, :, column_nr], mode='lines', name='Eingabesequenz {}'.format(column_name)))
    
    # Plot the target variables
    fig.add_trace(go.Scatter(x=time_series_output, y=y[index, :, column_nr], mode='lines', name='Zielsequenz {}'.format(column_name), line=dict(dash='dash')))
    
    # Adjust layout
    fig.update_layout(title='Sequenz für Vorhersage {}'.format(date_str), title_x=0.5, xaxis_title='Zeit', yaxis_title='Skalierte Werte', template='plotly_white', legend_x=0.8, legend_y=0.95)
    
    # Display the plot
    fig.show()


def plot_with_classification(data, title, train_split=0.7, val_split=0.85, y_secondary=False, combine=False):
    """
    Plots the data series with separate sections for training, validation, and testing sets, with an option to combine training and validation sets.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be plotted.
    - title (str): Title for the plot.
    - train_split (float): Fraction of data to be used for the training set.
    - val_split (float): Fraction of data to be used for the validation set (rest is used for the testing set).
    - y_secondary (bool): If True, plots PF_TOT on a secondary Y-axis.
    - combine (bool): If True, combines training and validation data in one plot section.
    """
    df = data.copy()
    # Convert Unix timestamps to readable datetime
    df['index'] = pd.to_datetime(df['index'], unit='s')

    # Calculate split indices
    train_end_idx = int(len(df) * train_split)
    val_end_idx = int(len(df) * val_split)

    # Create subplot
    fig = make_subplots(specs=[[{"secondary_y": y_secondary}]])

    if combine:
        # Plot P_TOT time series for training and validation combined
        fig.add_trace(
            go.Scatter(x=df['index'][:train_end_idx], y=df['P_TOT'][:train_end_idx], name='Trainings- und Validierungsdaten (P_TOT)'),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=df['index'][train_end_idx:], y=df['P_TOT'][train_end_idx:], name='Testdaten (P_TOT)'),
            secondary_y=False,
        )
    else:
        # Plot P_TOT time series for validation
        fig.add_trace(
            go.Scatter(x=df['index'][train_end_idx:val_end_idx], y=df['P_TOT'][train_end_idx:val_end_idx], name='Validierungsdaten (P_TOT)'),
            secondary_y=False,
        )
        # Plot P_TOT time series for testing
        fig.add_trace(
            go.Scatter(x=df['index'][val_end_idx:], y=df['P_TOT'][val_end_idx:], name='Testdaten (P_TOT)'),
            secondary_y=False,
        )

    if y_secondary:
        # Plot PF_TOT time series
        fig.add_trace(
            go.Scatter(x=df['index'][:train_end_idx], y=df['PF_TOT'][:train_end_idx], name='Trainingsdaten (PF_TOT)', line=dict(dash='dot')),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(x=df['index'][train_end_idx:val_end_idx], y=df['PF_TOT'][train_end_idx:val_end_idx], name='Validierungsdaten (PF_TOT)', line=dict(dash='dot')),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(x=df['index'][val_end_idx:], y=df['PF_TOT'][val_end_idx:], name='Testdaten (PF_TOT)', line=dict(dash='dot')),
            secondary_y=True,
        )
        fig.update_yaxes(title_text='Leistungsfaktor PF_TOT', secondary_y=True)

    # Set axis titles
    fig.update_xaxes(title_text='Zeitstempel')
    fig.update_yaxes(title_text='Wirkleistung P [W]', secondary_y=False)

    # Update layout for plotly white style
    fig.update_layout(template='plotly_white', title='Darstellung der {}-Zeitreihe mit Trainings-, Validierungs- und Testdaten'.format(title), title_x=0.5, legend_x=0.65)

    # Display the plot
    fig.show()

#####################
# 3. Hyperparameteroptimization


def plot_validation_losses_and_durations(trained_models_list, titles_list, durations_list):
    """
    Plots the training and validation loss, and training durations of different models for comparison.

    Parameters:
    - trained_models_list (list): List of trained model histories.
    - titles_list (list): List of titles for each model.
    - durations_list (list): List of training durations for each model.
    """
    # Create a Plotly figure with subplots: 1 row, 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Trainings- und Validierungsverlust', 'Trainingsdauer'),
                        column_widths=[0.8, 0.2])

    # Define colors for the models
    colors = ['blue', 'green', 'red', 'orange', "purple"]  # Extend this list if more than five models are present

    for i, trained_model in enumerate(trained_models_list):
        # Add training loss for each model to the first subplot
        fig.add_trace(go.Scatter(
            y=trained_model.history['loss'],
            mode='lines',
            name=f'Trainingsverlust {titles_list[i]}',
            line=dict(color=colors[i]),
            showlegend=True
        ), row=1, col=1)

        # Add validation loss for each model to the first subplot
        fig.add_trace(go.Scatter(
            y=trained_model.history['val_loss'],
            mode='lines',
            name=f'Validierungsverlust {titles_list[i]}',
            line=dict(color=colors[i], dash='dash'),
            showlegend=True  # Show legend only once
        ), row=1, col=1)

    # Add bar charts for training durations to the second subplot
    for i, duration in enumerate(durations_list):
        fig.add_trace(go.Bar(
            x=[titles_list[i]],
            y=[duration],
            name=f'Trainingsdauer {titles_list[i]}',
            marker_color=colors[i],
            showlegend=False,
        ), row=1, col=2)

    # Update layout
    fig.update_layout(
        title='', #'Trainings- und Validierungsverlust mit Trainingsdauer',
        title_x=0.5,
        title_font=dict(size=24),  # Adjust title font size
        legend=dict(
            font=dict(size=16),  # Adjust legend font size
        ),
        xaxis_title='Epochen',
        yaxis_title='Verlust',
        xaxis_title_font=dict(size=18),
        yaxis_title_font=dict(size=18),
        xaxis_tickfont=dict(size=14),
        yaxis_tickfont=dict(size=14),
        xaxis2_title='Modelle',
        yaxis2_title='Dauer (s)',
        xaxis2_title_font=dict(size=18),
        yaxis2_title_font=dict(size=18),
        xaxis2_tickfont=dict(size=14),
        yaxis2_tickfont=dict(size=14),
        template='plotly_white',
        width=1200,
        height=500,
    )

    # Display the plot
    fig.show()

###############
# 5. Evaluation

def plot_training_history(model_P, model_PF):
    """
    Plots the training and validation loss for two models representing power (P) and power factor (PF).

    Parameters:
    - model_P: Model trained to predict power (P).
    - model_PF: Model trained to predict power factor (PF).
    """
    # Create a subplot layout with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Wirkleistung P", "Leistungsfaktor cos(φ)"))

    # Colors for the lines
    colors = ['blue', 'red']  # One color for each model

    # Plot training and validation loss for the P model
    fig.add_trace(go.Scatter(y=model_P.history['loss'], mode='lines', name='Trainingsverlust', line=dict(color=colors[0])), row=1, col=1)
    fig.add_trace(go.Scatter(y=model_P.history['val_loss'], mode='lines', name='Validierungsverlust', line=dict(color=colors[1], dash='dot')), row=1, col=1)

    # Plot training and validation loss for the PF model
    fig.add_trace(go.Scatter(y=model_PF.history['loss'], mode='lines', name='Trainingsverlust', line=dict(color=colors[0]), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(y=model_PF.history['val_loss'], mode='lines', name='Validierungsverlust', line=dict(color=colors[1], dash='dot'), showlegend=False), row=1, col=2)

    # Determine the epochs with the minimum validation loss for both models
    min_val_loss_epoch_P = np.argmin(model_P.history['val_loss'])
    min_val_loss_epoch_PF = np.argmin(model_PF.history['val_loss'])

    # Add a vertical line for the epoch with the minimum validation loss
    fig.add_trace(go.Scatter(x=[min_val_loss_epoch_P, min_val_loss_epoch_P], y=[min(model_P.history['loss']), max(model_P.history['val_loss'])], mode='lines', name='Optimum', line=dict(color='black', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[min_val_loss_epoch_PF, min_val_loss_epoch_PF], y=[min(model_PF.history['loss']), max(model_PF.history['loss'])], mode='lines', name='Optimum', line=dict(color='black', dash='dash'), showlegend=False), row=1, col=2)

    # Update layout
    fig.update_layout(
        title='Trainings- und Validierungsverluste der optimierten Modelle',
        title_x=0.225,
        legend_x=0.8,
        legend_y=0.95,
        xaxis_title='Epochen',
        yaxis_title='Verlust (MSE)',
        xaxis2_title='Epochen',
        template='plotly_white',
        title_font=dict(size=18),
        legend=dict(font=dict(size=12)),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        xaxis_tickfont=dict(size=14),
        yaxis_tickfont=dict(size=14),
        xaxis2_title_font=dict(size=14),
        yaxis2_title_font=dict(size=14),
        xaxis2_tickfont=dict(size=14),
        yaxis2_tickfont=dict(size=14),
        width=900
    )

    # Display the plot
    fig.show()


def plot_predictions_vs_test_sequence(X_test_actual, actual_values, predictions, n):
    """
    Displays a plot comparing the input sequence, actual test data, and predicted data for a specific prediction.

    Parameters:
    - X_test_actual (list): List of DataFrames containing the actual input features with timestamps.
    - actual_values (list): List of DataFrames containing the actual target values.
    - predictions (list): List of DataFrames containing the predicted values.
    - n (int): Index of the prediction set to display.
    """

    # Retrieve the specific test data and predictions
    df_X_test = X_test_actual[n]
    df_y_test = actual_values[n].to_frame()
    df_y_predicted = predictions[n].to_frame()

    # Set negative predictions to 0
    df_y_predicted[df_y_predicted < 0] = 0

    # Get the timestamp for the start of the test data and create a time series for the plot
    first_timestamp = df_X_test["index"].iloc[0]
    start_time = pd.to_datetime(first_timestamp, unit='s')
    start_time_prediction = start_time + pd.Timedelta(days=3)
    date_str = (start_time_prediction + pd.Timedelta(hours=1)).strftime('%d-%m-%Y')
    time_series = [start_time_prediction + pd.Timedelta(minutes=15*i) for i in range(len(df_y_test.values))]
    time_series_complete = [start_time + pd.Timedelta(minutes=15*i) for i in range(len(df_X_test.values))]

    # Get building information from the test data
    sfh = int(df_X_test["building"].iloc[0].round(0))

    # Create the plot
    print(df_y_test)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_series_complete, y=df_X_test["P_TOT"], mode='lines', name='Eingabesequenz', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=time_series, y=df_y_test["P_TOT"], mode='lines', name='Testdaten', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=time_series, y=df_y_predicted["P_TOT"], mode='lines', name='Vorhersagedaten', line=dict(color='green')))

    fig.update_layout(title=f'Ergebnisse mit Eingabesequenz am {date_str} - SFH{sfh}',
                      title_x=0.5,
                      xaxis_title='Zeit',
                      yaxis_title="P_TOT",
                      template='plotly_white',
                      width=900,
                      legend={'orientation': 'h', 'y': 1.1, 'x': 0.2})

    # Calculate and print the RMSE between the test data and the predicted data for the selected day
    rmse_original_predicted = np.sqrt(np.mean((df_y_test["P_TOT"].values - df_y_predicted["P_TOT"].values) ** 2))
    print(f'RMSE zwischen der ursprünglichen und der vorhergesagten Zeitreihe für Tag {n + 1}: {rmse_original_predicted}')

    fig.show()


def plot_predictions_vs_test(actual_values, predictions, X_test_actual, n, multiplier=3.5, window_length=12, traces={"smoothing": True, "boosting":  True}):
    """
    Plots the comparison of actual values, predictions, and optionally smoothed and boosted predictions.

    Parameters:
    - actual_values (list): List containing actual values for each prediction.
    - predictions (list): List containing predicted values.
    - X_test_actual (list): List containing original input features with timestamps.
    - n (int): Index of the prediction to plot.
    - multiplier (float): Multiplier for boosting the prediction deltas.
    - window_length (int): Window length for the Savitzky-Golay filter.
    - traces (dict): Dictionary to control the display of smoothing and boosting traces.
    """

    # Reshape arrays for easier handling
    actual_values_array = np.stack(actual_values).reshape(-1, config.PREDICTION_LENGTH * 96)
    predictions_array = np.stack(predictions).reshape(-1, config.PREDICTION_LENGTH * 96)

    # Smoothing using Savitzky-Golay filter
    if traces["smoothing"]:
        smoothed_data = savgol_filter(actual_values_array, window_length=window_length, polyorder=0, axis=1)

    # Boosting prediction deltas
    if traces["boosting"]:
        prediction_deltas = np.diff(predictions_array, axis=1, prepend=predictions_array[:, 0:1])
        enhanced_deltas = prediction_deltas * multiplier
        enhanced_predictions = np.cumsum(enhanced_deltas, axis=1)
        enhanced_predictions[enhanced_predictions < 0] = 0  # Set negative values to 0

    # Prepare time series for plotting
    first_timestamp = X_test_actual[n]["index"].iloc[0]
    start_time = pd.to_datetime(first_timestamp, unit='s')
    date_str = start_time.strftime('%d-%m-%Y')
    time_series = [start_time + pd.Timedelta(minutes=15 * i) for i in range(predictions_array.shape[1])]

    # Get building information
    sfh = int(X_test_actual[n]["building"].iloc[0].round(0))

    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_series, y=actual_values_array[n, :], mode='lines', name='Testdaten'))
    fig.add_trace(go.Scatter(x=time_series, y=predictions_array[n, :], mode='lines', name='Vorhersagedaten'))

    if traces["smoothing"]:
        fig.add_trace(go.Scatter(x=time_series, y=smoothed_data[n, :], mode='lines', name='Geglättete Testdaten'))

    if traces["boosting"]:
        fig.add_trace(go.Scatter(x=time_series, y=enhanced_predictions[n, :], mode='lines', name='Verstärkte Vorhersagedaten'))

    fig.update_layout(
        title=f'Vergleich der Zeitreihen für {date_str} - SFH{sfh}',
        xaxis_title='Zeit',
        yaxis_title='Last P (W)',
        legend={'orientation': 'h', 'y': 1.1, 'x': 0.3},
        template='plotly_white',
        width=900
    )

    fig.show()

    # Calculate and print RMSE metrics
    rmse_original_predicted = np.sqrt(mean_squared_error(actual_values_array[n, :], predictions_array[n, :]))
    print(f'RMSE zwischen der ursprünglichen und der vorhergesagten Zeitreihe für Tag {n + 1}: {rmse_original_predicted:.2f}')

    if traces["smoothing"]:
        rmse_smoothed_predicted = np.sqrt(mean_squared_error(smoothed_data[n, :], predictions_array[n, :]))
        print(f'RMSE zwischen der geglätteten und der vorhergesagten Zeitreihe für Tag {n + 1}: {rmse_smoothed_predicted:.2f}')

    if traces["boosting"]:
        rmse_enhanced_predicted = np.sqrt(mean_squared_error(actual_values_array[n, :], enhanced_predictions[n, :]))
        print(f'RMSE zwischen den Testdaten und den verstärkten Vorhersagedaten für Tag {n + 1}: {rmse_enhanced_predicted:.2f}')


def plot_comparison(actual_values, predictions, X_test_actual, actual_values_2, predictions_2, X_test_actual_2, n):
    """
    Plots a comparison of actual and predicted values for two sets of data (e.g., power and power factor) for a specific prediction.

    Parameters:
    - actual_values (list): List containing actual values for the first set of predictions.
    - predictions (list): List containing predicted values for the first set.
    - X_test_actual (list): List containing original input features with timestamps for the first set.
    - actual_values_2 (list): List containing actual values for the second set of predictions.
    - predictions_2 (list): List containing predicted values for the second set.
    - X_test_actual_2 (list): List containing original input features with timestamps for the second set.
    - n (int): Index of the prediction to plot.
    """

    # Reshape arrays for easier handling
    actual_values_array_P = np.stack(actual_values).reshape(-1, config.PREDICTION_LENGTH * 96)
    predictions_array_P = np.stack(predictions).reshape(-1, config.PREDICTION_LENGTH * 96)
    actual_values_array_PF = np.stack(actual_values_2).reshape(-1, config.PREDICTION_LENGTH * 96)
    predictions_array_PF = np.stack(predictions_2).reshape(-1, config.PREDICTION_LENGTH * 96)

    # Adjust x-axis
    first_timestamp = X_test_actual[n]["index"].iloc[-1]  # Last value of input sequence
    start_time = pd.to_datetime(first_timestamp, unit='s')
    date_str = start_time.strftime('%d-%m-%Y')
    time_series = [start_time + pd.Timedelta(minutes=15 * i) for i in range(len(actual_values_array_P[n, :]))]

    # Get building information
    sfh = int(X_test_actual[n]["building"].iloc[0].round(0))

    # Create plot with subplots for both P and PF
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot actual and predicted values for P
    fig.add_trace(go.Scatter(x=time_series, y=actual_values_array_P[n, :], line=dict(color="blue"), mode='lines', name='Testdaten P'), secondary_y=False)
    fig.add_trace(go.Scatter(x=time_series, y=predictions_array_P[n, :], line=dict(color="blue", dash='dot'), mode='lines', name='Vorhersagedaten P'), secondary_y=False)

    # Plot actual and predicted values for PF
    fig.add_trace(go.Scatter(x=time_series, y=actual_values_array_PF[n, :], line=dict(color="red"), mode='lines', name='Testdaten cos(φ)'), secondary_y=True)
    fig.add_trace(go.Scatter(x=time_series, y=predictions_array_PF[n, :], line=dict(color="red", dash='dot'), mode='lines', name='Vorhersagedaten cos(φ)'), secondary_y=True)

    # Update layout
    fig.update_layout(
        title=f'Vergleich der Zeitreihen für {date_str} - SFH{sfh}',
        title_x=0.5,
        xaxis_title='Zeit',
        xaxis=dict(tickformat='%H:%M', showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis_title='Last P (W)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        width=900,
        legend={'orientation': 'h', 'y': 1.1, 'x': 0.1}
    )

    # Set axis labels for the secondary y-axis
    fig.update_yaxes(title_text="Leistungsfaktor cos(φ)", secondary_y=True)

    # Display the plot
    fig.show()

####################
# Aggregated model
    
def plot_results(actual_values, predictions, X_test_actual, n, multiplier=3.5, window_length=12, traces={"actual": True, "prediction": True, "smoothing": True, "boosting": True}, legend_x=0.1):
    """
    Plots a comparison of actual values, predictions, and optionally smoothed and boosted predictions.

    Parameters:
    - actual_values (list): List of actual values.
    - predictions (list): List of predicted values.
    - X_test_actual (list): List of DataFrames containing the actual input features with timestamps.
    - n (int): Index of the prediction to be plotted.
    - multiplier (float): Multiplier for boosting prediction deltas.
    - window_length (int): Window length for smoothing.
    - traces (dict): Dictionary indicating which traces to plot.
    - legend_x (float): X position of the legend.
    """
    # Reshape arrays for easier handling
    actual_values_array = np.stack(actual_values).reshape(-1, config.PREDICTION_LENGTH * 96)
    predictions_array = np.stack(predictions).reshape(-1, config.PREDICTION_LENGTH * 96)

    # Smoothing using Savitzky-Golay filter
    if traces["smoothing"]:
        smoothed_data = np.zeros_like(actual_values_array)
        for i in range(actual_values_array.shape[0]):
            smoothed_data[i, :] = savgol_filter(actual_values_array[i, :], window_length=window_length, polyorder=0)

    # Boosting predictions
    if traces["boosting"]:
        prediction_deltas = np.diff(predictions_array[n, :], prepend=predictions_array[n, 0])
        enhanced_deltas = prediction_deltas * multiplier
        enhanced_predictions = np.cumsum(enhanced_deltas)
        enhanced_predictions[enhanced_predictions < 0] = 0  # Set negative values to 0

    # Adjust x-axis
    first_timestamp = X_test_actual[n]["index"].iloc[-1]  # Last value of input sequence
    start_time = pd.to_datetime(first_timestamp, unit='s')
    date_str = start_time.strftime('%d-%m-%Y')
    time_series = [start_time + pd.Timedelta(minutes=15*i) for i in range(len(actual_values_array[n, :]))]

    # Create plot
    fig = go.Figure()

    if traces["actual"]:
        fig.add_trace(go.Scatter(x=time_series, y=actual_values_array[n, :], mode='lines', name='Testdaten', line=dict(color="blue")))
    if traces["prediction"]:
        fig.add_trace(go.Scatter(x=time_series, y=predictions_array[n, :], mode='lines', name='Vorhersagedaten', line=dict(color="red", dash='dash')))
    if traces["smoothing"]:
        fig.add_trace(go.Scatter(x=time_series, y=smoothed_data[n, :], mode='lines', name='Geglättete Testdaten', line=dict(color="green")))
    if traces["boosting"]:
        fig.add_trace(go.Scatter(x=time_series, y=enhanced_predictions, mode='lines', name='Angepasste Vorhersagedaten', line=dict(color="purple")))

    fig.update_layout(
        title=f'Vergleich der Zeitreihen für {date_str}',
        xaxis_title='Zeit',
        yaxis_title='Last P (W)',
        legend=dict(x=legend_x, y=1.1, orientation='h'),
        template="plotly_white"
    )

    fig.show()

    # Calculate and print RMSE for original, smoothed, and boosted predictions
    rmse_original = np.sqrt(np.mean((actual_values_array[n, :] - predictions_array[n, :]) ** 2))
    print(f'RMSE Original vs. Vorhersage: {rmse_original:.2f}')
    if traces["smoothing"]:
        rmse_smoothed = np.sqrt(np.mean((smoothed_data[n, :] - predictions_array[n, :]) ** 2))
        print(f'RMSE Geglättet vs. Vorhersage: {rmse_smoothed:.2f}')
    if traces["boosting"]:
        rmse_boosted = np.sqrt(np.mean((actual_values_array[n, :] - enhanced_predictions) ** 2))
        print(f'RMSE Original vs. Angepasste Vorhersage: {rmse_boosted:.2f}')
