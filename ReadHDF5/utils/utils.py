# This file contains code for different stages of the bachelor thesis 
# on the prediction of baseline consumption of heat pumps

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Modelling

## Creating sequences
def create_daily_sequences(df, input_seq_len, output_seq_len):
    """
    Erstellt tägliche Sequenzen für das Training eines GRU-Modells aus einem pandas 
    DataFrame
    
    :param df: DataFrame mit Zeitreihendaten
    :param input_seq_len: Länge der Eingabesequenzen in Tagen
    :param output_seq_len: Länge der Ausgabesequenzen (Vorhersagehorizont) in Tagen
    :return: Tuple aus Eingabesequenzen (X) und Ausgabesequenzen (y)
    """
    data_points_per_day = 96  # 15-Minuten-Intervalle in einem Tag
    total_seq_len = input_seq_len + output_seq_len
    num_days = len(df) // data_points_per_day

    # Anpassung der Anzahl der Sequenzen
    num_sequences = num_days - total_seq_len + 1

    # Initialisierung der Arrays für Eingabe- und Ausgabesequenzen
    X = np.zeros((num_sequences, input_seq_len * data_points_per_day, 
                  df.shape[1]), dtype=np.float32)
    y = np.zeros((num_sequences, output_seq_len * data_points_per_day, 
                  2), dtype=np.float32)  # 2 für P_TOT und PF_TOT

    # Erstellung der Sequenzen
    for i in range(num_sequences):
        start_idx = i * data_points_per_day
        end_idx = start_idx + total_seq_len * data_points_per_day

        X[i] = df.iloc[start_idx:start_idx + input_seq_len * data_points_per_day, 
                       :].values
        y[i] = df.iloc[start_idx + input_seq_len * data_points_per_day:end_idx, 
                       :2].values

    return X, y

def test_sequences(X, y, input_sequence_length, prediction_length, column_nr=0, column_name='P_TOT', index=0):
    """
    Funktion zum Testen und Darstellen einer erstellten Sequenz und ihrer Zielvariablen.

    :param X: Eingabesequenzen.
    :param y: Ausgabesequenzen.
    :param index: Index der Sequenz, die dargestellt werden soll.
    """
    plt.figure(figsize=(14, 5))
    
    # Eingabesequenz darstellen
    plt.plot(range(input_sequence_length), X[index, :, column_nr], label='Eingabesequenz {}'.format(column_name))
    
    # Zielvariablen darstellen
    plt.plot(range(input_sequence_length, input_sequence_length + prediction_length), y[index, :, column_nr], label='Zielsequenz  {}'.format(column_name), linestyle='--')
    plt.legend()
    plt.title(f'Sequenz #{index} und Zielvariable')
    plt.xlabel('Zeitschritte')
    plt.ylabel('Skalierte Werte')
    plt.show()

# Hyperparametertuning
def get_optimization_results(tuner, num_trials):
    # Ergebnisse der Versuche
    completed_trials = tuner.oracle.get_best_trials(num_trials=num_trials)

    # Daten für die Visualisierung vorbereiten
    param_values = [trial.hyperparameters.values for trial in completed_trials]
    performances = [trial.score for trial in completed_trials]

    # Sammeln der Hyperparameter und Leistungswerte in einer Liste von Dictionaries
    data_list = []
    for values, performance in zip(param_values, performances):
        # Kopieren Sie die Hyperparameter und fügen Sie die Leistung hinzu
        trial_data = values.copy()
        trial_data['val_loss'] = performance
        data_list.append(trial_data)
    # Erstellen des DataFrames aus der Liste von Dictionaries
    results_df = pd.DataFrame(data_list)

    # Anzeigen des DataFrames
    return results_df.sort_values(by="val_loss", ascending=True).head(20)

def plot_validation_loss(trained_model, title_spez=""):
    # Erstellung des Plotly-Diagramms
    fig = go.Figure()

    # Hinzufügen des Trainingsverlustes
    fig.add_trace(go.Scatter(
        y=trained_model.history['loss'], 
        mode='lines', 
        name='Trainingsverlust'
    ))

    # Hinzufügen des Validierungsverlustes
    fig.add_trace(go.Scatter(
        y=trained_model.history['val_loss'], 
        mode='lines', 
        name='Validierungsverlust'
    ))

    # Aktualisierung des Layouts
    fig.update_layout(
        title='Trainings- und Validierungsverlust {}'.format(title_spez),
        title_x=0.5,
        legend_x=0.8,
        xaxis_title='Epochen',
        yaxis_title='Verlust',
        template='plotly_white',
        width=1000,
        height=500
    )

    # Anzeigen des Diagramms
    return fig.show()

def print_metrics(y_test, predicted_test):
    # Berechnung von RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predicted_test))

    # Berechnung von MAE
    mae = mean_absolute_error(y_test, predicted_test)

    # Berechnung von R² Score
    r2 = r2_score(y_test, predicted_test)

    # Berechnung von MAPE
    mape = mean_absolute_percentage_error(y_test, predicted_test)

    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")
    print(f"Coefficient of Determination (R² Score): {r2}")

def quick_result_plot(predictions, test_data):
    index = list(range(1, 97))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=index, y=predictions[0:96], mode='lines', name='P Vorhersagewerte', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=index, y=test_data[0:96], mode='lines', name='P Messwerte', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=index, y=predictions[96:], mode='lines', name='PF Vorhersagewerte', line=dict(color='blue'), yaxis='y2'))
    fig.add_trace(go.Scatter(x=index, y=test_data[96:], mode='lines', name='PF Messwerte', line=dict(color='blue', dash='dash'), yaxis='y2'))
    # Aktualisierung des Layouts
    fig.update_layout(
        title=f'Vergleich von tatsächlichen und vorhergesagten Werten',
        xaxis_title='Zeitstempel',
        yaxis=dict(title='Wirkleistung P [W]', color='red'),
        yaxis2=dict(title='Leistungsfaktor PF', color='blue', overlaying='y', side='right'),
        template='plotly_white',
        legend=dict(x=1.05),
        title_x=0.8,
        width=1100,
        height=600
    )
    fig.show()

def calculate_nrmse(actual, predicted):
    """ Berechnet den normierten Root Mean Squared Error (nRMSE) zwischen den tatsächlichen und vorhergesagten Werten. """
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    
    # Normierung des RMSE
    # Hier wird der RMSE durch den Bereich der tatsächlichen Werte normiert
    range_of_actual = np.max(actual) - np.min(actual)
    
    # Vermeidung einer Division durch Null
    if range_of_actual == 0:
        return np.inf

    nrmse = rmse / range_of_actual
    return nrmse*100