import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np

def plot_consumption_resampled(df, columns, name, intervalls):

    df_copy = df.copy()
    df_copy.index = pd.to_datetime(df_copy.index, unit='s')

    # Resample der Daten auf 24-Stunden-Intervalle und berechne den Durchschnitt
    df_resampled = df_copy.resample('24H').mean().reset_index()

    # Erstelle eine Figur mit sekundärer Y-Achse
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Verläufe der übergebenen Größen
    axis_title = ""
    for column in columns:
        if column == 'PF_TOT':
            fig.add_trace(
            go.Scatter(x=df_resampled['index'], y=df_resampled[column], name=column, line=dict(dash='dot')),
            secondary_y=True,
            )
            fig.update_yaxes(title_text='PF_TOT', secondary_y=True)
        else:
            fig.add_trace(
            go.Scatter(x=df_resampled['index'], y=df_resampled['P_TOT'], name='P_TOT'),
            secondary_y=False,
            )
            axis_title += column

    # Vertikale Linien für ergänzte Bereiche
    for intervall in intervalls: 
        intervall_1 = pd.to_datetime(intervall[0],unit='s')
        intervall_2 = pd.to_datetime(intervall[1],unit='s')
        fig.add_vline(x=intervall_1, line_dash='dash')
        fig.add_vline(x=intervall_2, line_dash='dash')
    # Benenne die Achsen
    fig.update_xaxes(title_text='Zeit')
    fig.update_yaxes(title_text=axis_title, secondary_y=False)
    

    # Füge einen Titel hinzu und passe das Layout an
    fig.update_layout(
        title_text='Zeitliche Darstellung der Werte mit 24-Stunden-Intervallen - {}'.format(name),
        xaxis=dict(
            tickmode='auto',
            nticks=20,
            ticks='outside',
            tickson='boundaries',
            ticklen=20
        )
    )
    # Zeige die Figur an
    return fig.show()

def plot_metrics_lr(df_metrics):
    df = df_metrics.reset_index().rename(columns={'index': 'SFH'})
    # Erstellen des Plots
    fig = go.Figure(data=[
        go.Bar(name='RMSE', x=df['SFH'], y=df['RMSE'], yaxis='y', offsetgroup=1),
        #go.Bar(name='MSE', x=df['SFH'], y=df['MSE'], yaxis='y2', offsetgroup=2),
        go.Scatter(name='R2', x=df['SFH'], y=df['R2'], yaxis='y3', mode='lines+markers')
    ])

    # Aktualisieren des Layouts mit drei verschiedenen Y-Achsen
    fig.update_layout(
        title='Vergleich der Fehlermetriken für die interpolierten Verbrauchswerte',
        title_x=0.5,
        xaxis=dict(title='Datensatz'),
        yaxis=dict(title='RMSE in W', side='left'),
    # yaxis2=dict(title='MSE', overlaying='y', side='right', position=0.95),
        yaxis3=dict(title='R2', overlaying='y', side='right', position=0.85),
        legend=dict(x=0.01, y=0.99, bordercolor='Black', borderwidth=1)
    )

    # Hinzufügen einer sekundären Y-Achse für die MSE-Werte und eine dritte für R2
    fig.update_layout(
        yaxis2=dict(title='MSE', anchor="free", overlaying='y', side='right', position=0.95),
        yaxis3=dict(title='R2', anchor="x", overlaying='y', side='right', position=0.85, range=[0, 1])
    )

    # Zeige den Plot an
    fig.show()

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


def plot_pred_vs_act_cons(y_test, pred_test, timestamps, specified_date):
    # Konvertieren der Unix Zeitstempel in pandas Timestamps
    formatted_timestamps = pd.to_datetime(timestamps, unit='s')

    # Konvertieren des spezifizierten Datums in ein pandas Timestamp
    specified_datetime = pd.to_datetime(specified_date)

    # Finden des Indexbereichs für das spezifizierte Datum
    start_index = next(i for i, ts in enumerate(formatted_timestamps) if ts.date() == specified_datetime.date())
    end_index = start_index + 96  # 96 Zeitpunkte für einen Tag bei 15-minütiger Auflösung

    # Extrahieren der Daten für das spezifizierte Datum
    selected_timestamps = formatted_timestamps[start_index:end_index]
    actual_day_1 = y_test[start_index:end_index, 0]
    actual_day_2 = y_test[start_index:end_index, 1]
    predicted_day_1 = pred_test[start_index:end_index, 0]
    predicted_day_2 = pred_test[start_index:end_index, 1]

    # Erstellen des Plots mit Plotly
    fig = go.Figure()

    # Hinzufügen der tatsächlichen und vorhergesagten Werte
    fig.add_trace(go.Scatter(x=selected_timestamps, y=actual_day_1, mode='lines', name='P Messwerte', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=selected_timestamps, y=predicted_day_1, mode='lines', name='P Vorhersagewerte', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=selected_timestamps, y=actual_day_2, mode='lines', name='PF Messwerte', line=dict(color='blue'), yaxis='y2'))
    fig.add_trace(go.Scatter(x=selected_timestamps, y=predicted_day_2, mode='lines', name='PF Vorhersagewerte', line=dict(color='blue', dash='dash'), yaxis='y2'))

    # Aktualisierung des Layouts
    fig.update_layout(
        title=f'Vergleich von tatsächlichen und vorhergesagten Werten für {specified_datetime.strftime("%Y-%m-%d")}',
        xaxis_title='Uhrzeit',
        xaxis=dict(
            tickformat='%H:%M'
        ),
        yaxis=dict(title='Wirkleistung P [W]', color='red'),
        yaxis2=dict(title='Leistungsfaktor PF', color='blue', overlaying='y', side='right'),
        template='plotly_white',
        legend=dict(x=1.05),
        title_x=0.8,
        width=1100,
        height=600
    )

    # Anzeigen des Diagramms
    fig.show()