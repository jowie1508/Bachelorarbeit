import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go

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
    fig.show()

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