from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np

import config

# 2. Data Preparation

def train_and_predict(load_dict, weather_data, missing_intervals, list_incomplete, include_time_features=False):
    """
    Trains a linear regression model and predicts missing values for load data using weather data and optional time features.

    Parameters:
    - load_dict (dict): Dictionary containing load data for each house.
    - weather_data (pd.DataFrame): DataFrame containing weather data.
    - missing_intervals (dict): Dictionary containing the intervals of missing data for each house.
    - list_incomplete (list): List of keys for houses with incomplete data.
    - include_time_features (bool, optional): Flag to include time features in the model. Default is False.

    Returns:
    - dict: Dictionary containing the filled data for each house.
    - pd.DataFrame: DataFrame containing the RMSE, MSE, and R2 metrics for each house.
    """
    dict_result = {}
    df_metrics = pd.DataFrame(columns=['RMSE', 'MSE', 'R2'], index=list_incomplete)

    for key in list_incomplete:
        df_house = load_dict[key].set_index('index')
        df_house = df_house[df_house.index > INDEX_START]
        df_house = df_house[['P_TOT', 'PF_TOT']]

        # Cleaning negative values
        for column in df_house.columns:
            df_house.loc[df_house[column] < 0, column] = 0

        data = pd.merge(left=df_house, right=weather_data, how='inner', left_index=True, right_index=True)
        data = data.dropna()

        if include_time_features:
            data = add_time_features(data)

        for interval in missing_intervals[key]:
            features_to_predict = weather_data.loc[interval[0]: interval[1]]
            if include_time_features:
                features_to_predict = add_time_features(features_to_predict)

            X = data.drop(['P_TOT', 'PF_TOT'], axis=1)
            y = data[['P_TOT', 'PF_TOT']]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            model = LinearRegression()
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            
            r2 = r2_score(y_test, predictions)
            rmse = mean_squared_error(y_test, predictions, squared=False)
            mse = mean_squared_error(y_test, predictions, squared=True)

            predicted_values = model.predict(features_to_predict)
            df_house.loc[interval[0]:interval[1]] = predicted_values

        df_metrics.loc[key] = [rmse, mse, r2]
        dict_result[key] = df_house

    return dict_result, df_metrics

def add_time_features(data):
    """
    Adds time features (minute, hour, day, month, year) to the DataFrame.

    Parameters:
    - data (pd.DataFrame): DataFrame to which time features will be added.

    Returns:
    - pd.DataFrame: DataFrame with added time features.
    """
    data['minute'] = pd.to_datetime(data.index, unit='s').minute
    data['hour'] = pd.to_datetime(data.index, unit='s').hour
    data['day'] = pd.to_datetime(data.index, unit='s').day
    data['month'] = pd.to_datetime(data.index, unit='s').month
    data['year'] = pd.to_datetime(data.index, unit='s').year
    return data

def create_daily_sequences(df, input_seq_len, output_seq_len, num_target_var):
    """
    Creates daily sequences for the training of a GRU model
    Inputs:
    df:             DataFrame mit Zeitreihendaten
    input_seq_len:  length of input sequence in days
    output_seq_len: length of output/prediction sequence in days 
    num_target_var: Number of target variables
    Returns:
    Tuple with input sequences (X) and output sequences (y) 
    """
    points_per_day = 96  # number of 15 minute intervalls per day
    total_seq_len = input_seq_len + output_seq_len
    num_days = len(df) // points_per_day
    num_sequences = int(num_days - total_seq_len + 1)

    # initialize arrays for input and output sequences
    X = np.zeros((num_sequences, int(input_seq_len * points_per_day), 
                    df.shape[1]), dtype=np.float32)
    y = np.zeros((num_sequences, int(output_seq_len * points_per_day), 
                  num_target_var), dtype=np.float32)  
    # create sequences
    for i in range(num_sequences):
        start_idx = i * points_per_day
        end_idx = start_idx + int(total_seq_len * points_per_day)
        X[i] = df.iloc[start_idx:start_idx + int(input_seq_len * points_per_day), 
                       :].values
        y[i] = df.iloc[start_idx + int(input_seq_len * points_per_day):end_idx, 
                       :num_target_var].values

    return X, y

def train_test_val_data(df_scaled, num_households, num_target_variables, 
                        sequence_length=config.SEQUENZE_LENGTH, 
                        prediction_length=config.PREDICTION_LENGTH):
    ''' 
    Creates train, test and validation data sets by chronologically 
    splitting data into training/validation and test data. In order for the 
    validation data to be representative, the training/validation split is 
    performed randomly. 
    Inputs: 
    df_scaled:              scaled data
    len_dataset:            number of different households within the dataset
    num_target_variables:   number of target variables
    sequence_length:        length of sequence used for prediction
    prediction_length:      length of prediction 
    Returns: 
    X_train, X_val, X_test, y_train, y_val, y_test
    '''
    
    all_X, all_y, X_train_val, X_test, y_train_val, y_test = [], [], [], [], [], []

    # Create sequences for all buildings, append to one X and y variable
    for building in df_scaled["building"].unique():
        X_building, y_building = create_daily_sequences(
                                    df_scaled[df_scaled["building"]==building], 
                                    sequence_length, prediction_length, 
                                    num_target_variables
                                    )
        all_X.append(X_building)
        all_y.append(y_building)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # create train test val split for each house
    len_data = int(len(X)/num_households)               
    len_train = int(len(X)/num_households * config.TRAIN_VAL_SPLIT) 

    # chronological split - first half for training, second half for test
    for i in range(0, num_households): 
        X_train_val.append(X[int(i*len_data):int(i*len_data+len_train)])
        X_test.append(X[int(i*len_data+len_train):int((i+1)*len_data)])
        y_train_val.append(y[int(i*len_data):int(i*len_data+len_train)])
        y_test.append(y[int(i*len_data+len_train):int((i+1)*len_data)])

    X_train_val = np.concatenate(X_train_val, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_train_val = np.concatenate(y_train_val, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # split training data into training and validation data 
    X_train, X_val, y_train, y_val = train_test_split(
                                        X_train_val, y_train_val, 
                                        test_size=0.2, random_state=42
                                        )

    # adjust shape to [number of examples, number of output neurons]
    y_train = y_train.reshape(-1, int(prediction_length*96) * num_target_variables)  
    y_val= y_val.reshape(-1, int(prediction_length*96) * num_target_variables)        
    y_test = y_test.reshape(-1, int(prediction_length*96) * num_target_variables)    

    return X_train, X_val, X_test, y_train, y_val, y_test

######################
# 3. Hyperparameteroptimization

def get_optimization_results(tuner, num_trials):
    """
    Retrieves and organizes the optimization results from the tuner into a DataFrame.

    Parameters:
    - tuner: The tuning object containing the results of the hyperparameter optimization.
    - num_trials (int): Number of top trials to retrieve.

    Returns:
    - pd.DataFrame: DataFrame containing the hyperparameters and validation loss of the top trials, sorted by validation loss.
    """
    # Retrieve results of the completed trials
    completed_trials = tuner.oracle.get_best_trials(num_trials=num_trials)

    # Prepare data for visualization
    param_values = [trial.hyperparameters.values for trial in completed_trials]
    performances = [trial.score for trial in completed_trials]

    # Collect hyperparameters and performance values in a list of dictionaries
    data_list = []
    for values, performance in zip(param_values, performances):
        # Copy the hyperparameters and add the performance
        trial_data = values.copy()
        trial_data['val_loss'] = performance
        data_list.append(trial_data)
        
    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(data_list)

    # Display the DataFrame
    return results_df.sort_values(by="val_loss", ascending=True)

######################
# 5. Evaluation 

def inverse_transform(y_predicted, y_test, X_test, scaler, columns, num_target_variables=2, target_values=["P_TOT", "PF_TOT"]):
    """
    Performs inverse transformation on the predicted and actual values to convert them back to their original scale.

    Parameters:
    - y_predicted (np.array): The predicted values.
    - y_test (np.array): The actual target values from the test set.
    - X_test (np.array): The input features from the test set.
    - scaler (sklearn.preprocessing object): The scaler used for the initial transformation.
    - columns (list): list of columns for inverse transforming 
    - num_target_variables (int): Number of target variables predicted.
    - target_values (list): List of target variable names.

    Returns:
    - list: A list containing the actual values, predictions, and input features all converted back to their original scale.
    """
    predictions = []
    actual_values = []
    X_test_actual = []

    # Inverse transform predictions
    for prediction in y_predicted:
        temp_df = pd.DataFrame(0, index=np.arange(96*config.PREDICTION_LENGTH), columns=columns)
        temp_df[target_values] = prediction.reshape(-1, 96, num_target_variables)[0]
        predictions.append(pd.DataFrame(scaler.inverse_transform(temp_df), columns=columns)[target_values])

    # Inverse transform actual y values
    for actual_value in y_test:
        temp_df = pd.DataFrame(0, index=np.arange(96*config.PREDICTION_LENGTH), columns=columns)
        temp_df[target_values] = actual_value.reshape(-1, 96, num_target_variables)[0]
        actual_values.append(pd.DataFrame(scaler.inverse_transform(temp_df), columns=columns)[target_values])

    # Inverse transform X test data
    for test_data in X_test:
        X_test_actual.append(pd.DataFrame(scaler.inverse_transform(test_data), columns=columns))

    return actual_values, predictions, X_test_actual

def calculate_metrics(actual_values, predictions, n, column="P_TOT"):
    """
    Calculates various metrics to evaluate prediction performance.

    Parameters:
    - actual: Actual values.
    - predicted: Predicted values.

    Returns:
    - A list containing MSE, RMSE, NRMSE, MAE, and MAPE.
    """
    df_evaluation = pd.concat([actual_values[n], predictions[n]], axis=1)
    df_evaluation.columns = ["Actual", "Predicted"]
    mse = np.mean((df_evaluation["Actual"].values - df_evaluation["Predicted"].values) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse/len(df_evaluation)
    mae = mean_absolute_error(df_evaluation["Actual"].values, df_evaluation["Predicted"].values)
    mape = mean_absolute_percentage_error(df_evaluation["Actual"].values, df_evaluation["Predicted"].values)
    return [mse, rmse, nrmse, mae, mape]

def get_monthly_metrics(X_test_actual, actual_values, predictions):
    """
    Computes monthly metrics for predictions.

    Parameters:
    - X_test_actual: List of DataFrames containing the actual input features with timestamps.
    - actual_values: List of DataFrames containing the actual target values.
    - predictions: List of DataFrames containing the predicted values.

    Returns:
    - DataFrame containing monthly RMSE, MAE, and MAPE metrics.
    """
    index_dict = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: [],
        10: [],
        11: [],
        12: [],
    }

    for i in range(0, len(X_test_actual)):
        start_index_month = pd.to_datetime(X_test_actual[i]["index"][0], unit="s").month
        index_dict[start_index_month].append(i)
        
    monthly_metrics = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: [],
        10: [],
        11: [],
        12: [],
    }

    monthly_error_metrics = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: [],
        10: [],
        11: [],
        12: [],
    }

    for month in index_dict.keys():
        indizes = index_dict[month]
        df_metrics = pd.DataFrame(columns=["mse", "rmse", "nrmse", "mae", "mape"])
        for n in indizes:
            df_metrics.loc[n] = calculate_metrics(actual_values, predictions, n, column="P_TOT")
        monthly_metrics[month] = df_metrics.mean().values#.to_frame().T

    results = pd.DataFrame(monthly_metrics).T
    results.columns=["MSE", "RMSE", "nRMSE", "MAE", "MAPE"]
    results = results[["RMSE", "MAE", "MAPE"]]
    results = results.T
    results.columns=["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"]

    # calculate mape
    mape_values = []
    for month in index_dict.keys():
        indizes = index_dict[month]
        error_count = 0
        for n in indizes:
            mape = mean_absolute_percentage_error(actual_values[n].values, predictions[n].values)
        
            if mape < 100000:
                mape_values.append(mape)
            else:
                error_count += 1
                continue
        monthly_metrics[month] = np.mean(mape_values)
        monthly_error_metrics[month] = error_count
    results.loc["MAPE"] = monthly_metrics.values()

    # calculate mape
    nrmse_values = []
    for month in index_dict.keys():
        indizes = index_dict[month]
        error_count = 0
        for n in indizes:
            mse = np.mean((actual_values[n].values - predictions[n].values) ** 2)
            rmse = np.sqrt(mse)
            nrmse = rmse/(np.max(actual_values[n].values))

            if nrmse*100 < 500:
                nrmse_values.append(nrmse)
            else:
                #print("i")
                error_count += 1
                continue

        monthly_metrics[month] = np.mean(nrmse_values)*100
        #monthly_error_metrics[month] = error_count

    results.loc["nRMSE"] = monthly_metrics.values()

    results["Mittelwert"] = results.mean(axis=1)

    print(monthly_error_metrics)
    return results.round(2)


