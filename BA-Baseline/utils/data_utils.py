import pandas as pd
import pickle
import re

def data_loader(columns, timespan=None):
    """
    Loads, merges, and filters data from multiple sources, including cleaned heat pump data, weather data, and building information.

    Parameters:
    - columns (list): List of column names to be included in the final dataset.
    - timespan (int, optional): Number of days to limit the data to. If not specified, all available data is included.

    Returns:
    - pandas.DataFrame: A DataFrame containing the specified columns of data for each building, optionally limited to a certain timespan.
    """

    # Load cleaned heat pump data
    with open('data/cleaned/data_heatpump_cleaned_v1.pkl', 'rb') as f:
        load_dict = pickle.load(f)

    # Load weather data
    with open('Data/cleaned/data_weather_v1.pkl', 'rb') as f:
        weather_data = pickle.load(f)

    # Load building information and set the index
    building_info = pd.read_excel("data/cleaned/Gebaeudeinformationen_cleaned.xlsx", index_col=0)
    building_info.set_index("Building number", inplace=True)

    load_dict_sorted = {}

    # Add building information and merge with weather data
    for house in sorted(load_dict, key=lambda x: int(re.findall(r'\d+', x)[0])):
        id = int(re.findall(r'\d+', house)[0])

        # Add building area, number of inhabitants, and building id to each house's data
        load_dict[house]["area"] = building_info.loc[id]["Building area"]
        load_dict[house]["inhabitants"] = building_info.loc[id]["Number of inhabitants"]
        load_dict[house]["building"] = id
        
        # Filter weather data and merge with house data
        weather_data_filtered = weather_data[weather_data.index >= 1528965900]
        load_dict[house] = pd.concat([load_dict[house], weather_data_filtered], axis=1)
        load_dict[house].reset_index(inplace=True)
        load_dict[house] = load_dict[house][load_dict[house]["index"] > 1546298100]

        # If a timespan is specified, limit the data to that timespan
        if timespan:
            load_dict_sorted[house] = load_dict[house][columns].iloc[:4*24*timespan]
        else:
            load_dict_sorted[house] = load_dict[house][columns]

    # Concatenate data for all houses
    data = pd.concat(load_dict_sorted)

    # Reset index and set the new index to the house id, dropping the old index column
    return data.reset_index().set_index("level_0").drop(columns="level_1")
