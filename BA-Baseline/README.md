# Baseline-Prediction of Individual Heat Pumps

This repository contains the code created during the bachelor thesis "Baseline Prediction of Decentral Consumers: A case study on Heat Pumps" by Jonas Wiendl. 

Based on heat pump consumption data and historic weather data a GRU model is developed to predict future consumption, using the previous three days of load and weather information. 

# Installation

Access raw data
- As data source is the WPUQ data used, which can be accessed via ZENODO (https://zenodo.org/records/5642902). 
- save the raw hdf5-files in "data/WPUQ/..."

Install python packages
- create venv: python3 -m venv venv
- activate venv: 
    - Windows: .\venv\Scripts\activate
    - Linux/MacOS: source venv/bin/activate
- install all packages: pip install -r requirements.txt

# Execution

To run the project and obtain predictions, follow these chronological steps by executing the provided Jupyter Notebooks:

1. **Data Exploration:**
   - Start with `01_data_exploration.ipynb` to explore and understand the dataset. This notebook provides initial insights into the data.

2. **Data Preparation:**
   - Proceed with `02_data_preparation.ipynb` to prepare the data for modeling. This involves cleaning, normalization, feature selection, to structure the data suitably for the prediction model.

3. **Hyperparameter Optimization:**
   - Use `03_hyperparameter_optimization.ipynb` to find the optimal set of hyperparameters for the model. This notebook employs random search to explore various hyperparameter combinations.

4. **Modeling:**
   - In `04_modeling.ipynb`, the actual model is built, trained, and validated using the prepared dataset and the optimized hyperparameters identified in the previous step.

5. **Evaluation:**
   - After training the model, `05_evaluation.ipynb` is used to evaluate its performance. This notebook provides metrics such as MAE, RMSE and nRMSE relevant to the model's performance, along with visualizations.

6. **Model Aggregated Load:**
   - Additionaly, `06_Model_aggregated_load.ipynb` evaluates the performance of a simple GRU model on an aggregated load model.

Each notebook is self-contained with detailed instructions and explanations guiding you through the process. Ensure that you have all necessary packages installed as per the `requirements.txt` file and that the raw data is placed in the specified directory before beginning.

# Reference
In case of any questions, contact Jonas Wiendl. 




