import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error


'''Creates training and test sets for performance analyis.
X: inputs of the model
y: ouputs of the model
split: percent of data that is allocated for training the model.'''
def create_train_test(X, y, split):
    split_index = int(len(X) * split) 
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]
    return X_train, y_train, X_test, y_test 

'''Checks the stationarity of the time series data set. Used to determine if the data should be detrended.'''
def check_stationarity(timeseries):
    # Perform the Dickey-Fuller test
    result = adfuller(timeseries, autolag='AIC')
    p_value = result[1]
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {p_value}')
    print('Stationary' if p_value < 0.05 else 'Non-Stationary')


'''Calculates the performance metrics for the '''
def calculate_metrics(y_all, y_pred, split): 
    split_index = int(split * len(y_all)) 
    y_actual_split = y_all[split_index:]
    y_pred_split = y_pred[split_index:]

    mse = mean_squared_error(y_actual_split, y_pred_split)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual_split, y_pred_split)
    
    print(f"Mean Squared Error (MSE): {mse}")
print(f'Mean Absolute Error (MAE): {mae}')
print(f"Root Mean Squared Error (RMSE): {rmse}")

    return mse, rmse, mae