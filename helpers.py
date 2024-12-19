import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def create_train_test(X, y, split):
    '''Creates training and test sets for performance analyis.
    X: inputs of the model
    y: ouputs of the model
    split: percent of data that is allocated for training the model.'''
    X_train, X_test = train_test_split(X, train_size= split, random_state=42)
    y_train, y_test = train_test_split(y, train_size= split, random_state=42)
    return X_train, y_train, X_test, y_test 


# ToDO finish this
def calculate_metrics(y_test, y_pred): 
    '''Calculates the performance metrics for a prediction and prints them.
        y_test: actual values
        y_pred: predicted values
    '''
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse}")
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    return mse, rmse, mae

