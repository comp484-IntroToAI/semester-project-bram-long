import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

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


# https://www.geeksforgeeks.org/sarima-seasonal-autoregressive-integrated-moving-average/
def get_SARIMA_predictions(df):
    return 

'''Checks the stationarity of the time series data set. Used to determine if the data should be detrended.'''
def check_stationarity(timeseries):
    # Perform the Dickey-Fuller test
    result = adfuller(timeseries, autolag='AIC')
    p_value = result[1]
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {p_value}')
    print('Stationary' if p_value < 0.05 else 'Non-Stationary')


def find_best_random_forest_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42)

    param_grid = {
    'n_estimators': [100],  
    'max_depth': [10, 20],     
    'min_samples_split': [2, 5],          
    'min_samples_leaf': [1, 2],           
    'max_features': [None, 'sqrt', 'log2'],  
    'bootstrap': [True]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)
    return best_model


def get_rf_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions
    

def get_LSTM_predictions(df):

# def get_Transformer_predictions(df):

