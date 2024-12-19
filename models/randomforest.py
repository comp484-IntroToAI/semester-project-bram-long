import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import helpers as hp


def find_important_features(df, feature_columns, target):
    '''
     Identify the most important features in a dataset (df) for predicting a target variable (target). 
     Returns a dataframe (feature_importance_df) ranking the features by their importance and 
     displays a bar chart showing the feature importances.
     df: dataset provided by user
     target: target variable that the model is predicting.
     feature_columns: input/feature variables that predict the model.
     returns dataframe containing the feature importance values.
    '''
    df_copy = df.copy()
    X = df_copy[feature_columns]
    y = df_copy[target]

    model = RandomForestRegressor(n_estimators=200, random_state=42)

    model.fit(X, y)

    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    })

    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Predicting Maximum Temperature')
    plt.gca().invert_yaxis()  
    plt.show()

    return feature_importance_df



def find_best_random_forest_model(X_train, y_train):
    '''Finds the best random forest model utilizing
     a grid search algorithm.
    X_train: dataframe of the input values for the train set. 
    y_train: dataframe of the target value for the train set.
    returns the model with the best cross-validation score
     '''

    model = RandomForestRegressor(random_state=42)
    param_grid = {
    'n_estimators': [100, 300, 500],  
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
    '''Gets the predictions of a specified random forest 
    model for the test set.
    model: random forest model.
    X_test: dataframe of inputs of the test set.
    returns random forest model predictions
    '''
    predictions = model.predict(X_test)
    return predictions

def plot_predictions(y_pred, y_original):
    """
    Plots predictions vs actual values for a random forest model.
    y_pred: dataframe of predictions.
    y_original: dataframe of original values.
    """
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))

    # Scatter plot of predictions vs actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_original, y_pred, alpha=0.5)
    plt.plot([y_original.min(), y_original.max()], 
            [y_original.min(), y_original.max()], 
            'r--', lw=2)
    plt.title('Random Forest Predictions vs Actual Temperature (F)')
    plt.xlabel('Actual Temperature (F)')
    plt.ylabel('Predicted Temperature (F)')

    # Prediction errors histogram
    errors = y_pred - y_original
    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=30, edgecolor='black')
    plt.title('Random Forest Prediction Errors Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    hp.calculate_metrics(y_original, y_pred)
