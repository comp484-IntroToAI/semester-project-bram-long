import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def find_important_features(df, feature_columns, target):
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



'''Finds the best random forest model utilizing a grid search algorithm.'''
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

'''Gets the predictions of a model for the test set y.'''
def get_rf_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions
