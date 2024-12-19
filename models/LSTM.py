import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Dropout, Embedding, LSTM, Bidirectional
import helpers as hp

def prepare_data(X_train, y_train, X_test, y_test, window_size=12):
    """
    Prepares time series data for training and testing an LSTM model.

    This function processes the input data by standardizing the features and target values
    and then using the sliding window technique. 
    X_train: dataframe for training set of X variables
    y_train: dataframe for train set of y variables
    X_test: dataframe for test set of X variables
    y_test: dataframe for test set of y variables
    window_size: size of the windows
    returns prepared dataframe
    """
    feature_scaler = StandardScaler()
    X_np_train = X_train.values
    X_scaled_train = feature_scaler.fit_transform(X_np_train)
    
    target_scaler = StandardScaler()
    y_np_train = y_train.values
    y_scaled_train = target_scaler.fit_transform(y_np_train.reshape(-1, 1))
    
    X_windowed = []
    y_windowed = []
    for i in range(len(X_scaled_train) - window_size + 1):
        X_window = X_scaled_train[i:i+window_size]
        y_window = y_scaled_train[i+window_size-1]
        X_windowed.append(X_window)
        y_windowed.append(y_window)
    
    X_windowed = np.array(X_windowed)
    y_windowed = np.array(y_windowed)
    
    X_np_test = X_test.values
    X_scaled_test = feature_scaler.transform(X_np_test)
    y_np_test = y_test.values
    y_scaled_test = target_scaler.transform(y_np_test.reshape(-1, 1))

    X_test_windowed = []
    y_test_windowed = []
    for i in range(len(X_scaled_test) - window_size + 1):
        X_window = X_scaled_test[i:i+window_size]
        y_window = y_scaled_test[i+window_size-1] # drops first observations because doesn't have full window (preventing data leakage)
        X_test_windowed.append(X_window)
        y_test_windowed.append(y_window)
    
    X_test_windowed = np.array(X_test_windowed)
    y_test_windowed = np.array(y_test_windowed)
    
    return X_windowed, y_windowed, X_test_windowed, y_test_windowed, feature_scaler, target_scaler


def train_LSTM(X_train, y_train, units, batch_size, epochs, checkpoint_path, verbose=2, learning_rate=0.001):
    """
    Trains an LSTM model for time series forecasting.
    
    This function creates and trains a LSTM model using the provided
    training data. It includes multiple LSTM and Dense layers, with dropout for
    regularization. Early stopping and model checkpointing are used for training
    optimization.
    """
    model = Sequential()
    #===== Add LSTM layers
    model.add(LSTM(units = units, return_sequences=True,activation='relu',
                    input_shape=(X_train.shape[1], X_train.shape[2])))
    #===== Hidden layer
    model.add(LSTM(units = units))
    #=== output layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    #==== Compiling the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    #====== Fit Model
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 10)

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    history = model.fit(X_train, y_train, epochs = epochs, validation_split = 0.2,
                        batch_size = batch_size, shuffle = False, callbacks = [early_stop, checkpoint],verbose=verbose)

    modelN='LSTM'
    return(history,modelN,model)


def train_BiLSTM(X_train, y_train, units, batch_size, epochs, checkpoint_path, verbose=2, learning_rate=0.001):
    """
    Trains a Bidirectional LSTM model for time series forecasting.

    This function creates and trains a Bidirectional LSTM model using the provided
    training data. It includes multiple Bidirectional LSTM and Dense layers, with dropout for
    regularization. Early stopping and model checkpointing are used for training
    optimization.
    """

    model = Sequential()
    model.add(Bidirectional(LSTM(30, return_sequences=True, activation='tanh',
                                    input_shape=(X_train.shape[1], X_train.shape[2]))))
    model.add(Bidirectional(LSTM(20, return_sequences=True, activation='tanh')))
    model.add(Bidirectional(LSTM(10, return_sequences=False, activation='tanh')))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))


    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])


    # model.compile(optimizer='adam', loss='mape',  metrics=['MeanSquaredError', 'MeanAbsoluteError','RootMeanSquaredError'],)
    #====== Fit Model
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    history = model.fit(X_train, y_train, epochs = epochs, validation_split = 0.2,
                        batch_size = batch_size,  shuffle = False, callbacks = [early_stop, checkpoint], verbose=verbose)

    modelN='BiLSTM'
    return(history,modelN,model)

def plot_loss(history, name):
    """
    This function takes a history object returned by the tensorflow model.
    It plots and labels the training and validation loss curves of the model. 
    history: history of training and validaiton loss from tensorflow model
    name: Name of model
    """
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f'{name} - Loss Curve')
    plt.show()


def get_predictions(model, X_test_windowed):
    """
    This function takes a trained model and a set of test inputs and returns the model's predictions on those inputs.
    
    Parameters:
    model (tf.keras.Model): The trained model whose predictions we want to get.
    X_test_windowed (array): The test inputs, already windowed.
    
    Returns:
    y_pred (array): The model's predictions on X_test_windowed.
    """
    y_pred = model.predict(X_test_windowed)
    return y_pred

def plot_predictions(model_name, X_test_windowed, y_test_windowed, y_pred, target_scaler):

    """
    This function takes a trained model, the model name, the windowed test inputs, 
    the windowed test outputs, and the target scaler as parameters and plots 
    the model's predictions on the test set against the actual values. The 
    predictions and actual values are inverse transformed back to their original
    scale. The function also plots a histogram of the prediction errors.
    """
    # Inverse transform predictions and actual values

    y_pred_original = target_scaler.inverse_transform(y_pred)
    y_test_original = target_scaler.inverse_transform(y_test_windowed)

    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))

    # Scatter plot of predictions vs actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_original, y_pred_original, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()], 
            [y_test_original.min(), y_test_original.max()], 
            'r--', lw=2)
    plt.title(f'{model_name} Predictions vs Actual Temperature (F)')
    plt.xlabel('Actual Temperature (F)')
    plt.ylabel('Predicted Temperature (F)')

    # # Prediction errors histogram
    errors = y_pred_original - y_test_original
    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=30, edgecolor='black')
    plt.title(f'{model_name} Prediction Errors Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    hp.calculate_metrics(y_test_original, y_pred_original)
