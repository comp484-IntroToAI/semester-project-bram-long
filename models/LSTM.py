import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Dropout, Embedding, LSTM, Bidirectional
import helpers as hp

def prepare_data(X_train, y_train, X_test, y_test, window_size=12):
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




def train_LSTM(X_train, y_train,units,batch_size,epochs, verbose=2, learning_rate=0.001):
    #==Define model architecture
    # model = Sequential()
    # #===== Add LSTM layers
    # model.add(LSTM(units = units, return_sequences=True,activation='relu',
    #                input_shape=(X_train.shape[1], X_train.shape[2])))
    # #===== Hidden layer
    # model.add(LSTM(units = units))
    # #=== output layer
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(1))

    model = Sequential()
    
    # First LSTM layer with return_sequences=True
    model.add(LSTM(units=units, 
                   return_sequences=True,
                   activation='relu',
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    
    # Second LSTM layer
    model.add(LSTM(units=units))
    
    # Output dense layers with dropout
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(32, activation='relu'))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    
    # Final output layer
    model.add(Dense(1))
    #==== Compiling the model
    model.compile(optimizer='adam', loss='mean_squared_error') 
    #====== Fit Model
    early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',patience = 10)
    history = model.fit(X_train, y_train, epochs = epochs, 
                        validation_split = 0.2,
                        batch_size = batch_size, shuffle = False,
                        callbacks = [early_stop],verbose=0
                        )   
    
    modelN='LSTM'
    return(history,modelN,model)


def train_BiLSTM(X_train,y_train,units,batch_size,epochs, verbose=2, learning_rate=0.001):
    # model = Sequential()
    # model.add(Bidirectional(LSTM(30, return_sequences=True, activation='tanh', 
    #                               input_shape=(X_train.shape[1], X_train.shape[2]))))
    # model.add(Bidirectional(LSTM(20, return_sequences=True, activation='tanh')))
    # model.add(Bidirectional(LSTM(10, return_sequences=False, activation='tanh')))
    model = Sequential()
    model.add(Bidirectional(LSTM(20, return_sequences=False, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2]))))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    # model.compile(optimizer='adam', loss='mape',  metrics=['MeanSquaredError', 'MeanAbsoluteError','RootMeanSquaredError'],) 
    #====== Fit Model
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    history = model.fit(X_train, y_train, epochs = epochs, validation_split = 0.2,
                        batch_size = batch_size,  shuffle = False, callbacks = [early_stop],verbose=verbose)
    
    modelN='BiLSTM'
    return(history,modelN,model)

def plot_loss(history, name):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f'{name} - Loss Curve')
    plt.show()


def get_predictions(model, X_test_windowed):
    y_pred = model.predict(X_test_windowed)
    return y_pred

def plot_predictions(model_name, X_test_windowed, y_test_windowed, y_pred, target_scaler):

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
    plt.title(f'{model_name} Predictions vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

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

    plt.figure(figsize=(12, 6))
    plt.plot(y_pred_original , label='Actual', color='blue')
    plt.plot(y_pred_original, label='Predicted', color='red', linestyle='--')
    plt.title(f'{model_name} Actual vs Predicted Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.show()