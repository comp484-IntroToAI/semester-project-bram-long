import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Embedding, LSTM, Bidirectional


def scale_inputs(X_train, X_test):
    # Initialize the scaler
    scaler_X_lstm = StandardScaler()
    # Fit and transform the training data, then transform the test data
    X_train_scaled = scaler_X_lstm.fit_transform(X_train)
    X_test_scaled = scaler_X_lstm.transform(X_test)
    print(X_train_scaled)
    return X_train_scaled, X_test_scaled, scaler_X_lstm

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
        y_window = y_scaled_test[i+window_size-1]
        X_test_windowed.append(X_window)
        y_test_windowed.append(y_window)
    
    X_test_windowed = np.array(X_test_windowed)
    y_test_windowed = np.array(y_test_windowed)
    
    return X_windowed, y_windowed, X_test_windowed, y_test_windowed, feature_scaler, target_scaler
def create_sliding_window(X, y, window_size=3):
    X_windowed = []
    y_windowed = []
    
    for i in range(len(X) - window_size):
        X_window = X[i:i+window_size]
        y_window = y[i+window_size]
        
        X_windowed.append(X_window)
        y_windowed.append(y_window)


def Train_BiLSTM(X_train,y_train,units,batch_size,epochs, verbose=2, learning_rate=0.001):
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
    history = model.fit(X_train, y_train, epochs = epochs, validation_split = 0.2,
                        batch_size = batch_size,  shuffle = False, callbacks = [early_stop],verbose=verbose)
    
    modelN='BiLSTM'
    return(history,modelN,model)

y_lstm = new_lstm_df['actual_max_temp']

X_lstm = new_lstm_df[['temperature_2m_mean (°F)_lag1',
 'apparent_temperature_mean (°F)_lag1',
 'apparent_temperature_min (°F)_lag1',
 'apparent_temperature_max (°F)_lag1',
 'actual_max_temp_lag1',
 'wind_direction_10m_dominant (°)_lag1',
 'temperature_2m_max (°F)_lag1']]
 
X_lstm_train, y_lstm_train, X_lstm_test,  y_lstm_test = create_train_test(X_lstm, y_lstm, '2022-01-01')

X_windowed, y_windowed, X_test_windowed, y_test_windowed, feature_scaler, target_scaler = prepare_data(X_lstm_train, y_lstm_train, X_lstm_test,  y_lstm_test, window_size=3)

print(X_windowed.shape, y_windowed.shape, X_test_windowed.shape, y_test_windowed.shape)

units = 64  
batch_size = 64
epochs = 10

history, biLSTM_model_name, trained_bilstm_model = Train_BiLSTM(X_windowed, y_windowed, units, batch_size, epochs, verbose=2, learning_rate = .0007)

# Model summary to check the architecture
trained_bilstm_model.summary()

# Plot loss curve
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title(f'{model_name} - Loss Curve')
plt.show()
print('Number of elements in y_train_scaled:', len(y_train_scaled))



y_pred = trained_bilstm_model.predict(X_test_windowed)

# # Inverse transform predictions and actual values
y_pred_original = y_scaler.inverse_transform(y_pred)
y_test_original = y_scaler.inverse_transform(y_test_windowed)

# Plot predictions vs actual
plt.figure(figsize=(12, 6))

# # Scatter plot of predictions vs actual
plt.subplot(1, 2, 1)
plt.scatter(y_test_original, y_pred_original, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], 
         [y_test_original.min(), y_test_original.max()], 
         'r--', lw=2)
plt.title('Predictions vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# # Prediction errors histogram
errors = y_pred_original - y_test_original
plt.subplot(1, 2, 2)
plt.hist(errors, bins=30, edgecolor='black')
plt.title('Prediction Errors Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Calculate error metrics
mae = np.mean(np.abs(errors))
mse = np.mean(errors**2)
rmse = np.sqrt(mse)

print("Prediction Error Metrics:")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(y_test_original, label='Actual', color='blue')
plt.plot(y_pred_original, label='Predicted', color='red', linestyle='--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.legend()
plt.show()
y_test_original 
