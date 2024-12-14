import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Embedding, LSTM, Bidirectional



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
        y_window = y_scaled_test[i+window_size-1] #drops first two observations
        X_test_windowed.append(X_window)
        y_test_windowed.append(y_window)
    
    X_test_windowed = np.array(X_test_windowed)
    y_test_windowed = np.array(y_test_windowed)
    
    return X_windowed, y_windowed, X_test_windowed, y_test_windowed, feature_scaler, target_scaler




def train_LSTM(X_train, y_train,units,batch_size,epochs, verbose=2, learning_rate=0.001):
    #==Define model architecture
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
    early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',patience = 10)
    history = model.fit(X_trainn, y_trainn, epochs = epochs, 
                        validation_split = 0.2,
                        batch_size = batch_size, shuffle = False,
                        callbacks = [early_stop],verbose=0
                        )   
    
    modelN='LSTM'
    return(history,modelN,model)





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