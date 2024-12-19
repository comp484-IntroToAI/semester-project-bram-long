# Weather Temperature Prediction for Weather Trading

## Project Overview
This project leverages Artificial Intelligence and Machine Learning techniques to predict the maximum weather temperature for a given day in Miami, Florida. The goal is to aid in weather forecasting which has positive impacts on human livelihood, the economy, and businesses.

## Features
- Predicts maximum daily temperatures based on lagged values of historical weather data.
- Combines the strengths of traditional machine learning and deep learning models.
- Provides insights for weather-based trading strategies.

## Methods
We employ three robust models:
1. **Random Forest** - A ensemble-based method that averages decision tree predictions and captures non-linearity.
2. **LSTM (Long Short-Term Memory)** - A type of recurrent neural network which captures short-term and long-term patterns in time-series data.
3. **BiLSTM** - A type of LSTM that reads through the data forwards and backwards enabling it to more comprehensively understand the data.

## Installation
- Install Python 3.9
- Install Anaconda
- 


## Results
We then utilize an ensemble approach with all three models to combine our predictions with the OpenMeteo forecast and find improved forecasting results. We find that the random forest and LSTM trained on all four predictions outperforms the Open Meteo forecast. These results hold when altering the train-test size showing robust results.

## Sources
- Data Bank: Open Meteo API, Nation Weather Service
