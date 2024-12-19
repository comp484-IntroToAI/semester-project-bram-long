# Weather Temperature Prediction
### By Bram Nutt and Long Truong
## Project Overview
This project leverages Artificial Intelligence and Machine Learning techniques to predict the maximum weather temperature for a given day in Miami, Florida. The goal is to aid in weather forecasting which has positive impacts on human livelihood, the economy, and businesses.

## Features
- Predicts maximum daily temperatures based on lagged values of historical weather data.
- Combines the strengths of traditional machine learning and deep learning models.
- Provides insights for weather-based trading strategies.

## Methods
We employ three models:
1. **Random Forest** - A ensemble-based method that averages decision tree predictions and captures non-linearity.
2. **LSTM (Long Short-Term Memory)** - A type of recurrent neural network which captures short-term and long-term patterns in time-series data.
3. **BiLSTM** - A type of LSTM that reads through the data forwards and backwards enabling it to more comprehensively understand the data.

## Installation
- Install Python 3.9
- Install Anaconda
- Ensure all the required packages listed in the `requirements.txt` are installed by using `conda install `


## Results
We utilize an ensemble approach -predicting off our models and OpenMeteo's forecasts- and find improved forecasting results. We find that the ensemble LSTM and random forest trained on all four predictions outperforms the Open Meteo forecast. These results hold when altering the train-test size showing robust results.

## Sources
- Data Sources: Open Meteo API, Nation Weather Service
