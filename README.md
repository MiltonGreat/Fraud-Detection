# Fraud-Detection
# Stock Price Prediction Using LSTM

## Overview

This project implements a Long Short-Term Memory (LSTM) neural network model to predict the stock price movement of PayPal (PYPL) using historical stock data. The model leverages the power of LSTMs, which are well-suited for time series forecasting tasks due to their ability to learn long-term dependencies in sequential data.

## Dataset Information
- The project uses the yfinance library to download historical stock price data for PayPal from January 1, 2015, to January 1, 2023.
  
### Key Features:
- Data Preprocessing: The stock prices are scaled using MinMaxScaler, and sequences are created for LSTM input.
- Model Training: An LSTM model is built and trained to predict future stock prices based on past price sequences.
- Evaluation Metrics: The model's performance is evaluated using Mean Absolute Error (MAE) and R-squared (R²).
- Backtesting Strategy: A simple backtesting strategy simulates trading based on the model's predictions and evaluates the final balance.

### Insuights from the Output

1. Mean Absolute Error (MAE):

The value of 6.77 indicates that the model's predictions are, on average, about $6.77 off from the actual closing prices. This shows a reasonable level of accuracy for a stock price prediction model.

2. R-squared (R²):

An R² value of approximately 0.97 suggests that the model explains about 97% of the variance in the stock prices. This high value indicates a strong fit and suggests that the model is capturing the underlying trends in the stock price movements effectively.

3. Backtesting Result:

The backtesting outcome indicates a loss of about $118.62 from an initial balance of $10,000. This suggests that while the model predicts stock prices well, the trading strategy based on those predictions may not be profitable. This could be due to the simplistic trading logic used in the backtesting phase, and it highlights the need for a more refined strategy.

### Future Work

- Implement advanced trading strategies and incorporate transaction costs.
- Explore feature engineering by adding more indicators (e.g., moving averages, RSI).
- Experiment with different LSTM architectures or other machine learning algorithms to improve predictions.
- Conduct further backtesting across different market conditions to evaluate the robustness of the trading strategy.
