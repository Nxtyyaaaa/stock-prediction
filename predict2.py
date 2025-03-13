import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model('stock_model.h5')

# Stock ticker to predict (Change this to any stock symbol)
ticker = 'AAPL'

# Fetch latest stock data
df = yf.download(ticker, start='2023-01-01', end='2024-03-13')

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Use the last 'sequence length' days to predict the next day
seq_length = 50
input_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)

# Predict the next day's price
predicted_price = model.predict(input_sequence)
predicted_price = scaler.inverse_transform(predicted_price)

print(f"Predicted next day's price for {ticker}: ${predicted_price[0][0]:.2f}")
