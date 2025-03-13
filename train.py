import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from dataloader import fetch_data, preprocess_data
from model import build_model

ticker = 'AAPL'
df = fetch_data(ticker, '2015-01-01', '2024-01-01')

X_train, y_train, X_test, y_test, scaler, test_data = preprocess_data(df)

model = build_model(seq_length=50)
model.fit(X_train, y_train, batch_size=32, epochs=25)

model.save('stock_model.h5')
