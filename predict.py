import numpy as np
from tensorflow.keras.models import load_model
from dataloader import fetch_data, preprocess_data

model = load_model('stock_model.h5')

ticker = 'AAPL'
df = fetch_data(ticker, '2015-01-01', '2024-01-01')
_, _, X_test, y_test, scaler, test_data = preprocess_data(df)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

print(f"Predicted next day's price: ${predictions[-1][0]:.2f}")
