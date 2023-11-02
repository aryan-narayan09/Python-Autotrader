import bot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

data = pd.read_csv('/content/Final ADAUSDT Data.csv')

# Use 'close' column as the target variable for time series forecasting
data = data[['close']]

# Normalize the data to scale values between 0 and 1
scaler = MinMaxScaler()
data['close'] = scaler.fit_transform(data[['close']])

# Convert the dataset into sequences for time series forecasting
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i + seq_length]
        sequences.append(sequence)
    return np.array(sequences)

# Define the sequence length and split the data into train and test sets
sequence_length = 15
sequences = create_sequences(data, sequence_length)
train_size = int(len(sequences) * 0.8)
train_data = sequences[:train_size]
test_data = sequences[train_size:]

# Split the data into features (X) and target (y)
X_train = train_data[:, :-1]
y_train = train_data[:, -1]
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# Build an LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Inverse transform the predictions to get actual values
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)


# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate RMSE (Root Mean Squared Error) to evaluate the model
rmse = sqrt(mean_squared_error(y_test, y_pred))

# Calculate R-squared (coefficient of determination)

r2 = r2_score(y_test, y_pred)


def feedData():

    # Prepare new data (one or more rows)
    new_data = pd.DataFrame({
        'open_time': [bot.open_time],
        'open': [bot.open],
        'high': [bot.high],
        'low': [bot.low],
        'close': [bot.close],
        'volume': [bot.volume],
        'close_time': [bot.close_time],
        'quote_volume': [bot.quote_volume],
        'count': [bot.count],
        'taker_buy_volume': [bot.taker_buy_volume],
        'taker_buy_quote_volume': [bot.taker_buy_quote_volume],
        'ignore': [bot.ignore]
    })

    # Normalize the new data using the same MinMaxScaler used for training
    scaler = MinMaxScaler()
    new_data['close'] = scaler.fit_transform(new_data[['close']])

    # Create sequences for prediction
    sequence_length = 15  # Same as the model's sequence length
    new_sequence = new_data[-sequence_length:].values  # Take the latest sequence

    # Reshape the new sequence to match the model's input shape
    new_sequence = new_sequence.reshape(1, sequence_length, -1)

    # Make a prediction for the next candle's close price
    predicted_normalized_close = model.predict(new_sequence)

    # Inverse transform the prediction to get the actual close price
    predicted_close = scaler.inverse_transform(predicted_normalized_close)

    # Generate buy/sell signal based on your trading strategy
    threshold = 0.01  # Define your threshold for buy/sell signals

    # Compare the predicted close price with the previous close price
    previous_close = new_data['close'].values[-1]
    ppc = ((predicted_close - previous_close) / previous_close)*100

    if ppc > threshold:
        signal = 1  # Buy signal
    else:
        signal = -1  # No action (optional)
