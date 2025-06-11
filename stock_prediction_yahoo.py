import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# Call this function before training your model
set_seeds()

def preprocess_data_lstm(df, frequency):
    # Fill in missing dates with the specified frequency
    df = df.asfreq(frequency)

    # Forward-fill and backward-fill missing values
    df['Close'] = df['Close'].ffill().bfill()

    # Replace infinite values with NaN and drop rows with NaN values
    df['Close'] = df['Close'].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Close'])

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    return scaled_data, scaler

def train_lstm_model(df, frequency, look_back=60, epochs=75, batch_size=32):
    scaled_data, scaler = preprocess_data_lstm(df, frequency)

    X_train, y_train = [], []
    for i in range(look_back, len(scaled_data)):
        X_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # # Get training predictions
    # train_predictions = model.predict(X_train)
    # train_predictions = scaler.inverse_transform(train_predictions)
    #
    # return model, scaler, train_predictions, y_train

    return model, scaler

def predict_future_lstm(model, scaler, df, steps, look_back=60):
    scaled_data, _ = preprocess_data_lstm(df, 'D')
    X_test = scaled_data[-look_back:]
    X_test = np.reshape(X_test, (1, X_test.shape[0], 1))

    predictions = []
    for _ in range(steps):
        pred = model.predict(X_test)
        predictions.append(pred[0, 0])
        X_test = np.append(X_test[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

def plot_forecast(df, forecast, steps, frequency):
    # Ensure the forecast index is a DatetimeIndex and naive (without timezone information)
    if not isinstance(forecast.index, pd.DatetimeIndex):
        forecast.index = pd.to_datetime(forecast.index, format='%m/%d/%Y', dayfirst=True)
    forecast.index = forecast.index.tz_localize(None)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index, df['Close'], label='Historical Close Price')
    ax.plot(forecast.index, forecast, label='Forecasted Close Price', linestyle='--')
    ax.set_title('Stock Price Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    return fig