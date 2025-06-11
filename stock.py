import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import streamlit as st
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(symbol, duration):
    stock = yf.Ticker(symbol)
    end_date = datetime.today()

    if duration == "6 months":
        start_date = end_date - timedelta(days=6*30)
    elif duration == "1 year":
        start_date = end_date - timedelta(days=365)
    elif duration == "2 years":
        start_date = end_date - timedelta(days=2*365)
    elif duration == "3 years":
        start_date = end_date - timedelta(days=3*365)

    df = stock.history(start=start_date, end=end_date)
    df.index = pd.to_datetime(df.index)  # Ensure the index is in datetime format

    # Infer frequency from the date index
    if len(df.index) >= 3:
        frequency = pd.infer_freq(df.index)
    else:
        frequency = 'D'  # Default frequency (daily) or any other appropriate default

    return df, frequency

def plot_comparison(stock1, stock2, df1, df2):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df1['Close'], label=f'{stock1} Close Price')
    ax.plot(df2['Close'], label=f'{stock2} Close Price')
    ax.set_title(f'Comparison of {stock1} and {stock2} Stock Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    st.pyplot(fig)

def plot_peaks(symbol, df):
    close_prices = df['Close']
    peaks, _ = find_peaks(close_prices)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index, close_prices, label='Close Price')
    ax.plot(df.index[peaks], close_prices.iloc[peaks], 'x', label='Peaks')
    ax.set_title(f'{symbol} Stock Price and Peaks')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    st.pyplot(fig)
    st.write(f'Number of peaks for {symbol}: {len(peaks)}')