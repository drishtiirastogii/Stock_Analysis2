import streamlit as st
import pandas as pd
from stock import fetch_stock_data, plot_comparison, plot_peaks
from stock_prediction_yahoo import train_lstm_model, predict_future_lstm, plot_forecast

def main():
    st.title("Stock Analysis")
    option = st.selectbox("Choose an option:", ["Compare two stocks", "See peaks for a single stock", "Predict future stock price", "Display all details for a stock"])
    duration = st.selectbox("Select the time duration for analysis:", ["6 months", "1 year", "2 years", "3 years"])

    if option == "Compare two stocks":
        stock1 = st.text_input("Enter the first stock symbol:")
        stock2 = st.text_input("Enter the second stock symbol:")
        if st.button("Compare"):
            try:
                df1, freq1 = fetch_stock_data(stock1, duration)
                df2, freq2 = fetch_stock_data(stock2, duration)
                plot_comparison(stock1, stock2, df1, df2)
            except KeyError as e:
                st.error(e)
    elif option == "See peaks for a single stock":
        stock = st.text_input("Enter the stock symbol:")
        if st.button("Show Peaks"):
            try:
                df, freq = fetch_stock_data(stock, duration)
                plot_peaks(stock, df)
            except KeyError as e:
                st.error(e)
    elif option == "Predict future stock price":
        stock = st.text_input("Enter the stock symbol:")
        if st.button("Predict"):
            try:
                df, freq = fetch_stock_data(stock, duration)
                lstm_model, scaler = train_lstm_model(df, freq)

                # Calculate steps based on the selected duration
                if duration == "6 months":
                    steps = 6 * 30
                elif duration == "1 year":
                    steps = 365
                elif duration == "2 years":
                    steps = 2 * 365
                elif duration == "3 years":
                    steps = 3 * 365

                lstm_forecast = predict_future_lstm(lstm_model, scaler, df, steps)

                st.write(f"Predicted stock price for the next {duration}:")
                future_dates = pd.date_range(start=df.index[-1], periods=steps + 1, freq='D')[1:]

                # Ensure the lengths match
                if len(future_dates) > len(lstm_forecast):
                    future_dates = future_dates[:len(lstm_forecast)]
                elif len(lstm_forecast) > len(future_dates):
                    lstm_forecast = lstm_forecast[:len(future_dates)]

                lstm_forecast = pd.Series(lstm_forecast.flatten(), index=future_dates.strftime('%m/%d/%Y'))
                st.write(lstm_forecast.to_frame(name='Predicted Price'))
                fig = plot_forecast(df, lstm_forecast, steps, freq)
                st.pyplot(fig)
            except KeyError as e:
                st.error(e)
    elif option == "Display all details for a stock":
        stock = st.text_input("Enter the stock symbol:")
        if st.button("Display"):
            try:
                df, freq = fetch_stock_data(stock, duration)
                df.index = df.index.strftime('%m/%d/%Y')  # Format the date index to show only the date
                st.write(f"Displaying all details for {stock} for the duration: {duration}")
                st.dataframe(df)
            except KeyError as e:
                st.error(e)

if __name__ == "__main__":
    main()