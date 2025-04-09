import os
import time
from fsspec import Callback
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.models import load_model # type: ignore

class StreamlitCallback(Callback):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.logs.append(logs)
        self.placeholder.write(self.logs)

def intro(ticker):
    st.title("Stock Prediction Web App")
    #ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")

    if ticker:
        try:
            # Fetch Historical Data from Yahoo Finance
            start_date = "1990-01-01"
            end_date = "2025-02-28"
            data = yf.download(ticker, start=start_date, end=end_date)

            if data.empty:
                st.error("No data found for the given ticker. Please check the ticker symbol.")
            else:
                # Basic Stock Details
                st.write("Imported DataFrame:")
                
                stock_info = yf.Ticker(ticker).info
                st.subheader(f"Stock Details for {ticker}")
                if 'longName' in stock_info:
                    st.write(f"Company: {stock_info['longName']}")
                if 'sector' in stock_info:
                    st.write(f"Sector: {stock_info['sector']}")
                if 'industry' in stock_info:
                    st.write(f"Industry: {stock_info['industry']}")

                # Tabbed Visualization
                st.subheader("Historical Stock Data Visualization")
                tabs = st.tabs(["Close", "High", "Low", "Open", "Volume"])

                for i, col in enumerate(data.columns):
                    with tabs[i]:
                        fig = go.Figure(data=go.Scatter(x=data.index, y=data[col], mode='lines'))
                        fig.update_layout(title=f"{col} for {ticker}", xaxis_title="Date", yaxis_title=f"{col}")
                        st.plotly_chart(fig)
            st.dataframe(data)
        except Exception as e:
            st.error(f"An error occurred: {e}")

    else:
        st.write("Please enter a stock ticker to view data.")  

def plot_actual_vs_predicted(df, y_test_rescaled, predictions):
    # Ensure index is in datetime format
    df = df.sort_index()  # Ensure chronological order
    df.index = pd.to_datetime(df.index)

    # Split into training and testing data
    train_size = int(len(df) * 0.8)

    df_train = df.iloc[:train_size].copy()  # First 80% (training) with .copy()
    df_test = pd.DataFrame(y_test_rescaled, columns=df.columns).copy()  # Last 20% (actual test)
    df_pred = pd.DataFrame(predictions, columns=df.columns).copy()  # Last 20% (predictions)

    # Assign Date Index
    df_train["Date"] = df.index[:train_size]  # Training data dates
    df_test["Date"] = df.index[-len(y_test_rescaled):]  # Testing data dates
    df_pred["Date"] = df_test["Date"]  # Predictions match test data

    # Ensure Date is DateTime format
    df_train["Date"] = pd.to_datetime(df_train["Date"])
    df_test["Date"] = pd.to_datetime(df_test["Date"])
    df_pred["Date"] = pd.to_datetime(df_pred["Date"])

    # Group by Year
    df_train["Year"] = df_train["Date"].dt.year
    df_test["Year"] = df_test["Date"].dt.year
    df_pred["Year"] = df_pred["Date"].dt.year

    df_train_yearly = df_train.groupby("Year").mean()
    df_test_yearly = df_test.groupby("Year").mean()
    df_pred_yearly = df_pred.groupby("Year").mean()

    # Merge Train and Test for continuity
    df_actual = pd.concat([df_train_yearly, df_test_yearly])  # Combined actual data

    # Plotting using Plotly
    prediction_start_year = df_pred_yearly.index.min()

    st.subheader("Actual vs. Predicted Yearly Averages")
    tabs = st.tabs(df.columns.tolist()) #convert to list.

    for i, col in enumerate(df.columns):
        with tabs[i]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_actual.index, y=df_actual[col], mode='lines', name=f'Actual {col}'))
            fig.add_trace(go.Scatter(x=df_pred_yearly.index, y=df_pred_yearly[col], mode='lines', name=f'Predicted {col}'))
            fig.add_vline(x=prediction_start_year, line_width=1.5, line_dash="dash", line_color="white", annotation_text="Prediction Start")
            fig.update_layout(title=f'{col} Price Prediction (Yearly)', xaxis_title='Year', yaxis_title=f'{col} Price')
            st.plotly_chart(fig)

def load_and_preprocess_data(ticker):
    """Loads and preprocesses stock data."""
    ticker_yf = yf.Ticker(ticker)

    df = ticker_yf.history(start="1990-01-01", end="2025-02-28", auto_adjust=False)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df['Log_Volume'] = np.log(df['Volume'])
    df.drop(columns=['Volume'], inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    st.dataframe(df)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df[df.columns])
    scaled_df = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

    window_size = 60

    def create_sequence(data, window_size):
        X = []
        y = []
        for i in range(window_size, len(data)):
            X.append(data.iloc[i - window_size:i].values)
            y.append(data.iloc[i].values)
        return np.array(X), np.array(y)

    X, y = create_sequence(scaled_df, window_size)
    st.write(f"X shape: {X.shape}, y shape: {y.shape}")
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    st.write(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test, scaler, df

def open_trained_model(ticker):
    st.title("Open Trained Model (Long-Term)")
    st.write("This page will load and display predictions from a pre-trained model.")
    st.write(f"Ticker: {ticker}")

    X_train, X_test, y_train, y_test, scaler, df = load_and_preprocess_data(ticker)

    model_path = os.path.join("/Users/jeffrinmathew/Desktop/folder/models", f"{ticker}_lstm_model1.h5")
    model = load_model(model_path)
    st.write("Model loaded successfully!")
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test)
    loss, rmse = model.evaluate(X_test, y_test)
    st.write(f"Test Loss: {loss}")
    st.write(f"Test RMSE: {rmse}")
    mean_actual = np.mean(y_test)
    accuracy = 100 - (rmse / mean_actual * 100)
    st.write(f"Model Accuracy: {accuracy:.2f}%")

    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    test_indices = df.index[-len(y_test_rescaled):]
    plot_actual_vs_predicted(df,y_test_rescaled,predictions)


def train_model(ticker):
    st.title("Train Model (Long-Term)")
    st.write("This page will train a new model using historical data.")
    st.write(f"Ticker: {ticker}")

    X_train, X_test, y_train, y_test, scaler, df = load_and_preprocess_data(ticker)

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        keras.layers.LSTM(units=50, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(units=50, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(units=50, return_sequences=False),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(y_train.shape[1])
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['RootMeanSquaredError'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    log_placeholder = st.empty()
    streamlit_callback = StreamlitCallback(log_placeholder)

    history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=3, callbacks=[early_stopping, streamlit_callback], verbose=0)

    st.write("Final Training Log:")
    st.write(history.history)
    model_path = os.path.join("/Users/jeffrinmathew/Desktop/folder/models", f"{ticker}_lstm_model1.h5")
    model.save(model_path)
    st.write(f"Model saved at: {model_path}")

def long_term_prediction(ticker):
    st.title("Long-Term Prediction")
    choice = st.radio("Select Model Action:", ("Open Trained Model", "Train Model"))

    if choice == "Open Trained Model":
        open_trained_model(ticker)
    elif choice == "Train Model":
        train_model(ticker)

def plot_actual_vs_predicted_withnews(df, y_test_rescaled_prices, predictions_rescaled_prices):

    # Ensure index is in datetime format
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)

    # Split into training and testing data
    train_size = int(len(df) * 0.8)

    df_train = df.iloc[:train_size].copy()
    df_test = pd.DataFrame(y_test_rescaled_prices, columns=df.columns).copy()
    df_pred = pd.DataFrame(predictions_rescaled_prices, columns=df.columns).copy()

    # Assign Date Index
    df_train["Date"] = df.index[:train_size]
    df_test["Date"] = df.index[-len(y_test_rescaled_prices):]
    df_pred["Date"] = df_test["Date"]

    # Ensure Date is DateTime format
    df_train["Date"] = pd.to_datetime(df_train["Date"])
    df_test["Date"] = pd.to_datetime(df_test["Date"])
    df_pred["Date"] = pd.to_datetime(df_pred["Date"])

    # Group by Year
    df_train["Year"] = df_train["Date"].dt.year
    df_test["Year"] = df_test["Date"].dt.year
    df_pred["Year"] = df_pred["Date"].dt.year

    df_train_yearly = df_train.groupby("Year").mean()
    df_test_yearly = df_test.groupby("Year").mean()
    df_pred_yearly = df_pred.groupby("Year").mean()

    # Merge Train and Test for continuity
    df_actual = pd.concat([df_train_yearly, df_test_yearly])

    # Plot using Plotly
    prediction_start_year = df_pred_yearly.index.min()

    st.subheader("Actual vs. Predicted Yearly Averages (Interactive)")
    tabs = st.tabs(df.columns.tolist()) # Convert df.columns to list here.

    for i, col in enumerate(df.columns):
        with tabs[i]:
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=df_actual.index, y=df_actual[col], mode='lines', name=f'Actual {col}'))
            fig.add_trace(go.Scatter(x=df_pred_yearly.index, y=df_pred_yearly[col], mode='lines', name=f'Predicted {col}'))
            fig.add_vline(x=prediction_start_year, line_width=1.5, line_dash="dash", line_color="white", annotation_text="Prediction Start")

            fig.update_layout(title=f'{col} Price Prediction (Yearly)', xaxis_title='Year', yaxis_title=f'{col} Price')

            st.plotly_chart(fig)


def open_trained_model_with_news(ticker):
    st.title("Open Trained Model (With News)")
    st.write("This page will load and display predictions from a pre-trained model with news sentiment analysis.")
    ticker2 = ticker
    ticker = yf.Ticker(ticker)

    # Fetch historical data from 1990-01-01 to 2025-02-28
    df = ticker.history(start="1990-01-01", end="2025-02-28", auto_adjust=False)
    df.reset_index(inplace=True)

    # Convert 'Date' to a standard format (remove timezone)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    st.dataframe(df)

    df['Date'] = pd.to_datetime(df['Date'])
    # making the 'Date' col as index
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    # Apply log transformation to Volume to stabilize variance
    df['Log_Volume'] = np.log(df['Volume'])
    # deleting the original column of volume
    df.drop(columns=['Volume'], inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")  # Get the user's desktop path
    file_path = os.path.join(desktop_path, "folder", f"{ticker2}", f"{ticker2}stocknews.csv")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split("|") for line in f]  # Split by '|' instead of ','

    except FileNotFoundError:
        st.write(f"Error: File not found at {file_path}")
        return  # Exit if file not found
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return  # exit if other error occurs.

    cleaned_data = [[col.strip() for col in row if col.strip()] for row in lines]

    dfnews = pd.DataFrame(cleaned_data)
    st.dataframe(dfnews)
    if len(dfnews.columns) == 9:
        dfnews.drop(dfnews.columns[8], axis=1, inplace=True)
    dfnews.columns = ["datetime", "title", "source", "link", "negative_score", "neutral_score", "positive_score",
                      "compound_score"]
    dfnews = dfnews.drop([0, 1]).reset_index(drop=True)
    dfnews["datetime"] = pd.to_datetime(dfnews["datetime"]).dt.date
    dfnews["compound_score"] = pd.to_numeric(dfnews["compound_score"], errors='coerce')

    # Group sentiment data by date and calculate the average compound sentiment score per day
    sentiment_avg = dfnews.groupby('datetime')['compound_score'].mean().reset_index()
    sentiment_avg.rename(columns={'datetime': 'Date'}, inplace=True)

    df1 = df
    df1 = df1.reset_index()
    df = df1
    df.set_index('Date', inplace=True)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df[df.columns])
    scaled_df = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)
    scaled_df = scaled_df.reset_index()
    merged_df = pd.concat([scaled_df.set_index('Date'), sentiment_avg.set_index('Date')], axis=1)
    merged_df.dropna(inplace=True)
    merged_df = merged_df.reset_index()
    merged_df["Date"] = pd.to_datetime(merged_df["Date"])
    merged_df["sentiment_change"] = merged_df["compound_score"].diff()

    scaled_df1 = merged_df
    scaled_df1.set_index('Date', inplace=True)
    scaled_df1 = scaled_df1[scaled_df1.index >= "1990-01-01"]
    scaled_df1.drop("sentiment_change", axis=1, inplace=True)

    # train model
    window_size = 60

    def create_sequence(data, window_size):
        X = []
        y = []
        for i in range(window_size, len(data)):
            X.append(data.iloc[i - window_size:i].values)
            y.append(data.iloc[i].values)
        return np.array(X), np.array(y)

    X, y = create_sequence(scaled_df1, window_size)
    st.write(f"X shape: {X.shape}, y shape: {y.shape}")
    split_ratio = 0.8  # 80% train, 20% test

    split_index = int(len(X) * split_ratio)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    st.write(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    model_path = os.path.join("/Users/jeffrinmathew/Desktop/folder/models", f"{ticker2}_withnews_lstm_model1.h5")
    model = load_model(model_path)

    st.write("Model loaded successfully!")

    predictions = model.predict(X_test)
    scaled_columns = predictions.shape[1] - 1  # Exclude the last column if it's sentiment
    predictions_prices_scaled = predictions[:, :scaled_columns]
    predictions_rescaled_prices = scaler.inverse_transform(predictions_prices_scaled)
    y_test_prices_scaled = y_test[:, :scaled_columns]
    y_test_rescaled_prices = scaler.inverse_transform(y_test_prices_scaled)

    loss, rmse = model.evaluate(X_test, y_test)
    st.write(f"Test Loss: {loss}")
    st.write(f"Test RMSE: {rmse}")
    mean_actual = np.mean(y_test)

    # Calculate accuracy
    accuracy = 100 - (rmse / mean_actual * 100)
    st.write(f"Model Accuracy: {accuracy:.2f}%")
    plot_actual_vs_predicted_withnews(df, y_test_rescaled_prices, predictions_rescaled_prices)



#training model with news
def train_model_with_news(ticker):
    st.title("Train Model (With News)")
    st.write("This page will train a new model using historical data and news sentiment analysis.")
    ticker2=ticker
    # Define stock ticker
    ticker = yf.Ticker(ticker)

    # Fetch historical data from 1990-01-01 to 2025-02-28
    df = ticker.history(start="1990-01-01", end="2025-02-28", auto_adjust=False)
    df.reset_index(inplace=True)

    # Convert 'Date' to a standard format (remove timezone)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    st.dataframe(df)

    df['Date'] = pd.to_datetime(df['Date'])
    # making the 'Date' col as index
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    # Apply log transformation to Volume to stabilize variance
    df['Log_Volume'] = np.log(df['Volume'])
    # deleting the original column of volume
    df.drop(columns=['Volume'], inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")  # Get the user's desktop path
    file_path = os.path.join(desktop_path, "folder", f"{ticker2}", f"{ticker2}stocknews.csv")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split("|") for line in f]  # Split by '|' instead of ','

    except FileNotFoundError:
        st.write(f"Error: File not found at {file_path}")
        return  # Exit if file not found
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return #exit if other error occurs.

    cleaned_data = [[col.strip() for col in row if col.strip()] for row in lines]

    dfnews = pd.DataFrame(cleaned_data)
    st.dataframe(dfnews)

    if len(dfnews.columns) == 9:
        dfnews.drop(dfnews.columns[8], axis=1, inplace=True)

    dfnews.columns = ["datetime", "title", "source", "link", "negative_score", "neutral_score", "positive_score",
                      "compound_score"]
    dfnews = dfnews.drop([0, 1]).reset_index(drop=True)
    dfnews["datetime"] = pd.to_datetime(dfnews["datetime"]).dt.date
    dfnews["compound_score"] = pd.to_numeric(dfnews["compound_score"], errors='coerce')

    # Group sentiment data by date and calculate the average compound sentiment score per day
    sentiment_avg = dfnews.groupby('datetime')['compound_score'].mean().reset_index()
    sentiment_avg.rename(columns={'datetime': 'Date'}, inplace=True)

    df1 = df
    df1 = df1.reset_index()
    df = df1
    df.set_index('Date', inplace=True)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df[df.columns])
    scaled_df = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)
    scaled_df = scaled_df.reset_index()
    merged_df = pd.concat([scaled_df.set_index('Date'), sentiment_avg.set_index('Date')], axis=1)
    merged_df.dropna(inplace=True)
    merged_df = merged_df.reset_index()
    merged_df["Date"] = pd.to_datetime(merged_df["Date"])
    merged_df["sentiment_change"] = merged_df["compound_score"].diff()

    scaled_df1 = merged_df
    scaled_df1.set_index('Date', inplace=True)
    scaled_df1 = scaled_df1[scaled_df1.index >= "1990-01-01"]
    scaled_df1.drop("sentiment_change", axis=1, inplace=True)

    # train model
    window_size = 60

    def create_sequence(data, window_size):
        X = []
        y = []
        for i in range(window_size, len(data)):
            X.append(data.iloc[i - window_size:i].values)
            y.append(data.iloc[i].values)
        return np.array(X), np.array(y)

    X, y = create_sequence(scaled_df1, window_size)
    st.write(f"X shape: {X.shape}, y shape: {y.shape}")
    split_ratio = 0.8  # 80% train, 20% test

    split_index = int(len(X) * split_ratio)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    st.write(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    model = keras.Sequential([
        # Adding the first LSTM layer with Dropout
        keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        keras.layers.Dropout(0.3),

        # Adding the second LSTM layer with Dropout
        keras.layers.LSTM(units=50, return_sequences=True),
        keras.layers.Dropout(0.3),

        # Adding the third LSTM layer with Dropout
        keras.layers.LSTM(units=50, return_sequences=False),
        keras.layers.Dropout(0.3),

        # Adding a Dense output layer
        keras.layers.Dense(y_train.shape[1])
    ])
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['RootMeanSquaredError'])
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=15,
                                   restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        validation_split=0.2,
                        epochs=100,
                        batch_size=3,
                        callbacks=[early_stopping])

    model_path = os.path.join("/Users/jeffrinmathew/Desktop/folder/models", f"{ticker2}_withnews_lstm_model1.h5")
    model.save(model_path)
    st.write(f"Model saved at: {model_path}")
 
    # Add your code to train a new model with news data here.

def long_term_prediction_with_news(ticker):
    st.title("Long-Term Prediction (With News)")
    choice = st.radio("Select Model Action:", ("Open Trained Model", "Train Model"))

    if choice == "Open Trained Model":
        open_trained_model_with_news(ticker)
    elif choice == "Train Model":
        train_model_with_news(ticker)

#short-term prediction code



def get_stock_data(ticker, period="2d"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval="1m")
    data = data[data.index.date == pd.to_datetime("2025-04-07").date()]
    data = data.iloc[:, :-2]
    return data

def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    data['RSI'] = rsi
    return data

def calculate_zscore(data, window=20):
    mean = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()

    data['ZScore'] = (data['Close'] - mean) / std
    return data

def calculate_sma(data, periods=[5, 10]):
    for period in periods:
        data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
    return data

def generate_signals(data, rsi_low=30, rsi_high=70, zscore_low=-1, zscore_high=1):
    data['Signal_MA'] = 0
    data['Signal_RSI_Z'] = 0

    sma_crossover_buy = (data['SMA_5'] > data['SMA_10']) & (data['SMA_5'].shift(1) <= data['SMA_10'].shift(1))
    sma_crossover_sell = (data['SMA_5'] < data['SMA_10']) & (data['SMA_5'].shift(1) >= data['SMA_10'].shift(1))

    data.loc[sma_crossover_buy, 'Signal_MA'] = 1
    data.loc[sma_crossover_sell, 'Signal_MA'] = -1

    data.loc[(data['RSI'] < rsi_low) & (data['ZScore'] < zscore_low), 'Signal_RSI_Z'] = 1
    data.loc[(data['RSI'] > rsi_high) & (data['ZScore'] > zscore_high), 'Signal_RSI_Z'] = -1

    return data

def plot_signals_plotly(data):
    """Plots signals using Plotly for interactivity."""

    tab1, tab2 = st.tabs(["SMA Signals", "RSI & Z-score Signals"])

    # SMA Signals Plot
    with tab1:
        st.dataframe(data) # added dataframe display
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Stock Price', line=dict(color='blue')))
        fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA_5'], mode='lines', name='SMA (5)', line=dict(color='orange', dash='dash')))
        fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA_10'], mode='lines', name='SMA (10)', line=dict(color='purple', dash='dash')))

        buy_signals_ma = data[data['Signal_MA'] == 1]
        sell_signals_ma = data[data['Signal_MA'] == -1]

        fig_ma.add_trace(go.Scatter(x=buy_signals_ma.index, y=buy_signals_ma['Close'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Buy Signal (MA)'))
        fig_ma.add_trace(go.Scatter(x=sell_signals_ma.index, y=sell_signals_ma['Close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell Signal (MA)'))

        fig_ma.update_layout(title='Stock Price with SMA Buy/Sell Signals', xaxis_title='Time', yaxis_title='Price')
        st.plotly_chart(fig_ma)

    # RSI & Z-score Signals Plot
    with tab2:
        st.dataframe(data) # added dataframe display
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Stock Price', line=dict(color='blue')))

        buy_signals_rsi = data[data['Signal_RSI_Z'] == 1]
        sell_signals_rsi = data[data['Signal_RSI_Z'] == -1]

        fig_rsi.add_trace(go.Scatter(x=buy_signals_rsi.index, y=buy_signals_rsi['Close'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Buy Signal (RSI & Z-score)'))
        fig_rsi.add_trace(go.Scatter(x=sell_signals_rsi.index, y=sell_signals_rsi['Close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell Signal (RSI & Z-score)'))

        fig_rsi.update_layout(title='Stock Price with RSI & Z-score Buy/Sell Signals', xaxis_title='Time', yaxis_title='Price')
        st.plotly_chart(fig_rsi)

#Live stock data code


def get_live_stock_data(ticker, period="1d", interval="1m"):
    """Fetches live stock data."""
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return data

def plot_live_dashboard(ticker):
    """Plots a live dashboard with OHLCV data."""
    st.title(f"Live OHLCV Dashboard for {ticker}")
    placeholder = st.empty()  # Create a placeholder for live updates

    while True:
        live_data = get_live_stock_data(ticker)

        if not live_data.empty:
            fig = go.Figure(data=[go.Candlestick(x=live_data.index,
                                                 open=live_data['Open'],
                                                 high=live_data['High'],
                                                 low=live_data['Low'],
                                                 close=live_data['Close'])])

            fig.add_trace(go.Bar(x=live_data.index, y=live_data['Volume'], name='Volume', yaxis='y2'))

            fig.update_layout(
                yaxis2=dict(overlaying='y', side='right'),
                title=f"Live OHLCV for {ticker}",
                xaxis_title="Time",
                yaxis_title="Price",
                yaxis2_title="Volume",
                height=600,
                xaxis_rangeslider_visible=True,
            )

            with placeholder.container():
                st.plotly_chart(fig, use_container_width=True)

        time.sleep(60)  # Update every minute

def plot_live_signals(ticker):
    """Plots live buy/sell signals."""
    st.title(f"Live Buy/Sell Signals for {ticker}")
    placeholder_ma = st.empty()
    placeholder_rsi = st.empty()
    placeholder_data = st.empty()

    while True:
        live_data = get_live_stock_data(ticker)

        if not live_data.empty:
            live_data = calculate_rsi(live_data)
            live_data = calculate_zscore(live_data)
            live_data = calculate_sma(live_data)
            live_data = generate_signals(live_data)

            # SMA Signals Plot
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=live_data.index, y=live_data['Close'], mode='lines', name='Stock Price', line=dict(color='blue')))
            fig_ma.add_trace(go.Scatter(x=live_data.index, y=live_data['SMA_5'], mode='lines', name='SMA (5)', line=dict(color='orange', dash='dash')))
            fig_ma.add_trace(go.Scatter(x=live_data.index, y=live_data['SMA_10'], mode='lines', name='SMA (10)', line=dict(color='purple', dash='dash')))

            buy_signals_ma = live_data[live_data['Signal_MA'] == 1]
            sell_signals_ma = live_data[live_data['Signal_MA'] == -1]

            fig_ma.add_trace(go.Scatter(x=buy_signals_ma.index, y=buy_signals_ma['Close'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Buy Signal (MA)'))
            fig_ma.add_trace(go.Scatter(x=sell_signals_ma.index, y=sell_signals_ma['Close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell Signal (MA)'))

            fig_ma.update_layout(title='Live SMA Buy/Sell Signals', xaxis_title='Time', yaxis_title='Price')

            # RSI & Z-score Signals Plot
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=live_data.index, y=live_data['Close'], mode='lines', name='Stock Price', line=dict(color='blue')))

            buy_signals_rsi = live_data[live_data['Signal_RSI_Z'] == 1]
            sell_signals_rsi = live_data[live_data['Signal_RSI_Z'] == -1]

            fig_rsi.add_trace(go.Scatter(x=buy_signals_rsi.index, y=buy_signals_rsi['Close'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Buy Signal (RSI & Z-score)'))
            fig_rsi.add_trace(go.Scatter(x=sell_signals_rsi.index, y=sell_signals_rsi['Close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell Signal (RSI & Z-score)'))

            fig_rsi.update_layout(title='Live RSI & Z-score Buy/Sell Signals', xaxis_title='Time', yaxis_title='Price')

            with placeholder_ma.container():
                st.plotly_chart(fig_ma, use_container_width=True)

            with placeholder_rsi.container():
                st.plotly_chart(fig_rsi, use_container_width=True)

            with placeholder_data.container():
                st.dataframe(live_data)

        time.sleep(60)

# Main Streamlit app
def short_term_live(ticker):
    st.sidebar.title("Stock Dashboard")
    mode = st.sidebar.radio("Select Mode", ["Live OHLCV Candle-Stick Graph", "Live Signals"])

    if mode == "Live OHLCV Candle-Stick Graph":
        plot_live_dashboard(ticker)
    elif mode == "Live Signals":
        plot_live_signals(ticker)

def short_term_prediction(ticker):
    # (Your existing short_term_prediction function remains here)
    st.title("Stock Buy/Sell Signal Dashboard")

    if ticker:
        data = get_stock_data(ticker, period="2d")
        data = calculate_rsi(data)
        data = calculate_zscore(data)
        data = calculate_sma(data)
        data = generate_signals(data)
        plot_signals_plotly(data)

# Main Streamlit app
def short_term(ticker):
    st.sidebar.title("Stock Dashboard")
    ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL")
    mode = st.sidebar.radio("Select Mode", ["Short-Term Prediction", "Live OHLCV"])

    if mode == "Short-Term Prediction":
        short_term_prediction(ticker)
    elif mode == "Live OHLCV":
        short_term_live(ticker)

    

def main():
    st.title("Stock Prediction Web App")
    choice = st.radio("Select Page:", ("Introduction", "Long-Term Prediction", "Long-Term Prediction (With News)", "Short-Term Prediction"))
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):")

    if ticker:
        if choice == "Introduction":
            intro(ticker)
        elif choice == "Long-Term Prediction":
            long_term_prediction(ticker)
        elif choice == "Long-Term Prediction (With News)":
            long_term_prediction_with_news(ticker)
        elif choice == "Short-Term Prediction":
            short_term(ticker)
    else:
        st.write("Please enter a stock ticker to view data.")

if __name__ == "__main__":
    main()