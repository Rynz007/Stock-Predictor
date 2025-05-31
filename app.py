import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
import plotly.graph_objects as go
import datetime as dt

st.set_page_config(layout="wide", page_title="Advanced Stock Predictor")

# ---------------------
# Data download & indicators
# ---------------------
def download_data(tickers, start, end):
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end)
        data[ticker] = df[['Adj Close', 'Close', 'High', 'Low', 'Volume']].rename(columns={'Adj Close': ticker})
    combined = pd.concat(data.values(), axis=1, keys=data.keys())
    return combined.dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_technical_indicators(df, ticker):
    df = df.copy()
    df['MA_20'] = df[ticker]['Close'].rolling(window=20).mean()
    df['MA_50'] = df[ticker]['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df[ticker]['Close'])
    return df

def add_macd(df, ticker):
    exp1 = df[ticker]['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df[ticker]['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    return df

def add_bollinger_bands(df, ticker, window=20):
    sma = df[ticker]['Close'].rolling(window=window).mean()
    std = df[ticker]['Close'].rolling(window=window).std()
    df['BB_upper'] = sma + (std * 2)
    df['BB_lower'] = sma - (std * 2)
    return df

# ---------------------
# Dataset prep & model
# ---------------------
def create_dataset(data, ticker, window_size=60):
    features = data[ticker][['Close', 'Volume']]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------------
# Signals & backtesting
# ---------------------
def generate_signals(df):
    signals = []
    for i in range(1, len(df)):
        if df['MA_20'].iloc[i] > df['MA_50'].iloc[i] and df['MA_20'].iloc[i-1] <= df['MA_50'].iloc[i-1]:
            signals.append('Buy')
        elif df['MA_20'].iloc[i] < df['MA_50'].iloc[i] and df['MA_20'].iloc[i-1] >= df['MA_50'].iloc[i-1]:
            signals.append('Sell')
        else:
            signals.append('Hold')
    signals.insert(0, 'Hold')
    return signals

def backtest_strategy(df, ticker):
    df = df.copy()
    df['Returns'] = df[ticker]['Close'].pct_change()
    df['Strategy'] = df['Signal'].shift(1)
    df['Strategy_Returns'] = np.where(df['Strategy'] == 'Buy', df['Returns'], 0)
    df['Cumulative_Market'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
    
    trades = []
    position = None
    entry_price = 0
    for i in range(len(df)):
        if df['Signal'].iloc[i] == 'Buy' and position != 'long':
            position = 'long'
            entry_price = df[ticker]['Close'].iloc[i]
            trades.append({'Date': df.index[i], 'Type': 'Buy', 'Price': entry_price})
        elif df['Signal'].iloc[i] == 'Sell' and position == 'long':
            exit_price = df[ticker]['Close'].iloc[i]
            profit = exit_price - entry_price
            trades.append({'Date': df.index[i], 'Type': 'Sell', 'Price': exit_price, 'Profit': profit})
            position = None
    trades_df = pd.DataFrame(trades)
    return df, trades_df

# ---------------------
# Advanced chart with volume, MACD, Bollinger Bands, RSI, signals
# ---------------------
def enhanced_stock_chart_advanced(df, ticker, date_range=None, dark_mode=False):
    if date_range:
        df = df.loc[date_range[0]:date_range[1]]

    template = "plotly_dark" if dark_mode else "plotly_white"

    fig = go.Figure()

    # Price & Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[ticker]['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue'),
        yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_upper'],
        mode='lines',
        name='Bollinger Upper',
        line=dict(color='lightblue', dash='dot'),
        yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_lower'],
        mode='lines',
        name='Bollinger Lower',
        line=dict(color='lightblue', dash='dot'),
        yaxis='y1'
    ))

    # MAs
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA_20'],
        mode='lines',
        name='MA 20',
        line=dict(color='orange', dash='dash'),
        yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA_50'],
        mode='lines',
        name='MA 50',
        line=dict(color='green', dash='dash'),
        yaxis='y1'
    ))

    # Buy/Sell markers
    buy_signals = df[df['Signal'] == 'Buy']
    sell_signals = df[df['Signal'] == 'Sell']

    fig.add_trace(go.Scatter(
        x=buy_signals.index,
        y=buy_signals[ticker]['Close'],
        mode='markers',
        name='Buy Signal',
        marker=dict(symbol='triangle-up', size=12, color='green'),
        yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=sell_signals.index,
        y=sell_signals[ticker]['Close'],
        mode='markers',
        name='Sell Signal',
        marker=dict(symbol='triangle-down', size=12, color='red'),
        yaxis='y1'
    ))

    # RSI (secondary y)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='purple'),
        yaxis='y2'
    ))

    # MACD (tertiary y)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        mode='lines',
        name='MACD',
        line=dict(color='brown'),
        yaxis='y3'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD_Signal'],
        mode='lines',
        name='MACD Signal',
        line=dict(color='pink', dash='dash'),
        yaxis='y3'
    ))

    # Volume bars (fourth y)
    fig.add_trace(go.Bar(
        x=df.index,
        y=df[ticker]['Volume'],
        name='Volume',
        marker_color='lightgrey',
        yaxis='y4',
        opacity=0.5
    ))

    fig.update_layout(
        template=template,
        title=f"{ticker} Advanced Chart: Price, MA, BB, RSI, MACD & Volume",
        xaxis=dict(domain=[0, 1]),
        yaxis=dict(
            title='Price',
            domain=[0.4, 1],
            side='left',
            showgrid=True,
            zeroline=False
        ),
        yaxis2=dict(
            title='RSI',
            domain=[0.2, 0.35],
            side='right',
            showgrid=False,
            zeroline=False
        ),
        yaxis3=dict(
            title='MACD',
            domain=[0.05, 0.18],
            side='right',
            showgrid=False,
            zeroline=False
        ),
        yaxis4=dict(
            title='Volume',
            domain=[0, 0.04],
            showgrid=False,
            zeroline=False
        ),
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=40, r=40, t=60, b=40),
        height=900
    )
    return fig

# ---------------------
# Main app
# ---------------------

st.title("📈 Advanced Real-Time Stock Predictor with Backtesting & Trade Logs")

# Sidebar for settings
st.sidebar.header("Settings")
tickers = st.sidebar.multiselect("Select Tickers", ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'], default=['AAPL'])
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=dt.datetime.now())
dark_mode = st.sidebar.checkbox("Dark Mode", value=False)

if len(tickers) == 0:
    st.warning("Please select at least one ticker.")
    st.stop()

if start_date >= end_date:
    st.warning("Start date must be before end date.")
    st.stop()

if st.sidebar.button("Train & Predict"):

    with st.spinner("Downloading data and computing indicators..."):
        df = download_data(tickers, start_date, end_date)

        for ticker in tickers:
            df = add_technical_indicators(df, ticker)
            df = add_macd(df, ticker)
            df = add_bollinger_bands(df, ticker)
            df['Signal'] = generate_signals(df)

            # Prepare dataset
            X, y, scaler = create_dataset(df, ticker)
            X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = build_model((X_train.shape[1], X_train.shape[2]))
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

            y_pred = model.predict(X_test)

            # Backtest strategy and get trades
            backtest_df, trades_df = backtest_strategy(df, ticker)

            st.markdown(f"## Predictions & Analysis for {ticker}")

            # Date range slider for charts
            date_slider = st.slider(
                "Filter date range for charts:",
                min_value=df.index.min().date(),
                max_value=df.index.max().date(),
                value=(df.index.min().date(), df.index.max().date()),
                format="YYYY-MM-DD"
            )

            # Show advanced chart
            chart_fig = enhanced_stock_chart_advanced(backtest_df, ticker, date_range=date_slider, dark_mode=dark_mode)
            st.plotly_chart(chart_fig, use_container_width=True)

            # Backtest cumulative returns chart
            st.markdown("### Backtest: Strategy vs Market Returns")
            bt_fig = go.Figure()
            bt_fig.add_trace(go.Scatter(y=backtest_df['Cumulative_Market'], mode='lines', name='Market'))
            bt_fig.add_trace(go.Scatter(y=backtest_df['Cumulative_Strategy'], mode='lines', name='Strategy'))
            bt_fig.update_layout(title="Backtest Cumulative Returns", xaxis_title="Date", yaxis_title="Returns", template="plotly_dark" if dark_mode else "plotly_white")
            st.plotly_chart(bt_fig, use_container_width=True)

            # Latest signal
            st.markdown(f"### Latest Trading Signal: **{df['Signal'].iloc[-1]}**")

            # Show last 10 rows with indicators and signals
            st.dataframe(df[[ticker, 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal', 'BB_upper', 'BB_lower', 'Signal']].tail(10))

            # Show trades log
            st.markdown("### Trade Logs")
            if trades_df.empty:
                st.info("No trades executed in this period.")
            else:
                st.dataframe(trades_df)

                # Export trades as CSV
                csv = trades_df.to_csv(index=False).encode()
                st.download_button(
                    label="Export Trade Logs as CSV",
                    data=csv,
                    file_name=f"{ticker}_trade_logs.csv",
                    mime="text/csv",
                )
