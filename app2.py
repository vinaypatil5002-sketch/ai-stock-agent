import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Get Stock Data
# -----------------------------
def get_data(ticker):
    df = yf.download(ticker, period="2y")

    # Fix multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


# -----------------------------
# Add Features (MA + RSI)
# -----------------------------
def add_features(df):
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()

    # RSI calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df


# -----------------------------
# Train Model
# -----------------------------
def train_model(df):
    df = df.dropna()

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    X = df[['MA50', 'MA200', 'RSI']]
    y = df['Target']

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    return model, df


# -----------------------------
# Predict
# -----------------------------
def predict_stock(ticker):
    df = get_data(ticker)

    if df.empty:
        return None, None

    df = add_features(df)
    model, df = train_model(df)

    latest = df[['MA50', 'MA200', 'RSI']].iloc[-1:]
    prediction = model.predict(latest)[0]

    result = "BUY 📈" if prediction == 1 else "SELL 📉"

    return result, df


# -----------------------------
# Improved AI Explanation
# -----------------------------
def get_free_ai_advice(df, result):
    ma50 = df['MA50'].iloc[-1]
    ma200 = df['MA200'].iloc[-1]
    rsi = df['RSI'].iloc[-1]

    trend = "Bullish 📈" if ma50 > ma200 else "Bearish 📉"

    if rsi < 30:
        rsi_signal = "Oversold (Possible BUY opportunity)"
    elif rsi > 70:
        rsi_signal = "Overbought (Possible SELL pressure)"
    else:
        rsi_signal = "Neutral"

    return f"""
🤖 AI Analysis Report

📊 Trend: {trend}
- MA50: {ma50:.2f}
- MA200: {ma200:.2f}

📈 RSI: {rsi:.2f}
- Signal: {rsi_signal}

💡 Final Recommendation: {result}

⚠️ Note:
This prediction is based on technical indicators (Moving Averages + RSI).
Always consider market news and risks before investing.
"""


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 AI Stock Market Agent")

st.write("AI-powered stock analysis using Machine Learning + Technical Indicators 🤖")

stock = st.text_input("Enter Stock (e.g., TCS.NS, INFY.NS, RELIANCE.NS)")

if st.button("Get Recommendation"):
    if stock:
        result, df = predict_stock(stock)

        if df is None:
            st.error("Invalid stock or no data found ❌")
        else:
            st.success(f"Recommendation for {stock}: {result}")

            # AI Explanation
            advice = get_free_ai_advice(df, result)
            st.subheader("🤖 AI Analysis")
            st.write(advice)

            # RSI display
            st.write("📊 RSI Value:", round(df['RSI'].iloc[-1], 2))

            # Show data
            st.subheader("Recent Data")
            st.write(df.tail())

            # Chart
            st.subheader("📈 Stock Price Chart")
            st.line_chart(df[['Close']])

    else:
        st.warning("Please enter a stock name")