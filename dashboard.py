# market_dashboard.py


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Market Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# HEADER
# -------------------------------
st.title("📈 Market Intelligence Dashboard (Offline Demo)")
st.markdown("A mock dashboard with sample data for stocks, news, and macro indicators — no API required.")
st.markdown("---")

# -------------------------------
# MOCK DATA FUNCTIONS
# -------------------------------
def generate_stock_data(symbol, days=30):
    base_price = random.uniform(1000, 3000)
    data = []
    for i in range(days):
        date = datetime.now() - timedelta(days=days - i)
        open_p = base_price + random.uniform(-50, 50)
        close_p = open_p + random.uniform(-30, 30)
        high_p = max(open_p, close_p) + random.uniform(5, 20)
        low_p = min(open_p, close_p) - random.uniform(5, 20)
        volume = random.randint(500000, 2000000)
        data.append({
            "date": date,
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2),
            "volume": volume
        })
    return pd.DataFrame(data)

def generate_news_data():
    sample_news = [
        ("Reliance Q2 results beat expectations", "positive"),
        ("TCS faces slowdown in US market", "negative"),
        ("HDFC Bank announces digital expansion", "positive"),
        ("Infosys launches new AI platform", "positive"),
        ("ICICI Bank reports stable growth", "neutral"),
        ("ITC increases dividend payout", "positive"),
        ("SBI under investigation for NPA issue", "negative"),
        ("Wipro collaborates with Google Cloud", "positive"),
    ]
    return pd.DataFrame(sample_news, columns=["Title", "Sentiment"])

def generate_macro_data():
    months = pd.date_range(end=datetime.today(), periods=24, freq='M')
    gdp = [7 + random.uniform(-0.5, 0.5) for _ in months]
    inflation = [5 + random.uniform(-0.7, 0.7) for _ in months]
    unemployment = [6 + random.uniform(-0.5, 0.5) for _ in months]

    return {
        "Real GDP (%)": pd.DataFrame({"Date": months, "Value": gdp}),
        "Inflation (%)": pd.DataFrame({"Date": months, "Value": inflation}),
        "Unemployment (%)": pd.DataFrame({"Date": months, "Value": unemployment}),
    }

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📰 News & Sentiment",
    "📊 Stock Prices",
    "🌍 Macro Indicators",
    "📈 Price Analysis"
])

# -------------------------------
# TAB 1: NEWS
# -------------------------------
with tab1:
    st.header("📰 Market News & Sentiment (Sample Data)")
    df_news = generate_news_data()

    # Sentiment metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Positive", len(df_news[df_news['Sentiment'] == 'positive']))
    col2.metric("⚪ Neutral", len(df_news[df_news['Sentiment'] == 'neutral']))
    col3.metric("🔴 Negative", len(df_news[df_news['Sentiment'] == 'negative']))

    st.markdown("---")

    # Chart
    sentiment_counts = df_news['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment",
                 color_discrete_map={"positive": "green", "neutral": "gray", "negative": "red"},
                 title="Sentiment Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    for _, row in df_news.iterrows():
        emoji = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}[row.Sentiment]
        st.markdown(f"**{emoji} {row.Title}**")

# -------------------------------
# TAB 2: STOCK PRICES
# -------------------------------
with tab2:
    st.header("📊 Stock Prices (Sample Data)")
    stocks = ["Reliance", "TCS", "HDFC Bank", "Infosys", "ICICI Bank", "ITC", "SBI", "Wipro"]
    selected_stock = st.selectbox("Select Stock", stocks)

    df_stock = generate_stock_data(selected_stock)

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Price", f"₹{df_stock['close'].iloc[-1]:.2f}")
    col2.metric("30-Day High", f"₹{df_stock['high'].max():.2f}")
    col3.metric("30-Day Low", f"₹{df_stock['low'].min():.2f}")

    fig = go.Figure(data=[go.Candlestick(
        x=df_stock['date'], open=df_stock['open'], high=df_stock['high'],
        low=df_stock['low'], close=df_stock['close']
    )])
    fig.update_layout(title=f"{selected_stock} - 30 Day Price Trend",
                      xaxis_title="Date", yaxis_title="Price (₹)", height=500)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 3: MACRO INDICATORS
# -------------------------------
with tab3:
    st.header("🌍 Macroeconomic Indicators (Sample Data)")
    macro_data = generate_macro_data()
    selected_indicator = st.selectbox("Select Indicator", list(macro_data.keys()))

    df_macro = macro_data[selected_indicator]
    latest_value = df_macro.iloc[-1]['Value']

    st.metric("Latest Value", f"{latest_value:.2f}")
    fig = px.line(df_macro, x='Date', y='Value', title=f"{selected_indicator} Trend", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 4: PRICE ANALYSIS
# -------------------------------
with tab4:
    st.header("📈 Compare Multiple Stocks")
    selected_stocks = st.multiselect("Select Stocks", stocks, default=stocks[:3])

    comparison = []
    for s in selected_stocks:
        df = generate_stock_data(s)
        change_pct = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
        comparison.append({
            "Stock": s,
            "Price": df['close'].iloc[-1],
            "30D Return %": round(change_pct, 2),
            "Volume": df['volume'].mean()
        })

    df_comp = pd.DataFrame(comparison)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    fig = px.bar(df_comp, x="Stock", y="30D Return %", color="30D Return %",
                 color_continuous_scale=["red", "yellow", "green"],
                 title="Stock Performance Comparison", text="30D Return %")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("📊 Demo Dashboard | Offline version without any APIs")
st.caption("Created by Market Alert System – Sample data generated locally")
