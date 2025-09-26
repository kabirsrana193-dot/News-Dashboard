# dashboard.py
import datetime
import pandas as pd
from newsapi import NewsApiClient
from transformers import pipeline
import streamlit as st

# ------------------------
# 1. Setup NewsAPI client
# ------------------------
# Get a free API key at https://newsapi.org
newsapi = NewsApiClient(api_key='80414492c8f54330bf95a3e5de8d1034')

# Finance-related keywords
finance_keywords = [
    "stock market", "NSE", "BSE", "Sensex", "Nifty",
    "RBI", "banking", "earnings", "IPO", "finance",
    "quarterly results", "dividend", "merger", "acquisition",
    "sales", "revenue", "inflation", "interest rate"
]

# ------------------------
# 2. Load FinBERT sentiment model
# ------------------------
finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", tokenizer="yiyanghkust/finbert-tone")

def get_sentiment_finbert(text):
    try:
        result = finbert(text[:512])[0]  # truncate to 512 tokens
        return result['label']   # 'positive', 'negative', 'neutral'
    except Exception as e:
        return f"Error: {e}"

# ------------------------
# 3. Fetch news function
# ------------------------
def fetch_finance_news():
    all_articles = []
    query = " OR ".join(finance_keywords)  # combine keywords with OR
    articles = newsapi.get_everything(q=query, language='en', sort_by='publishedAt', page_size=40)
    
    if articles.get('articles'):
        all_articles.extend(articles['articles'])
    
    # Prepare DataFrame
    data = []
    for article in all_articles:
        title = article['title']
        source = article['source']['name'] if article.get('source') else "Unknown"
        url = article['url']
        sentiment = get_sentiment_finbert(title)
        data.append({"Title": title, "Source": source, "Sentiment": sentiment, "Link": url})
    
    df = pd.DataFrame(data)
    return df

# ------------------------
# 4. Streamlit app
# ------------------------
st.set_page_config(page_title="Finance News Dashboard", layout="wide")
st.title("📊 Finance News Dashboard")
st.write(f"Last updated: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

if st.button("Refresh News"):
    with st.spinner("Fetching latest news..."):
        df = fetch_finance_news()
        if not df.empty:
            st.dataframe(df)
        else:
            st.info("No news articles found.")
