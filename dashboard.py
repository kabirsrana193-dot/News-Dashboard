import streamlit as st
import feedparser
from transformers import pipeline
import pandas as pd
import plotly.express as px
import re

# Page config
st.set_page_config(
    page_title="Nifty 200 News & Twitter Dashboard",
    page_icon="📈",
    layout="wide"
)

# --------------------------
# Config
# --------------------------
NIFTY_200_STOCKS = [
    "Reliance", "TCS", "HDFC Bank", "Infosys", "ICICI Bank", "Bharti Airtel", "ITC",
    "State Bank", "SBI", "Hindustan Unilever", "HUL", "Bajaj Finance", "Kotak Mahindra",
    "LIC", "Axis Bank", "Larsen & Toubro", "L&T", "Asian Paints", "Maruti Suzuki",
    "Titan", "Sun Pharma", "HCL Tech", "Ultratechn Cement", "Nestle", "Adani",
    "Tata Motors", "Wipro", "Power Grid", "NTPC", "Bajaj Finserv", "Tata Steel",
    "Grasim", "Hindalco", "IndusInd Bank", "Mahindra", "M&M", "Coal India",
    "JSW Steel", "Tata Consumer", "Eicher Motors", "BPCL", "Tech Mahindra",
    "Dr Reddy", "Cipla", "UPL", "Shree Cement", "Havells", "Pidilite", "Britannia",
    "Divi's Lab", "SBI Life", "HDFC Life", "Berger Paints", "Bandhan Bank",
    "Adani Ports", "Adani Green", "Adani Total Gas", "Adani Power", "Adani Enterprises",
    "ONGC", "IOC", "Vedanta", "Godrej Consumer", "Bajaj Auto", "TVS Motor",
    "Hero MotoCorp", "Ashok Leyland", "Tata Power", "GAIL", "Ambuja Cement",
    "ACC", "UltraTech", "Shriram Finance", "SBI Cards", "Zomato", "Paytm",
    "Nykaa", "Policybazaar", "Trent", "Avenue Supermarts", "DMart", "Jubilant",
    "Page Industries", "MRF", "Apollo Hospitals", "Fortis Healthcare", "Max Healthcare",
    "Lupin", "Torrent Pharma", "Biocon", "Aurobindo Pharma", "Alkem Labs",
    "ICICI Lombard", "ICICI Prudential", "Bajaj Allianz", "PNB", "Bank of Baroda",
    "Canara Bank", "Union Bank", "Indian Bank", "IDFC First", "Federal Bank",
    "AU Small Finance", "RBL Bank", "Yes Bank", "DLF", "Prestige Estates",
    "Godrej Properties", "Oberoi Realty", "Phoenix Mills", "Brigade Enterprises",
    "InterGlobe Aviation", "IndiGo", "SpiceJet", "Zydus Lifesciences", "Mankind Pharma"
]

# Twitter accounts to fetch from
TWITTER_RSS_FEEDS = [
    ("https://rsshub.app/twitter/user/CNBCTV18News", "CNBC-TV18"),
    ("https://rsshub.app/twitter/user/business", "Bloomberg Business"),
    ("https://rsshub.app/twitter/user/markets", "Markets"),
    ("https://rsshub.app/twitter/user/RedboxGlobal", "Redbox Global"),
    ("https://rsshub.app/twitter/user/first_quake", "First Quake"),
]

ARTICLES_PER_REFRESH = 10

# --------------------------
# Initialize session state
# --------------------------
if 'news_articles' not in st.session_state:
    st.session_state.news_articles = []
if 'tweets' not in st.session_state:
    st.session_state.tweets = []

# --------------------------
# Cache FinBERT model
# --------------------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

finbert = load_model()

# --------------------------
# Functions
# --------------------------
def check_nifty_200_mention(text):
    """Check if text mentions any Nifty 200 stock"""
    text_upper = text.upper()
    for stock in NIFTY_200_STOCKS:
        if stock.upper() in text_upper:
            return True
    return False

def fetch_news(num_articles=10):
    """Fetch news articles mentioning Nifty 200 stocks"""
    all_articles = []
    seen_titles = {article['Title'] for article in st.session_state.news_articles}
    
    search_terms = [
        "NSE India", "BSE India", "Sensex", "Nifty", "stock market India",
        "Indian stocks", "Mumbai stock exchange", "earnings India",
        "quarterly results India", "Indian companies"
    ]
    
    for term in search_terms:
        try:
            url = f"https://news.google.com/rss/search?q={term}+when:1d&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)
            
            for entry in feed.entries:
                title = entry.title
                
                # Skip if already seen or doesn't mention Nifty 200 stocks
                if title in seen_titles or not check_nifty_200_mention(title):
                    continue
                
                all_articles.append(entry)
                seen_titles.add(title)
                
                if len(all_articles) >= num_articles:
                    break
        except Exception as e:
            continue
        
        if len(all_articles) >= num_articles:
            break
    
    return all_articles[:num_articles]

def fetch_tweets(num_tweets=10):
    """Fetch tweets from financial accounts using RSSHub"""
    all_tweets = []
    seen_content = {tweet['Title'] for tweet in st.session_state.tweets}
    
    for feed_url, account_name in TWITTER_RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            
            if not feed.entries:
                continue
            
            for entry in feed.entries:
                # Get tweet content
                if hasattr(entry, 'title'):
                    content = entry.title
                elif hasattr(entry, 'summary'):
                    content = entry.summary[:200]
                else:
                    continue
                
                # Skip if already seen
                if content in seen_content:
                    continue
                
                all_tweets.append({
                    'content': content,
                    'link': entry.link if hasattr(entry, 'link') else '#',
                    'account': account_name
                })
                seen_content.add(content)
                
                if len(all_tweets) >= num_tweets:
                    break
                    
        except Exception as e:
            st.sidebar.error(f"Error fetching from {account_name}: {str(e)}")
            continue
        
        if len(all_tweets) >= num_tweets:
            break
    
    return all_tweets

def process_news(articles):
    """Process news articles with sentiment analysis"""
    records = []
    
    for art in articles:
        title = art.title
        source = getattr(art, "source", {}).get("title", "Unknown") if hasattr(art, "source") else "Unknown"
        url = art.link
        
        sentiment_result = finbert(title[:512])[0]
        sentiment = sentiment_result["label"]
        score = sentiment_result["score"]
        
        records.append({
            "Title": title,
            "Source": source,
            "Sentiment": sentiment,
            "Score": round(score, 2),
            "Link": url,
            "Type": "News"
        })
    
    return records

def process_tweets(tweets):
    """Process tweets with sentiment analysis"""
    records = []
    
    for tweet_data in tweets:
        content = tweet_data['content']
        account = tweet_data['account']
        link = tweet_data['link']
        
        sentiment_result = finbert(content[:512])[0]
        sentiment = sentiment_result["label"]
        score = sentiment_result["score"]
        
        records.append({
            "Title": content,
            "Source": account,
            "Sentiment": sentiment,
            "Score": round(score, 2),
            "Link": link,
            "Type": "Tweet"
        })
    
    return records

# --------------------------
# Streamlit App
# --------------------------
st.title("📈 Nifty 200 News & Twitter Dashboard")
st.markdown("*Real-time news and tweets about Nifty 200 stocks with sentiment analysis*")
st.markdown("---")

# Refresh button
col1, col2, col3 = st.columns([1, 2, 3])
with col1:
    if st.button("🔄 Refresh", type="primary", use_container_width=True):
        with st.spinner("Fetching latest updates..."):
            news_count = 0
            tweet_count = 0
            
            # Fetch news
            new_articles = fetch_news(ARTICLES_PER_REFRESH)
            if new_articles:
                processed_news = process_news(new_articles)
                st.session_state.news_articles = processed_news + st.session_state.news_articles
                news_count = len(processed_news)
            
            # Fetch tweets
            new_tweets = fetch_tweets(ARTICLES_PER_REFRESH)
            if new_tweets:
                processed_tweets = process_tweets(new_tweets)
                st.session_state.tweets = processed_tweets + st.session_state.tweets
                tweet_count = len(processed_tweets)
            
            st.success(f"✅ Added {news_count} news + {tweet_count} tweets!")
            st.rerun()

# Load initial content if empty
if not st.session_state.news_articles and not st.session_state.tweets:
    with st.spinner("Loading initial content..."):
        initial_news = fetch_news(ARTICLES_PER_REFRESH)
        initial_tweets = fetch_tweets(ARTICLES_PER_REFRESH)
        
        if initial_news:
            st.session_state.news_articles = process_news(initial_news)
        if initial_tweets:
            st.session_state.tweets = process_tweets(initial_tweets)

# Combine all content for overall sentiment
all_content = st.session_state.news_articles + st.session_state.tweets

if all_content:
    df_all = pd.DataFrame(all_content)
    
    # Display overall metrics
    st.subheader("📊 Overall Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    total_items = len(df_all)
    news_count = len(st.session_state.news_articles)
    tweet_count = len(st.session_state.tweets)
    positive_count = len(df_all[df_all['Sentiment'] == 'positive'])
    neutral_count = len(df_all[df_all['Sentiment'] == 'neutral'])
    negative_count = len(df_all[df_all['Sentiment'] == 'negative'])
    
    with col1:
        st.metric("Total Items", total_items)
    with col2:
        st.metric("News Articles", news_count)
    with col3:
        st.metric("Tweets", tweet_count)
    with col4:
        st.metric("🟢 Positive", positive_count)
    with col5:
        st.metric("⚪ Neutral", neutral_count)
    with col6:
        st.metric("🔴 Negative", negative_count)
    
    st.markdown("---")
    
    # Overall Sentiment Chart
    st.subheader("📊 Overall Sentiment Distribution")
    sentiment_counts = df_all['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    
    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_discrete_map={
            "positive": "green",
            "neutral": "gray",
            "negative": "red"
        },
        title="Sentiment Analysis of All Content",
        text="Count"
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Two column layout for News and Tweets
    col_news, col_tweets = st.columns(2)
    
    # News Column
    with col_news:
        st.subheader("📰 News Articles (Nifty 200)")
        
        if st.session_state.news_articles:
            for article in st.session_state.news_articles:
                with st.container():
                    sentiment_color = {
                        "positive": "#28a745",
                        "neutral": "#6c757d",
                        "negative": "#dc3545"
                    }
                    
                    sentiment_emoji = {
                        "positive": "🟢",
                        "neutral": "⚪",
                        "negative": "🔴"
                    }
                    
                    st.markdown(f"**[{article['Title']}]({article['Link']})**")
                    
                    # Sentiment badge with confidence
                    sentiment_text = f"{sentiment_emoji[article['Sentiment']]} {article['Sentiment'].upper()} (confidence: {article['Score']})"
                    st.markdown(f"<span style='background-color: {sentiment_color[article['Sentiment']]}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;'>{sentiment_text}</span>", unsafe_allow_html=True)
                    
                    st.caption(f"Source: {article['Source']}")
                    st.markdown("---")
        else:
            st.info("No news articles yet. Click Refresh!")
    
    # Tweets Column
    with col_tweets:
        st.subheader("🐦 Latest Tweets")
        
        if st.session_state.tweets:
            for tweet in st.session_state.tweets:
                with st.container():
                    sentiment_color = {
                        "positive": "#28a745",
                        "neutral": "#6c757d",
                        "negative": "#dc3545"
                    }
                    
                    sentiment_emoji = {
                        "positive": "🟢",
                        "neutral": "⚪",
                        "negative": "🔴"
                    }
                    
                    st.markdown(f"**[{tweet['Title'][:150]}...]({tweet['Link']})**")
                    
                    # Sentiment badge with confidence
                    sentiment_text = f"{sentiment_emoji[tweet['Sentiment']]} {tweet['Sentiment'].upper()} (confidence: {tweet['Score']})"
                    st.markdown(f"<span style='background-color: {sentiment_color[tweet['Sentiment']]}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;'>{sentiment_text}</span>", unsafe_allow_html=True)
                    
                    st.caption(f"@{tweet['Source']}")
                    st.markdown("---")
        else:
            st.info("No tweets yet. Click Refresh! (Note: Twitter feeds may take time to load)")

else:
    st.info("👆 Click 'Refresh' to load content.")

# Footer
st.markdown("---")
st.caption("💡 Dashboard updates with latest Nifty 200 stock news and financial tweets | Sentiment confidence scores show model certainty")
