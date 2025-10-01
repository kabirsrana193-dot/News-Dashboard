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

TWITTER_ACCOUNTS = {
    "cnbctv18news": "CNBC-TV18",
    "CNBC": "CNBC",
    "business": "Business News",
    "FirstQuake": "First Quake",
    "markets": "Markets"
}

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
        except Exception:
            continue
        
        if len(all_articles) >= num_articles:
            break
    
    return all_articles[:num_articles]

def fetch_tweets(num_tweets=10):
    """Fetch tweets from specified accounts"""
    all_tweets = []
    seen_titles = {tweet['Title'] for tweet in st.session_state.tweets}
    
    # Using Nitter instances for RSS (free Twitter access)
    nitter_instances = [
        "https://nitter.poast.org",
        "https://nitter.privacydev.net",
        "https://nitter.net"
    ]
    
    for account, display_name in TWITTER_ACCOUNTS.items():
        for nitter in nitter_instances:
            try:
                url = f"{nitter}/{account}/rss"
                feed = feedparser.parse(url)
                
                if not feed.entries:
                    continue
                
                for entry in feed.entries[:num_tweets]:
                    title = entry.title if hasattr(entry, 'title') else entry.summary[:100]
                    
                    if title in seen_titles:
                        continue
                    
                    all_tweets.append({
                        'entry': entry,
                        'account': display_name
                    })
                    seen_titles.add(title)
                    
                    if len(all_tweets) >= num_tweets:
                        break
                
                break  # Success, move to next account
                
            except Exception:
                continue
        
        if len(all_tweets) >= num_tweets:
            break
    
    return all_tweets[:num_tweets]

def process_news(articles):
    """Process news articles with sentiment analysis"""
    records = []
    
    for art in articles:
        title = art.title
        source = getattr(art, "source", {}).get("title", "Unknown") if hasattr(art, "source") else "Unknown"
        url = art.link
        
        sentiment = finbert(title[:512])[0]["label"]
        
        records.append({
            "Title": title,
            "Source": source,
            "Sentiment": sentiment,
            "Link": url,
            "Type": "News"
        })
    
    return records

def process_tweets(tweets):
    """Process tweets with sentiment analysis"""
    records = []
    
    for tweet_data in tweets:
        entry = tweet_data['entry']
        account = tweet_data['account']
        
        title = entry.title if hasattr(entry, 'title') else entry.summary[:100]
        url = entry.link
        
        sentiment = finbert(title[:512])[0]["label"]
        
        records.append({
            "Title": title,
            "Source": account,
            "Sentiment": sentiment,
            "Link": url,
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
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🔄 Refresh", type="primary"):
        with st.spinner("Fetching latest updates..."):
            # Fetch news
            new_articles = fetch_news(ARTICLES_PER_REFRESH)
            if new_articles:
                processed_news = process_news(new_articles)
                st.session_state.news_articles = processed_news + st.session_state.news_articles
            
            # Fetch tweets
            new_tweets = fetch_tweets(ARTICLES_PER_REFRESH)
            if new_tweets:
                processed_tweets = process_tweets(new_tweets)
                st.session_state.tweets = processed_tweets + st.session_state.tweets
            
            if new_articles or new_tweets:
                st.success(f"Added {len(new_articles) if new_articles else 0} news + {len(new_tweets) if new_tweets else 0} tweets!")
            else:
                st.info("No new updates found.")

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
    
    with col1:
        st.metric("Total Items", len(df_all))
    with col2:
        st.metric("News Articles", len(st.session_state.news_articles))
    with col3:
        st.metric("Tweets", len(st.session_state.tweets))
    with col4:
        positive_count = len(df_all[df_all['Sentiment'] == 'positive'])
        st.metric("Positive", positive_count)
    with col5:
        neutral_count = len(df_all[df_all['Sentiment'] == 'neutral'])
        st.metric("Neutral", neutral_count)
    with col6:
        negative_count = len(df_all[df_all['Sentiment'] == 'negative'])
        st.metric("Negative", negative_count)
    
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
        title="Sentiment Analysis of All Content"
    )
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
                    sentiment_emoji = {
                        "positive": "🟢",
                        "neutral": "⚪",
                        "negative": "🔴"
                    }
                    
                    st.markdown(f"{sentiment_emoji.get(article['Sentiment'], '⚪')} **[{article['Title']}]({article['Link']})**")
                    st.caption(f"Source: {article['Source']} | {article['Sentiment'].title()}")
                    st.markdown("---")
        else:
            st.info("No news articles yet. Click Refresh!")
    
    # Tweets Column
    with col_tweets:
        st.subheader("🐦 Latest Tweets")
        
        if st.session_state.tweets:
            for tweet in st.session_state.tweets:
                with st.container():
                    sentiment_emoji = {
                        "positive": "🟢",
                        "neutral": "⚪",
                        "negative": "🔴"
                    }
                    
                    st.markdown(f"{sentiment_emoji.get(tweet['Sentiment'], '⚪')} **[{tweet['Title']}]({tweet['Link']})**")
                    st.caption(f"@{tweet['Source']} | {tweet['Sentiment'].title()}")
                    st.markdown("---")
        else:
            st.info("No tweets yet. Click Refresh!")

else:
    st.info("Click 'Refresh' to load content.")

# Footer
st.markdown("---")
st.caption("💡 Dashboard updates with latest Nifty 200 stock news and financial tweets")
