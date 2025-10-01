import streamlit as st
import feedparser
from transformers import pipeline
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import time

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

# Alternative RSS feeds for financial news (Twitter/X RSS feeds are deprecated)
FINANCIAL_RSS_FEEDS = [
    ("https://feeds.feedburner.com/ndtvprofit-latest", "NDTV Profit"),
    ("https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms", "ET Markets"),
    ("https://www.moneycontrol.com/rss/latestnews.xml", "Moneycontrol"),
]

ARTICLES_PER_REFRESH = 15
NEWS_AGE_LIMIT_HOURS = 48  # 2 days

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
def is_recent(published_time, hours_limit=NEWS_AGE_LIMIT_HOURS):
    """Check if article is within the time limit"""
    try:
        if not published_time:
            return True  # Include if no timestamp available
        
        # Parse the published time
        pub_time = None
        if hasattr(published_time, 'tm_year'):
            pub_time = datetime(*published_time[:6])
        elif isinstance(published_time, str):
            # Try common date formats
            for fmt in ['%a, %d %b %Y %H:%M:%S %Z', '%Y-%m-%dT%H:%M:%S%z']:
                try:
                    pub_time = datetime.strptime(published_time, fmt)
                    break
                except:
                    continue
        
        if pub_time:
            # Make timezone-naive for comparison
            if pub_time.tzinfo:
                pub_time = pub_time.replace(tzinfo=None)
            
            cutoff_time = datetime.now() - timedelta(hours=hours_limit)
            return pub_time >= cutoff_time
        
        return True  # Include if we can't parse the date
    except Exception as e:
        return True  # Include if there's any error

def check_nifty_200_mention(text):
    """Check if text mentions any Nifty 200 stock"""
    text_upper = text.upper()
    for stock in NIFTY_200_STOCKS:
        if stock.upper() in text_upper:
            return True
    return False

def fetch_news(num_articles=15):
    """Fetch news articles mentioning Nifty 200 stocks from last 48 hours"""
    all_articles = []
    seen_titles = {article['Title'] for article in st.session_state.news_articles}
    
    # Priority stocks for focused searching
    priority_stocks = NIFTY_200_STOCKS[:30]  # Top 30 stocks
    
    for stock in priority_stocks:
        try:
            # Search for each stock with date filter
            url = f"https://news.google.com/rss/search?q={stock}+stock+india+when:2d&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:2]:  # Top 2 articles per stock
                title = entry.title
                
                # Skip if already seen
                if title in seen_titles:
                    continue
                
                # Check if recent (last 48 hours)
                published = getattr(entry, 'published_parsed', None)
                if not is_recent(published):
                    continue
                
                all_articles.append(entry)
                seen_titles.add(title)
                
                if len(all_articles) >= num_articles:
                    break
        except Exception as e:
            continue
        
        if len(all_articles) >= num_articles:
            break
    
    # Also fetch from financial RSS feeds
    for feed_url, source_name in FINANCIAL_RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:10]:
                title = entry.title if hasattr(entry, 'title') else ""
                
                # Skip if already seen
                if title in seen_titles:
                    continue
                
                # Check if mentions Nifty 200 stocks
                full_text = title + " " + getattr(entry, 'summary', '')
                if not check_nifty_200_mention(full_text):
                    continue
                
                # Check if recent
                published = getattr(entry, 'published_parsed', None)
                if not is_recent(published):
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
    """Fetch from financial news RSS feeds (Twitter RSS deprecated)"""
    all_items = []
    seen_content = {tweet['Title'] for tweet in st.session_state.tweets}
    
    # Use financial news feeds as alternative
    feed_urls = [
        ("https://www.business-standard.com/rss/markets-106.rss", "Business Standard"),
        ("https://feeds.feedburner.com/ndtvprofit-latest", "NDTV Profit"),
    ]
    
    for feed_url, source_name in feed_urls:
        try:
            feed = feedparser.parse(feed_url)
            
            if not feed.entries:
                continue
            
            for entry in feed.entries[:5]:
                # Get content
                if hasattr(entry, 'title'):
                    content = entry.title
                elif hasattr(entry, 'summary'):
                    content = entry.summary[:200]
                else:
                    continue
                
                # Skip if already seen
                if content in seen_content:
                    continue
                
                # Check if mentions Nifty 200 stocks
                if not check_nifty_200_mention(content):
                    continue
                
                # Check if recent
                published = getattr(entry, 'published_parsed', None)
                if not is_recent(published):
                    continue
                
                all_items.append({
                    'content': content,
                    'link': entry.link if hasattr(entry, 'link') else '#',
                    'account': source_name
                })
                seen_content.add(content)
                
                if len(all_items) >= num_tweets:
                    break
                    
        except Exception as e:
            continue
        
        if len(all_items) >= num_tweets:
            break
    
    return all_items

def process_news(articles):
    """Process news articles with sentiment analysis"""
    records = []
    
    for art in articles:
        title = art.title
        source = getattr(art, "source", {}).get("title", "Unknown") if hasattr(art, "source") else "Unknown"
        url = art.link
        
        # Get published time
        published = getattr(art, 'published', 'Unknown')
        
        sentiment_result = finbert(title[:512])[0]
        sentiment = sentiment_result["label"].lower()
        score = sentiment_result["score"]
        
        records.append({
            "Title": title,
            "Source": source,
            "Sentiment": sentiment,
            "Score": round(score, 2),
            "Link": url,
            "Type": "News",
            "Published": published
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
        sentiment = sentiment_result["label"].lower()
        score = sentiment_result["score"]
        
        records.append({
            "Title": content,
            "Source": account,
            "Sentiment": sentiment,
            "Score": round(score, 2),
            "Link": link,
            "Type": "News Feed",
            "Published": "Recent"
        })
    
    return records

# --------------------------
# Streamlit App
# --------------------------
st.title("📈 Nifty 200 News Dashboard (Last 48 Hours)")
st.markdown("*Real-time news about Nifty 200 stocks with sentiment analysis*")
st.markdown(f"**Showing news from last 2 days** | **{len(NIFTY_200_STOCKS)} stocks tracked**")
st.markdown("---")

# Refresh button
col1, col2, col3 = st.columns([1, 2, 3])
with col1:
    if st.button("🔄 Refresh", type="primary", use_container_width=True):
        with st.spinner("Fetching latest updates from last 48 hours..."):
            news_count = 0
            feed_count = 0
            
            # Fetch news
            new_articles = fetch_news(ARTICLES_PER_REFRESH)
            if new_articles:
                processed_news = process_news(new_articles)
                st.session_state.news_articles = processed_news + st.session_state.news_articles
                # Keep only unique articles
                seen = set()
                unique_articles = []
                for article in st.session_state.news_articles:
                    if article['Title'] not in seen:
                        unique_articles.append(article)
                        seen.add(article['Title'])
                st.session_state.news_articles = unique_articles[:50]  # Keep last 50
                news_count = len(processed_news)
            
            # Fetch from news feeds
            new_feeds = fetch_tweets(10)
            if new_feeds:
                processed_feeds = process_tweets(new_feeds)
                st.session_state.tweets = processed_feeds + st.session_state.tweets
                # Keep only unique items
                seen = set()
                unique_feeds = []
                for item in st.session_state.tweets:
                    if item['Title'] not in seen:
                        unique_feeds.append(item)
                        seen.add(item['Title'])
                st.session_state.tweets = unique_feeds[:50]  # Keep last 50
                feed_count = len(processed_feeds)
            
            st.success(f"✅ Added {news_count} news articles + {feed_count} feed items (last 48 hours)!")
            st.rerun()

# Load initial content if empty
if not st.session_state.news_articles and not st.session_state.tweets:
    with st.spinner("Loading initial content from last 48 hours..."):
        initial_news = fetch_news(ARTICLES_PER_REFRESH)
        initial_tweets = fetch_tweets(10)
        
        if initial_news:
            st.session_state.news_articles = process_news(initial_news)
        if initial_tweets:
            st.session_state.tweets = process_tweets(initial_tweets)

# Combine all content for overall sentiment
all_content = st.session_state.news_articles + st.session_state.tweets

if all_content:
    df_all = pd.DataFrame(all_content)
    
    # Display overall metrics
    st.subheader("📊 Overall Metrics (Last 48 Hours)")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    total_items = len(df_all)
    news_count = len(st.session_state.news_articles)
    feed_count = len(st.session_state.tweets)
    positive_count = len(df_all[df_all['Sentiment'].str.lower() == 'positive'])
    neutral_count = len(df_all[df_all['Sentiment'].str.lower() == 'neutral'])
    negative_count = len(df_all[df_all['Sentiment'].str.lower() == 'negative'])
    
    with col1:
        st.metric("Total Items", total_items)
    with col2:
        st.metric("News Articles", news_count)
    with col3:
        st.metric("News Feeds", feed_count)
    with col4:
        st.metric("🟢 Positive", positive_count)
    with col5:
        st.metric("⚪ Neutral", neutral_count)
    with col6:
        st.metric("🔴 Negative", negative_count)
    
    st.markdown("---")
    
    # Overall Sentiment Chart
    st.subheader("📊 Sentiment Distribution")
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
        title="Sentiment Analysis of All Content (Last 48 Hours)",
        text="Count"
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Two column layout for News and Feeds
    col_news, col_feeds = st.columns(2)
    
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
                    
                    st.caption(f"Source: {article['Source']} | {article.get('Published', 'Recent')}")
                    st.markdown("---")
        else:
            st.info("No news articles yet. Click Refresh!")
    
    # News Feeds Column
    with col_feeds:
        st.subheader("📡 Financial News Feeds")
        
        if st.session_state.tweets:
            for item in st.session_state.tweets:
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
                    
                    st.markdown(f"**[{item['Title'][:150]}...]({item['Link']})**")
                    
                    # Sentiment badge with confidence
                    sentiment_text = f"{sentiment_emoji[item['Sentiment']]} {item['Sentiment'].upper()} (confidence: {item['Score']})"
                    st.markdown(f"<span style='background-color: {sentiment_color[item['Sentiment']]}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;'>{sentiment_text}</span>", unsafe_allow_html=True)
                    
                    st.caption(f"Source: {item['Source']}")
                    st.markdown("---")
        else:
            st.info("No feed items yet. Click Refresh!")

else:
    st.info("👆 Click 'Refresh' to load content from the last 48 hours.")

# Footer
st.markdown("---")
st.caption("💡 Dashboard shows news from last 48 hours for Nifty 200 stocks | Sentiment confidence scores show model certainty")
st.caption("⚠️ Note: X/Twitter RSS feeds are no longer available. Using alternative financial news sources.")
