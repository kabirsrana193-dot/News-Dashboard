import streamlit as st
import torch
import feedparser
from transformers import pipeline
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(
    page_title="Nifty 200 News Dashboard",
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

# Alternative RSS feeds for financial news
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
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = "All Stocks"

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

def convert_to_ist(published_time):
    """Convert published time to IST"""
    try:
        if not published_time or published_time == "Unknown":
            return "Recent"
        
        pub_time = None
        if hasattr(published_time, 'tm_year'):
            pub_time = datetime(*published_time[:6])
        elif isinstance(published_time, str):
            # Try common date formats
            for fmt in ['%a, %d %b %Y %H:%M:%S %Z', '%a, %d %b %Y %H:%M:%S %z', '%Y-%m-%dT%H:%M:%S%z']:
                try:
                    pub_time = datetime.strptime(published_time, fmt)
                    break
                except:
                    continue
        
        if pub_time:
            # Convert to IST (UTC+5:30)
            if pub_time.tzinfo:
                pub_time = pub_time.replace(tzinfo=None)
            
            # Add 5 hours 30 minutes for IST
            ist_time = pub_time + timedelta(hours=5, minutes=30)
            
            # Format nicely
            now = datetime.now()
            time_diff = now - ist_time
            
            if time_diff.days == 0:
                if time_diff.seconds < 3600:
                    minutes = time_diff.seconds // 60
                    return f"{minutes} minutes ago"
                else:
                    hours = time_diff.seconds // 3600
                    return f"{hours} hours ago"
            elif time_diff.days == 1:
                return "Yesterday " + ist_time.strftime("%I:%M %p IST")
            else:
                return ist_time.strftime("%d %b %Y, %I:%M %p IST")
        
        return "Recent"
    except Exception as e:
        return "Recent"

def check_nifty_200_mention(text):
    """Check if text mentions any Nifty 200 stock"""
    text_upper = text.upper()
    for stock in NIFTY_200_STOCKS:
        if stock.upper() in text_upper:
            return True
    return False

def check_specific_stock_mention(text, stock_name):
    """Check if text mentions a specific stock"""
    text_upper = text.upper()
    return stock_name.upper() in text_upper

def get_mentioned_stocks(text):
    """Get list of stocks mentioned in the text"""
    text_upper = text.upper()
    mentioned = []
    for stock in NIFTY_200_STOCKS:
        if stock.upper() in text_upper:
            if stock not in mentioned:  # Avoid duplicates
                mentioned.append(stock)
    return mentioned if mentioned else ["Other"]  # Return "Other" if no stock foundd = []
    for stock in NIFTY_200_STOCKS:
        if stock.upper() in text_upper:
            mentioned.append(stock)
    return mentioned

def fetch_news(num_articles=15, specific_stock=None):
    """Fetch news articles mentioning Nifty 200 stocks from last 48 hours"""
    all_articles = []
    seen_titles = {article['Title'] for article in st.session_state.news_articles}
    
    # If specific stock is selected, prioritize it
    if specific_stock and specific_stock != "All Stocks":
        priority_stocks = [specific_stock] + [s for s in NIFTY_200_STOCKS[:30] if s != specific_stock]
        num_articles = num_articles * 2  # Fetch more articles for specific stock
    else:
        # Priority stocks for focused searching
        priority_stocks = NIFTY_200_STOCKS[:30]  # Top 30 stocks
    
    for stock in priority_stocks:
        try:
            # Search for each stock with date filter
            url = f"https://news.google.com/rss/search?q={stock}+stock+india+when:2d&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)
            
            # Get more articles for specific stock
            articles_per_stock = 5 if specific_stock == stock else 2
            
            for entry in feed.entries[:articles_per_stock]:
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

def process_news(articles):
    """Process news articles with sentiment analysis"""
    records = []
    
    for art in articles:
        title = art.title
        source = getattr(art, "source", {}).get("title", "Unknown") if hasattr(art, "source") else "Unknown"
        url = art.link
        
        # Get published time
        published = getattr(art, 'published', 'Unknown')
        
        # Get mentioned stocks
        mentioned_stocks = get_mentioned_stocks(title + " " + getattr(art, 'summary', ''))
        
        sentiment_result = finbert(title[:512])[0]
        sentiment = sentiment_result["label"].lower()
        score = sentiment_result["score"]
        
        records.append({
            "Title": title,
            "Source": source,
            "Sentiment": sentiment,
            "Score": round(score, 2),
            "Link": url,
            "Published": published,
            "Stocks": mentioned_stocks
        })
    
    return records

def filter_news_by_stock(news_articles, stock_name):
    """Filter news articles by specific stock"""
    if stock_name == "All Stocks":
        return news_articles
    
    filtered = []
    for article in news_articles:
        if stock_name in article.get('Stocks', []):
            filtered.append(article)
    
    return filtered

# --------------------------
# Streamlit App
# --------------------------
st.title("📈 Nifty 200 News Dashboard (Last 48 Hours)")
st.markdown("*Real-time news about Nifty 200 stocks with sentiment analysis*")
st.markdown(f"**Showing news from last 2 days** | **{len(NIFTY_200_STOCKS)} stocks tracked**")
st.markdown("---")

# Search/Filter and Refresh section
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    # Stock filter dropdown
    stock_options = ["All Stocks"] + sorted(NIFTY_200_STOCKS)
    selected_stock = st.selectbox(
        "🔍 Filter by Stock",
        options=stock_options,
        index=stock_options.index(st.session_state.selected_stock),
        key="stock_filter"
    )
    st.session_state.selected_stock = selected_stock

with col2:
    if st.button("🔄 Refresh News", type="primary", use_container_width=True):
        with st.spinner(f"Fetching latest updates for {st.session_state.selected_stock}..."):
            news_count = 0
            
            # Fetch news - pass selected stock for targeted fetching
            new_articles = fetch_news(ARTICLES_PER_REFRESH, st.session_state.selected_stock)
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
                st.session_state.news_articles = unique_articles[:100]  # Keep last 100
                news_count = len(processed_news)
            
            st.success(f"✅ Added {news_count} news articles for {st.session_state.selected_stock}!")
            st.rerun()

with col3:
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.news_articles = []
        st.success("✅ Cleared all news!")
        st.rerun()

# Load initial content if empty
if not st.session_state.news_articles:
    with st.spinner(f"Loading initial content for {st.session_state.selected_stock}..."):
        initial_news = fetch_news(ARTICLES_PER_REFRESH, st.session_state.selected_stock)
        
        if initial_news:
            st.session_state.news_articles = process_news(initial_news)

# Filter news based on selected stock
filtered_articles = filter_news_by_stock(st.session_state.news_articles, st.session_state.selected_stock)

if filtered_articles:
    df_all = pd.DataFrame(filtered_articles)
    
    # Display overall metrics
    st.subheader(f"📊 Metrics for {st.session_state.selected_stock}")
    col1, col2, col3, col4 = st.columns(4)
    
    total_items = len(df_all)
    positive_count = len(df_all[df_all['Sentiment'].str.lower() == 'positive'])
    neutral_count = len(df_all[df_all['Sentiment'].str.lower() == 'neutral'])
    negative_count = len(df_all[df_all['Sentiment'].str.lower() == 'negative'])
    
    with col1:
        st.metric("Total Articles", total_items)
    with col2:
        st.metric("🟢 Positive", positive_count)
    with col3:
        st.metric("⚪ Neutral", neutral_count)
    with col4:
        st.metric("🔴 Negative", negative_count)
    
    st.markdown("---")
    
    # Sentiment Chart
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
        title=f"Sentiment Analysis for {st.session_state.selected_stock} (Last 48 Hours)",
        text="Count"
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # News Articles
    st.subheader(f"📰 News Articles for {st.session_state.selected_stock}")
    
    for article in filtered_articles:
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
            
            # Show mentioned stocks
            if article.get('Stocks'):
                stocks_text = ", ".join(article['Stocks'][:5])  # Show first 5 stocks
                st.caption(f"📊 Stocks mentioned: {stocks_text}")
            
            st.caption(f"Source: {article['Source']} | {article.get('Published', 'Recent')}")
            st.markdown("---")

else:
    if st.session_state.selected_stock == "All Stocks":
        st.info("👆 Click 'Refresh News' to load content from the last 48 hours.")
    else:
        st.warning(f"No news found for {st.session_state.selected_stock}. Try refreshing or select 'All Stocks'.")

# Footer
st.markdown("---")
st.caption("💡 Dashboard shows news from last 48 hours for Nifty 200 stocks | Sentiment confidence scores show model certainty")
st.caption("🔍 Use the filter dropdown to search for specific stocks")
