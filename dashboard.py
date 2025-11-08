import streamlit as st
import feedparser
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import time
import yfinance as yf
import re

# Page config
st.set_page_config(
    page_title="Nifty 200 Dashboard",
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

# Stock ticker mapping
STOCK_TICKER_MAP = {
    "Reliance": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS", "ICICI Bank": "ICICIBANK.NS", "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS", "State Bank": "SBIN.NS", "SBI": "SBIN.NS",
    "Hindustan Unilever": "HINDUNILVR.NS", "HUL": "HINDUNILVR.NS",
    "Bajaj Finance": "BAJFINANCE.NS", "Kotak Mahindra": "KOTAKBANK.NS",
    "Axis Bank": "AXISBANK.NS", "Larsen & Toubro": "LT.NS", "L&T": "LT.NS",
    "Asian Paints": "ASIANPAINT.NS", "Maruti Suzuki": "MARUTI.NS",
    "Titan": "TITAN.NS", "Sun Pharma": "SUNPHARMA.NS", "HCL Tech": "HCLTECH.NS",
    "Nestle": "NESTLEIND.NS", "Adani": "ADANIENT.NS", "Tata Motors": "TATAMOTORS.NS",
    "Wipro": "WIPRO.NS", "Power Grid": "POWERGRID.NS", "NTPC": "NTPC.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS", "Tata Steel": "TATASTEEL.NS",
    "Grasim": "GRASIM.NS", "Hindalco": "HINDALCO.NS", "IndusInd Bank": "INDUSINDBK.NS",
    "Mahindra": "M&M.NS", "M&M": "M&M.NS", "Coal India": "COALINDIA.NS",
    "JSW Steel": "JSWSTEEL.NS", "Tata Consumer": "TATACONSUM.NS",
    "Eicher Motors": "EICHERMOT.NS", "BPCL": "BPCL.NS", "Tech Mahindra": "TECHM.NS",
    "Dr Reddy": "DRREDDY.NS", "Cipla": "CIPLA.NS", "UPL": "UPL.NS",
    "Shree Cement": "SHREECEM.NS", "Havells": "HAVELLS.NS", "Pidilite": "PIDILITIND.NS",
    "Britannia": "BRITANNIA.NS", "Divi's Lab": "DIVISLAB.NS", "ONGC": "ONGC.NS",
    "IOC": "IOC.NS", "Vedanta": "VEDL.NS", "Bajaj Auto": "BAJAJ-AUTO.NS"
}

FINANCIAL_RSS_FEEDS = [
    ("https://feeds.feedburner.com/ndtvprofit-latest", "NDTV Profit"),
    ("https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms", "ET Markets"),
    ("https://www.moneycontrol.com/rss/latestnews.xml", "Moneycontrol"),
]

ARTICLES_PER_REFRESH = 15
NEWS_AGE_LIMIT_HOURS = 48

# Sentiment keywords (simplified sentiment without heavy ML)
POSITIVE_WORDS = ['surge', 'rally', 'gain', 'profit', 'growth', 'high', 'rise', 'up', 'bullish', 
                  'strong', 'beats', 'outperform', 'success', 'jumps', 'soars', 'positive']
NEGATIVE_WORDS = ['fall', 'drop', 'loss', 'decline', 'weak', 'down', 'crash', 'bearish',
                  'concern', 'worry', 'risk', 'plunge', 'slump', 'miss', 'negative']

# --------------------------
# Initialize session state
# --------------------------
if 'news_articles' not in st.session_state:
    st.session_state.news_articles = []
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = "All Stocks"
if 'earnings_data' not in st.session_state:
    st.session_state.earnings_data = []
if 'last_earnings_fetch' not in st.session_state:
    st.session_state.last_earnings_fetch = None

# --------------------------
# Lightweight sentiment analysis
# --------------------------
def analyze_sentiment(text):
    """Simple keyword-based sentiment analysis"""
    text_lower = text.lower()
    positive_count = sum(1 for word in POSITIVE_WORDS if word in text_lower)
    negative_count = sum(1 for word in NEGATIVE_WORDS if word in text_lower)
    
    if positive_count > negative_count:
        sentiment = "positive"
        score = min(0.6 + (positive_count * 0.1), 0.95)
    elif negative_count > positive_count:
        sentiment = "negative"
        score = min(0.6 + (negative_count * 0.1), 0.95)
    else:
        sentiment = "neutral"
        score = 0.5
    
    return sentiment, round(score, 2)

# --------------------------
# NEWS Functions
# --------------------------
def is_recent(published_time, hours_limit=NEWS_AGE_LIMIT_HOURS):
    """Check if article is within the time limit"""
    try:
        if not published_time:
            return True
        
        pub_time = None
        if hasattr(published_time, 'tm_year'):
            pub_time = datetime(*published_time[:6])
        elif isinstance(published_time, str):
            for fmt in ['%a, %d %b %Y %H:%M:%S %Z', '%Y-%m-%dT%H:%M:%S%z']:
                try:
                    pub_time = datetime.strptime(published_time, fmt)
                    break
                except:
                    continue
        
        if pub_time:
            if pub_time.tzinfo:
                pub_time = pub_time.replace(tzinfo=None)
            cutoff_time = datetime.now() - timedelta(hours=hours_limit)
            return pub_time >= cutoff_time
        
        return True
    except:
        return True

def convert_to_ist(published_time):
    """Convert published time to IST"""
    try:
        if not published_time or published_time == "Unknown":
            return "Recent"
        
        pub_time = None
        if hasattr(published_time, 'tm_year'):
            pub_time = datetime(*published_time[:6])
        elif isinstance(published_time, str):
            for fmt in ['%a, %d %b %Y %H:%M:%S %Z', '%a, %d %b %Y %H:%M:%S %z', '%Y-%m-%dT%H:%M:%S%z']:
                try:
                    pub_time = datetime.strptime(published_time, fmt)
                    break
                except:
                    continue
        
        if pub_time:
            if pub_time.tzinfo:
                pub_time = pub_time.replace(tzinfo=None)
            ist_time = pub_time + timedelta(hours=5, minutes=30)
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
    except:
        return "Recent"

def check_nifty_200_mention(text):
    """Check if text mentions any Nifty 200 stock"""
    text_upper = text.upper()
    for stock in NIFTY_200_STOCKS:
        if stock.upper() in text_upper:
            return True
    return False

def get_mentioned_stocks(text):
    """Get list of stocks mentioned in the text"""
    text_upper = text.upper()
    mentioned = []
    for stock in NIFTY_200_STOCKS:
        if stock.upper() in text_upper:
            if stock not in mentioned:
                mentioned.append(stock)
    return mentioned if mentioned else ["Other"]

def fetch_news(num_articles=15, specific_stock=None):
    """Fetch news articles mentioning Nifty 200 stocks"""
    all_articles = []
    seen_titles = {article['Title'] for article in st.session_state.news_articles}
    
    if specific_stock and specific_stock != "All Stocks":
        priority_stocks = [specific_stock] + [s for s in NIFTY_200_STOCKS[:30] if s != specific_stock]
        num_articles = num_articles * 2
    else:
        priority_stocks = NIFTY_200_STOCKS[:30]
    
    for stock in priority_stocks:
        try:
            url = f"https://news.google.com/rss/search?q={stock}+stock+india+when:2d&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)
            articles_per_stock = 5 if specific_stock == stock else 2
            
            for entry in feed.entries[:articles_per_stock]:
                title = entry.title
                if title in seen_titles:
                    continue
                published = getattr(entry, 'published_parsed', None)
                if not is_recent(published):
                    continue
                all_articles.append(entry)
                seen_titles.add(title)
                if len(all_articles) >= num_articles:
                    break
        except:
            continue
        if len(all_articles) >= num_articles:
            break
    
    for feed_url, source_name in FINANCIAL_RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                title = entry.title if hasattr(entry, 'title') else ""
                if title in seen_titles:
                    continue
                full_text = title + " " + getattr(entry, 'summary', '')
                if not check_nifty_200_mention(full_text):
                    continue
                published = getattr(entry, 'published_parsed', None)
                if not is_recent(published):
                    continue
                all_articles.append(entry)
                seen_titles.add(title)
                if len(all_articles) >= num_articles:
                    break
        except:
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
        published = getattr(art, 'published', 'Unknown')
        mentioned_stocks = get_mentioned_stocks(title + " " + getattr(art, 'summary', ''))
        
        # Use lightweight sentiment analysis
        sentiment, score = analyze_sentiment(title)
        
        records.append({
            "Title": title,
            "Source": source,
            "Sentiment": sentiment,
            "Score": score,
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
# EARNINGS Functions
# --------------------------
@st.cache_data(ttl=3600)
def fetch_earnings_data(stocks_to_fetch=50):
    """Fetch earnings data for Nifty 200 stocks"""
    earnings_list = []
    
    progress_placeholder = st.empty()
    
    for idx, stock_name in enumerate(NIFTY_200_STOCKS[:stocks_to_fetch]):
        ticker = STOCK_TICKER_MAP.get(stock_name)
        if not ticker:
            continue
        
        try:
            progress_placeholder.text(f"Fetching {stock_name}... ({idx+1}/{stocks_to_fetch})")
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get earnings dates
            earnings_date = info.get('earningsDate', None)
            if earnings_date and isinstance(earnings_date, list):
                earnings_date = earnings_date[0] if len(earnings_date) > 0 else None
            
            next_earnings = 'Not Scheduled'
            if earnings_date:
                try:
                    if hasattr(earnings_date, 'strftime'):
                        next_earnings = earnings_date.strftime('%Y-%m-%d')
                    else:
                        next_earnings = str(earnings_date)
                except:
                    next_earnings = str(earnings_date)
            
            # Get quarterly financials
            try:
                quarterly = stock.quarterly_financials
                latest_quarter = 'N/A'
                revenue = 'N/A'
                
                if not quarterly.empty:
                    latest_quarter = quarterly.columns[0]
                    if hasattr(latest_quarter, 'date'):
                        latest_quarter = latest_quarter.date().strftime('%Y-%m-%d')
                    else:
                        latest_quarter = str(latest_quarter)
                    
                    if 'Total Revenue' in quarterly.index:
                        rev_value = quarterly.loc['Total Revenue'].iloc[0]
                        revenue = f"₹{rev_value/10000000:.2f} Cr"
            except:
                pass
            
            earnings_list.append({
                'Stock': stock_name,
                'Symbol': ticker.replace('.NS', ''),
                'Next Earnings': next_earnings,
                'Latest Quarter': latest_quarter,
                'Revenue': revenue,
                'EPS': f"₹{info.get('trailingEps', 'N/A')}" if info.get('trailingEps') else 'N/A',
                'PE Ratio': f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else 'N/A',
                'Market Cap': f"₹{info.get('marketCap', 0)/10000000:.2f} Cr" if info.get('marketCap') else 'N/A'
            })
            
            time.sleep(0.3)
            
        except Exception as e:
            continue
    
    progress_placeholder.empty()
    return earnings_list

# --------------------------
# Streamlit App
# --------------------------

# Main tabs
tab1, tab2 = st.tabs(["📰 News Dashboard", "📅 Earnings Calendar"])

# --------------------------
# TAB 1: NEWS DASHBOARD
# --------------------------
with tab1:
    st.title("📈 Nifty 200 News Dashboard (Last 48 Hours)")
    st.markdown("*Real-time news about Nifty 200 stocks with sentiment analysis*")
    st.markdown(f"**Showing news from last 2 days** | **{len(NIFTY_200_STOCKS)} stocks tracked**")
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
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
                new_articles = fetch_news(ARTICLES_PER_REFRESH, st.session_state.selected_stock)
                if new_articles:
                    processed_news = process_news(new_articles)
                    st.session_state.news_articles = processed_news + st.session_state.news_articles
                    seen = set()
                    unique_articles = []
                    for article in st.session_state.news_articles:
                        if article['Title'] not in seen:
                            unique_articles.append(article)
                            seen.add(article['Title'])
                    st.session_state.news_articles = unique_articles[:100]
                    news_count = len(processed_news)
                st.success(f"✅ Added {news_count} news articles!")
                st.rerun()

    with col3:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.news_articles = []
            st.success("✅ Cleared all news!")
            st.rerun()

    if not st.session_state.news_articles:
        with st.spinner(f"Loading initial content..."):
            initial_news = fetch_news(ARTICLES_PER_REFRESH, st.session_state.selected_stock)
            if initial_news:
                st.session_state.news_articles = process_news(initial_news)

    filtered_articles = filter_news_by_stock(st.session_state.news_articles, st.session_state.selected_stock)

    if filtered_articles:
        df_all = pd.DataFrame(filtered_articles)
        
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
            title=f"Sentiment Analysis for {st.session_state.selected_stock}",
            text="Count"
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
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
                sentiment_text = f"{sentiment_emoji[article['Sentiment']]} {article['Sentiment'].upper()} (confidence: {article['Score']})"
                st.markdown(f"<span style='background-color: {sentiment_color[article['Sentiment']]}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;'>{sentiment_text}</span>", unsafe_allow_html=True)
                
                if article.get('Stocks'):
                    stocks_text = ", ".join(article['Stocks'][:5])
                    st.caption(f"📊 Stocks mentioned: {stocks_text}")
                
                st.caption(f"Source: {article['Source']} | {article.get('Published', 'Recent')}")
                st.markdown("---")

    else:
        if st.session_state.selected_stock == "All Stocks":
            st.info("👆 Click 'Refresh News' to load content from the last 48 hours.")
        else:
            st.warning(f"No news found for {st.session_state.selected_stock}. Try refreshing or select 'All Stocks'.")

# --------------------------
# TAB 2: EARNINGS CALENDAR
# --------------------------
with tab2:
    st.title("📅 Earnings Calendar & Results")
    st.markdown("*Upcoming earnings dates and latest quarterly results for Nifty 200 stocks*")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        stocks_to_load = st.selectbox(
            "📊 Number of Stocks",
            options=[20, 50, 100],
            index=1,
            key="earnings_limit"
        )
    
    with col2:
        if st.button("🔄 Refresh Earnings", type="primary", use_container_width=True, key="refresh_earnings"):
            with st.spinner(f"Fetching earnings data for {stocks_to_load} stocks..."):
                st.cache_data.clear()
                earnings = fetch_earnings_data(stocks_to_load)
                st.session_state.earnings_data = earnings
                st.session_state.last_earnings_fetch = datetime.now()
                st.success(f"✅ Loaded earnings data for {len(earnings)} stocks!")
                st.rerun()
    
    with col3:
        if st.session_state.last_earnings_fetch:
            time_ago = datetime.now() - st.session_state.last_earnings_fetch
            minutes_ago = int(time_ago.total_seconds() / 60)
            st.info(f"⏱️ Updated {minutes_ago}m ago")
    
    if not st.session_state.earnings_data:
        with st.spinner(f"Loading earnings data..."):
            earnings = fetch_earnings_data(stocks_to_load)
            st.session_state.earnings_data = earnings
            st.session_state.last_earnings_fetch = datetime.now()
    
    if st.session_state.earnings_data:
        df_earnings = pd.DataFrame(st.session_state.earnings_data)
        
        st.subheader("📊 Earnings Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        scheduled = len(df_earnings[df_earnings['Next Earnings'] != 'Not Scheduled'])
        
        with col1:
            st.metric("Total Stocks", len(df_earnings))
        with col2:
            st.metric("Scheduled Earnings", scheduled)
        with col3:
            try:
                pe_values = []
                for val in df_earnings['PE Ratio']:
                    if val != 'N/A':
                        try:
                            pe_values.append(float(val))
                        except:
                            pass
                avg_pe = sum(pe_values) / len(pe_values) if pe_values else 0
                st.metric("Avg PE Ratio", f"{avg_pe:.2f}" if avg_pe > 0 else "N/A")
            except:
                st.metric("Avg PE Ratio", "N/A")
        with col4:
            st.metric("Data Points", len(df_earnings) * 8)
        
        st.markdown("---")
        
        search_earnings = st.text_input("🔍 Search by Stock Name or Symbol", "")
        
        if search_earnings:
            mask = df_earnings.apply(lambda row: row.astype(str).str.contains(search_earnings, case=False).any(), axis=1)
            filtered_earnings = df_earnings[mask]
        else:
            filtered_earnings = df_earnings
        
        st.info(f"Showing {len(filtered_earnings)} stocks")
        
        st.dataframe(
            filtered_earnings,
            use_container_width=True,
            height=600
        )
        
        csv_earnings = filtered_earnings.to_csv(index=False)
        st.download_button(
            label="📥 Download Earnings Data (CSV)",
            data=csv_earnings,
            file_name=f"nifty_earnings_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("👆 Click 'Refresh Earnings' to load earnings data.")

st.markdown("---")
st.caption("💡 Dashboard shows news from last 48 hours and earnings data for Nifty 200 stocks")
st.caption("🔍 Sentiment analysis uses keyword-based approach | Data refreshes hourly")
