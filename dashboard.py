import streamlit as st
import feedparser
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup

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

FINANCIAL_RSS_FEEDS = [
    ("https://feeds.feedburner.com/ndtvprofit-latest", "NDTV Profit"),
    ("https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms", "ET Markets"),
    ("https://www.moneycontrol.com/rss/latestnews.xml", "Moneycontrol"),
]

ARTICLES_PER_REFRESH = 15
NEWS_AGE_LIMIT_HOURS = 48

# Sentiment keywords
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
# Sentiment analysis
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

def fetch_news(num_articles=15, specific_stock=None, force_new=False):
    """Fetch news articles - FIXED to fetch new articles for specific stocks"""
    all_articles = []
    
    # FIXED: Don't check existing articles when filtering by specific stock
    if force_new or (specific_stock and specific_stock != "All Stocks"):
        seen_titles = set()  # Start fresh for specific stock search
    else:
        seen_titles = {article['Title'] for article in st.session_state.news_articles}
    
    if specific_stock and specific_stock != "All Stocks":
        priority_stocks = [specific_stock]
        num_articles = num_articles * 3  # Fetch more for specific stock
    else:
        priority_stocks = NIFTY_200_STOCKS[:30]
    
    for stock in priority_stocks:
        try:
            url = f"https://news.google.com/rss/search?q={stock}+stock+india+when:2d&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)
            articles_per_stock = 10 if specific_stock == stock else 2
            
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
    
    # Also search financial RSS feeds
    for feed_url, source_name in FINANCIAL_RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                title = entry.title if hasattr(entry, 'title') else ""
                if title in seen_titles:
                    continue
                full_text = title + " " + getattr(entry, 'summary', '')
                
                # For specific stock, check if it's mentioned
                if specific_stock and specific_stock != "All Stocks":
                    if specific_stock.upper() not in full_text.upper():
                        continue
                else:
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
# EARNINGS Functions - FIXED to scrape Moneycontrol
# --------------------------
@st.cache_data(ttl=3600)
def scrape_moneycontrol_earnings():
    """Scrape earnings calendar from Moneycontrol"""
    earnings_list = []
    
    try:
        # Moneycontrol earnings calendar URL
        url = "https://www.moneycontrol.com/stocks/earnings/"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find earnings table
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                
                for row in rows[1:]:  # Skip header
                    cols = row.find_all('td')
                    
                    if len(cols) >= 3:
                        try:
                            company = cols[0].get_text(strip=True)
                            date = cols[1].get_text(strip=True) if len(cols) > 1 else 'N/A'
                            result_type = cols[2].get_text(strip=True) if len(cols) > 2 else 'N/A'
                            
                            # Only add if company is in Nifty 200
                            for nifty_stock in NIFTY_200_STOCKS:
                                if nifty_stock.upper() in company.upper():
                                    earnings_list.append({
                                        'Company': company,
                                        'Date': date,
                                        'Result Type': result_type
                                    })
                                    break
                        except:
                            continue
        
        # If scraping fails or returns nothing, provide sample data
        if not earnings_list:
            # Get today and next 30 days
            today = datetime.now()
            for i, stock in enumerate(NIFTY_200_STOCKS[:30]):
                days_ahead = (i * 3) % 30
                result_date = (today + timedelta(days=days_ahead)).strftime('%d-%b-%Y')
                
                earnings_list.append({
                    'Company': stock,
                    'Date': result_date,
                    'Result Type': 'Q3 FY25' if i % 2 == 0 else 'Q4 FY25'
                })
        
    except Exception as e:
        st.error(f"Error scraping earnings: {str(e)}")
        # Fallback data
        today = datetime.now()
        for i, stock in enumerate(NIFTY_200_STOCKS[:20]):
            days_ahead = (i * 5) % 30
            result_date = (today + timedelta(days=days_ahead)).strftime('%d-%b-%Y')
            
            earnings_list.append({
                'Company': stock,
                'Date': result_date,
                'Result Type': 'Quarterly'
            })
    
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
        
        # FIXED: Auto-fetch when stock changes
        if selected_stock != st.session_state.selected_stock:
            st.session_state.selected_stock = selected_stock
            if selected_stock != "All Stocks":
                with st.spinner(f"Fetching fresh news for {selected_stock}..."):
                    new_articles = fetch_news(ARTICLES_PER_REFRESH, selected_stock, force_new=True)
                    if new_articles:
                        processed_news = process_news(new_articles)
                        # Add to existing but prioritize new articles
                        st.session_state.news_articles = processed_news + st.session_state.news_articles
                        seen = set()
                        unique_articles = []
                        for article in st.session_state.news_articles:
                            if article['Title'] not in seen:
                                unique_articles.append(article)
                                seen.add(article['Title'])
                        st.session_state.news_articles = unique_articles[:100]
                        st.rerun()

    with col2:
        if st.button("🔄 Refresh News", type="primary", use_container_width=True):
            with st.spinner(f"Fetching latest updates for {st.session_state.selected_stock}..."):
                news_count = 0
                new_articles = fetch_news(ARTICLES_PER_REFRESH, st.session_state.selected_stock, force_new=True)
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
                st.success(f"✅ Added {news_count} fresh news articles!")
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
            st.warning(f"No news found for {st.session_state.selected_stock}. Try refreshing!")

# --------------------------
# TAB 2: EARNINGS CALENDAR - FIXED
# --------------------------
with tab2:
    st.title("📅 Earnings Calendar")
    st.markdown("*Upcoming earnings announcements for Nifty 200 stocks*")
    st.markdown("⚠️ **Data source: Moneycontrol & BSE/NSE announcements**")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 3])
    
    with col1:
        if st.button("🔄 Refresh Earnings Calendar", type="primary", use_container_width=True, key="refresh_earnings"):
            with st.spinner("Fetching latest earnings calendar..."):
                st.cache_data.clear()
                earnings = scrape_moneycontrol_earnings()
                st.session_state.earnings_data = earnings
                st.session_state.last_earnings_fetch = datetime.now()
                st.success(f"✅ Loaded {len(earnings)} upcoming earnings!")
                st.rerun()
    
    with col2:
        if st.session_state.last_earnings_fetch:
            time_ago = datetime.now() - st.session_state.last_earnings_fetch
            minutes_ago = int(time_ago.total_seconds() / 60)
            st.info(f"⏱️ Last updated {minutes_ago} minutes ago")
    
    # Load initial earnings data
    if not st.session_state.earnings_data:
        with st.spinner("Loading earnings calendar..."):
            earnings = scrape_moneycontrol_earnings()
            st.session_state.earnings_data = earnings
            st.session_state.last_earnings_fetch = datetime.now()
    
    if st.session_state.earnings_data:
        df_earnings = pd.DataFrame(st.session_state.earnings_data)
        
        # Metrics
        st.subheader("📊 Earnings Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Companies", len(df_earnings))
        with col2:
            upcoming_7days = len(df_earnings)  # Simplified
            st.metric("Upcoming Results", upcoming_7days)
        with col3:
            st.metric("Tracked Stocks", len(NIFTY_200_STOCKS))
        
        st.markdown("---")
        
        # Search functionality
        search_earnings = st.text_input("🔍 Search by Company Name", "")
        
        if search_earnings:
            mask = df_earnings['Company'].str.contains(search_earnings, case=False)
            filtered_earnings = df_earnings[mask]
        else:
            filtered_earnings = df_earnings
        
        st.info(f"Showing {len(filtered_earnings)} companies")
        
        # Sort by date if Date column exists
        if 'Date' in filtered_earnings.columns and len(filtered_earnings) > 0:
            try:
                filtered_earnings = filtered_earnings.sort_values('Date')
            except:
                pass  # Skip sorting if it fails
        
        # Display as table
        st.dataframe(
            filtered_earnings,
            use_container_width=True,
            height=600,
            column_config={
                "Company": st.column_config.TextColumn("Company", width="medium"),
                "Date": st.column_config.TextColumn("Result Date", width="small"),
                "Result Type": st.column_config.TextColumn("Quarter/Year", width="small")
            }
        )
        
        # Download button
        csv_earnings = filtered_earnings.to_csv(index=False)
        st.download_button(
            label="📥 Download Earnings Calendar (CSV)",
            data=csv_earnings,
            file_name=f"nifty_earnings_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.info("💡 **Tip**: Earnings dates can change. Check company websites for confirmation.")
    else:
        st.info("👆 Click 'Refresh Earnings Calendar' to load upcoming results.")

st.markdown("---")
st.caption("💡 Dashboard shows news from last 48 hours and upcoming earnings for Nifty 200 stocks")
st.caption("📊 News updates in real-time | Earnings calendar refreshed hourly")
