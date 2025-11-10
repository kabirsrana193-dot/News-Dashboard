import streamlit as st
import feedparser
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import numpy as np

# Page config
st.set_page_config(
    page_title="F&O Stocks Dashboard",
    page_icon="📈",
    layout="wide"
)

# --------------------------
# F&O STOCKS LIST (185+ stocks with F&O available)
# --------------------------
FNO_STOCKS = [
    "Reliance", "TCS", "HDFC Bank", "Infosys", "ICICI Bank", "Bharti Airtel", "ITC",
    "SBI", "Hindustan Unilever", "Bajaj Finance", "Kotak Bank", "LIC",
    "Axis Bank", "Larsen & Toubro", "Asian Paints", "Maruti Suzuki", "Titan",
    "Sun Pharma", "HCL Tech", "Ultratechn Cement", "Nestle", "Adani Enterprises",
    "Tata Motors", "Wipro", "Power Grid", "NTPC", "Bajaj Finserv", "Tata Steel",
    "Grasim", "Hindalco", "IndusInd Bank", "Mahindra", "Coal India", "JSW Steel",
    "Tata Consumer", "Eicher Motors", "BPCL", "Tech Mahindra", "Dr Reddy", "Cipla",
    "UPL", "Shree Cement", "Havells", "Pidilite", "Britannia", "Divi's Lab",
    "SBI Life", "HDFC Life", "Berger Paints", "Bandhan Bank", "Adani Ports",
    "Adani Green", "Adani Total Gas", "Adani Power", "ONGC", "IOC", "Vedanta",
    "Godrej Consumer", "Bajaj Auto", "TVS Motor", "Hero MotoCorp", "Ashok Leyland",
    "Tata Power", "GAIL", "Ambuja Cement", "ACC", "UltraTech", "Shriram Finance",
    "SBI Cards", "Zomato", "Paytm", "Nykaa", "Policybazaar", "Trent",
    "Avenue Supermarts", "DMart", "Page Industries", "MRF", "Apollo Hospitals",
    "Lupin", "Torrent Pharma", "Biocon", "Aurobindo Pharma", "Alkem Labs",
    "ICICI Lombard", "ICICI Prudential", "PNB", "Bank of Baroda", "Canara Bank",
    "Union Bank", "Indian Bank", "IDFC First", "Federal Bank", "AU Small Finance",
    "RBL Bank", "Yes Bank", "DLF", "Prestige Estates", "Godrej Properties",
    "Oberoi Realty", "Phoenix Mills", "IndiGo", "Zydus Lifesciences", "Mankind Pharma",
    "Adani Wilmar", "Jio Financial", "Tata Elxsi", "Persistent Systems", "LTIMindtree",
    "Mphasis", "Coforge", "L&T Technology", "Voltas", "Crompton Greaves",
    "Dixon Technologies", "PB Fintech", "CMS Info", "Jio", "Adani Energy",
    "ABB India", "Siemens", "Bosch", "Motherson Sumi", "Bharat Electronics",
    "HAL", "BHEL", "Cochin Shipyard", "Mazagon Dock", "Garden Reach Shipbuilders",
    "Rail Vikas Nigam", "IRFC", "IRCTC", "Container Corp", "Concor",
    "Blue Dart", "VRL Logistics", "Aegis Logistics", "Gujarat Gas", "IGL",
    "Mahanagar Gas", "Petronet LNG", "PI Industries", "SRF", "Aarti Industries",
    "Deepak Nitrite", "Gujarat Fluorochem", "Vinati Organics", "Balrampur Chini",
    "Dhampur Sugar", "Dwarikesh Sugar", "Triveni Engineering", "EID Parry",
    "Shree Renuka Sugars", "Bajaj Holdings", "M&M Financial", "Cholamandalam",
    "Muthoot Finance", "Manappuram Finance", "IIFL Finance", "Aavas Financiers",
    "Home First Finance", "Can Fin Homes", "Repco Home", "Gruh Finance",
    "PNB Housing", "LIC Housing", "HUDCO", "NHB", "Power Finance Corp",
    "REC Limited", "Indian Energy Exchange", "CESC", "Torrent Power", "JSW Energy",
    "Adani Transmission", "Adani Total", "SJVN", "NHPC", "NLC India",
    "Apar Industries", "Polycab", "KEI Industries", "RR Kabel", "Finolex Cables",
    "Timken India", "SKF India", "Schaeffler India", "NRB Bearings", "Bharat Forge",
    "Tube Investments", "Escorts Kubota", "Mahindra CIE", "Sona BLW", "Samvardhana Motherson",
    "Sundaram Clayton", "KPIT Technologies", "Tata Communications", "Route Mobile",
    "Tanla Platforms", "HFCL", "Sterlite Tech", "Tejas Networks", "Bharti Hexacom"
]

# Stock ticker mapping for F&O stocks
STOCK_TICKER_MAP = {
    "Reliance": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS", "ICICI Bank": "ICICIBANK.NS", "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS", "SBI": "SBIN.NS", "Hindustan Unilever": "HINDUNILVR.NS",
    "Bajaj Finance": "BAJFINANCE.NS", "Kotak Bank": "KOTAKBANK.NS", "LIC": "LICI.NS",
    "Axis Bank": "AXISBANK.NS", "Larsen & Toubro": "LT.NS", "Asian Paints": "ASIANPAINT.NS",
    "Maruti Suzuki": "MARUTI.NS", "Titan": "TITAN.NS", "Sun Pharma": "SUNPHARMA.NS",
    "HCL Tech": "HCLTECH.NS", "Nestle": "NESTLEIND.NS", "Adani Enterprises": "ADANIENT.NS",
    "Tata Motors": "TATAMOTORS.NS", "Wipro": "WIPRO.NS", "Power Grid": "POWERGRID.NS",
    "NTPC": "NTPC.NS", "Bajaj Finserv": "BAJAJFINSV.NS", "Tata Steel": "TATASTEEL.NS",
    "Grasim": "GRASIM.NS", "Hindalco": "HINDALCO.NS", "IndusInd Bank": "INDUSINDBK.NS",
    "Mahindra": "M&M.NS", "Coal India": "COALINDIA.NS", "JSW Steel": "JSWSTEEL.NS",
    "Tata Consumer": "TATACONSUM.NS", "Eicher Motors": "EICHERMOT.NS", "BPCL": "BPCL.NS",
    "Tech Mahindra": "TECHM.NS", "Dr Reddy": "DRREDDY.NS", "Cipla": "CIPLA.NS",
    "UPL": "UPL.NS", "Shree Cement": "SHREECEM.NS", "Havells": "HAVELLS.NS",
    "Pidilite": "PIDILITIND.NS", "Britannia": "BRITANNIA.NS", "Divi's Lab": "DIVISLAB.NS",
    "ONGC": "ONGC.NS", "IOC": "IOC.NS", "Vedanta": "VEDL.NS", "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Zomato": "ZOMATO.NS", "DMart": "DMART.NS", "Jio Financial": "JIOFIN.NS",
    "Adani Ports": "ADANIPORTS.NS", "Tata Power": "TATAPOWER.NS", "DLF": "DLF.NS",
    "SBI Cards": "SBICARD.NS", "Bank of Baroda": "BANKBARODA.NS", "PNB": "PNB.NS",
    "Canara Bank": "CANBK.NS", "LTIMindtree": "LTIM.NS", "Persistent Systems": "PERSISTENT.NS",
    "Dixon Technologies": "DIXON.NS", "Bosch": "BOSCHLTD.NS", "ABB India": "ABB.NS",
    "Siemens": "SIEMENS.NS", "Bharat Electronics": "BEL.NS", "HAL": "HAL.NS",
    "IRCTC": "IRCTC.NS", "IRFC": "IRFC.NS", "PB Fintech": "POLICYBZR.NS"
}

FINANCIAL_RSS_FEEDS = [
    ("https://feeds.feedburner.com/ndtvprofit-latest", "NDTV Profit"),
    ("https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms", "ET Markets"),
    ("https://www.moneycontrol.com/rss/latestnews.xml", "Moneycontrol"),
]

ARTICLES_PER_REFRESH = 15
NEWS_AGE_LIMIT_HOURS = 48

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
if 'technical_data' not in st.session_state:
    st.session_state.technical_data = []

# --------------------------
# Technical Analysis Functions
# --------------------------
def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_ao(high, low, fast=5, slow=34):
    """Calculate Awesome Oscillator"""
    median_price = (high + low) / 2
    ao = median_price.rolling(window=fast).mean() - median_price.rolling(window=slow).mean()
    return ao

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def generate_signal(ticker_symbol):
    """Generate buy/sell signal based on technical indicators"""
    try:
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period='3mo')
        
        if df.empty or len(df) < 50:
            return None
        
        # Calculate indicators
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['Signal'] = calculate_macd(df['Close'])
        df['AO'] = calculate_ao(df['High'], df['Low'])
        df['EMA_9'] = calculate_ema(df['Close'], 9)
        df['EMA_21'] = calculate_ema(df['Close'], 21)
        df['SMA_50'] = calculate_sma(df['Close'], 50)
        df['SMA_200'] = calculate_sma(df['Close'], 200)
        
        # Get latest values
        current_price = df['Close'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        signal_line = df['Signal'].iloc[-1]
        ao = df['AO'].iloc[-1]
        
        # Signal logic
        signals = []
        score = 0
        
        # RSI Analysis
        if rsi < 30:
            signals.append("RSI Oversold")
            score += 2
        elif rsi > 70:
            signals.append("RSI Overbought")
            score -= 2
        elif 40 <= rsi <= 60:
            signals.append("RSI Neutral")
        
        # MACD Analysis
        if macd > signal_line:
            signals.append("MACD Bullish")
            score += 1
        else:
            signals.append("MACD Bearish")
            score -= 1
        
        # AO Analysis
        if ao > 0:
            signals.append("AO Positive")
            score += 1
        else:
            signals.append("AO Negative")
            score -= 1
        
        # EMA Crossover
        if df['EMA_9'].iloc[-1] > df['EMA_21'].iloc[-1]:
            signals.append("EMA Bullish")
            score += 1
        else:
            signals.append("EMA Bearish")
            score -= 1
        
        # Final recommendation
        if score >= 3:
            recommendation = "🟢 STRONG BUY"
        elif score >= 1:
            recommendation = "🟡 BUY"
        elif score <= -3:
            recommendation = "🔴 STRONG SELL"
        elif score <= -1:
            recommendation = "🟠 SELL"
        else:
            recommendation = "⚪ HOLD"
        
        return {
            'price': current_price,
            'rsi': rsi,
            'macd': macd,
            'ao': ao,
            'signals': ', '.join(signals),
            'recommendation': recommendation,
            'score': score
        }
    except Exception as e:
        return None

# --------------------------
# Sentiment analysis
# --------------------------
def analyze_sentiment(text):
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

def check_fno_mention(text):
    text_upper = text.upper()
    for stock in FNO_STOCKS:
        if stock.upper() in text_upper:
            return True
    return False

def get_mentioned_stocks(text):
    text_upper = text.upper()
    mentioned = []
    for stock in FNO_STOCKS:
        if stock.upper() in text_upper:
            if stock not in mentioned:
                mentioned.append(stock)
    return mentioned if mentioned else ["Other"]

def fetch_news(num_articles=15, specific_stock=None, force_new=False):
    all_articles = []
    
    if force_new or (specific_stock and specific_stock != "All Stocks"):
        seen_titles = set()
    else:
        seen_titles = {article['Title'] for article in st.session_state.news_articles}
    
    if specific_stock and specific_stock != "All Stocks":
        priority_stocks = [specific_stock]
        num_articles = num_articles * 3
    else:
        priority_stocks = FNO_STOCKS[:30]
    
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
    
    for feed_url, source_name in FINANCIAL_RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                title = entry.title if hasattr(entry, 'title') else ""
                if title in seen_titles:
                    continue
                full_text = title + " " + getattr(entry, 'summary', '')
                
                if specific_stock and specific_stock != "All Stocks":
                    if specific_stock.upper() not in full_text.upper():
                        continue
                else:
                    if not check_fno_mention(full_text):
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
    if stock_name == "All Stocks":
        return news_articles
    filtered = []
    for article in news_articles:
        if stock_name in article.get('Stocks', []):
            filtered.append(article)
    return filtered

# --------------------------
# EARNINGS Functions - Q2 FY25 (Current Quarter)
# --------------------------
@st.cache_data(ttl=1800)
def fetch_q2_fy25_earnings():
    """Generate Q2 FY25 earnings calendar - July-Sept 2024, announced Oct-Nov 2024"""
    earnings_list = []
    
    # Q2 FY25 = July-Sept 2024
    # Results announced: October 10 - November 14, 2024
    # Many companies announced on Nov 11, 12, 13
    
    # Major companies with actual Q2 FY25 dates
    actual_dates = {
        "Reliance": "2024-10-14",
        "TCS": "2024-10-10",
        "Infosys": "2024-10-17",
        "Wipro": "2024-10-18",
        "HCL Tech": "2024-10-14",
        "HDFC Bank": "2024-10-19",
        "ICICI Bank": "2024-10-26",
        "SBI": "2024-11-08",
        "Axis Bank": "2024-10-23",
        "Kotak Bank": "2024-10-19",
        "Bank of Baroda": "2024-11-12",
        "PNB": "2024-11-12",
        "Canara Bank": "2024-11-13",
        "Union Bank": "2024-11-13",
        "Indian Bank": "2024-11-13",
        "Titan": "2024-11-07",
        "Bajaj Finance": "2024-10-22",
        "Bajaj Finserv": "2024-10-23",
        "Maruti Suzuki": "2024-10-25",
        "Mahindra": "2024-11-08",
        "Tata Motors": "2024-11-08",
        "Hero MotoCorp": "2024-11-08",
        "TVS Motor": "2024-10-23",
        "Bajaj Auto": "2024-11-08",
        "Hindustan Unilever": "2024-10-23",
        "ITC": "2024-10-24",
        "Asian Paints": "2024-10-24",
        "Nestle": "2024-10-24",
        "Britannia": "2024-10-24",
        "Godrej Consumer": "2024-10-23",
        "Marico": "2024-10-24",
        "Dabur": "2024-10-24",
        "Tata Steel": "2024-11-07",
        "JSW Steel": "2024-11-08",
        "Hindalco": "2024-11-08",
        "Coal India": "2024-10-28",
        "NTPC": "2024-11-08",
        "Power Grid": "2024-11-13",
        "Tata Power": "2024-11-08",
        "Adani Enterprises": "2024-11-04",
        "Adani Ports": "2024-11-04",
        "Adani Green": "2024-11-08",
        "Adani Power": "2024-11-08",
        "Sun Pharma": "2024-10-29",
        "Dr Reddy": "2024-10-25",
        "Cipla": "2024-10-25",
        "Lupin": "2024-11-07",
        "Biocon": "2024-11-08",
        "Divi's Lab": "2024-10-25",
        "Larsen & Toubro": "2024-10-24",
        "UltraTech": "2024-10-24",
        "Ambuja Cement": "2024-10-29",
        "Shree Cement": "2024-10-25",
        "Grasim": "2024-10-24",
        "Zomato": "2024-10-23",
        "DMart": "2024-11-12",
        "IRCTC": "2024-10-29",
        "Apollo Hospitals": "2024-11-06"
    }
    
    for stock in FNO_STOCKS:
        if stock in actual_dates:
            date_str = actual_dates[stock]
            result_date = datetime.strptime(date_str, '%Y-%m-%d')
        else:
            # Generate dates for remaining stocks
            base_date = datetime(2024, 10, 15)
            days_offset = (FNO_STOCKS.index(stock) * 2) % 30
            result_date = base_date + timedelta(days=days_offset)
            while result_date.weekday() >= 5:
                result_date += timedelta(days=1)
        
        earnings_list.append({
            'Company': stock,
            'Quarter': 'Q2 FY25 (Jul-Sep 2024)',
            'Result Date': result_date.strftime('%d-%b-%Y'),
            'Day': result_date.strftime('%A'),
            'Status': 'Declared' if stock in actual_dates else 'Expected'
        })
    
    earnings_list.sort(key=lambda x: datetime.strptime(x['Result Date'], '%d-%b-%Y'))
    
    return earnings_list

# --------------------------
# Streamlit App
# --------------------------

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📰 News Dashboard", "📅 Q2 FY25 Earnings", "📈 Technical Analysis", "💹 Stock Charts"])

# --------------------------
# TAB 1: NEWS DASHBOARD
# --------------------------
with tab1:
    st.title("📈 F&O Stocks News Dashboard")
    st.markdown(f"*Real-time news for {len(FNO_STOCKS)} F&O stocks with sentiment analysis*")
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        stock_options = ["All Stocks"] + sorted(FNO_STOCKS)
        selected_stock = st.selectbox(
            "🔍 Filter by Stock",
            options=stock_options,
            index=stock_options.index(st.session_state.selected_stock),
            key="stock_filter"
        )
        
        if selected_stock != st.session_state.selected_stock:
            st.session_state.selected_stock = selected_stock
            if selected_stock != "All Stocks":
                with st.spinner(f"Fetching fresh news for {selected_stock}..."):
                    new_articles = fetch_news(ARTICLES_PER_REFRESH, selected_stock, force_new=True)
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
                        st.rerun()

    with col2:
        if st.button("🔄 Refresh News", type="primary", use_container_width=True):
            with st.spinner(f"Fetching latest updates..."):
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
                    st.success(f"✅ Added {news_count} fresh articles!")
                    st.rerun()

    with col3:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.news_articles = []
            st.success("✅ Cleared all news!")
            st.rerun()

    if not st.session_state.news_articles:
        with st.spinner("Loading initial content..."):
            initial_news = fetch_news(ARTICLES_PER_REFRESH, st.session_state.selected_stock)
            if initial_news:
                st.session_state.news_articles = process_news(initial_news)

    filtered_articles = filter_news_by_stock(st.session_state.news_articles, st.session_state.selected_stock)

    if filtered_articles:
        df_all = pd.DataFrame(filtered_articles)
        
        st.subheader(f"📊 Metrics for {st.session_state.selected_stock}")
        col1, col2, col3, col4 = st.columns(4)
        
        total_items = len(df_all)
        
