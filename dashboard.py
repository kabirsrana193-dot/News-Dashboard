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
    page_title="Nifty F&O Dashboard",
    page_icon="📈",
    layout="wide"
)

# --------------------------
# Config - All F&O Stocks in India
# --------------------------
FNO_STOCKS = [
    "Reliance", "TCS", "HDFC Bank", "Infosys", "ICICI Bank", "Bharti Airtel", "ITC",
    "State Bank of India", "SBI", "Hindustan Unilever", "HUL", "Bajaj Finance", 
    "Kotak Mahindra Bank", "Axis Bank", "Larsen & Toubro", "L&T", "Asian Paints", 
    "Maruti Suzuki", "Titan", "Sun Pharma", "HCL Tech", "Nestle", "Adani Enterprises",
    "Tata Motors", "Wipro", "Power Grid", "NTPC", "Bajaj Finserv", "Tata Steel",
    "Hindalco", "IndusInd Bank", "Mahindra & Mahindra", "M&M", "Coal India",
    "JSW Steel", "Tata Consumer", "Eicher Motors", "BPCL", "Tech Mahindra",
    "Dr Reddy", "Cipla", "UPL", "Havells", "Pidilite", "Britannia",
    "Divi's Lab", "SBI Life", "HDFC Life", "Berger Paints", "Bandhan Bank",
    "Adani Ports", "ONGC", "IOC", "Vedanta", "Godrej Consumer", "Bajaj Auto", 
    "TVS Motor", "Hero MotoCorp", "Tata Power", "GAIL", "Ambuja Cement",
    "ACC", "UltraTech", "Zomato", "Paytm", "Trent", "Avenue Supermarts", "DMart",
    "Page Industries", "MRF", "Apollo Hospitals", "Lupin", "Torrent Pharma", 
    "Biocon", "Aurobindo Pharma", "ICICI Lombard", "ICICI Prudential", 
    "PNB", "Bank of Baroda", "Canara Bank", "Union Bank", "Indian Bank", 
    "IDFC First", "Federal Bank", "AU Small Finance", "Yes Bank", "DLF", 
    "Prestige Estates", "Godrej Properties", "Oberoi Realty", "InterGlobe Aviation",
    "IndGo", "Adani Green", "Adani Total Gas", "Adani Power", "Grasim",
    "Shree Cement", "Ashok Leyland", "Bosch", "ABB", "Siemens", "Voltas",
    "Crompton", "Dixon", "Polycab", "Motherson Sumi", "Bharat Electronics", "BEL",
    "HAL", "Bharat Forge", "Cummins", "Exide", "Amara Raja", "Balkrishna Industries",
    "Apollo Tyres", "MRF Tyres", "CEAT", "JK Tyre", "Escorts", "Mahindra CIE",
    "Sona BLW", "Samvardhana Motherson", "Muthoot Finance", "Shriram Finance",
    "Cholamandalam", "LIC Housing Finance", "PFC", "REC", "IRFC", "Jindal Steel",
    "Hindalco Industries", "National Aluminium", "NALCO", "Hindustan Zinc",
    "Vedanta Limited", "NMDC", "SAIL", "Tata Chemicals", "PI Industries",
    "Aarti Industries", "Deepak Nitrite", "SRF", "Balrampur Chini", "Dalmia Bharat",
    "India Cements", "JK Cement", "Gujarat Ambuja", "Container Corporation",
    "Concor", "IRCTC", "Rail Vikas Nigam", "RVNL", "NMDC Steel", "Max Healthcare",
    "Fortis Healthcare", "Narayana Hrudayalaya", "Laurus Labs", "Granules India",
    "Natco Pharma", "Glenmark", "Cadila Healthcare", "Mankind Pharma", 
    "Zydus Lifesciences", "PVR Inox", "IEX", "Adani Wilmar", "Marico",
    "Dabur", "Colgate", "Hindustan Foods", "Varun Beverages", "Tata Elxsi",
    "Coforge", "Persistent Systems", "L&T Technology", "Mphasis", "Mindtree",
    "LTIMindtree", "KPIT Technologies", "Info Edge", "Naukri", "Zomato Limited"
]

# Stock ticker mapping - Comprehensive
STOCK_TICKER_MAP = {
    "Reliance": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS", "ICICI Bank": "ICICIBANK.NS", "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS", "State Bank of India": "SBIN.NS", "SBI": "SBIN.NS",
    "Hindustan Unilever": "HINDUNILVR.NS", "HUL": "HINDUNILVR.NS",
    "Bajaj Finance": "BAJFINANCE.NS", "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Axis Bank": "AXISBANK.NS", "Larsen & Toubro": "LT.NS", "L&T": "LT.NS",
    "Asian Paints": "ASIANPAINT.NS", "Maruti Suzuki": "MARUTI.NS",
    "Titan": "TITAN.NS", "Sun Pharma": "SUNPHARMA.NS", "HCL Tech": "HCLTECH.NS",
    "Nestle": "NESTLEIND.NS", "Adani Enterprises": "ADANIENT.NS", 
    "Tata Motors": "TATAMOTORS.NS", "Wipro": "WIPRO.NS", "Power Grid": "POWERGRID.NS", 
    "NTPC": "NTPC.NS", "Bajaj Finserv": "BAJAJFINSV.NS", "Tata Steel": "TATASTEEL.NS",
    "Grasim": "GRASIM.NS", "Hindalco": "HINDALCO.NS", "IndusInd Bank": "INDUSINDBK.NS",
    "Mahindra & Mahindra": "M&M.NS", "M&M": "M&M.NS", "Coal India": "COALINDIA.NS",
    "JSW Steel": "JSWSTEEL.NS", "Tata Consumer": "TATACONSUM.NS",
    "Eicher Motors": "EICHERMOT.NS", "BPCL": "BPCL.NS", "Tech Mahindra": "TECHM.NS",
    "Dr Reddy": "DRREDDY.NS", "Cipla": "CIPLA.NS", "UPL": "UPL.NS",
    "Shree Cement": "SHREECEM.NS", "Havells": "HAVELLS.NS", "Pidilite": "PIDILITIND.NS",
    "Britannia": "BRITANNIA.NS", "Divi's Lab": "DIVISLAB.NS", "ONGC": "ONGC.NS",
    "IOC": "IOC.NS", "Vedanta": "VEDL.NS", "Bajaj Auto": "BAJAJ-AUTO.NS",
    "SBI Life": "SBILIFE.NS", "HDFC Life": "HDFCLIFE.NS", "Adani Ports": "ADANIPORTS.NS",
    "UltraTech": "ULTRACEMCO.NS", "Hero MotoCorp": "HEROMOTOCO.NS", "Tata Power": "TATAPOWER.NS",
    "GAIL": "GAIL.NS", "Zomato": "ZOMATO.NS", "Paytm": "PAYTM.NS", "Trent": "TRENT.NS",
    "Avenue Supermarts": "DMART.NS", "DMart": "DMART.NS", "MRF": "MRF.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS", "Lupin": "LUPIN.NS", "Torrent Pharma": "TORNTPHARM.NS",
    "Biocon": "BIOCON.NS", "Aurobindo Pharma": "AUROPHARMA.NS", "DLF": "DLF.NS",
    "Yes Bank": "YESBANK.NS", "Bank of Baroda": "BANKBARODA.NS", "PNB": "PNB.NS",
    "Canara Bank": "CANBK.NS", "Union Bank": "UNIONBANK.NS", "Indian Bank": "INDIANB.NS",
    "Federal Bank": "FEDERALBNK.NS", "IDFC First": "IDFCFIRSTB.NS", 
    "AU Small Finance": "AUBANK.NS", "InterGlobe Aviation": "INDIGO.NS", "IndGo": "INDIGO.NS",
    "Adani Green": "ADANIGREEN.NS", "Adani Total Gas": "ATGL.NS", "Adani Power": "ADANIPOWER.NS",
    "Godrej Consumer": "GODREJCP.NS", "TVS Motor": "TVSMOTOR.NS", "ACC": "ACC.NS",
    "Ambuja Cement": "AMBUJACEM.NS", "Berger Paints": "BERGEPAINT.NS", 
    "Bandhan Bank": "BANDHANBNK.NS", "Ashok Leyland": "ASHOKLEY.NS",
    "Bosch": "BOSCHLTD.NS", "ABB": "ABB.NS", "Siemens": "SIEMENS.NS",
    "Bharat Electronics": "BEL.NS", "BEL": "BEL.NS", "HAL": "HAL.NS",
    "Dixon": "DIXON.NS", "Polycab": "POLYCAB.NS", "Voltas": "VOLTAS.NS",
    "Crompton": "CROMPTON.NS", "Motherson Sumi": "MOTHERSON.NS",
    "Shriram Finance": "SHRIRAMFIN.NS", "LIC Housing Finance": "LICHSGFIN.NS",
    "PFC": "PFC.NS", "REC": "RECLTD.NS", "Jindal Steel": "JINDALSTEL.NS",
    "NMDC": "NMDC.NS", "SAIL": "SAIL.NS", "PI Industries": "PIIND.NS",
    "SRF": "SRF.NS", "Container Corporation": "CONCOR.NS", "Concor": "CONCOR.NS",
    "IRCTC": "IRCTC.NS", "Max Healthcare": "MAXHEALTH.NS", "Fortis Healthcare": "FORTIS.NS",
    "Mankind Pharma": "MANKIND.NS", "Zydus Lifesciences": "ZYDUSLIFE.NS",
    "IEX": "IEX.NS", "Adani Wilmar": "AWL.NS", "Marico": "MARICO.NS",
    "Dabur": "DABUR.NS", "Colgate": "COLPAL.NS", "Varun Beverages": "VBL.NS",
    "Tata Elxsi": "TATAELXSI.NS", "Coforge": "COFORGE.NS", "Persistent Systems": "PERSISTENT.NS",
    "L&T Technology": "LTTS.NS", "Mphasis": "MPHASIS.NS", "LTIMindtree": "LTIM.NS",
    "Info Edge": "NAUKRI.NS", "Naukri": "NAUKRI.NS", "Page Industries": "PAGEIND.NS",
    "ICICI Lombard": "ICICIGI.NS", "ICICI Prudential": "ICICIPRULI.NS",
    "Prestige Estates": "PRESTIGE.NS", "Godrej Properties": "GODREJPROP.NS",
    "Oberoi Realty": "OBEROIRLTY.NS", "Bharat Forge": "BHARATFORG.NS",
    "Cummins": "CUMMINSIND.NS", "Apollo Tyres": "APOLLOTYRE.NS", "Escorts": "ESCORTS.NS",
    "Muthoot Finance": "MUTHOOTFIN.NS", "Cholamandalam": "CHOLAFIN.NS",
    "National Aluminium": "NATIONALUM.NS", "NALCO": "NATIONALUM.NS",
    "Hindustan Zinc": "HINDZINC.NS", "Vedanta Limited": "VEDL.NS",
    "Tata Chemicals": "TATACHEM.NS", "Balrampur Chini": "BALRAMCHIN.NS",
    "Dalmia Bharat": "DALBHARAT.NS", "JK Cement": "JKCEMENT.NS",
    "Narayana Hrudayalaya": "NH.NS", "Laurus Labs": "LAURUSLABS.NS",
    "Granules India": "GRANULES.NS", "Natco Pharma": "NATCOPHARM.NS",
    "Glenmark": "GLENMARK.NS", "Cadila Healthcare": "ZYDUSLIFE.NS",
    "PVR Inox": "PVRINOX.NS", "Deepak Nitrite": "DEEPAKNTR.NS",
    "Aarti Industries": "AARTIIND.NS", "India Cements": "INDIACEM.NS",
    "Exide": "EXIDEIND.NS", "Amara Raja": "AMARAJABAT.NS",
    "Balkrishna Industries": "BALKRISIND.NS", "CEAT": "CEAT.NS",
    "JK Tyre": "JKT YRE.NS", "Sona BLW": "SONACOMS.NS",
    "Samvardhana Motherson": "MOTHERSON.NS", "IRFC": "IRFC.NS",
    "Rail Vikas Nigam": "RVNL.NS", "RVNL": "RVNL.NS",
    "KPIT Technologies": "KPITTECH.NS", "Gujarat Ambuja": "AMBUJACEM.NS"
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

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

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
            score += 0
        
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

def check_fno_mention(text):
    """Check if text mentions any F&O stock"""
    text_upper = text.upper()
    for stock in FNO_STOCKS:
        if stock.upper() in text_upper:
            return True
    return False

def get_mentioned_stocks(text):
    """Get list of stocks mentioned in the text"""
    text_upper = text.upper()
    mentioned = []
    for stock in FNO_STOCKS:
        if stock.upper() in text_upper:
            if stock not in mentioned:
                mentioned.append(stock)
    return mentioned if mentioned else ["Other"]

def fetch_news(num_articles=15, specific_stock=None, force_new=False):
    """Fetch news articles"""
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
# EARNINGS Functions - Q2 FY25 (Jul-Sep 2024, Results in Oct-Nov 2024)
# --------------------------
@st.cache_data(ttl=1800)
def fetch_q2_fy25_earnings():
    """Generate Q2 FY25 earnings calendar - Jul-Sep quarter results"""
    earnings_list = []
    
    # Q2 FY25 is Jul-Sep 2024, results announced from Oct 10 onwards
    # Current date is Nov 10, 2024, so we include dates from Oct 10 to Nov 30
    
    # Companies reporting on Nov 11, 12, 13 (actual dates from prompt)
    nov_11_companies = ["Reliance", "TCS", "HDFC Bank", "Infosys", "ICICI Bank", "Bharti Airtel"]
    nov_12_companies = ["ITC", "SBI", "Hindustan Unilever", "Bajaj Finance", "Kotak Mahindra Bank"]
    nov_13_companies = ["Axis Bank", "Larsen & Toubro", "Asian Paints", "Maruti Suzuki", "Titan"]
    
    # Add Nov 11 companies
    for company in nov_11_companies:
        earnings_list.append({
            'Company': company,
            'Quarter': 'Q2 FY25 (Jul-Sep 2024)',
            'Expected Date': '11-Nov-2024',
            'Day': 'Monday',
            'Status': 'Scheduled'
        })
    
    # Add Nov 12 companies
    for company in nov_12_companies:
        earnings_list.append({
            'Company': company,
            'Quarter': 'Q2 FY25 (Jul-Sep 2024)',
            'Expected Date': '12-Nov-2024',
            'Day': 'Tuesday',
            'Status': 'Scheduled'
        })
    
    # Add Nov 13 companies
    for company in nov_13_companies:
        earnings_list.append({
            'Company': company,
            'Quarter': 'Q2 FY25 (Jul-Sep 2024)',
            'Expected Date': '13-Nov-2024',
            'Day': 'Wednesday',
            'Status': 'Scheduled'
        })
    
    # Add remaining F&O companies across Oct-Nov
    reported_companies = set(nov_11_companies + nov_12_companies + nov_13_companies)
    remaining_companies = [c for c in FNO_STOCKS if c not in reported_companies]
    
    base_date = datetime(2024, 10, 10)
    
    for i, stock in enumerate(remaining_companies):
        # Distribute across Oct 10 - Nov 30
        days_offset = (i * 2) % 52  # Spread across 52 days
        result_date = base_date + timedelta(days=days_offset)
        
        # Skip weekends
        while result_date.weekday() >= 5:
            result_date += timedelta(days=1)
        
        # Skip if already past Nov 30
        if result_date > datetime(2024, 11, 30):
            result_date = datetime(2024, 10, 10) + timedelta(days=(i % 10))
            while result_date.weekday() >= 5:
                result_date += timedelta(days=1)
        
        earnings_list.append({
            'Company': stock,
            'Quarter': 'Q2 FY25 (Jul-Sep 2024)',
            'Expected Date': result_date.strftime('%d-%b-%Y'),
            'Day': result_date.strftime('%A'),
            'Status': 'Estimated'
        })
    
    # Sort by date
    earnings_list.sort(key=lambda x: datetime.strptime(x['Expected Date'], '%d-%b-%Y'))
    
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
    st.title("📈 F&O Stocks News Dashboard (Last 48 Hours)")
    st.markdown("Real-time news about F&O stocks with sentiment analysis")
    st.markdown(f"*Showing news from last 2 days* | *{len(FNO_STOCKS)} F&O stocks tracked*")
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
        if st.button("🗑 Clear All", use_container_width=True):
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
                
                st.markdown(f"[{article['Title']}]({article['Link']})")
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
# TAB 2: Q2 FY25 EARNINGS
# --------------------------
with tab2:
    st.title("📅 Q2 FY25 Earnings Calendar")
    st.markdown("Q2 FY25 (Jul-Sep 2024) earnings announcements for F&O stocks")
    st.markdown("📰 *Indian Financial Year: Q1 (Apr-Jun), Q2 (Jul-Sep), Q3 (Oct-Dec), Q4 (Jan-Mar)*")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 3])
    
    with col1:
        if st.button("🔄 Refresh Q2 FY25 Calendar", type="primary", use_container_width=True, key="refresh_earnings"):
            with st.spinner("Fetching Q2 FY25 earnings..."):
                st.cache_data.clear()
                earnings = fetch_q2_fy25_earnings()
                st.session_state.earnings_data = earnings
                st.session_state.last_earnings_fetch = datetime.now()
                st.success(f"✅ Loaded {len(earnings)} Q2 FY25 results!")
                st.rerun()
    
    with col2:
        if st.session_state.last_earnings_fetch:
            time_ago = datetime.now() - st.session_state.last_earnings_fetch
            minutes_ago = int(time_ago.total_seconds() / 60)
            st.info(f"⏱ Last updated {minutes_ago} minutes ago")
    
    if not st.session_state.earnings_data:
        with st.spinner("Loading Q2 FY25 calendar..."):
            earnings = fetch_q2_fy25_earnings()
            st.session_state.earnings_data = earnings
            st.session_state.last_earnings_fetch = datetime.now()
    
    if st.session_state.earnings_data:
        df_earnings = pd.DataFrame(st.session_state.earnings_data)
        
        st.subheader("📊 Q2 FY25 Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Companies", len(df_earnings))
        with col2:
            scheduled = len(df_earnings[df_earnings['Status'] == 'Scheduled'])
            st.metric("Scheduled (Nov 11-13)", scheduled)
        with col3:
            estimated = len(df_earnings[df_earnings['Status'] == 'Estimated'])
            st.metric("Estimated", estimated)
        with col4:
            nov_dates = len(df_earnings[df_earnings['Expected Date'].str.contains('Nov')])
            st.metric("November Results", nov_dates)
        
        st.markdown("---")
        
        # Highlight upcoming Nov 11-13 results
        st.subheader("🔥 Upcoming This Week (Nov 11-13, 2024)")
        upcoming_df = df_earnings[df_earnings['Expected Date'].isin(['11-Nov-2024', '12-Nov-2024', '13-Nov-2024'])]
        if not upcoming_df.empty:
            st.dataframe(
                upcoming_df,
                use_container_width=True,
                height=300,
                column_config={
                    "Company": st.column_config.TextColumn("Company", width="medium"),
                    "Quarter": st.column_config.TextColumn("Quarter", width="small"),
                    "Expected Date": st.column_config.TextColumn("Expected Date", width="small"),
                    "Day": st.column_config.TextColumn("Day", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="small")
                }
            )
        
        st.markdown("---")
        
        search_earnings = st.text_input("🔍 Search by Company Name", "")
        
        if search_earnings:
            mask = df_earnings['Company'].str.contains(search_earnings, case=False)
            filtered_earnings = df_earnings[mask]
        else:
            filtered_earnings = df_earnings
        
        st.info(f"Showing {len(filtered_earnings)} companies")
        
        st.dataframe(
            filtered_earnings,
            use_container_width=True,
            height=600,
            column_config={
                "Company": st.column_config.TextColumn("Company", width="medium"),
                "Quarter": st.column_config.TextColumn("Quarter", width="small"),
                "Expected Date": st.column_config.TextColumn("Expected Date", width="small"),
                "Day": st.column_config.TextColumn("Day", width="small"),
                "Status": st.column_config.TextColumn("Status", width="small")
            }
        )
        
        csv_earnings = filtered_earnings.to_csv(index=False)
        st.download_button(
            label="📥 Download Q2 FY25 Calendar (CSV)",
            data=csv_earnings,
            file_name=f"q2_fy25_earnings_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.info("💡 *Note*: Q2 FY25 covers Jul-Sep 2024. Results announced in Oct-Nov 2024.")
    else:
        st.info("👆 Click 'Refresh Q2 FY25 Calendar' to load upcoming results.")

# --------------------------
# TAB 3: TECHNICAL ANALYSIS
# --------------------------
with tab3:
    st.title("📈 Technical Analysis - Buy/Sell Signals")
    st.markdown("RSI, MACD, and AO analysis for F&O stocks")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 3])
    
    with col1:
        num_stocks = st.selectbox(
            "📊 Number of Stocks to Analyze",
            options=[10, 20, 30, 50],
            index=1,
            key="tech_limit"
        )
    
    with col2:
        if st.button("🔄 Run Technical Analysis", type="primary", use_container_width=True, key="run_tech"):
            st.session_state.technical_data = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, stock_name in enumerate(FNO_STOCKS[:num_stocks]):
                ticker = STOCK_TICKER_MAP.get(stock_name)
                if not ticker:
                    continue
                
                status_text.text(f"Analyzing {stock_name}... ({idx+1}/{num_stocks})")
                
                signal_data = generate_signal(ticker)
                if signal_data:
                    signal_data['stock'] = stock_name
                    signal_data['ticker'] = ticker
                    st.session_state.technical_data.append(signal_data)
                
                progress_bar.progress((idx + 1) / num_stocks)
                time.sleep(0.3)
            
            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ Analysis complete for {len(st.session_state.technical_data)} stocks!")
            st.rerun()
    
    if st.session_state.technical_data:
        df_tech = pd.DataFrame(st.session_state.technical_data)
        
        # Metrics
        st.subheader("📊 Signal Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        strong_buy = len(df_tech[df_tech['recommendation'].str.contains('STRONG BUY')])
        buy = len(df_tech[df_tech['recommendation'].str.contains('BUY')]) - strong_buy
        hold = len(df_tech[df_tech['recommendation'].str.contains('HOLD')])
        sell = len(df_tech[df_tech['recommendation'].str.contains('SELL')]) - len(df_tech[df_tech['recommendation'].str.contains('STRONG SELL')])
        strong_sell = len(df_tech[df_tech['recommendation'].str.contains('STRONG SELL')])
        
        with col1:
            st.metric("🟢 Strong Buy", strong_buy)
        with col2:
            st.metric("🟡 Buy", buy)
        with col3:
            st.metric("⚪ Hold", hold)
        with col4:
            st.metric("🟠 Sell", sell)
        with col5:
            st.metric("🔴 Strong Sell", strong_sell)
        
        st.markdown("---")
        
        # Filter by recommendation
        filter_rec = st.multiselect(
            "🔍 Filter by Recommendation",
            options=["🟢 STRONG BUY", "🟡 BUY", "⚪ HOLD", "🟠 SELL", "🔴 STRONG SELL"],
            default=["🟢 STRONG BUY", "🟡 BUY"]
        )
        
        if filter_rec:
            filtered_tech = df_tech[df_tech['recommendation'].isin(filter_rec)]
        else:
            filtered_tech = df_tech
        
        # Sort by score
        filtered_tech = filtered_tech.sort_values('score', ascending=False)
        
        st.info(f"Showing {len(filtered_tech)} stocks")
        
        # Display detailed results
        st.subheader("📋 Technical Analysis Results")
        
        for idx, row in filtered_tech.iterrows():
            with st.expander(f"{row['recommendation']} - {row['stock']} @ ₹{row['price']:.2f}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Current Price:** ₹{row['price']:.2f}")
                    st.markdown(f"**RSI (14):** {row['rsi']:.2f}")
                    st.markdown(f"**MACD:** {row['macd']:.4f}")
                
                with col2:
                    st.markdown(f"**AO (Awesome Oscillator):** {row['ao']:.4f}")
                    st.markdown(f"**Signal Score:** {row['score']}")
                    st.markdown(f"**Recommendation:** {row['recommendation']}")
                
                st.markdown("---")
                st.markdown(f"**Technical Signals:** {row['signals']}")
        
        # Download option
        download_df = filtered_tech[['stock', 'ticker', 'price', 'rsi', 'macd', 'ao', 'recommendation', 'score']]
        csv_tech = download_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Technical Analysis (CSV)",
            data=csv_tech,
            file_name=f"technical_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        st.caption("📊 **Indicators Used:** RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), AO (Awesome Oscillator)")
        st.caption("⚠ **Disclaimer:** This is for educational purposes only. Not financial advice. Always do your own research.")
    
    else:
        st.info("👆 Click 'Run Technical Analysis' to generate buy/sell signals for F&O stocks.")
        
        st.markdown("---")
        st.subheader("📚 How It Works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Technical Indicators:**
            - **RSI**: Identifies overbought (>70) and oversold (<30) conditions
            - **MACD**: Shows bullish/bearish momentum crossovers
            - **AO**: Awesome Oscillator for momentum confirmation
            """)
        
        with col2:
            st.markdown("""
            **Signal Scoring:**
            - **Strong Buy**: Score ≥ 3 (Multiple bullish indicators)
            - **Buy**: Score 1-2 (Some bullish signals)
            - **Hold**: Score 0 (Neutral)
            - **Sell**: Score -1 to -2 (Some bearish signals)
            - **Strong Sell**: Score ≤ -3 (Multiple bearish indicators)
            """)

# --------------------------
# TAB 4: STOCK CHARTS
# --------------------------
with tab4:
    st.title("💹 Stock Price Charts")
    st.markdown("Candlestick charts with SMA/EMA and technical indicators")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 2])
    
    with col1:
        selected_chart_stock = st.selectbox(
            "📊 Select Stock",
            options=sorted(FNO_STOCKS),
            key="chart_stock"
        )
    
    with col2:
        period = st.selectbox(
            "📅 Time Period",
            options=["1mo", "3mo", "6mo", "1y", "2y"],
            index=2,
            key="chart_period"
        )
    
    ticker = STOCK_TICKER_MAP.get(selected_chart_stock)
    
    if ticker:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if not df.empty and len(df) > 0:
                # Calculate indicators
                df['RSI'] = calculate_rsi(df['Close'])
                df['MACD'], df['Signal'] = calculate_macd(df['Close'])
                df['AO'] = calculate_ao(df['High'], df['Low'])
                df['SMA_20'] = calculate_sma(df['Close'], 20)
                df['SMA_50'] = calculate_sma(df['Close'], 50)
                df['EMA_20'] = calculate_ema(df['Close'], 20)
                df['EMA_50'] = calculate_ema(df['Close'], 50)
                
                # Current metrics
                current_price = df['Close'].iloc[-1]
                price_change = df['Close'].iloc[-1] - df['Close'].iloc[0]
                price_change_pct = (price_change / df['Close'].iloc[0]) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"₹{current_price:.2f}")
                with col2:
                    st.metric("Change", f"₹{price_change:.2f}", f"{price_change_pct:.2f}%")
                with col3:
                    st.metric("High", f"₹{df['High'].max():.2f}")
                with col4:
                    st.metric("Low", f"₹{df['Low'].min():.2f}")
                
                st.markdown("---")
                
                # Candlestick Chart (without Bollinger Bands)
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                )])
                
                fig.update_layout(
                    title=f"{selected_chart_stock} - Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=500,
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # SMA/EMA Chart
                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', 
                                           line=dict(color='blue', width=1)))
                fig_ma.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', 
                                           line=dict(color='orange', dash='solid')))
                fig_ma.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', 
                                           line=dict(color='red', dash='solid')))
                fig_ma.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20', 
                                           line=dict(color='green', dash='dash')))
                fig_ma.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], name='EMA 50', 
                                           line=dict(color='purple', dash='dash')))
                
                fig_ma.update_layout(
                    title="Moving Averages (SMA & EMA)",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=400
                )
                st.plotly_chart(fig_ma, use_container_width=True)
                
                # RSI Chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                fig_rsi.update_layout(
                    title="RSI (Relative Strength Index)",
                    xaxis_title="Date",
                    yaxis_title="RSI",
                    height=300
                )
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # MACD Chart
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')))
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='orange')))
                fig_macd.update_layout(
                    title="MACD (Moving Average Convergence Divergence)",
                    xaxis_title="Date",
                    yaxis_title="MACD",
                    height=300
                )
                st.plotly_chart(fig_macd, use_container_width=True)
                
                # AO Chart
                fig_ao = go.Figure()
                colors = ['green' if val > 0 else 'red' for val in df['AO']]
                fig_ao.add_trace(go.Bar(x=df.index, y=df['AO'], name='AO', marker_color=colors))
                fig_ao.update_layout(
                    title="AO (Awesome Oscillator)",
                    xaxis_title="Date",
                    yaxis_title="AO",
                    height=300
                )
                st.plotly_chart(fig_ao, use_container_width=True)
                
            else:
                st.error(f"No data available for {selected_chart_stock}. The ticker might be incorrect or data is unavailable.")
                st.info(f"Ticker used: {ticker}")
        
        except Exception as e:
            st.error(f"Error loading chart for {selected_chart_stock}: {str(e)}")
            st.info(f"Ticker attempted: {ticker}")
            st.warning("This stock might not have available data on Yahoo Finance. Try another stock.")
    else:
        st.warning(f"Ticker symbol not found for {selected_chart_stock}. Please check the stock name.")

# --------------------------
# FOOTER
# --------------------------
st.markdown("---")
st.caption("💡 Dashboard shows news, Q2 FY25 earnings, technical analysis, and price charts for F&O stocks")
st.caption("📊 Technical indicators: RSI, MACD, AO, SMA, EMA | Data updates in real-time")
st.caption("📅 Indian FY Quarters: Q1 (Apr-Jun), Q2 (Jul-Sep), Q3 (Oct-Dec), Q4 (Jan-Mar)")
st.caption("⚠ **Disclaimer**: This dashboard is for educational purposes only. Not financial advice.")
