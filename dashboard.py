import streamlit as st
import feedparser
import pandas as pd
from datetime import datetime, timedelta
import time
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

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

# Stock ticker mapping
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
    "JK Tyre": "JKTYRE.NS", "Sona BLW": "SONACOMS.NS",
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

# --------------------------
# Initialize session state
# --------------------------
if 'news_articles' not in st.session_state:
    st.session_state.news_articles = []
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = "All Stocks"
if 'technical_data' not in st.session_state:
    st.session_state.technical_data = []
if 'watchlist_stocks' not in st.session_state:
    st.session_state.watchlist_stocks = [
        "Reliance", "TCS", "HDFC Bank", "Infosys", 
        "ICICI Bank", "Bharti Airtel", "ITC", "SBI",
        "Hindustan Unilever", "Bajaj Finance"
    ]
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'finbert_model' not in st.session_state:
    st.session_state.finbert_model = None
if 'finbert_tokenizer' not in st.session_state:
    st.session_state.finbert_tokenizer = None

# --------------------------
# Load FinBERT Model
# --------------------------
@st.cache_resource
def load_finbert():
    """Load FinBERT model for financial sentiment analysis"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        return tokenizer, model
    except Exception as e:
        st.warning(f"Could not load FinBERT: {e}. Using fallback sentiment analysis.")
        return None, None

# Load model at startup
if st.session_state.finbert_tokenizer is None:
    with st.spinner("Loading FinBERT sentiment model..."):
        tokenizer, model = load_finbert()
        st.session_state.finbert_tokenizer = tokenizer
        st.session_state.finbert_model = model

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
        
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['Signal'] = calculate_macd(df['Close'])
        df['AO'] = calculate_ao(df['High'], df['Low'])
        
        current_price = df['Close'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        signal_line = df['Signal'].iloc[-1]
        ao = df['AO'].iloc[-1]
        
        signals = []
        score = 0
        
        if rsi < 30:
            signals.append("RSI Oversold")
            score += 2
        elif rsi > 70:
            signals.append("RSI Overbought")
            score -= 2
        elif 40 <= rsi <= 60:
            signals.append("RSI Neutral")
            score += 0
        
        if macd > signal_line:
            signals.append("MACD Bullish")
            score += 1
        else:
            signals.append("MACD Bearish")
            score -= 1
        
        if ao > 0:
            signals.append("AO Positive")
            score += 1
        else:
            signals.append("AO Negative")
            score -= 1
        
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
# FinBERT Sentiment Analysis
# --------------------------
def analyze_sentiment_finbert(text):
    """Analyze sentiment using FinBERT model"""
    tokenizer = st.session_state.finbert_tokenizer
    model = st.session_state.finbert_model
    
    if tokenizer is None or model is None:
        # Fallback to simple keyword-based
        return analyze_sentiment_fallback(text)
    
    try:
        # Tokenize and predict
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get sentiment scores
        scores = predictions[0].tolist()
        labels = ['positive', 'negative', 'neutral']
        
        # Get the sentiment with highest score
        max_idx = scores.index(max(scores))
        sentiment = labels[max_idx]
        confidence = scores[max_idx]
        
        return sentiment, round(confidence, 2)
    
    except Exception as e:
        return analyze_sentiment_fallback(text)

def analyze_sentiment_fallback(text):
    """Fallback keyword-based sentiment analysis"""
    POSITIVE_WORDS = ['surge', 'rally', 'gain', 'profit', 'growth', 'high', 'rise', 'up', 'bullish', 
                      'strong', 'beats', 'outperform', 'success', 'jumps', 'soars', 'positive']
    NEGATIVE_WORDS = ['fall', 'drop', 'loss', 'decline', 'weak', 'down', 'crash', 'bearish',
                      'concern', 'worry', 'risk', 'plunge', 'slump', 'miss', 'negative']
    
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
    seen_titles = set()
    
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
    """Process news articles with FinBERT sentiment analysis"""
    records = []
    for art in articles:
        title = art.title
        source = getattr(art, "source", {}).get("title", "Unknown") if hasattr(art, "source") else "Unknown"
        url = art.link
        published = getattr(art, 'published', 'Unknown')
        mentioned_stocks = get_mentioned_stocks(title + " " + getattr(art, 'summary', ''))
        
        # Use FinBERT for sentiment analysis
        sentiment, score = analyze_sentiment_finbert(title)
        
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
# Streamlit App
# --------------------------

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📰 News Dashboard", "📈 Technical Analysis", "💹 Stock Charts", "📊 Live Multi-Chart"])

# --------------------------
# TAB 1: NEWS DASHBOARD with FinBERT
# --------------------------
with tab1:
    st.title("📈 F&O Stocks News Dashboard (Last 48 Hours)")
    st.markdown("Real-time news with **FinBERT** AI sentiment analysis")
    st.markdown(f"🤖 Powered by FinBERT | {len(FNO_STOCKS)} F&O stocks tracked")
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
            with st.spinner(f"Fetching news for {selected_stock}..."):
                new_articles = fetch_news(ARTICLES_PER_REFRESH * 2, selected_stock, force_new=True)
                if new_articles:
                    processed_news = process_news(new_articles)
                    st.session_state.news_articles = processed_news
                    st.session_state.last_refresh = datetime.now()
                    st.rerun()

    with col2:
        if st.button("🔄 Refresh News", type="primary", use_container_width=True):
            with st.spinner(f"Fetching latest updates..."):
                new_articles = fetch_news(ARTICLES_PER_REFRESH * 2, st.session_state.selected_stock, force_new=True)
                if new_articles:
                    processed_news = process_news(new_articles)
                    all_articles = processed_news + st.session_state.news_articles
                    seen = set()
                    unique_articles = []
                    for article in all_articles:
                        if article['Title'] not in seen:
                            unique_articles.append(article)
                            seen.add(article['Title'])
                    st.session_state.news_articles = unique_articles[:100]
                    st.session_state.last_refresh = datetime.now()
                    news_count = len(processed_news)
                    st.success(f"✅ Added {news_count} fresh articles!")
                    time.sleep(1)
                    st.rerun()

    with col3:
        if st.button("🗑 Clear All", use_container_width=True):
            st.session_state.news_articles = []
            st.session_state.last_refresh = None
            st.success("✅ Cleared all news!")
            time.sleep(1)
            st.rerun()

    if st.session_state.last_refresh:
        time_ago = datetime.now() - st.session_state.last_refresh
        minutes_ago = int(time_ago.total_seconds() / 60)
        st.caption(f"⏱ Last refreshed {minutes_ago} minutes ago | 🤖 FinBERT AI Sentiment Analysis")

    if not st.session_state.news_articles:
        with st.spinner("Loading initial content with FinBERT analysis..."):
            initial_news = fetch_news(ARTICLES_PER_REFRESH, st.session_state.selected_stock)
            if initial_news:
                st.session_state.news_articles = process_news(initial_news)
                st.session_state.last_refresh = datetime.now()

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
        
        st.subheader("📊 FinBERT Sentiment Distribution")
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
            title=f"AI Sentiment Analysis for {st.session_state.selected_stock}",
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
                sentiment_text = f"{sentiment_emoji[article['Sentiment']]} {article['Sentiment'].upper()} (FinBERT: {article['Score']})"
                st.markdown(f"<span style='background-color: {sentiment_color[article['Sentiment']]}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;'>{sentiment_text}</span>", unsafe_allow_html=True)
                
                if article.get('Stocks'):
                    stocks_text = ", ".join(article['Stocks'][:5])
                    st.caption(f"📊 Stocks mentioned: {stocks_text}")
                
                st.caption(f"Source: {article['Source']} | {article.get('Published', 'Recent')}")
                st.markdown("---")
    else:
        st.info("No articles found. Try selecting a different stock or click Refresh News.")

# --------------------------
# TAB 2: TECHNICAL ANALYSIS
# --------------------------
with tab2:
    st.title("📈 Technical Analysis - Buy/Sell Signals")
    st.markdown("RSI, MACD, and AO analysis for F&O stocks")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 3])
    
    with col1:
        analysis_options = ["10 Stocks", "20 Stocks", "30 Stocks", "50 Stocks", "All Stocks"]
        selected_analysis = st.selectbox(
            "📊 Number of Stocks to Analyze",
            options=analysis_options,
            index=1,
            key="tech_limit"
        )
    
    with col2:
        if st.button("🔄 Run Technical Analysis", type="primary", use_container_width=True, key="run_tech"):
            st.session_state.technical_data = []
            
            if selected_analysis == "All Stocks":
                num_stocks = len(FNO_STOCKS)
            else:
                num_stocks = int(selected_analysis.split()[0])
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            stocks_to_analyze = FNO_STOCKS[:num_stocks]
            
            for idx, stock_name in enumerate(stocks_to_analyze):
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
                time.sleep(0.1)
            
            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ Analysis complete for {len(st.session_state.technical_data)} stocks!")
            st.rerun()
    
    if st.session_state.technical_data:
        df_tech = pd.DataFrame(st.session_state.technical_data)
        
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
        
        filter_rec = st.multiselect(
            "🔍 Filter by Recommendation",
            options=["🟢 STRONG BUY", "🟡 BUY", "⚪ HOLD", "🟠 SELL", "🔴 STRONG SELL"],
            default=["🟢 STRONG BUY", "🟡 BUY"]
        )
        
        if filter_rec:
            filtered_tech = df_tech[df_tech['recommendation'].isin(filter_rec)]
        else:
            filtered_tech = df_tech
        
        filtered_tech = filtered_tech.sort_values('score', ascending=False)
        
        st.info(f"Showing {len(filtered_tech)} stocks")
        
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

# --------------------------
# TAB 3: STOCK CHARTS
# --------------------------
with tab3:
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
                df['RSI'] = calculate_rsi(df['Close'])
                df['MACD'], df['Signal'] = calculate_macd(df['Close'])
                df['AO'] = calculate_ao(df['High'], df['Low'])
                
                df['SMA_20'] = calculate_sma(df['Close'], 20)
                df['SMA_50'] = calculate_sma(df['Close'], 50)
                df['SMA_200'] = calculate_sma(df['Close'], 200)
                
                df['EMA_9'] = calculate_ema(df['Close'], 9)
                df['EMA_20'] = calculate_ema(df['Close'], 20)
                df['EMA_50'] = calculate_ema(df['Close'], 50)
                
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
                
                fig_sma = go.Figure()
                fig_sma.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', 
                                           line=dict(color='blue', width=1)))
                fig_sma.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', 
                                           line=dict(color='orange', dash='solid')))
                fig_sma.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', 
                                           line=dict(color='red', dash='solid')))
                fig_sma.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', 
                                           line=dict(color='purple', dash='solid')))
                
                fig_sma.update_layout(
                    title="Simple Moving Averages (SMA 20, 50, 200)",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=400
                )
                st.plotly_chart(fig_sma, use_container_width=True)
                
                fig_ema = go.Figure()
                fig_ema.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', 
                                           line=dict(color='blue', width=1)))
                fig_ema.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], name='EMA 9', 
                                           line=dict(color='green', dash='dash')))
                fig_ema.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20', 
                                           line=dict(color='yellow', dash='dash')))
                fig_ema.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], name='EMA 50', 
                                           line=dict(color='cyan', dash='dash')))
                
                fig_ema.update_layout(
                    title="Exponential Moving Averages (EMA 9, 20, 50)",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=400
                )
                st.plotly_chart(fig_ema, use_container_width=True)
                
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
                st.error(f"No data available for {selected_chart_stock}.")
                st.info(f"Ticker used: {ticker}")
        
        except Exception as e:
            st.error(f"Error loading chart for {selected_chart_stock}: {str(e)}")
            st.info(f"Ticker attempted: {ticker}")

# --------------------------
# TAB 4: LIVE MULTI-CHART (CUSTOMIZABLE GRID)
# --------------------------
with tab4:
    st.title("📊 Live Multi-Chart Dashboard")
    st.markdown("Monitor multiple stocks simultaneously with customizable live charts")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
    
    with col1:
        st.markdown("📋 **Manage Your Watchlist**")
        
        max_stocks = st.number_input("Max stocks to display", min_value=1, max_value=20, value=10, step=1)
        
        selected_watchlist = st.multiselect(
            "Select stocks to monitor",
            options=sorted(FNO_STOCKS),
            default=st.session_state.watchlist_stocks[:max_stocks],
            max_selections=max_stocks,
            key="watchlist_selector"
        )
        
        if selected_watchlist != st.session_state.watchlist_stocks:
            st.session_state.watchlist_stocks = selected_watchlist
    
    with col2:
        chart_period_multi = st.selectbox(
            "📅 Period",
            options=["1d", "5d", "1mo"],
            index=0,
            key="multi_chart_period"
        )
        
        chart_interval = st.selectbox(
            "⏱ Interval",
            options=["1m", "5m", "15m", "30m", "60m"],
            index=2,
            key="multi_chart_interval"
        )
    
    with col3:
        chart_height = st.number_input(
            "📏 Height",
            min_value=150,
            max_value=400,
            value=250,
            step=25,
            key="chart_height"
        )
    
    with col4:
        if st.button("🔄 Refresh", type="primary", use_container_width=True):
            st.rerun()
        
        st.caption(f"**{len(selected_watchlist)}/{max_stocks}** stocks")
    
    st.markdown("---")
    
    if not selected_watchlist:
        st.info("👆 Select stocks from the dropdown to start monitoring")
    else:
        # Dynamic columns based on number of stocks
        if len(selected_watchlist) <= 2:
            num_cols = 2
        elif len(selected_watchlist) <= 4:
            num_cols = 2
        elif len(selected_watchlist) <= 9:
            num_cols = 3
        else:
            num_cols = 2  # Changed from 4 to 2 for better spacing with 10 stocks
        
        num_stocks = len(selected_watchlist)
        num_rows = (num_stocks + num_cols - 1) // num_cols
        
        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col_idx, col in enumerate(cols):
                stock_idx = row * num_cols + col_idx
                
                if stock_idx < num_stocks:
                    stock_name = selected_watchlist[stock_idx]
                    ticker = STOCK_TICKER_MAP.get(stock_name)
                    
                    with col:
                        try:
                            stock = yf.Ticker(ticker)
                            df = stock.history(period=chart_period_multi, interval=chart_interval)
                            
                            if not df.empty and len(df) > 0:
                                current_price = df['Close'].iloc[-1]
                                prev_price = df['Close'].iloc[0]
                                price_change = current_price - prev_price
                                price_change_pct = (price_change / prev_price) * 100
                                
                                if price_change >= 0:
                                    color = "green"
                                    arrow = "🟢"
                                else:
                                    color = "red"
                                    arrow = "🔴"
                                
                                st.markdown(f"### {arrow} **{stock_name}**")
                                st.metric(
                                    label="Price",
                                    value=f"₹{current_price:.2f}",
                                    delta=f"{price_change_pct:.2f}%"
                                )
                                
                                fig_mini = go.Figure()
                                fig_mini.add_trace(go.Scatter(
                                    x=df.index,
                                    y=df['Close'],
                                    mode='lines',
                                    line=dict(color=color, width=2),
                                    fill='tozeroy',
                                    fillcolor=f'rgba({"0,255,0" if color == "green" else "255,0,0"},0.1)',
                                    name='Price'
                                ))
                                
                                fig_mini.update_layout(
                                    height=chart_height,
                                    margin=dict(l=10, r=10, t=10, b=10),
                                    xaxis=dict(showgrid=True, showticklabels=True, gridcolor='rgba(128,128,128,0.2)'),
                                    yaxis=dict(showgrid=True, showticklabels=True, gridcolor='rgba(128,128,128,0.2)'),
                                    showlegend=False,
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig_mini, use_container_width=True, config={'displayModeBar': False})
                                
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.caption(f"📈 High: ₹{df['High'].max():.2f}")
                                with col_b:
                                    st.caption(f"📉 Low: ₹{df['Low'].min():.2f}")
                                
                                st.caption(f"📊 Vol: {df['Volume'].iloc[-1]:,.0f}")
                            
                            else:
                                st.warning(f"⚠️ No data for {stock_name}")
                        
                        except Exception as e:
                            st.error(f"❌ {stock_name}")
                            st.caption(f"Error: {str(e)[:50]}")
        
        st.markdown("---")
        st.caption("💡 **Tip:** Adjust the number of stocks, chart height, period, and interval using the controls above")
        st.caption("📊 **Live Data:** Charts show real-time price movements. Click 'Refresh' to update all charts")
        st.caption("⚡ **Performance:** For best performance, limit to 10 stocks or fewer")

# --------------------------
# FOOTER
# --------------------------
st.markdown("---")
st.caption("💡 Dashboard with **FinBERT AI** sentiment analysis, technical indicators, and live price charts")
st.caption("📊 Technical: RSI, MACD, AO | SMA: 20, 50, 200 | EMA: 9, 20, 50")
st.caption("🤖 Powered by FinBERT (ProsusAI) for financial sentiment analysis")
st.caption("⚠ **Disclaimer:** For educational purposes only. Not financial advice.")
