# --------------------------
# EARNINGS Functions - Q3 FY26 (Oct-Dec 2025, Results in Jan-Feb 2026)
# --------------------------
@st.cache_data(ttl=1800)
def fetch_q3_fy26_earnings():
    """Generate Q3 FY26 earnings calendar - Oct-Dec quarter results"""
    earnings_list = []
    
    # Q3 FY26 is Oct-Dec 2025, results announced from Jan 10, 2026 onwards
    # Current date is Nov 12, 2025, so we include dates from Jan 10 to Feb 28, 2026
    
    # Companies reporting on Jan 13, 14, 15, 2026
    jan_13_companies = ["Reliance", "TCS", "HDFC Bank", "Infosys", "ICICI Bank", "Bharti Airtel"]
    jan_14_companies = ["ITC", "SBI", "Hindustan Unilever", "Bajaj Finance", "Kotak Mahindra Bank"]
    jan_15_companies = ["Axis Bank", "Larsen & Toubro", "Asian Paints", "Maruti Suzuki", "Titan"]
    
    # Add Jan 13 companies
    for company in jan_13_companies:
        earnings_list.append({
            'Company': company,
            'Quarter': 'Q3 FY26 (Oct-Dec 2025)',
            'Expected Date': '13-Jan-2026',
            'Day': 'Tuesday',
            'Status': 'Scheduled'
        })
    
    # Add Jan 14 companies
    for company in jan_14_companies:
        earnings_list.append({
            'Company': company,
            'Quarter': 'Q3 FY26 (Oct-Dec 2025)',
            'Expected Date': '14-Jan-2026',
            'Day': 'Wednesday',
            'Status': 'Scheduled'
        })
    
    # Add Jan 15 companies
    for company in jan_15_companies:
        earnings_list.append({
            'Company': company,
            'Quarter': 'Q3 FY26 (Oct-Dec 2025)',
            'Expected Date': '15-Jan-2026',
            'Day': 'Thursday',
            'Status': 'Scheduled'
        })
    
    # Add remaining F&O companies across Jan-Feb 2026
    reported_companies = set(jan_13_companies + jan_14_companies + jan_15_companies)
    remaining_companies = [c for c in FNO_STOCKS if c not in reported_companies]
    
    base_date = datetime(2026, 1, 10)
    
    for i, stock in enumerate(remaining_companies):
        # Distribute across Jan 10 - Feb 28
        days_offset = (i * 2) % 50  # Spread across 50 days
        result_date = base_date + timedelta(days=days_offset)
        
        # Skip weekends
        while result_date.weekday() >= 5:
            result_date += timedelta(days=1)
        
        # Skip if already past Feb 28
        if result_date > datetime(2026, 2, 28):
            result_date = datetime(2026, 1, 10) + timedelta(days=(i % 10))
            while result_date.weekday() >= 5:
                result_date += timedelta(days=1)
        
        earnings_list.append({
            'Company': stock,
            'Quarter': 'Q3 FY26 (Oct-Dec 2025)',
            'Expected Date': result_date.strftime('%d-%b-%Y'),
            'Day': result_date.strftime('%A'),
            'Status': 'Estimated'
        })
    
    # Sort by date
    earnings_list.sort(key=lambda x: datetime.strptime(x['Expected Date'], '%d-%b-%Y'))
    
    return earnings_list
