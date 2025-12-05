import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from textblob import TextBlob

# --- PAGE CONFIG ---
st.set_page_config(page_title="Stock AI Dashboard", layout="wide")

# --- FUNCTIONS ---

def get_sentiment(ticker):
    """
    Fetches news and returns: Score, Label, and Article Count
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        # DEBUG: Check if news is empty
        if not news: 
            return 0, "No News Found", 0
        
        polarity_sum = 0
        count = 0
        
        for article in news:
            title = article.get('title', '')
            # simple clean up
            if title:
                analysis = TextBlob(title)
                polarity_sum += analysis.sentiment.polarity
                count += 1
            
        if count == 0: 
            return 0, "Neutral", 0
        
        avg_polarity = polarity_sum / count
        
        if avg_polarity > 0.05: status = "POSITIVE"
        elif avg_polarity < -0.05: status = "NEGATIVE"
        else: status = "NEUTRAL"
            
        return avg_polarity, status, count
    except Exception as e:
        print(e)
        return 0, "Error", 0

def get_data_summary(ticker):
    stock = yf.Ticker(ticker)
    # Fetch 1 year to ensure we have enough for EMA/SMA calculations
    df = stock.history(period="1y") 
    
    if df.empty: return None
    
    current_price = df['Close'].iloc[-1]
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    info = stock.info
    return {
        "Price": current_price,
        "RSI": current_rsi,
        "PE": info.get('trailingPE', "N/A"),
        "Sector": info.get('sector', 'Unknown')
    }

def run_ml_prediction(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="max")
    
    if len(df) < 250: return "Not enough data", 0.0
    
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    
    horizons = [2, 5, 60, 250]
    predictors = []
    
    for horizon in horizons:
        rolling = df.rolling(horizon).mean()
        ratio_col = f"Ratio_{horizon}"
        df[ratio_col] = df["Close"] / rolling["Close"]
        predictors.append(ratio_col)
        
    df = df.dropna()
    train = df.iloc[:-100]
    test = df.iloc[-100:]
    
    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    
    precision = precision_score(test["Target"], preds, zero_division=0)
    
    latest = df.iloc[[-1]][predictors]
    prediction = model.predict(latest)[0]
    direction = "UP" if prediction == 1 else "DOWN"
    
    return direction, precision

# --- ADVANCED CHART FUNCTION FOR STREAMLIT ---

def plot_advanced_chart(ticker):
    """
    Generates the matplotlib figure with Bollinger Bands and EMA.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    
    if df.empty:
        st.error("No historical data found.")
        return

    # Calculate Indicators
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA25'] = df['Close'].ewm(span=25, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])

    # Create the Plot (using fig, ax approach for Streamlit)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Price
    ax.plot(df.index, df['Close'], label='Price', color='black', alpha=0.6)
    
    # Plot Moving Averages
    ax.plot(df.index, df['EMA25'], label='EMA 25', color='green', linewidth=1.5)
    ax.plot(df.index, df['SMA50'], label='SMA 50', color='blue', linestyle='--', alpha=0.5)
    
    # Plot Bollinger Bands
    ax.plot(df.index, df['BB_Upper'], label='BB Upper', color='gray', alpha=0.3, linewidth=0.5)
    ax.plot(df.index, df['BB_Lower'], label='BB Lower', color='gray', alpha=0.3, linewidth=0.5)
    ax.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.1)
    
    ax.set_title(f"{ticker} - Advanced Technicals")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    return fig

# --- MAIN APP LAYOUT ---

st.title("📈 AI Stock Dashboard")
st.write("Enter a stock ticker to get Real-time Data, News Sentiment, and AI Predictions.")

# Sidebar
ticker_input = st.sidebar.text_input("Enter Ticker", value="NVDA").upper()
run_btn = st.sidebar.button("Run Analysis")

if run_btn:
    with st.spinner(f"Analyzing {ticker_input}..."):
        
        # 1. Fetch Data
        data = get_data_summary(ticker_input)
        
        if not data:
            st.error("Could not find data. Check ticker.")
        else:
            # 2. Run AI & Sentiment
            sent_score, sent_label, sent_count = get_sentiment(ticker_input)
            ml_dir, ml_conf = run_ml_prediction(ticker_input)

            # 3. Display Top Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Price", f"${data['Price']:.2f}")
            col2.metric("RSI", f"{data['RSI']:.1f}")
            
            # Show sentiment with article count
            col3.metric("Sentiment", f"{sent_label}", 
                        delta=f"{sent_score:.2f} ({sent_count} articles)")
            
            col4.metric("AI Forecast", ml_dir, 
                        delta=f"{ml_conf*100:.0f}% Conf.")

            st.markdown("---")

            # 4. Display Advanced Chart
            st.subheader(f"📊 Advanced Technical Chart: {ticker_input}")
            fig = plot_advanced_chart(ticker_input)
            st.pyplot(fig)  # <--- This is how we render matplotlib in Streamlit
            
            # 5. Analysis Logic
            st.subheader("📝 AI Verdict")
            
            score = 0
            reasons = []
            
            if data['RSI'] < 30: 
                score += 1
                reasons.append("RSI is Oversold (Good entry)")
            elif data['RSI'] < 70 and ml_dir == "UP":
                score += 1
                reasons.append("Healthy Momentum + AI Uptrend")
                
            if ml_dir == "UP" and ml_conf > 0.55: 
                score += 1
                reasons.append("Strong AI Prediction")
                
            if sent_label == "POSITIVE": 
                score += 1
                reasons.append("Positive News Coverage")

            if score >= 3:
                st.success(f"**STRONG BUY**: {', '.join(reasons)}")
            elif score == 2:
                st.info(f"**MODERATE BUY**: {', '.join(reasons)}")
            else:
                st.warning("**HOLD / WATCH**: Waiting for clearer signals.")