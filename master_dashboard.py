import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from textblob import TextBlob
import sys

# ==========================================
# MODULE 1: SENTIMENT & FUNDAMENTALS
# ==========================================

def get_sentiment(ticker):
    """Scans news headlines for positive/negative sentiment."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0, "No News"
        
        polarity_sum = 0
        count = 0
        for article in news:
            title = article.get('title', '')
            analysis = TextBlob(title)
            polarity_sum += analysis.sentiment.polarity
            count += 1
            
        if count == 0: return 0, "Neutral"
        
        avg_polarity = polarity_sum / count
        if avg_polarity > 0.1: return avg_polarity, "POSITIVE"
        elif avg_polarity < -0.1: return avg_polarity, "NEGATIVE"
        else: return avg_polarity, "NEUTRAL"
    except:
        return 0, "Error"

def get_data_summary(ticker):
    """Fetches price, RSI, PE Ratio, and general info."""
    stock = yf.Ticker(ticker)
    
    # Get Price Data
    df = stock.history(period="6mo")
    if df.empty: return None
    
    current_price = df['Close'].iloc[-1]
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    # Get Fundamentals
    info = stock.info
    pe_ratio = info.get('trailingPE', None)
    peg_ratio = info.get('pegRatio', None)
    
    return {
        "Price": current_price,
        "RSI": current_rsi,
        "PE": pe_ratio,
        "PEG": peg_ratio,
        "Sector": info.get('sector', 'Unknown')
    }

# ==========================================
# MODULE 2: MACHINE LEARNING PREDICTOR
# ==========================================

def run_ml_prediction(ticker):
    """Trains a Random Forest model on the fly."""
    stock = yf.Ticker(ticker)
    df = stock.history(period="max")
    
    if len(df) < 250:
        return "Not enough data", 0.0

    # Data Prep
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
    
    # Train/Test Split (Last 100 days for validation)
    train = df.iloc[:-100]
    test = df.iloc[-100:]
    
    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    model.fit(train[predictors], train["Target"])
    
    # Evaluate Precision
    preds = model.predict(test[predictors])
    precision = precision_score(test["Target"], preds, zero_division=0)
    
    # Predict Tomorrow
    latest = df.iloc[[-1]][predictors]
    prediction = model.predict(latest)[0]
    
    direction = "UP" if prediction == 1 else "DOWN"
    return direction, precision

# ==========================================
# MODULE 3: VISUALIZER
# ==========================================

def show_chart(ticker):
    print(f"Generating advanced chart for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    
    if df.empty:
        print("No data to plot.")
        return

    # --- 1. CALCULATE INDICATORS ---
    
    # SMA (Simple Moving Averages)
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # EMA 25 (Exponential Moving Average)
    # span=25 makes it a 25-day EMA
    df['EMA25'] = df['Close'].ewm(span=25, adjust=False).mean()
    
    # BOLLINGER BANDS
    # Step 1: Calculate the middle band (usually 20-day SMA)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    # Step 2: Calculate Standard Deviation
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    # Step 3: Calculate Upper and Lower bands (2 std devs away)
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])

    # --- 2. PLOT THE CHART ---
    
    plt.figure(figsize=(12, 7))
    
    # Plot the Price
    plt.plot(df.index, df['Close'], label='Price', color='black', alpha=0.7, linewidth=1.5)
    
    # Plot SMA/EMA
    plt.plot(df.index, df['EMA25'], label='EMA 25 (Fast Trend)', color='green', linestyle='-', linewidth=1)
    plt.plot(df.index, df['SMA50'], label='SMA 50', color='blue', linestyle='--', alpha=0.5)
    # We often hide SMA200 if the chart gets too crowded, but let's keep it for context
    plt.plot(df.index, df['SMA200'], label='SMA 200 (Long Trend)', color='red', linestyle='--', alpha=0.5)

    # Plot Bollinger Bands
    # We plot the lines thin and transparent
    plt.plot(df.index, df['BB_Upper'], label='Bollinger Upper', color='gray', alpha=0.3, linewidth=0.8)
    plt.plot(df.index, df['BB_Lower'], label='Bollinger Lower', color='gray', alpha=0.3, linewidth=0.8)
    
    # FILL THE AREA between the bands (This makes it look professional)
    plt.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.1)

    # Styling
    plt.title(f"{ticker} - Advanced Technical Analysis")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN DASHBOARD CONTROLLER
# ==========================================

def main():
    while True:
        print("\n" + "="*50)
        print("   SUPER STOCK DASHBOARD   ")
        print("="*50)
        ticker = input("Enter Ticker (or 'q' to quit): ").upper()
        
        if ticker == 'Q':
            break
            
        print(f"\n[1/3] Gathering Fundamental & Sentiment Data for {ticker}...")
        data = get_data_summary(ticker)
        
        if not data:
            print("Error: Could not fetch data. Check ticker symbol.")
            continue
            
        sent_score, sent_label = get_sentiment(ticker)
        
        print(f"[2/3] Training AI Model (This may take a few seconds)...")
        ml_dir, ml_conf = run_ml_prediction(ticker)
        
        # --- DISPLAY REPORT ---
        print("\n" + "-"*50)
        print(f"REPORT FOR: {ticker} ({data['Sector']})")
        print("-"*50)
        
        # 1. Financials
        print(f"CURRENT PRICE:   ${data['Price']:.2f}")
        print(f"RSI (Momentum):  {data['RSI']:.1f} " + 
              ("(Oversold)" if data['RSI'] < 30 else "(Overbought)" if data['RSI'] > 70 else ""))
        print(f"P/E RATIO:       {data['PE'] if data['PE'] else 'N/A'}")
        
        # 2. Sentiment
        print(f"NEWS SENTIMENT:  {sent_label} (Score: {sent_score:.2f})")
        
        # 3. AI Prediction
        print(f"AI FORECAST:     Price will go {ml_dir}")
        print(f"AI CONFIDENCE:   {ml_conf*100:.1f}%")
        
        # 4. Final Verdict Logic
        signals = 0
        if data['RSI'] < 70 and data['RSI'] > 30: signals += 1 # Healthy momentum
        if ml_dir == "UP" and ml_conf > 0.55: signals += 1     # Strong AI signal
        if sent_label == "POSITIVE": signals += 1              # Good news
        
        print("-" * 50)
        if signals == 3:
            print(">>> OVERALL RATING: STRONG BUY (Confluence of factors)")
        elif signals == 2:
            print(">>> OVERALL RATING: MODERATE BUY")
        else:
            print(">>> OVERALL RATING: HOLD / WATCH")
        print("-" * 50)
        
        print("[3/3] Opening Chart...")
        show_chart(ticker)

if __name__ == "__main__":
    main()

