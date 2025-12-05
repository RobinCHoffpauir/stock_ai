import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# --- PART 1: TECHNICAL ANALYSIS FUNCTIONS ---

def calculate_sma(data, window):
    """Calculates Simple Moving Average (SMA)."""
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- PART 2: THE VISUALIZER ---

def visualize_stock(ticker, period="1y"):
    """
    Fetches data and displays a chart with Price, SMA-50, SMA-200 and Volume.
    """
    print(f"\n[Visualizer] Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        print(f"Error: No data found for {ticker}.")
        return

    # Calculate Indicators
    df['SMA50'] = calculate_sma(df, 50)
    df['SMA200'] = calculate_sma(df, 200)

    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Price & Moving Averages
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index, df['Close'], label='Price', color='black', alpha=0.6)
    ax1.plot(df.index, df['SMA50'], label='SMA 50 (Short-term Trend)', color='blue')
    ax1.plot(df.index, df['SMA200'], label='SMA 200 (Long-term Trend)', color='red')
    ax1.set_title(f"{ticker} Price Analysis ({period})")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Volume
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.bar(df.index, df['Volume'], color='gray', alpha=0.5)
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# --- PART 3: THE STOCK PICKER / ANALYZER ---

def analyze_stock_list(tickers):
    """
    Scans a list of tickers and picks those meeting specific criteria.
    Criteria:
    1. Uptrend: Current Price > 50-day SMA
    2. Value: RSI < 70 (Not overbought)
    """
    print(f"\n[Analyzer] Scanning {len(tickers)} stocks for opportunities...")
    picked_stocks = []

    for ticker in tickers:
        try:
            # Get data (fast download)
            df = yf.download(ticker, period="6mo", progress=False)
            
            if len(df) < 50: # Skip if not enough data
                continue

            # Calculate latest metrics
            current_price = df['Close'].iloc[-1]
            sma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
            
            # Calculate RSI manually for the last data point
            rsi_series = calculate_rsi(df)
            current_rsi = rsi_series.iloc[-1]

            # --- PICKER LOGIC ---
            # You can customize these conditions!
            condition_1 = current_price > sma_50  # Bullish Trend
            condition_2 = current_rsi < 70        # Not Overbought
            condition_3 = current_rsi > 30        # Not Oversold (Healthy momentum)

            status = "NEUTRAL"
            if condition_1 and condition_2 and condition_3:
                status = "BUY SIGNAL (Trend Up)"
                picked_stocks.append({
                    'Ticker': ticker,
                    'Price': round(float(current_price), 2),
                    'SMA_50': round(float(sma_50), 2),
                    'RSI': round(float(current_rsi), 2),
                    'Signal': status
                })
            
        except Exception as e:
            print(f"Could not analyze {ticker}: {e}")

    # Display Results
    if picked_stocks:
        print("\n--- STOCK PICKER RESULTS ---")
        results_df = pd.DataFrame(picked_stocks)
        # Format the dataframe nicely
        print(results_df[['Ticker', 'Price', 'SMA_50', 'RSI', 'Signal']].to_string(index=False))
    else:
        print("No stocks matched the criteria.")

# --- PART 4: MAIN MENU ---

def main():
    while True:
        print("\n=== STOCK MARKET TOOL ===")
        print("1. Visualize a Stock")
        print("2. Run Stock Picker (Scanner)")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ")

        if choice == '1':
            t = input("Enter Ticker Symbol (e.g., AAPL, TSLA, NVDA): ").upper()
            visualize_stock(t)
        
        elif choice == '2':
            # You can add more tickers to this list!
            default_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'AMD', 'INTC', 'F', 'DIS', 'KO']
            print(f"Scanning default list: {default_list}")
            analyze_stock_list(default_list)
            
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
