import schedule
import time
import datetime
import os
import sys

# Import functions from your other file
# This works because both files are in the same folder
from master_dashboard import get_data_summary, run_ml_prediction, get_sentiment

# Define the list of stocks to track automatically
WATCHLIST = ['AAPL', 'NVDA', 'TSLA', 'SPY', 'AMD', 'GOOGL']

def generate_daily_report():
    """Runs the analysis and saves it to a text file."""
    
    # Create a filename based on today's date
    today = datetime.date.today()
    filename = f"reports/report_{today}.txt"
    
    # Ensure the 'reports' folder exists
    os.makedirs("reports", exist_ok=True)
    
    print(f"\n[Auto-Runner] Starting analysis for {today}...")
    
    with open(filename, "w") as f:
        # Header
        f.write(f"DAILY STOCK REPORT: {today}\n")
        f.write("="*50 + "\n\n")
        
        for ticker in WATCHLIST:
            print(f"Analyzing {ticker}...", end="\r")
            
            try:
                # 1. Get Data
                data = get_data_summary(ticker)
                if not data:
                    f.write(f"Could not fetch data for {ticker}\n\n")
                    continue

                # 2. Get Sentiment
                sent_score, sent_label = get_sentiment(ticker)
                
                # 3. Get AI Prediction
                ml_dir, ml_conf = run_ml_prediction(ticker)
                
                # 4. Write to file
                f.write(f"TICKER: {ticker}\n")
                f.write(f"Price: ${data['Price']:.2f} | RSI: {data['RSI']:.1f}\n")
                f.write(f"Sentiment: {sent_label} ({sent_score:.2f})\n")
                f.write(f"AI Forecast: {ml_dir} ({ml_conf*100:.1f}% confidence)\n")
                
                # Simple logic for "Star Rating" in the report
                score = 0
                if ml_dir == "UP" and ml_conf > 0.55: score += 1
                if sent_label == "POSITIVE": score += 1
                if 30 < data['RSI'] < 70: score += 1
                
                if score == 3: f.write("RATING: *** STRONG BUY ***\n")
                elif score == 2: f.write("RATING: ** BUY **\n")
                else: f.write("RATING: HOLD/WATCH\n")
                
                f.write("-" * 30 + "\n")
                
            except Exception as e:
                f.write(f"Error analyzing {ticker}: {e}\n")
        
        f.write("\nEnd of Report.")
        
    print(f"\n[Success] Report saved to {filename}")

# --- SCHEDULING ---

# Schedule the job to run every day at 09:00 AM
# You can change this time!
schedule.every().day.at("09:00").do(generate_daily_report)

# Also run it once immediately so you can see if it works
print("Running immediate test...")
generate_daily_report()

print("\nScheduler is running. Waiting for 9:00 AM... (Press Ctrl+C to stop)")

while True:
    schedule.run_pending()
    time.sleep(60) # Check every minute

