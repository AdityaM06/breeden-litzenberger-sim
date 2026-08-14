import pandas as pd
import yfinance as yf
import numpy as np
from scipy.interpolate import UnivariateSpline
import datetime
import os

# --- 1. The Core Analysis Function (Modified to return data, not plot) ---
def analyze_sentiment(ticker_symbol, expiration_index=0):
    stock = yf.Ticker(ticker_symbol)
    try:
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        expirations = stock.options
    except:
        return None, None

    if not expirations:
        return None, None

    # Grab a slightly further out expiration to ensure liquidity
    # Using index 1 or 2 often gives better "swing trade" signals than 0 (which might be tomorrow)
    selected_date = expirations[min(expiration_index, len(expirations)-1)]
    
    try:
        opt_chain = stock.option_chain(selected_date)
        calls = opt_chain.calls
        calls['mid_price'] = (calls['bid'] + calls['ask']) / 2
        calls = calls[(calls['volume'] > 0) | (calls['openInterest'] > 0)]
        calls = calls[calls['mid_price'] > 0.01].sort_values('strike')
        
        if len(calls) < 5: return None, None

        strikes = calls['strike'].values
        prices = calls['mid_price'].values
        
        # Smoothing
        if len(strikes) > 50:
            smooth_factor = len(strikes) * 1.0
        else:
            smooth_factor = len(strikes) * 0.5
        spline = UnivariateSpline(strikes, prices, k=3, s=smooth_factor)
        k_range = np.linspace(strikes.min(), strikes.max(), 500)
        
        # Derivatives
        second_deriv = spline.derivative(n=2)(k_range)
        pdf = np.maximum(second_deriv, 0)
        
        # Find Peak (Consensus)
        peak_idx = np.argmax(pdf)
        consensus_price = k_range[peak_idx]
        
        return current_price, consensus_price
        
    except Exception as e:
        print(f"Calc Error: {e}")
        return None, None

# --- 2. The Portfolio Manager ---
class PaperTrader:
    def __init__(self, filename='paper_trades.csv'):
        self.filename = filename
        if not os.path.exists(self.filename):
            # Create the log file if it doesn't exist
            df = pd.DataFrame(columns=['Date', 'Ticker', 'Type', 'Entry_Price', 'Target_Price', 'Status', 'Exit_Price', 'PnL'])
            df.to_csv(self.filename, index=False)
    
    def place_trade(self, ticker, signal_threshold=0.02):
        # --- NEW LOGIC: Check for existing positions first ---
        try:
            df = pd.read_csv(self.filename)
            # Filter for this ticker AND status 'OPEN'
            existing_trade = df[(df['Ticker'] == ticker) & (df['Status'] == 'OPEN')]
            
            if not existing_trade.empty:
                # If we find one, we skip analysis entirely to save resources
                print(f"  --- Skipping {ticker}: Position already OPEN")
                return
        except (FileNotFoundError, pd.errors.EmptyDataError):
            # If file doesn't exist yet, just proceed
            pass

        # --- EXISTING LOGIC: Proceed with analysis ---
        # 1. Analyze fresh data
        # (Make sure to unpack 3 values now if you updated the return statement previously)
        try:
            current, consensus = analyze_sentiment(ticker, expiration_index=2)
        except ValueError:
            # Handle case where analyze_sentiment returns 3 values (price, consensus, date)
            current, consensus, _ = analyze_sentiment(ticker, expiration_index=2)
            
        if current is None:
            print(f"Skipping {ticker}: No data")
            return

        # 2. Calculate implied move
        diff_pct = (consensus - current) / current
        
        trade_type = "NEUTRAL"
        if diff_pct > signal_threshold: trade_type = "LONG"
        elif diff_pct < -signal_threshold: trade_type = "SHORT"
            
        if trade_type != "NEUTRAL":
            print(f"Ticker: {ticker} | Price: ${current:.2f} | Consensus: ${consensus:.2f} | Gap: {diff_pct:.2%}")
            self._log_trade(ticker, trade_type, current, consensus)
            print(f"  >>> TRADE OPENED: {trade_type} {ticker}")
        else:
            print(f"  --- No Signal for {ticker} (Gap too small)")
            
    def _log_trade(self, ticker, trade_type, entry, target):
        new_row = pd.DataFrame([{
            'Date': datetime.date.today(),
            'Ticker': ticker,
            'Type': trade_type,
            'Entry_Price': entry,
            'Target_Price': target,
            'Status': 'OPEN',
            'Exit_Price': 0.0,
            'PnL': 0.0
        }])
        # Append to CSV
        new_row.to_csv(self.filename, mode='a', header=False, index=False)

    def update_portfolio(self):
        """Check open trades to see if they hit targets or stop losses"""
        try:
            df = pd.read_csv(self.filename)
        except pd.errors.EmptyDataError:
            return

        if df.empty: return

        # Loop through OPEN trades
        for i, row in df.iterrows():
            if row['Status'] == 'OPEN':
                ticker = row['Ticker']
                try:
                    # Get live price
                    live_price = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
                    
                    # Simple Exit Logic: Close if we hit the Consensus Target OR if we lose 5% (Stop Loss)
                    entry = row['Entry_Price']
                    target = row['Target_Price']
                    trade_type = row['Type']
                    
                    close_trade = False
                    
                    # LONG Logic
                    if trade_type == 'LONG':
                        if live_price >= target: # Take Profit
                            close_trade = True
                        elif live_price < entry * 0.95: # Stop Loss
                            close_trade = True
                            
                    # SHORT Logic
                    elif trade_type == 'SHORT':
                        if live_price <= target: # Take Profit
                            close_trade = True
                        elif live_price > entry * 1.05: # Stop Loss
                            close_trade = True
                    
                    if close_trade:
                        df.at[i, 'Status'] = 'CLOSED'
                        df.at[i, 'Exit_Price'] = live_price
                        
                        # Calculate Profit
                        if trade_type == 'LONG':
                            pnl = (live_price - entry) / entry
                        else:
                            pnl = (entry - live_price) / entry
                        
                        df.at[i, 'PnL'] = pnl
                        print(f"Closing Trade: {ticker} | PnL: {pnl:.2%}")
                
                except Exception as e:
                    print(f"Could not update {ticker}: {e}")

        df.to_csv(self.filename, index=False)

# --- RUN THE SIMULATION ---
bot = PaperTrader()

# 1. Update old trades
bot.update_portfolio()

# 2. Look for new trades
watchlist = ['AC.TO', 'AVGO', 'TSLA', 'NVDA', 'AAPL', 'GOOG', 'MSFT', 'AMZN', 'ORCL', 'CIEN', 'AA', 'AMD', 'META', 'NFLX', 'SPY', 'QQQ', 'IWM']
print("\n--- Scanning Market ---")
for stock in watchlist:
    bot.place_trade(stock)