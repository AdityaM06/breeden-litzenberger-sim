import pandas as pd
import yfinance as yf
import numpy as np
from scipy.interpolate import UnivariateSpline
import datetime
import os
import warnings
import time # <--- REQUIRED for large watchlists

# --- Supress Warnings ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
PROXIMITY_THRESHOLD = 0.015
STOP_LOSS_PCT = 0.05

# --- Helper Function ---
def get_probability(pdf, k_range, lower_bound, upper_bound):
    idx = np.where((k_range >= lower_bound) & (k_range <= upper_bound))
    if len(idx[0]) == 0: return 0.0
    
    pdf_slice = pdf[idx]
    k_slice = k_range[idx]
    
    try:
        probability = np.trapezoid(pdf_slice, k_slice)
    except AttributeError:
        probability = np.trapz(pdf_slice, k_slice)
    
    return probability

def analyze_sentiment(ticker_symbol, expiration_index=0):
    stock = yf.Ticker(ticker_symbol)
    try:
        hist = stock.history(period="1d")
        if hist.empty: return None, None, None
        current_price = hist['Close'].iloc[-1]
        expirations = stock.options
        if not expirations: return None, None, None
    except: return None, None, None

    try:
        selected_date = expirations[min(expiration_index, len(expirations)-1)]
    except IndexError: return None, None, None
    
    try:
        opt_chain = stock.option_chain(selected_date)
        calls = opt_chain.calls
        calls['mid_price'] = (calls['bid'] + calls['ask']) / 2
        
        # Liquidity Filter
        calls = calls[(calls['volume'] > 0) | (calls['openInterest'] > 0)]
        calls = calls[calls['mid_price'] > 0.01]
        calls = calls.sort_values('strike')
        
        if len(calls) < 5: return None, None, None

        strikes = calls['strike'].values
        prices = calls['mid_price'].values
        
        if len(strikes) > 50: smooth_factor = len(strikes) * 1.0
        else: smooth_factor = len(strikes) * 0.5
            
        spline = UnivariateSpline(strikes, prices, k=3, s=smooth_factor)
        k_range = np.linspace(strikes.min(), strikes.max(), 500)
        second_deriv = spline.derivative(n=2)(k_range)
        pdf = np.maximum(second_deriv, 0)

        # Normalize PDF
        total_area = np.trapezoid(pdf, k_range) if hasattr(np, 'trapezoid') else np.trapz(pdf, k_range)
        if total_area > 0: pdf = pdf / total_area
        
        peak_idx = np.argmax(pdf)
        consensus_price = k_range[peak_idx]

        range_radius = consensus_price * 0.03
        prob_mass = get_probability(pdf, k_range, consensus_price - range_radius, consensus_price + range_radius)
        
        return current_price, consensus_price, prob_mass
        
    except Exception: return None, None, None

class DynamicTrader:
    def __init__(self, filename='dynamic_trades_2.0.csv'):
        self.filename = filename
        
        # CLEAN START: No migration logic. 
        # If file doesn't exist, create it with the full header immediately.
        if not os.path.exists(self.filename):
            df = pd.DataFrame(columns=[
                'Date', 'Ticker', 'Type', 'Entry_Price', 'Initial_Target', 
                'Status', 'Exit_Price', 'PnL', 'Confidence'
            ])
            df.to_csv(self.filename, index=False)
            print(f"Initialized new trade log: {self.filename}")

    def place_trade(self, ticker, signal_threshold=0.02, min_confidence=0.15):
        try:
            df = pd.read_csv(self.filename)
            # Check if ticker is already OPEN
            if not df.empty:
                existing_trade = df[(df['Ticker'] == ticker) & (df['Status'] == 'OPEN')]
                if not existing_trade.empty:
                    print(f"  --- Skipping {ticker}: Position already OPEN")
                    return
        except (FileNotFoundError, pd.errors.EmptyDataError): pass

        current, consensus, confidence = analyze_sentiment(ticker, expiration_index=2)
            
        if current is None: return

        diff_pct = (consensus - current) / current
        trade_type = "NEUTRAL"
        if diff_pct > signal_threshold: trade_type = "LONG"
        elif diff_pct < -signal_threshold: trade_type = "SHORT"
            
        if trade_type != "NEUTRAL":
            print(f"Ticker: {ticker:<6} | Gap: {diff_pct:>6.2%} | Conf: {confidence:.2%}")
            
            if confidence < min_confidence:
                print(f"  --- REJECTED: Low Confidence ({confidence:.2%} < {min_confidence:.2%})")
                return

            self._log_trade(ticker, trade_type, current, consensus, confidence)
            print(f"  >>> TRADE OPENED: {trade_type} {ticker} (High Conviction)")
        else:
            # print(f"  --- No Signal for {ticker}") # Uncomment for verbose output
            pass

    def _log_trade(self, ticker, trade_type, entry, target, confidence):
        new_row = pd.DataFrame([{
            'Date': datetime.date.today(),
            'Ticker': ticker,
            'Type': trade_type,
            'Entry_Price': entry,
            'Initial_Target': target,
            'Status': 'OPEN',
            'Exit_Price': 0.0,
            'PnL': 0.0,
            'Confidence': confidence
        }])
        # Append to CSV
        new_row.to_csv(self.filename, mode='a', header=False, index=False)

    def update_portfolio(self):
        try:
            df = pd.read_csv(self.filename)
            if df.empty: return
        except: return

        updates_made = False

        for i, row in df.iterrows():
            if row['Status'] == 'OPEN':
                ticker = row['Ticker']
                entry_price = row['Entry_Price']
                trade_type = row['Type']

                current_price, dynamic_target, _ = analyze_sentiment(ticker, expiration_index=2)
                
                if current_price is None:
                    print(f"Could not fetch update for {ticker}")
                    continue
                
                gap_to_target = abs(current_price - dynamic_target) / current_price
                close_trade = False
                exit_reason = ""

                if gap_to_target < PROXIMITY_THRESHOLD:
                    close_trade = True
                    exit_reason = "Target Hit"
                elif trade_type == 'LONG' and current_price < entry_price * (1 - STOP_LOSS_PCT):
                    close_trade = True
                    exit_reason = "Stop Loss"
                elif trade_type == 'SHORT' and current_price > entry_price * (1 + STOP_LOSS_PCT):
                    close_trade = True
                    exit_reason = "Stop Loss"

                if close_trade:
                    updates_made = True
                    df.at[i, 'Status'] = 'CLOSED'
                    df.at[i, 'Exit_Price'] = current_price
                    if trade_type == 'LONG':
                        pnl = (current_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - current_price) / entry_price
                    df.at[i, 'PnL'] = pnl
                    print(f"CLOSING {ticker} ({exit_reason}) | PnL: {pnl:.2%} | New Target Was: ${dynamic_target:.2f}")

        if updates_made:
            df.to_csv(self.filename, index=False)

# --- RUN ---
bot = DynamicTrader()

print("\n--- Updating Open Positions (Dynamic Check) ---")
bot.update_portfolio()

print("\n--- Scanning for New Trades ---")
watchlist = [
    # --- 1. THE MAGNIFICENT 7 & BIG TECH ---
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 
    'NFLX', 'ORCL', 'ADBE', 'CRM', 'IBM',

    # --- 2. SEMICONDUCTORS ---
    'AMD', 'AVGO', 'INTC', 'QCOM', 'MU', 'TSM', 'ARM', 'TXN', 'AMAT',

    # --- 3. INDICES & ETFS ---
    'SPY', 'QQQ', 'IWM', 'DIA',   
    'TLT', 'HYG',                 
    'GLD', 'SLV', 'GDX',          
    'XLE', 'XLF', 'XLK', 'XBI',   
    'ARKK', 'SMH',                

    # --- 4. FINANCIALS & FINTECH ---
    'JPM', 'BAC', 'C', 'WFC', 'GS', 'MS',  
    'V', 'MA', 'PYPL', 'SQ', 'COIN',       

    # --- 5. RETAIL & CONSUMER ---
    'WMT', 'TGT', 'COST', 'HD', 'LOW',     
    'NKE', 'SBUX', 'MCD', 'CMG',           
    'KO', 'PEP', 'PG',                     
    'DIS', 'ROKU',                         

    # --- 6. ENERGY & INDUSTRIALS ---
    'XOM', 'CVX', 'OXY', 'SLB',            
    'GE', 'CAT', 'BA', 'LMT', 'RTX', 'F', 'GM', 'UPS', 

    # --- 7. HEALTHCARE ---
    'LLY', 'UNH', 'JNJ', 'PFE', 'MRK', 'ABBV', 

    # --- 8. HIGH BETA / GROWTH ---
    'PLTR', 'UBER', 'LYFT', 'SHOP', 'SNOW', 'DDOG', 
    'DKNG', 'SOFI', 'HOOD', 'MARA', 'RIOT', 'GME', 'AMC',
    
    # --- 9. CHINA ADRs ---
    'BABA', 'JD', 'PDD', 'NIO',

    # --- 10. SPECIAL SITUATIONS ---
    'AC.TO', 'CIEN', 'AA', 'BE' 
]

print(f"Scanning {len(watchlist)} tickers for high-probability setups...")
print("-" * 60)

for stock in watchlist:
    bot.place_trade(stock)

print("\nScan Complete.")

# --- AA CHECK ---
current, consensus, conf = analyze_sentiment('AA', expiration_index=2)

if current is not None:
    print(f"\n--- AA STATUS CHECK ---")
    print(f"Current Price:    ${current:.2f}")
    print(f"Fresh Consensus:  ${consensus:.2f}")
    print(f"Confidence:       {conf:.2%}")
    print(f"Distance to Target: {abs(current - consensus):.2f}")
else:
    print("\n--- AA STATUS CHECK FAILED (No Data) ---")