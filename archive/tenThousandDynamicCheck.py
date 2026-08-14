import pandas as pd
import yfinance as yf

# --- CONFIGURATION ---
STARTING_BALANCE = 10000  # Example: You have $10k in the account
TRADE_SIZE = 1000         # Example: You put $1k into each trade

# Load file
try:
    df = pd.read_csv('dynamic_trades.csv')
except FileNotFoundError:
    print("No portfolio file found.")
    exit()

total_pnl_dollars = 0.0
win_count = 0
loss_count = 0
open_trades = 0

print(f"{'TICKER':<8} {'STATUS':<8} {'RESULT':<10} {'PnL ($)'}")
print("-" * 40)

for i, row in df.iterrows():
    pnl_pct = 0.0
    
    if row['Status'] == 'CLOSED':
        pnl_pct = row['PnL']
        
    elif row['Status'] == 'OPEN':
        open_trades += 1
        try:
            # Fetch live price for open trades
            current_price = yf.Ticker(row['Ticker']).history(period='1d')['Close'].iloc[-1]
            entry = row['Entry_Price']
            if row['Type'] == 'LONG':
                pnl_pct = (current_price - entry) / entry
            else:
                pnl_pct = (entry - current_price) / entry
        except:
            pnl_pct = 0.0 # Error fetching price

    # Calculate Dollar PnL based on fixed trade size
    pnl_dollars = pnl_pct * TRADE_SIZE
    total_pnl_dollars += pnl_dollars
    
    # Track Wins/Losses
    if pnl_pct > 0: win_count += 1
    elif pnl_pct < 0: loss_count += 1

    print(f"{row['Ticker']:<8} {row['Status']:<8} {pnl_pct:>8.2%}   ${pnl_dollars:>7.2f}")

print("-" * 40)
print(f"Total Trades:      {win_count + loss_count}")
print(f"Win/Loss Ratio:    {win_count}W - {loss_count}L")
print(f"Net Profit/Loss:   ${total_pnl_dollars:.2f}")
print(f"Account Growth:    {(total_pnl_dollars / STARTING_BALANCE):.2%}")
