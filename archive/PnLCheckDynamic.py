import pandas as pd
import yfinance as yf
import numpy as np

# --- USER SETTINGS ---
STARTING_BALANCE = 10000 
POSITION_SIZE_PCT = 0.20  # You bet 20% of your current cash per trade
# ---------------------

# 1. Load Your Trade History
try:
    df = pd.read_csv('dynamic_trades_2.0.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Handle optional columns
    if 'Exit_Date' not in df.columns: df['Exit_Date'] = pd.NaT
    df['Exit_Date'] = pd.to_datetime(df['Exit_Date'])
    
    # Sort by date to simulate the timeline correctly
    df = df.sort_values(by='Date').reset_index(drop=True)
except FileNotFoundError:
    print("Error: 'dynamic_trades.csv' not found. Please run your bot first.")
    exit()

if df.empty:
    print("No trades found in CSV.")
    exit()

# 2. Benchmark Setup (SPY Buy & Hold)
print("Fetching SPY data for benchmark comparison...")
spy = yf.Ticker("SPY")
spy_hist = spy.history(period="10y") # Get enough history
spy_hist.index = spy_hist.index.tz_localize(None)

# Find SPY Price on the day of your FIRST trade
first_trade_date = df['Date'].min()
try:
    # Try to find exact date, or nearest previous date
    spy_start_price = spy_hist.asof(first_trade_date)['Close']
    spy_current_price = spy_hist['Close'].iloc[-1]
except:
    # Fallback if history is too short or date issues
    spy_start_price = spy_hist['Close'].iloc[0] 
    spy_current_price = spy_hist['Close'].iloc[-1]

# Calculate "If I just bought SPY" value
spy_shares = STARTING_BALANCE / spy_start_price
spy_final_balance = spy_shares * spy_current_price
spy_total_return = (spy_final_balance - STARTING_BALANCE) / STARTING_BALANCE

# 3. Simulation Loop (Your Bot)
current_cash = STARTING_BALANCE

print(f"\n{'DATE':<12} {'TICKER':<6} {'TYPE':<6} {'RESULT':<10} {'ACCT BALANCE':<15}")
print("-" * 65)

for i, row in df.iterrows():
    # Determine the PnL of this specific trade
    if row['Status'] == 'CLOSED':
        trade_pnl_pct = row['PnL']
    else:
        # If trade is OPEN, calculate current floating PnL
        try:
            curr_price = yf.Ticker(row['Ticker']).history(period='1d')['Close'].iloc[-1]
            if row['Type'] == 'LONG':
                trade_pnl_pct = (curr_price - row['Entry_Price']) / row['Entry_Price']
            else:
                trade_pnl_pct = (row['Entry_Price'] - curr_price) / row['Entry_Price']
        except:
            trade_pnl_pct = 0.0

    # Calculate impact on account
    bet_amount = current_cash * POSITION_SIZE_PCT
    profit_dollars = bet_amount * trade_pnl_pct
    
    # Update Cash
    current_cash += profit_dollars
    
    # Visual Log
    status_str = f"{trade_pnl_pct:+.2%}"
    print(f"{row['Date'].strftime('%Y-%m-%d'):<12} {row['Ticker']:<6} {row['Type']:<6} {status_str:<10} ${current_cash:,.2f}")

# 4. Final Math
bot_total_return = (current_cash - STARTING_BALANCE) / STARTING_BALANCE
alpha = bot_total_return - spy_total_return

# 5. Final Report
print("\n" + "=" * 45)
print(f" FINAL PERFORMANCE REPORT")
print("=" * 45)
print(f"{'Metric':<20} {'YOUR BOT':<12} {'S&P 500 (SPY)'}")
print("-" * 45)
print(f"{'Start Date':<20} {first_trade_date.strftime('%Y-%m-%d'):<12} {first_trade_date.strftime('%Y-%m-%d')}")
print(f"{'Start Balance':<20} ${STARTING_BALANCE:,.2f}   ${STARTING_BALANCE:,.2f}")
print(f"{'End Balance':<20} ${current_cash:,.2f}   ${spy_final_balance:,.2f}")
print(f"{'Total Return':<20} {bot_total_return:+.2%}      {spy_total_return:+.2%}")
print("-" * 45)
print(f"NET ALPHA (Skill):   {alpha:+.2%}")
print("-" * 45)

# 6. The Verdict
diff = current_cash - spy_final_balance
if diff > 0:
    print(f"✅ SUCCESS: You beat the market by ${diff:,.2f}!")
    print(f"   Your Alpha of {alpha:+.2%} indicates your strategy adds value.")
else:
    print(f"❌ UNDERPERFORMANCE: You lagged the market by ${abs(diff):,.2f}.")
    print(f"   Negative Alpha ({alpha:+.2%}) suggests Buy & Hold was better.")
print("=" * 45)