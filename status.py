import pandas as pd
import yfinance as yf
import numpy as np

# --- USER SETTINGS ---
STARTING_BALANCE = 10000
POSITION_SIZE_PCT = 0.20  # You bet 20% of your current cash per trade
RISK_FREE_RATE_ANNUAL = 0.045  # update to current T-bill/SOFR rate as needed
# ---------------------

# 1. Load Your Trade History
try:
    df = pd.read_csv('dynamic_trades_2.0.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    if 'Exit_Date' not in df.columns:
        df['Exit_Date'] = pd.NaT
    df['Exit_Date'] = pd.to_datetime(df['Exit_Date'], errors='coerce')

    df = df.sort_values(by='Date').reset_index(drop=True)
except FileNotFoundError:
    print("Error: 'dynamic_trades_2.0.csv' not found. Please run your bot first.")
    exit()

if df.empty:
    print("No trades found in CSV.")
    exit()

# 2. Benchmark Setup (SPY Buy & Hold)
print("Fetching SPY data for benchmark comparison...")
spy = yf.Ticker("SPY")
spy_hist = spy.history(period="10y")
spy_hist.index = spy_hist.index.tz_localize(None)

first_trade_date = df['Date'].min()
try:
    spy_start_price = spy_hist.asof(first_trade_date)['Close']
    spy_current_price = spy_hist['Close'].iloc[-1]
except Exception:
    spy_start_price = spy_hist['Close'].iloc[0]
    spy_current_price = spy_hist['Close'].iloc[-1]

spy_shares = STARTING_BALANCE / spy_start_price
spy_final_balance = spy_shares * spy_current_price
spy_total_return = (spy_final_balance - STARTING_BALANCE) / STARTING_BALANCE

# 3. Simulation Loop (Your Bot) -- now also records per-trade step returns
#    for both the bot and SPY-over-the-same-window, for the beta regression.
current_cash = STARTING_BALANCE
today = pd.Timestamp.today().normalize()

bot_step_returns = []   # POSITION_SIZE_PCT * trade_pnl_pct, i.e. impact on total account
spy_step_returns = []   # SPY's % return over that same trade's holding window
step_days = []          # holding period length in calendar days, for rf scaling

print(f"\n{'DATE':<12} {'TICKER':<6} {'TYPE':<6} {'RESULT':<10} {'ACCT BALANCE':<15}")
print("-" * 65)

for i, row in df.iterrows():
    exit_date = row['Exit_Date'] if pd.notna(row['Exit_Date']) else today

    if row['Status'] == 'CLOSED':
        trade_pnl_pct = row['PnL']
    else:
        try:
            curr_price = yf.Ticker(row['Ticker']).history(period='1d')['Close'].iloc[-1]
            if row['Type'] == 'LONG':
                trade_pnl_pct = (curr_price - row['Entry_Price']) / row['Entry_Price']
            else:
                trade_pnl_pct = (row['Entry_Price'] - curr_price) / row['Entry_Price']
        except Exception:
            trade_pnl_pct = 0.0

    bet_amount = current_cash * POSITION_SIZE_PCT
    profit_dollars = bet_amount * trade_pnl_pct
    current_cash += profit_dollars

    # --- NEW: SPY's return over this same holding window, for beta ---
    try:
        spy_entry_px = spy_hist.asof(row['Date'])['Close']
        spy_exit_px = spy_hist.asof(exit_date)['Close']
        spy_window_return = (spy_exit_px - spy_entry_px) / spy_entry_px
    except Exception:
        spy_window_return = np.nan

    bot_step_returns.append(POSITION_SIZE_PCT * trade_pnl_pct)
    spy_step_returns.append(spy_window_return)
    step_days.append(max((exit_date - row['Date']).days, 1))

    status_str = f"{trade_pnl_pct:+.2%}"
    print(f"{row['Date'].strftime('%Y-%m-%d'):<12} {row['Ticker']:<6} {row['Type']:<6} {status_str:<10} ${current_cash:,.2f}")

# 4. Final Math
bot_total_return = (current_cash - STARTING_BALANCE) / STARTING_BALANCE
alpha = bot_total_return - spy_total_return

# --- NEW: Beta and Jensen's (risk-adjusted) Alpha ---
step_df = pd.DataFrame({
    'bot': bot_step_returns,
    'spy': spy_step_returns,
    'days': step_days
}).dropna()

if len(step_df) >= 5:
    rf_per_step = RISK_FREE_RATE_ANNUAL / 365 * step_df['days']
    excess_bot = step_df['bot'] - rf_per_step
    excess_spy = step_df['spy'] - rf_per_step

    beta, jensen_alpha_per_step = np.polyfit(excess_spy, excess_bot, 1)
    predicted = jensen_alpha_per_step + beta * excess_spy
    ss_res = np.sum((excess_bot - predicted) ** 2)
    ss_tot = np.sum((excess_bot - excess_bot.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    avg_days_per_step = step_df['days'].mean()
    steps_per_year = 365 / avg_days_per_step
    jensen_alpha_annual = (1 + jensen_alpha_per_step) ** steps_per_year - 1
else:
    beta, jensen_alpha_annual, r_squared = np.nan, np.nan, np.nan
    print("\n(Not enough closed/dated trades yet for a reliable beta regression -- need 5+.)")

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
print(f"NET ALPHA (naive):   {alpha:+.2%}   <- ignores risk taken")
print(f"BETA:                {beta:.2f}    <- sensitivity to SPY (1.0 = market-like)")
print(f"JENSEN ALPHA (ann.): {jensen_alpha_annual:+.2%}   <- excess return AFTER stripping out beta")
print(f"R-SQUARED:           {r_squared:.2f}    <- how much of your variance SPY explains")
print("-" * 45)

diff = current_cash - spy_final_balance
if diff > 0:
    print(f"✅ SUCCESS: You beat the market by ${diff:,.2f}!")
    print(f"   Naive Alpha of {alpha:+.2%} indicates outperformance.")
else:
    print(f"❌ UNDERPERFORMANCE: You lagged the market by ${abs(diff):,.2f}.")

if not np.isnan(beta) and beta > 1.3:
    print(f"\n⚠️  Beta of {beta:.2f} means a large chunk of your naive alpha is likely just")
    print(f"    leveraged market exposure, not skill. Jensen Alpha ({jensen_alpha_annual:+.2%})")
    print(f"    is the more honest number for 'value the strategy itself adds.'")
print("=" * 45)