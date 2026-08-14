import re

with open('status.py', 'r') as f:
    content = f.read()

# 1. Config
config_old = """STARTING_BALANCE = 10000.0        # Initial starting bankroll in USD
POSITION_SIZE_PCT = 0.20          # Fraction of current cash committed per trade (20%)
RISK_FREE_RATE_ANNUAL = 0.045     # Risk-free rate (4.5% US 3-Month T-Bill / SOFR)"""

config_new = """STARTING_BALANCE = 10000.0        # Initial starting bankroll in USD
POSITION_SIZE_PCT = 0.05          # 5% per trade (realistic for concurrent positions)
MAX_POSITIONS = 20                # Maximum concurrent open positions
CASH_RESERVE_PCT = 0.10           # Keep 10% cash reserve
SHORT_BORROW_COST_ANNUAL = 0.03   # 3% annualized borrow cost for shorts
RISK_FREE_RATE_ANNUAL = 0.045     # Risk-free rate (4.5% US 3-Month T-Bill / SOFR)"""

content = content.replace(config_old, config_new)

# 2. Dedup
dedup_old = """    # Sort strictly by entry date
    df = df.sort_values(by='Date').reset_index(drop=True)
    return df, csv_file"""
dedup_new = """    # Sort strictly by entry date
    df = df.sort_values(by='Date').reset_index(drop=True)
    # Deduplicate identical trades on same day
    df = df.drop_duplicates(subset=['Date', 'Ticker', 'Type', 'Entry_Price'], keep='first').reset_index(drop=True)
    return df, csv_file"""
content = content.replace(dedup_old, dedup_new)

with open('status.py', 'w') as f:
    f.write(content)
print("Applied initial patches.")
