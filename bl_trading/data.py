import os
import sys
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from .config import TradingConfig

def load_trade_data(config: TradingConfig) -> tuple[pd.DataFrame, str]:
    """Load and sanitize trade history data."""
    csv_file = config.trade_file if os.path.exists(config.trade_file) else config.fallback_trade_file
    if not os.path.exists(csv_file):
        print(f"❌ Error: Neither '{config.trade_file}' nor '{config.fallback_trade_file}' found.")
        print("Please run your trading bot first to generate trade logs.")
        sys.exit(1)

    print(f"📂 Loading trade history from: {csv_file}")
    df = pd.read_csv(csv_file)
    if df.empty:
        print("❌ Error: Trade history CSV is empty.")
        sys.exit(1)

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Required columns check
    required_cols = ['Date', 'Ticker', 'Type', 'Entry_Price', 'Status']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Error: Missing required column '{col}' in CSV.")
            sys.exit(1)

    # Add deduplication
    df = df.drop_duplicates(subset=['Date', 'Ticker', 'Type', 'Entry_Price'], keep='first')

    # Date parsing
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if 'Exit_Date' not in df.columns:
        df['Exit_Date'] = pd.NaT
    else:
        df['Exit_Date'] = pd.to_datetime(df['Exit_Date'], errors='coerce')

    # Default PnL if missing
    if 'PnL' not in df.columns:
        df['PnL'] = np.nan
    df['PnL'] = pd.to_numeric(df['PnL'], errors='coerce')

    # Add PnL validation
    if 'Exit_Price' in df.columns:
        closed_trades = df[(df['Status'] == 'Closed') & df['Entry_Price'].notna() & df['Exit_Price'].notna() & df['PnL'].notna()]
        for idx, row in closed_trades.iterrows():
            if row['Type'].upper() == 'LONG':
                calc_pnl = (row['Exit_Price'] - row['Entry_Price']) / row['Entry_Price']
            elif row['Type'].upper() == 'SHORT':
                calc_pnl = (row['Entry_Price'] - row['Exit_Price']) / row['Entry_Price']
            else:
                continue
            
            if abs(row['PnL'] - calc_pnl) > 0.01:
                warnings.warn(f"PnL deviation for {row['Ticker']} on {row['Date']}: Stated {row['PnL']:.4f}, Calculated {calc_pnl:.4f}")

    # Drop any row with invalid entry date or entry price
    df = df.dropna(subset=['Date', 'Entry_Price']).copy()
    df = df[df['Entry_Price'] > 0].copy()

    # Sort strictly by entry date
    df = df.sort_values(by='Date').reset_index(drop=True)
    return df, csv_file

def fetch_benchmark_data(start_date: str, config: TradingConfig) -> pd.DataFrame:
    """Fetch historical daily data for the benchmark."""
    ticker = config.benchmark_ticker
    period = f"{config.benchmark_history_years}y"
    print(f"📡 Fetching {ticker} benchmark history...")
    try:
        bench = yf.Ticker(ticker)
        hist = bench.history(period=period)
        if hist.empty:
            raise ValueError("Empty history returned")
        hist.index = hist.index.tz_localize(None)
        return hist
    except Exception as e:
        print(f"⚠️  Warning: Failed to fetch {ticker} live data: {e}")
        return pd.DataFrame()

def fetch_current_prices(tickers: list[str]) -> dict[str, float]:
    """Batch fetch latest market prices for all open tickers in 1 fast query."""
    if not tickers:
        return {}
    
    unique_tickers = list(set(tickers))
    print(f"⚡ Batch fetching latest market prices for {len(unique_tickers)} open positions...")
    latest_prices = {}

    try:
        data = yf.download(unique_tickers, period="5d", progress=False, group_by='ticker', threads=False)
        for ticker in unique_tickers:
            try:
                if len(unique_tickers) == 1:
                    close_series = data['Close'].dropna()
                else:
                    if ticker in data.columns.levels[0]:
                        close_series = data[ticker]['Close'].dropna()
                    else:
                        close_series = pd.Series()

                if not close_series.empty:
                    latest_prices[ticker] = float(close_series.iloc[-1])
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️  Batch download failed ({e}), falling back to single ticker queries...")
        for ticker in unique_tickers:
            try:
                t = yf.Ticker(ticker)
                h = t.history(period="1d")
                if not h.empty:
                    latest_prices[ticker] = float(h['Close'].iloc[-1])
            except Exception:
                pass

    return latest_prices

def fetch_historical_prices(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Batch fetch historical close prices for concurrent simulation mode."""
    if not tickers:
        return pd.DataFrame()
    unique_tickers = list(set(tickers))
    try:
        data = yf.download(unique_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if len(unique_tickers) == 1:
            df = data[['Close']].copy()
            df.columns = unique_tickers
        else:
            df = data['Close'].copy()
        
        # Forward-fill missing data (weekends, holidays)
        df = df.ffill()
        return df
    except Exception as e:
        print(f"⚠️  Batch historical download failed ({e}), falling back to single ticker queries...")
        df_list = []
        for ticker in unique_tickers:
            try:
                t = yf.download([ticker], start=start_date, end=end_date, auto_adjust=True, progress=False)
                if not t.empty:
                    t_close = t[['Close']].copy()
                    t_close.columns = [ticker]
                    df_list.append(t_close)
            except Exception:
                pass
        if df_list:
            df = pd.concat(df_list, axis=1)
            df = df.ffill()
            return df
        return pd.DataFrame()
