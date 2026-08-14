import pandas as pd
import yfinance as yf
import numpy as np
import scipy.stats as stats
import warnings
import sys
import os

warnings.filterwarnings("ignore")

# =====================================================================
# ⚙️ USER CONFIGURATION
# =====================================================================
STARTING_BALANCE = 10000.0        # Initial starting bankroll in USD
POSITION_SIZE_PCT = 0.20          # Fraction of current cash committed per trade (20%)
RISK_FREE_RATE_ANNUAL = 0.045     # Risk-free rate (4.5% US 3-Month T-Bill / SOFR)
BENCHMARK_TICKER = "SPY"          # Benchmark asset for market comparison
TRADES_CSV = "dynamic_trades_2.0.csv"  # Primary trade history file
FALLBACK_CSV = "dynamic_trades.csv"    # Fallback if primary is missing
MAX_DISPLAY_TRADES = 20           # Max trades shown in initial audit table (0 for all)
# =====================================================================


def fmt_pct(v, decimals=2, sign=False):
    """Format floating point number as percentage."""
    if v is None or (isinstance(v, (int, float, np.number)) and np.isnan(v)):
        return "N/A"
    try:
        fmt = f"+.{decimals}%" if sign else f".{decimals}%"
        return f"{float(v):{fmt}}"
    except Exception:
        return "N/A"


def fmt_curr(v, sign=False):
    """Format floating point number as currency."""
    if v is None or (isinstance(v, (int, float, np.number)) and np.isnan(v)):
        return "N/A"
    try:
        return f"${float(v):+,.2f}" if sign else f"${float(v):,.2f}"
    except Exception:
        return "N/A"


def fmt_num(v, decimals=2, sign=False):
    """Format floating point number as decimal string."""
    if v is None or (isinstance(v, (int, float, np.number)) and np.isnan(v)):
        return "N/A"
    try:
        fmt = f"+.{decimals}f" if sign else f".{decimals}f"
        return f"{float(v):{fmt}}"
    except Exception:
        return "N/A"


def load_trade_data():
    """Load and sanitize trade history data."""
    csv_file = TRADES_CSV if os.path.exists(TRADES_CSV) else FALLBACK_CSV
    if not os.path.exists(csv_file):
        print(f"❌ Error: Neither '{TRADES_CSV}' nor '{FALLBACK_CSV}' found.")
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

    # Drop any row with invalid entry date or entry price
    df = df.dropna(subset=['Date', 'Entry_Price']).copy()
    df = df[df['Entry_Price'] > 0].copy()

    # Sort strictly by entry date
    df = df.sort_values(by='Date').reset_index(drop=True)
    return df, csv_file


def fetch_benchmark_data(start_date, ticker="SPY"):
    """Fetch historical daily data for the benchmark."""
    print(f"📡 Fetching {ticker} benchmark history...")
    try:
        bench = yf.Ticker(ticker)
        hist = bench.history(period="10y")
        if hist.empty:
            raise ValueError("Empty history returned")
        hist.index = hist.index.tz_localize(None)
        return hist
    except Exception as e:
        print(f"⚠️  Warning: Failed to fetch {ticker} live data: {e}")
        return pd.DataFrame()


def fetch_open_prices_batch(open_tickers):
    """Batch fetch latest market prices for all open tickers in 1 fast query."""
    if not open_tickers:
        return {}
    
    unique_tickers = list(set(open_tickers))
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


def simulate_trading(df, spy_hist):
    """
    Run realistic sequential portfolio accounting:
    - Compounding position sizing (POSITION_SIZE_PCT of current cash)
    - Mark-to-market for open positions
    - Tracking step returns and matched benchmark returns
    """
    today = pd.Timestamp.today().normalize()
    
    # Identify open tickers and batch fetch current quotes
    open_mask = df['Status'] != 'CLOSED'
    open_tickers = df.loc[open_mask, 'Ticker'].dropna().tolist()
    live_prices = fetch_open_prices_batch(open_tickers)

    # State variables
    current_cash = STARTING_BALANCE
    equity_curve = [STARTING_BALANCE]
    equity_dates = [df['Date'].min()]
    
    trade_results = []
    bot_step_returns = []
    spy_step_returns = []
    holding_days_list = []
    
    spy_available = not spy_hist.empty
    if spy_available:
        try:
            spy_start_px = float(spy_hist.asof(df['Date'].min())['Close'])
            spy_latest_px = float(spy_hist['Close'].iloc[-1])
        except Exception:
            spy_start_px = float(spy_hist['Close'].iloc[0])
            spy_latest_px = float(spy_hist['Close'].iloc[-1])
    else:
        spy_start_px = 100.0
        spy_latest_px = 100.0

    print("\n" + "=" * 80)
    print(" 📋 REAL-TIME TRADE LOG & SIMULATION (Sample View)")
    print("=" * 80)
    print(f"{'#':<4} {'DATE':<11} {'TICKER':<6} {'TYPE':<6} {'STATUS':<7} {'ENTRY':<9} {'EXIT/CURR':<10} {'PnL %':<9} {'PROFIT ($)':<12} {'BALANCE':<12}")
    print("-" * 80)

    total_rows = len(df)
    for i, row in df.iterrows():
        is_closed = (row['Status'] == 'CLOSED')
        entry_price = float(row['Entry_Price'])
        trade_type = str(row['Type']).upper().strip()
        ticker = str(row['Ticker']).strip()
        entry_date = row['Date']
        
        # Determine exit date and price
        if is_closed:
            exit_date = row['Exit_Date'] if pd.notna(row['Exit_Date']) else (entry_date + pd.Timedelta(days=1))
            if pd.notna(row['PnL']) and row['PnL'] != 0.0:
                trade_pnl_pct = float(row['PnL'])
                if 'Exit_Price' in row and pd.notna(row['Exit_Price']) and float(row['Exit_Price']) > 0:
                    exit_price = float(row['Exit_Price'])
                else:
                    exit_price = entry_price * (1 + trade_pnl_pct if trade_type == 'LONG' else 1 - trade_pnl_pct)
            elif 'Exit_Price' in row and pd.notna(row['Exit_Price']) and float(row['Exit_Price']) > 0:
                exit_price = float(row['Exit_Price'])
                if trade_type == 'LONG':
                    trade_pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    trade_pnl_pct = (entry_price - exit_price) / entry_price
            else:
                trade_pnl_pct = 0.0
                exit_price = entry_price
        else:
            exit_date = today
            exit_price = live_prices.get(ticker, entry_price)
            if trade_type == 'LONG':
                trade_pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
            else:
                trade_pnl_pct = (entry_price - exit_price) / entry_price if entry_price > 0 else 0.0

        # Holding days
        h_days = max((exit_date - entry_date).days, 1)
        holding_days_list.append(h_days)

        # Portfolio math (compounded allocation)
        bet_amount = current_cash * POSITION_SIZE_PCT
        profit_dollars = bet_amount * trade_pnl_pct
        current_cash += profit_dollars
        
        equity_curve.append(current_cash)
        equity_dates.append(exit_date)
        
        # Step return of portfolio
        step_ret = profit_dollars / (current_cash - profit_dollars) if (current_cash - profit_dollars) > 0 else 0.0
        bot_step_returns.append(step_ret)

        # Matched SPY window return
        if spy_available:
            try:
                spy_entry = float(spy_hist.asof(entry_date)['Close'])
                spy_exit = float(spy_hist.asof(exit_date)['Close'])
                spy_win_ret = (spy_exit - spy_entry) / spy_entry
            except Exception:
                spy_win_ret = np.nan
        else:
            spy_win_ret = np.nan
        spy_step_returns.append(spy_win_ret)

        trade_record = {
            'Index': i + 1,
            'Date': entry_date,
            'Exit_Date': exit_date,
            'Ticker': ticker,
            'Type': trade_type,
            'Status': 'CLOSED' if is_closed else 'OPEN',
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'PnL_Pct': trade_pnl_pct,
            'Profit_Dollars': profit_dollars,
            'Balance_After': current_cash,
            'Holding_Days': h_days,
            'Step_Return': step_ret,
            'SPY_Window_Return': spy_win_ret
        }
        trade_results.append(trade_record)

        # Print visual row (first 10 and last 10 if long list)
        should_print = (MAX_DISPLAY_TRADES == 0 or total_rows <= MAX_DISPLAY_TRADES or 
                        i < 10 or i >= total_rows - 10)
        if should_print:
            status_tag = "CLOSED" if is_closed else "OPEN*"
            pnl_color_str = fmt_pct(trade_pnl_pct, sign=True)
            print(f"{i+1:<4} {entry_date.strftime('%Y-%m-%d'):<11} {ticker:<6} {trade_type:<6} {status_tag:<7} ${entry_price:<8.2f} ${exit_price:<9.2f} {pnl_color_str:<9} ${profit_dollars:+10.2f} ${current_cash:11.2f}")
        elif i == 10:
            print(f"... [{total_rows - 20} intermediate trades omitted for brevity] ...")

    print("-" * 80)
    print("(* Note: OPEN positions marked-to-market using latest live market prices)\n")

    sim_df = pd.DataFrame(trade_results)
    
    # SPY benchmark final calculations
    spy_shares = STARTING_BALANCE / spy_start_px
    spy_final_balance = spy_shares * spy_latest_px
    spy_total_return = (spy_final_balance - STARTING_BALANCE) / STARTING_BALANCE

    return sim_df, equity_curve, equity_dates, current_cash, spy_final_balance, spy_total_return, spy_start_px, spy_latest_px


def calculate_comprehensive_metrics(sim_df, equity_curve, equity_dates, final_cash, spy_final_balance, spy_total_return, spy_hist):
    """Calculate all quantitative trading performance, risk, and statistical metrics."""
    metrics = {}
    
    # -------------------------------------------------------------
    # 1. TIME HORIZON & CORE RETURNS
    # -------------------------------------------------------------
    start_date = sim_df['Date'].min()
    end_date = max(sim_df['Exit_Date'].max(), pd.Timestamp.today().normalize())
    total_days = max((end_date - start_date).days, 1)
    years = total_days / 365.25

    bot_total_return = (final_cash - STARTING_BALANCE) / STARTING_BALANCE
    bot_net_profit = final_cash - STARTING_BALANCE
    spy_net_profit = spy_final_balance - STARTING_BALANCE
    
    # CAGR
    if 1 + bot_total_return > 0:
        bot_cagr = (1 + bot_total_return) ** (1 / years) - 1
    else:
        bot_cagr = -1.0

    if 1 + spy_total_return > 0:
        spy_cagr = (1 + spy_total_return) ** (1 / years) - 1
    else:
        spy_cagr = -1.0

    naive_alpha = bot_total_return - spy_total_return
    annualized_active_return = bot_cagr - spy_cagr

    metrics['start_date'] = start_date
    metrics['end_date'] = end_date
    metrics['total_days'] = total_days
    metrics['years'] = years
    metrics['bot_total_return'] = bot_total_return
    metrics['bot_net_profit'] = bot_net_profit
    metrics['bot_cagr'] = bot_cagr
    metrics['spy_total_return'] = spy_total_return
    metrics['spy_net_profit'] = spy_net_profit
    metrics['spy_cagr'] = spy_cagr
    metrics['naive_alpha'] = naive_alpha
    metrics['annualized_active_return'] = annualized_active_return
    metrics['ending_balance'] = final_cash
    metrics['spy_final_balance'] = spy_final_balance

    # -------------------------------------------------------------
    # 2. TRADE COUNTS & WIN/LOSS RATIOS
    # -------------------------------------------------------------
    total_trades = len(sim_df)
    closed_trades = int((sim_df['Status'] == 'CLOSED').sum())
    open_trades = int((sim_df['Status'] == 'OPEN').sum())

    trade_pnls = sim_df['PnL_Pct']
    dollar_profits = sim_df['Profit_Dollars']

    winning_trades = sim_df[trade_pnls > 0]
    losing_trades = sim_df[trade_pnls < 0]
    breakeven_trades = sim_df[trade_pnls == 0]

    num_wins = len(winning_trades)
    num_losses = len(losing_trades)
    num_breakeven = len(breakeven_trades)

    win_rate = num_wins / total_trades if total_trades > 0 else 0.0
    loss_rate = num_losses / total_trades if total_trades > 0 else 0.0

    # Long vs Short
    long_trades = sim_df[sim_df['Type'] == 'LONG']
    short_trades = sim_df[sim_df['Type'] == 'SHORT']
    long_wins = len(long_trades[long_trades['PnL_Pct'] > 0])
    short_wins = len(short_trades[short_trades['PnL_Pct'] > 0])
    long_win_rate = long_wins / len(long_trades) if len(long_trades) > 0 else 0.0
    short_win_rate = short_wins / len(short_trades) if len(short_trades) > 0 else 0.0

    # Return averages
    avg_trade_pnl = float(trade_pnls.mean()) if total_trades > 0 else 0.0
    median_trade_pnl = float(trade_pnls.median()) if total_trades > 0 else 0.0
    std_trade_pnl = float(trade_pnls.std(ddof=1)) if total_trades > 1 else 0.0

    avg_win_pnl = float(winning_trades['PnL_Pct'].mean()) if num_wins > 0 else 0.0
    avg_loss_pnl = float(losing_trades['PnL_Pct'].mean()) if num_losses > 0 else 0.0

    avg_win_dollar = float(winning_trades['Profit_Dollars'].mean()) if num_wins > 0 else 0.0
    avg_loss_dollar = float(abs(losing_trades['Profit_Dollars'].mean())) if num_losses > 0 else 0.0

    # Payoff ratio (Avg Win / |Avg Loss|)
    payoff_ratio = (avg_win_pnl / abs(avg_loss_pnl)) if (avg_loss_pnl != 0 and not np.isnan(avg_loss_pnl)) else np.nan

    # Profit factor (Gross Profit / Gross Loss)
    gross_profit = float(winning_trades['Profit_Dollars'].sum()) if num_wins > 0 else 0.0
    gross_loss = float(abs(losing_trades['Profit_Dollars'].sum())) if num_losses > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else np.nan)

    # Mathematical Expectancy
    expectancy_pct = (win_rate * avg_win_pnl) - (loss_rate * abs(avg_loss_pnl))
    expectancy_dollar = (win_rate * avg_win_dollar) - (loss_rate * avg_loss_dollar)

    # Best & Worst Trade
    best_idx = trade_pnls.idxmax() if total_trades > 0 else None
    worst_idx = trade_pnls.idxmin() if total_trades > 0 else None
    best_trade = sim_df.loc[best_idx] if best_idx is not None else None
    worst_trade = sim_df.loc[worst_idx] if worst_idx is not None else None

    # Consecutive Wins & Losses
    outcomes = [1 if p > 0 else (-1 if p < 0 else 0) for p in trade_pnls]
    max_consec_wins = 0
    max_consec_losses = 0
    curr_wins = 0
    curr_losses = 0
    for o in outcomes:
        if o == 1:
            curr_wins += 1
            curr_losses = 0
            max_consec_wins = max(max_consec_wins, curr_wins)
        elif o == -1:
            curr_losses += 1
            curr_wins = 0
            max_consec_losses = max(max_consec_losses, curr_losses)
        else:
            curr_wins = 0
            curr_losses = 0

    # Holding times
    avg_holding_days = float(sim_df['Holding_Days'].mean()) if total_trades > 0 else 0.0
    avg_win_holding_days = float(winning_trades['Holding_Days'].mean()) if num_wins > 0 else 0.0
    avg_loss_holding_days = float(losing_trades['Holding_Days'].mean()) if num_losses > 0 else 0.0

    # Trade frequency
    trades_per_month = (total_trades / (total_days / 30.4375)) if total_days > 0 else 0.0
    trades_per_year = (total_trades / years) if years > 0 else 0.0

    metrics['total_trades'] = total_trades
    metrics['closed_trades'] = closed_trades
    metrics['open_trades'] = open_trades
    metrics['num_wins'] = num_wins
    metrics['num_losses'] = num_losses
    metrics['num_breakeven'] = num_breakeven
    metrics['win_rate'] = win_rate
    metrics['loss_rate'] = loss_rate
    metrics['long_trades_count'] = len(long_trades)
    metrics['short_trades_count'] = len(short_trades)
    metrics['long_win_rate'] = long_win_rate
    metrics['short_win_rate'] = short_win_rate
    metrics['avg_trade_pnl'] = avg_trade_pnl
    metrics['median_trade_pnl'] = median_trade_pnl
    metrics['std_trade_pnl'] = std_trade_pnl
    metrics['avg_win_pnl'] = avg_win_pnl
    metrics['avg_loss_pnl'] = avg_loss_pnl
    metrics['avg_win_dollar'] = avg_win_dollar
    metrics['avg_loss_dollar'] = avg_loss_dollar
    metrics['payoff_ratio'] = payoff_ratio
    metrics['gross_profit'] = gross_profit
    metrics['gross_loss'] = gross_loss
    metrics['profit_factor'] = profit_factor
    metrics['expectancy_pct'] = expectancy_pct
    metrics['expectancy_dollar'] = expectancy_dollar
    metrics['best_trade'] = best_trade
    metrics['worst_trade'] = worst_trade
    metrics['max_consec_wins'] = max_consec_wins
    metrics['max_consec_losses'] = max_consec_losses
    metrics['avg_holding_days'] = avg_holding_days
    metrics['avg_win_holding_days'] = avg_win_holding_days
    metrics['avg_loss_holding_days'] = avg_loss_holding_days
    metrics['trades_per_month'] = trades_per_month
    metrics['trades_per_year'] = trades_per_year

    # -------------------------------------------------------------
    # 3. RISK, VOLATILITY & DRAWDOWN METRICS
    # -------------------------------------------------------------
    eq_series = pd.Series(equity_curve)
    peak_series = eq_series.cummax()
    drawdown_series = (eq_series - peak_series) / peak_series
    max_drawdown = float(drawdown_series.min())
    avg_drawdown = float(drawdown_series[drawdown_series < 0].mean()) if (drawdown_series < 0).any() else 0.0
    peak_equity = float(peak_series.max())

    # Max Drawdown Duration (in steps)
    mdd_duration_steps = 0
    curr_dd_len = 0
    for dd in drawdown_series:
        if dd < 0:
            curr_dd_len += 1
            mdd_duration_steps = max(mdd_duration_steps, curr_dd_len)
        else:
            curr_dd_len = 0

    # Step returns volatility
    step_returns = sim_df['Step_Return']
    steps_per_year = total_trades / years if years > 0 else 252
    rf_per_step = (1 + RISK_FREE_RATE_ANNUAL) ** (1 / steps_per_year) - 1 if steps_per_year > 0 else 0.0

    ann_volatility = float(step_returns.std(ddof=1) * np.sqrt(steps_per_year)) if len(step_returns) > 1 else 0.0
    
    # Downside deviation (returns below step rf)
    downside_diffs = np.minimum(step_returns - rf_per_step, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside_diffs ** 2)) * np.sqrt(steps_per_year)) if len(step_returns) > 0 else 0.0

    # SPY Volatility & Drawdown
    if not spy_hist.empty and len(spy_hist) > 1:
        spy_daily_ret = spy_hist['Close'].pct_change().dropna()
        spy_ann_vol = float(spy_daily_ret.std(ddof=1) * np.sqrt(252))
        spy_peaks = spy_hist['Close'].cummax()
        spy_mdd = float(((spy_hist['Close'] - spy_peaks) / spy_peaks).min())
    else:
        spy_ann_vol = np.nan
        spy_mdd = np.nan

    # Value at Risk (Historical 95% & 99% per step)
    var_95 = float(np.percentile(step_returns, 5)) if len(step_returns) >= 20 else np.nan
    var_99 = float(np.percentile(step_returns, 1)) if len(step_returns) >= 100 else (var_95 if len(step_returns) >= 20 else np.nan)

    # Conditional VaR (Expected Shortfall)
    cvar_95 = float(step_returns[step_returns <= var_95].mean()) if not np.isnan(var_95) and len(step_returns[step_returns <= var_95]) > 0 else np.nan
    cvar_99 = float(step_returns[step_returns <= var_99].mean()) if not np.isnan(var_99) and len(step_returns[step_returns <= var_99]) > 0 else np.nan

    # Skewness & Kurtosis of Trade Returns
    if len(trade_pnls) >= 4:
        trade_skew = float(stats.skew(trade_pnls))
        trade_kurtosis = float(stats.kurtosis(trade_pnls))  # Fisher excess kurtosis
    else:
        trade_skew = np.nan
        trade_kurtosis = np.nan

    metrics['peak_equity'] = peak_equity
    metrics['max_drawdown'] = max_drawdown
    metrics['avg_drawdown'] = avg_drawdown
    metrics['mdd_duration_steps'] = mdd_duration_steps
    metrics['ann_volatility'] = ann_volatility
    metrics['downside_deviation'] = downside_deviation
    metrics['spy_ann_vol'] = spy_ann_vol
    metrics['spy_mdd'] = spy_mdd
    metrics['var_95'] = var_95
    metrics['var_99'] = var_99
    metrics['cvar_95'] = cvar_95
    metrics['cvar_99'] = cvar_99
    metrics['trade_skew'] = trade_skew
    metrics['trade_kurtosis'] = trade_kurtosis

    # -------------------------------------------------------------
    # 4. RISK-ADJUSTED PERFORMANCE RATIOS
    # -------------------------------------------------------------
    # Sharpe Ratio
    sharpe_ratio = ((bot_cagr - RISK_FREE_RATE_ANNUAL) / ann_volatility) if ann_volatility > 0 else np.nan
    spy_sharpe = ((spy_cagr - RISK_FREE_RATE_ANNUAL) / spy_ann_vol) if (not np.isnan(spy_ann_vol) and spy_ann_vol > 0) else np.nan

    # Sortino Ratio
    sortino_ratio = ((bot_cagr - RISK_FREE_RATE_ANNUAL) / downside_deviation) if downside_deviation > 0 else np.nan

    # Calmar Ratio (CAGR / |MDD|)
    calmar_ratio = (bot_cagr / abs(max_drawdown)) if (max_drawdown != 0 and not np.isnan(max_drawdown)) else np.nan

    # Sterling Ratio (CAGR / |Avg DD|)
    sterling_ratio = (bot_cagr / abs(avg_drawdown)) if (avg_drawdown != 0 and not np.isnan(avg_drawdown)) else np.nan

    # Omega Ratio
    pos_excess = step_returns[step_returns > rf_per_step] - rf_per_step
    neg_excess = rf_per_step - step_returns[step_returns < rf_per_step]
    sum_pos = pos_excess.sum()
    sum_neg = neg_excess.sum()
    omega_ratio = float(sum_pos / sum_neg) if sum_neg > 0 else (np.inf if sum_pos > 0 else np.nan)

    # Gain to Pain Ratio (Schwager)
    sum_step_gains = step_returns.sum()
    sum_step_losses = abs(step_returns[step_returns < 0].sum())
    gain_to_pain_ratio = float(sum_step_gains / sum_step_losses) if sum_step_losses > 0 else np.nan

    metrics['sharpe_ratio'] = sharpe_ratio
    metrics['spy_sharpe'] = spy_sharpe
    metrics['sortino_ratio'] = sortino_ratio
    metrics['calmar_ratio'] = calmar_ratio
    metrics['sterling_ratio'] = sterling_ratio
    metrics['omega_ratio'] = omega_ratio
    metrics['gain_to_pain_ratio'] = gain_to_pain_ratio

    # -------------------------------------------------------------
    # 5. BENCHMARK (SPY) REGRESSION & CAPM FACTOR ANALYSIS
    # -------------------------------------------------------------
    paired_df = sim_df[['Step_Return', 'SPY_Window_Return', 'Holding_Days']].dropna()
    if len(paired_df) >= 5:
        step_rf_series = (RISK_FREE_RATE_ANNUAL / 365.25) * paired_df['Holding_Days']
        excess_bot = paired_df['Step_Return'] - step_rf_series
        excess_spy = paired_df['SPY_Window_Return'] - step_rf_series

        # OLS Linear Regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(excess_spy, excess_bot)
        beta = float(slope)
        beta_stderr = float(std_err)
        beta_pvalue = float(p_value)
        correlation = float(r_value)
        r_squared = float(r_value ** 2)

        # Jensen's Alpha per step and annualized
        jensen_alpha_step = float(intercept)
        avg_days = paired_df['Holding_Days'].mean()
        steps_yr = 365.25 / avg_days if avg_days > 0 else 252
        jensen_alpha_annual = float((1 + jensen_alpha_step) ** steps_yr - 1)

        # Alpha standard error and t-test
        n_obs = len(paired_df)
        fitted = intercept + slope * excess_spy
        residuals = excess_bot - fitted
        sse = np.sum(residuals ** 2)
        s_err = np.sqrt(sse / (n_obs - 2)) if n_obs > 2 else 0.0
        x_mean = excess_spy.mean()
        ss_x = np.sum((excess_spy - x_mean) ** 2)
        alpha_se = s_err * np.sqrt(1 / n_obs + (x_mean ** 2) / ss_x) if ss_x > 0 else np.nan
        alpha_tstat = (jensen_alpha_step / alpha_se) if (not np.isnan(alpha_se) and alpha_se > 0) else np.nan
        alpha_pvalue = float(2 * (1 - stats.t.cdf(abs(alpha_tstat), df=n_obs - 2))) if not np.isnan(alpha_tstat) else np.nan

        # Tracking Error & Information Ratio
        excess_diff = paired_df['Step_Return'] - paired_df['SPY_Window_Return']
        tracking_error = float(excess_diff.std(ddof=1) * np.sqrt(steps_yr)) if len(excess_diff) > 1 else np.nan
        information_ratio = (annualized_active_return / tracking_error) if (not np.isnan(tracking_error) and tracking_error > 0) else np.nan

        # Treynor Ratio
        treynor_ratio = ((bot_cagr - RISK_FREE_RATE_ANNUAL) / beta) if (beta != 0 and not np.isnan(beta)) else np.nan

        # Up / Down Market Capture
        up_spy = paired_df[paired_df['SPY_Window_Return'] > 0]
        down_spy = paired_df[paired_df['SPY_Window_Return'] < 0]
        up_capture = float(up_spy['Step_Return'].mean() / up_spy['SPY_Window_Return'].mean()) if (len(up_spy) > 0 and up_spy['SPY_Window_Return'].mean() != 0) else np.nan
        down_capture = float(down_spy['Step_Return'].mean() / down_spy['SPY_Window_Return'].mean()) if (len(down_spy) > 0 and down_spy['SPY_Window_Return'].mean() != 0) else np.nan
    else:
        beta = np.nan
        beta_stderr = np.nan
        beta_pvalue = np.nan
        correlation = np.nan
        r_squared = np.nan
        jensen_alpha_annual = np.nan
        alpha_tstat = np.nan
        alpha_pvalue = np.nan
        tracking_error = np.nan
        information_ratio = np.nan
        treynor_ratio = np.nan
        up_capture = np.nan
        down_capture = np.nan

    metrics['beta'] = beta
    metrics['beta_stderr'] = beta_stderr
    metrics['beta_pvalue'] = beta_pvalue
    metrics['correlation'] = correlation
    metrics['r_squared'] = r_squared
    metrics['jensen_alpha_annual'] = jensen_alpha_annual
    metrics['alpha_tstat'] = alpha_tstat
    metrics['alpha_pvalue'] = alpha_pvalue
    metrics['tracking_error'] = tracking_error
    metrics['information_ratio'] = information_ratio
    metrics['treynor_ratio'] = treynor_ratio
    metrics['up_capture'] = up_capture
    metrics['down_capture'] = down_capture

    # -------------------------------------------------------------
    # 6. STATISTICAL SIGNIFICANCE & HYPOTHESIS TESTING
    # -------------------------------------------------------------
    n_sample = len(trade_pnls)
    if n_sample >= 2:
        # One-sample T-test on trade mean return (H0: mean <= 0)
        mean_pnl = trade_pnls.mean()
        se_pnl = std_trade_pnl / np.sqrt(n_sample)
        t_stat_pnl = float(mean_pnl / se_pnl) if se_pnl > 0 else 0.0
        df_deg = n_sample - 1
        p_val_2tail = float(2 * (1 - stats.t.cdf(abs(t_stat_pnl), df=df_deg)))
        p_val_1tail = float(1 - stats.t.cdf(t_stat_pnl, df=df_deg))  # Right-tail p-value
        
        # 95% Confidence Interval for Mean Trade Return
        ci_res = stats.t.interval(0.95, df=df_deg, loc=mean_pnl, scale=se_pnl)
        ci_low, ci_high = float(ci_res[0]), float(ci_res[1])

        # Binomial Test on Win Rate (H0: Win Rate <= 50%)
        binom_res = stats.binomtest(num_wins, n_sample, p=0.5, alternative='greater')
        binom_pvalue = float(binom_res.pvalue)

        # Probabilistic Sharpe Ratio (PSR) - Marcos López de Prado (2012)
        sr_benchmark = 0.0
        if not np.isnan(sharpe_ratio) and n_sample >= 4 and not np.isnan(trade_skew) and not np.isnan(trade_kurtosis):
            sr_step = float((step_returns.mean() - rf_per_step) / step_returns.std(ddof=1)) if step_returns.std(ddof=1) > 0 else 0.0
            denom_variance = 1.0 - (trade_skew * sr_step) + (((trade_kurtosis + 3) - 1.0) / 4.0) * (sr_step ** 2)
            if denom_variance > 0:
                psr_std = np.sqrt(denom_variance / (n_sample - 1))
                psr_z = (sr_step - sr_benchmark) / psr_std
                psr = float(stats.norm.cdf(psr_z))
                
                # Min Track Record Length (MinTRL) at 95% confidence (Z=1.645)
                if sr_step > 0:
                    min_trl = float(1 + (1.0 - trade_skew * sr_step + ((trade_kurtosis + 2) / 4.0) * (sr_step ** 2)) * ((1.645 / sr_step) ** 2))
                else:
                    min_trl = np.inf
            else:
                psr = np.nan
                min_trl = np.nan
        else:
            psr = np.nan
            min_trl = np.nan
    else:
        t_stat_pnl = np.nan
        p_val_1tail = np.nan
        p_val_2tail = np.nan
        ci_low, ci_high = np.nan, np.nan
        binom_pvalue = np.nan
        psr = np.nan
        min_trl = np.nan

    metrics['t_stat_pnl'] = t_stat_pnl
    metrics['p_val_1tail'] = p_val_1tail
    metrics['p_val_2tail'] = p_val_2tail
    metrics['ci_low'] = ci_low
    metrics['ci_high'] = ci_high
    metrics['binom_pvalue'] = binom_pvalue
    metrics['psr'] = psr
    metrics['min_trl'] = min_trl

    # -------------------------------------------------------------
    # 7. POSITION SIZING & KELLY CRITERION
    # -------------------------------------------------------------
    # Kelly fraction: K = W - (1 - W) / R = (W * (R + 1) - 1) / R
    if not np.isnan(payoff_ratio) and payoff_ratio > 0:
        kelly_fraction = float(win_rate - ((1.0 - win_rate) / payoff_ratio))
        half_kelly = float(kelly_fraction / 2.0)
    else:
        kelly_fraction = np.nan
        half_kelly = np.nan

    metrics['kelly_fraction'] = kelly_fraction
    metrics['half_kelly'] = half_kelly

    return metrics


def print_comprehensive_report(m):
    """Render a beautifully structured, institutional-grade quant report."""
    print("\n" + "═" * 85)
    print(" 🚀 QUANTITATIVE PERFORMANCE & STATISTICAL AUDIT REPORT")
    print("═" * 85)
    print(f" Strategy Period: {m['start_date'].strftime('%Y-%m-%d')}  to  {m['end_date'].strftime('%Y-%m-%d')} ({m['total_days']} calendar days / {m['years']:.2f} years)")
    print(f" Initial Capital: ${STARTING_BALANCE:,.2f}  |  Position Sizing: {POSITION_SIZE_PCT:.0%} per trade  |  Rf Rate: {RISK_FREE_RATE_ANNUAL:.2%}")
    print("═" * 85)

    # -----------------------------------------------------------------
    # SECTION 1: CAPITAL & RETURN OVERVIEW
    # -----------------------------------------------------------------
    print("\n📊 1. CAPITAL & RETURN PERFORMANCE")
    print("─" * 85)
    print(f"{'Metric':<35} {'YOUR BOT':<22} {'S&P 500 (SPY)':<20} {'SPREAD / ACTIVE':<15}")
    print("─" * 85)
    
    end_eq_bot = fmt_curr(m['ending_balance'])
    end_eq_spy = fmt_curr(m['spy_final_balance'])
    end_eq_spr = fmt_curr(m['ending_balance'] - m['spy_final_balance'], sign=True)
    print(f"{'Ending Equity':<35} {end_eq_bot:<22} {end_eq_spy:<20} {end_eq_spr:<15}")

    pnl_bot = fmt_curr(m['bot_net_profit'], sign=True)
    pnl_spy = fmt_curr(m['spy_net_profit'], sign=True)
    pnl_spr = fmt_curr(m['bot_net_profit'] - m['spy_net_profit'], sign=True)
    print(f"{'Net Dollar Profit':<35} {pnl_bot:<22} {pnl_spy:<20} {pnl_spr:<15}")

    tot_ret_bot = fmt_pct(m['bot_total_return'], sign=True)
    tot_ret_spy = fmt_pct(m['spy_total_return'], sign=True)
    tot_ret_spr = fmt_pct(m['naive_alpha'], sign=True)
    print(f"{'Cumulative Total Return':<35} {tot_ret_bot:<22} {tot_ret_spy:<20} {tot_ret_spr:<15}")

    cagr_bot = fmt_pct(m['bot_cagr'], sign=True)
    cagr_spy = fmt_pct(m['spy_cagr'], sign=True)
    cagr_spr = fmt_pct(m['annualized_active_return'], sign=True)
    print(f"{'CAGR (Annualized Return)':<35} {cagr_bot:<22} {cagr_spy:<20} {cagr_spr:<15}")

    peak_eq = fmt_curr(m['peak_equity'])
    print(f"{'Peak Equity (High Water Mark)':<35} {peak_eq:<22} {'-':<20} {'-'}")

    # -----------------------------------------------------------------
    # SECTION 2: TRADE EXECUTION & WIN/LOSS PROFILE
    # -----------------------------------------------------------------
    print("\n🎯 2. TRADE EXECUTION & WIN/LOSS PROFILE")
    print("─" * 85)
    print(f" Total Trades Evaluated: {m['total_trades']:<6} (Closed: {m['closed_trades']} | Open Mark-to-Market: {m['open_trades']})")
    print(f" Long Trades:            {m['long_trades_count']:<6} (Win Rate: {fmt_pct(m['long_win_rate'])})")
    print(f" Short Trades:           {m['short_trades_count']:<6} (Win Rate: {fmt_pct(m['short_win_rate'])})")
    print("─" * 85)
    print(f"{'Metric':<35} {'Value':<20} {'Metric':<25} {'Value'}")
    print("─" * 85)
    
    wins_summary = f"{m['num_wins']} ({fmt_pct(m['win_rate'])})"
    loss_summary = f"{m['num_losses']} ({fmt_pct(m['loss_rate'])})"
    print(f"{'Winning Trades':<35} {wins_summary:<20} {'Losing Trades':<25} {loss_summary}")

    avg_pnl_str = fmt_pct(m['avg_trade_pnl'], sign=True)
    med_pnl_str = fmt_pct(m['median_trade_pnl'], sign=True)
    print(f"{'Average Trade Return':<35} {avg_pnl_str:<20} {'Median Trade Return':<25} {med_pnl_str}")

    avg_win_pnl_str = fmt_pct(m['avg_win_pnl'], sign=True)
    avg_loss_pnl_str = fmt_pct(m['avg_loss_pnl'], sign=True)
    print(f"{'Average Win Return':<35} {avg_win_pnl_str:<20} {'Average Loss Return':<25} {avg_loss_pnl_str}")

    avg_win_dlr_str = fmt_curr(m['avg_win_dollar'])
    avg_loss_dlr_str = fmt_curr(m['avg_loss_dollar'])
    print(f"{'Average Win ($)':<35} {avg_win_dlr_str:<20} {'Average Loss ($)':<25} {avg_loss_dlr_str}")
    
    payoff_str = f"{fmt_num(m['payoff_ratio'])}x" if not np.isnan(m['payoff_ratio']) else "N/A"
    pf_str = fmt_num(m['profit_factor']) if not np.isnan(m['profit_factor']) else "N/A"
    print(f"{'Payoff Ratio (Avg Win / Loss)':<35} {payoff_str:<20} {'Profit Factor':<25} {pf_str}")

    exp_pct_str = fmt_pct(m['expectancy_pct'], sign=True)
    exp_dlr_str = fmt_curr(m['expectancy_dollar'], sign=True)
    print(f"{'Mathematical Expectancy (%)':<35} {exp_pct_str:<20} {'Expectancy ($ / Trade)':<25} {exp_dlr_str}")

    print(f"{'Max Consecutive Wins':<35} {str(m['max_consec_wins']):<20} {'Max Consecutive Losses':<25} {str(m['max_consec_losses'])}")
    
    hold_str = f"{m['avg_holding_days']:.1f} days"
    freq_str = f"{m['trades_per_month']:.1f}/mo ({m['trades_per_year']:.1f}/yr)"
    print(f"{'Average Holding Time':<35} {hold_str:<20} {'Trade Frequency':<25} {freq_str}")
    
    if m['best_trade'] is not None:
        best_str = f"{m['best_trade']['Ticker']} ({fmt_pct(m['best_trade']['PnL_Pct'], sign=True)}) on {m['best_trade']['Date'].strftime('%Y-%m-%d')}"
        print(f"{'Best Trade':<35} {best_str}")
    if m['worst_trade'] is not None:
        worst_str = f"{m['worst_trade']['Ticker']} ({fmt_pct(m['worst_trade']['PnL_Pct'], sign=True)}) on {m['worst_trade']['Date'].strftime('%Y-%m-%d')}"
        print(f"{'Worst Trade':<35} {worst_str}")

    # -----------------------------------------------------------------
    # SECTION 3: RISK, VOLATILITY & DRAWDOWN
    # -----------------------------------------------------------------
    print("\n🛡️ 3. RISK, VOLATILITY & TAIL METRICS")
    print("─" * 85)
    print(f"{'Metric':<35} {'YOUR BOT':<22} {'S&P 500 (SPY)':<20} {'STATUS / REMARK'}")
    print("─" * 85)
    
    vol_bot = fmt_pct(m['ann_volatility'])
    vol_spy = fmt_pct(m['spy_ann_vol'])
    print(f"{'Annualized Volatility (σ)':<35} {vol_bot:<22} {vol_spy:<20} {'Strategy risk'}")

    down_dev = fmt_pct(m['downside_deviation'])
    print(f"{'Downside Deviation (σ_down)':<35} {down_dev:<22} {'-':<20} {'Downside volatility only'}")

    mdd_bot = fmt_pct(m['max_drawdown'], sign=True)
    mdd_spy = fmt_pct(m['spy_mdd'], sign=True)
    print(f"{'Maximum Drawdown (MDD)':<35} {mdd_bot:<22} {mdd_spy:<20} {'Peak-to-trough drop'}")

    avg_dd = fmt_pct(m['avg_drawdown'], sign=True)
    print(f"{'Average Drawdown Depth':<35} {avg_dd:<22} {'-':<20} {'Mean underwater depth'}")

    mdd_dur = f"{m['mdd_duration_steps']} trades"
    print(f"{'Max Drawdown Duration':<35} {mdd_dur:<22} {'-':<20} {'Longest recovery span'}")

    var95_str = fmt_pct(m['var_95'], sign=True)
    var99_str = fmt_pct(m['var_99'], sign=True)
    cvar95_str = fmt_pct(m['cvar_95'], sign=True)
    print(f"{'Historical VaR (95% per trade)':<35} {var95_str:<22} {'-':<20} {'1-in-20 trade risk cutoff'}")
    print(f"{'Historical VaR (99% per trade)':<35} {var99_str:<22} {'-':<20} {'1-in-100 trade risk cutoff'}")
    print(f"{'Conditional VaR / CVaR (95%)':<35} {cvar95_str:<22} {'-':<20} {'Avg loss beyond 95% VaR'}")

    skew_str = fmt_num(m['trade_skew'])
    kurt_str = fmt_num(m['trade_kurtosis'])
    print(f"{'Return Distribution Skewness':<35} {skew_str:<22} {'-':<20} {'>0: Right tail (gains), <0: Left tail'}")
    print(f"{'Return Distribution Excess Kurtosis':<35} {kurt_str:<22} {'-':<20} {'>0: Fat tails / outlier risk'}")

    # -----------------------------------------------------------------
    # SECTION 4: RISK-ADJUSTED PERFORMANCE RATIOS
    # -----------------------------------------------------------------
    print("\n⚖️ 4. RISK-ADJUSTED RETURN RATIOS")
    print("─" * 85)
    print(f"{'Ratio':<35} {'YOUR BOT':<22} {'S&P 500 (SPY)':<20} {'BENCHMARK THRESHOLD'}")
    print("─" * 85)
    
    sr_str = fmt_num(m['sharpe_ratio'])
    spy_sr_str = fmt_num(m['spy_sharpe'])
    print(f"{'Sharpe Ratio (Annualized)':<35} {sr_str:<22} {spy_sr_str:<20} {'> 1.0 Good, > 2.0 Excellent'}")

    sortino_str = fmt_num(m['sortino_ratio'])
    print(f"{'Sortino Ratio (Downside)':<35} {sortino_str:<22} {'-':<20} {'> 1.5 Good, > 3.0 Excellent'}")

    calmar_str = fmt_num(m['calmar_ratio'])
    print(f"{'Calmar Ratio (CAGR / |MDD|)':<35} {calmar_str:<22} {'-':<20} {'> 1.0 Acceptable, > 3.0 Great'}")

    sterling_str = fmt_num(m['sterling_ratio'])
    print(f"{'Sterling Ratio (CAGR / Avg DD)':<35} {sterling_str:<22} {'-':<20} {'Return per average drawdown'}")

    omega_str = fmt_num(m['omega_ratio'])
    print(f"{'Omega Ratio (Threshold Rf)':<35} {omega_str:<22} {'-':<20} {'> 1.0 means positive edge'}")

    gpr_str = fmt_num(m['gain_to_pain_ratio'])
    print(f"{'Gain-to-Pain Ratio (Schwager)':<35} {gpr_str:<22} {'-':<20} {'> 1.0 Good, > 2.0 Excellent'}")

    ir_str = fmt_num(m['information_ratio'])
    print(f"{'Information Ratio (vs SPY)':<35} {ir_str:<22} {'-':<20} {'> 0.5 Good, > 1.0 Elite'}")

    treynor_str = fmt_num(m['treynor_ratio'])
    print(f"{'Treynor Ratio (CAGR / Beta)':<35} {treynor_str:<22} {'-':<20} {'Excess return per unit of beta'}")

    # -----------------------------------------------------------------
    # SECTION 5: BENCHMARK (SPY) & CAPM FACTOR ANALYSIS
    # -----------------------------------------------------------------
    print("\n📈 5. BENCHMARK (SPY) FACTOR REGRESSION & ATTRIBUTION")
    print("─" * 85)
    print(f"{'CAPM / Factor Metric':<35} {'Estimate':<22} {'Std Error / P-Val':<20} {'INTERPRETATION'}")
    print("─" * 85)
    
    beta_str = fmt_num(m['beta'])
    beta_sub = f"SE={fmt_num(m['beta_stderr'])} (p={fmt_num(m['beta_pvalue'], 3)})" if not np.isnan(m['beta_stderr']) else "-"
    print(f"{'Beta (β to SPY)':<35} {beta_str:<22} {beta_sub:<20} {'Market sensitivity (1.0 = SPY)'}")

    corr_str = fmt_num(m['correlation'])
    print(f"{'Correlation with SPY (r)':<35} {corr_str:<22} {'-':<20} {'Linear correlation with market'}")

    r2_str = fmt_pct(m['r_squared'])
    print(f"{'R-Squared (R²)':<35} {r2_str:<22} {'-':<20} {'% Variance explained by SPY'}")

    alpha_str = fmt_pct(m['jensen_alpha_annual'], sign=True)
    alpha_sub = f"t={fmt_num(m['alpha_tstat'])} (p={fmt_num(m['alpha_pvalue'], 3)})" if not np.isnan(m['alpha_tstat']) else "-"
    print(f"{'Annualized Jensen Alpha (α)':<35} {alpha_str:<22} {alpha_sub:<20} {'True risk-adjusted active alpha'}")

    te_str = fmt_pct(m['tracking_error'])
    print(f"{'Annualized Tracking Error':<35} {te_str:<22} {'-':<20} {'Volatility of active return'}")

    up_cap_str = fmt_pct(m['up_capture'])
    down_cap_str = fmt_pct(m['down_capture'])
    print(f"{'Up-Market Capture Ratio':<35} {up_cap_str:<22} {'-':<20} {'% of SPY gains captured'}")
    print(f"{'Down-Market Capture Ratio':<35} {down_cap_str:<22} {'-':<20} {'% of SPY losses captured'}")

    # -----------------------------------------------------------------
    # SECTION 6: STATISTICAL SIGNIFICANCE & HYPOTHESIS TESTING
    # -----------------------------------------------------------------
    print("\n🔬 6. STATISTICAL SIGNIFICANCE & HYPOTHESIS TESTING")
    print("─" * 85)
    print(f"{'Statistical Test':<35} {'Result / Statistic':<22} {'P-Value':<20} {'SIGNIFICANCE (α = 0.05)'}")
    print("─" * 85)
    
    # 1-sample t-test on trade returns
    t_stat_str = f"t = {fmt_num(m['t_stat_pnl'], 3, sign=True)}"
    p_val_str = f"p = {fmt_num(m['p_val_1tail'], 4)} (1-tail)"
    t_sig = "✅ Significant (Edge > 0)" if (not np.isnan(m['p_val_1tail']) and m['p_val_1tail'] < 0.05 and m['t_stat_pnl'] > 0) else "❌ Not Significant"
    print(f"{'Trade Return T-Test (H0: μ ≤ 0)':<35} {t_stat_str:<22} {p_val_str:<20} {t_sig}")
    
    ci_str = f"[{fmt_pct(m['ci_low'], sign=True)}, {fmt_pct(m['ci_high'], sign=True)}]"
    print(f"{'95% CI for Mean Trade Return':<35} {ci_str:<22} {'-':<20} {'True mean return bounds'}")

    # Binomial test on win rate
    binom_wins_str = f"Wins: {m['num_wins']}/{m['total_trades']}"
    binom_p_str = f"p = {fmt_num(m['binom_pvalue'], 4)}"
    binom_sig = "✅ Significant (Win Rate > 50%)" if (not np.isnan(m['binom_pvalue']) and m['binom_pvalue'] < 0.05) else "❌ Not Significant vs Coin Flip"
    print(f"{'Binomial Win Rate Test (H0: W ≤ 50%)':<35} {binom_wins_str:<22} {binom_p_str:<20} {binom_sig}")

    # Probabilistic Sharpe Ratio (PSR)
    psr_str = fmt_pct(m['psr'], 1)
    psr_sig = "✅ High Confidence (>95%)" if (not np.isnan(m['psr']) and m['psr'] >= 0.95) else "⚠️ Low Confidence (<95%)"
    print(f"{'Probabilistic Sharpe Ratio (PSR)':<35} {psr_str:<22} {'Target SR > 0':<20} {psr_sig}")
    
    min_trl_str = f"{int(np.ceil(m['min_trl']))} trades" if (not np.isnan(m['min_trl']) and m['min_trl'] < 100000) else "N/A"
    print(f"{'Min Track Record Length (MinTRL)':<35} {min_trl_str:<22} {'95% Conf. Level':<20} {'Required trades to prove skill'}")

    # -------------------------------------------------------------
    # SECTION 7: KELLY CRITERION & POSITION SIZING
    # -------------------------------------------------------------
    print("\n📐 7. POSITION SIZING & KELLY CRITERION")
    print("─" * 85)
    kelly_pct_str = fmt_pct(m['kelly_fraction'], sign=True)
    half_kelly_str = fmt_pct(m['half_kelly'], sign=True)
    print(f" Optimal Full Kelly Fraction (K*):   {kelly_pct_str}  (Aggressive growth, high volatility)")
    print(f" Recommended Half-Kelly Sizing:     {half_kelly_str}  (Institutional risk standard)")
    print(f" Current Bot Setting:                {POSITION_SIZE_PCT:.1%} of bankroll per trade")
    
    if not np.isnan(m['kelly_fraction']):
        if m['kelly_fraction'] <= 0:
            print(" ⚠️  WARNING: Strategy has negative expectancy. Mathematical recommendation is 0% allocation.")
        elif POSITION_SIZE_PCT > m['kelly_fraction']:
            print(f" ⚠️  WARNING: You are OVER-BETTING ({POSITION_SIZE_PCT:.1%} > Full Kelly {fmt_pct(m['kelly_fraction'])}), causing high risk of ruin.")
        elif POSITION_SIZE_PCT > m['half_kelly']:
            print(f" ℹ️  NOTE: Sizing is between Half-Kelly ({fmt_pct(m['half_kelly'])}) and Full Kelly ({fmt_pct(m['kelly_fraction'])}).")
        else:
            print(f" ✅ SAFE: Current sizing ({POSITION_SIZE_PCT:.1%}) is conservative and within Half-Kelly limits.")

    # -------------------------------------------------------------
    # SECTION 8: EXECUTIVE SUMMARY & BOT VERDICT
    # -------------------------------------------------------------
    print("\n" + "═" * 85)
    print(" 🏆 EXECUTIVE STRATEGY VERDICT")
    print("═" * 85)
    
    diff = m['ending_balance'] - m['spy_final_balance']
    if diff > 0:
        print(f" ✅ OUTPERFORMANCE: Your bot beat S&P 500 (SPY) by ${diff:,.2f} (+{m['naive_alpha']:.2%} active return).")
    else:
        print(f" ❌ UNDERPERFORMANCE: Your bot lagged S&P 500 (SPY) by ${abs(diff):,.2f} ({m['naive_alpha']:.2%} active return).")

    # Alpha & Beta diagnosis
    if not np.isnan(m['beta']):
        if m['beta'] > 1.3:
            print(f" ⚠️  High Beta Exposure (β = {m['beta']:.2f}): Returns are heavily driven by leveraged market movement.")
        elif m['beta'] < 0.3:
            print(f" 🛡️  Market Neutral / Low Beta (β = {m['beta']:.2f}): Strategy shows uncorrelated return profile.")

        if not np.isnan(m['jensen_alpha_annual']):
            if m['jensen_alpha_annual'] > 0 and (not np.isnan(m['alpha_pvalue']) and m['alpha_pvalue'] < 0.10):
                print(f" 🌟 Statistically Valid Alpha: Annualized Jensen's Alpha of {fmt_pct(m['jensen_alpha_annual'], sign=True)} is statistically meaningful.")
            elif m['jensen_alpha_annual'] <= 0:
                print(f" 🔻 Negative Alpha ({fmt_pct(m['jensen_alpha_annual'], sign=True)}): Once market risk (beta) is stripped out, the strategy has not added value.")

    # Statistical significance verdict
    if not np.isnan(m['p_val_1tail']) and m['p_val_1tail'] < 0.05 and m['t_stat_pnl'] > 0:
        print(f" 🎯 Statistical Edge Confirmed: Trade return t-stat ({fmt_num(m['t_stat_pnl'], 2, sign=True)}, p={fmt_num(m['p_val_1tail'], 4)}) rejects random chance at 95% confidence.")
    else:
        t_val = m['t_stat_pnl'] if not np.isnan(m['t_stat_pnl']) else 0.0
        p_val = m['p_val_1tail'] if not np.isnan(m['p_val_1tail']) else 1.0
        print(f" ⏳ Insufficient Statistical Proof: With t={fmt_num(t_val, 2, sign=True)} (p={fmt_num(p_val, 4)}), observed returns cannot yet rule out random luck.")

    print("═" * 85 + "\n")


def main():
    # 1. Load Data
    df, filename = load_trade_data()

    # 2. Benchmark Data
    spy_hist = fetch_benchmark_data(df['Date'].min(), BENCHMARK_TICKER)

    # 3. Simulate & Process Portfolio
    sim_df, equity_curve, equity_dates, final_cash, spy_final, spy_tot_ret, spy_start_px, spy_latest_px = simulate_trading(df, spy_hist)

    # 4. Math & Statistical Metrics
    metrics = calculate_comprehensive_metrics(sim_df, equity_curve, equity_dates, final_cash, spy_final, spy_tot_ret, spy_hist)

    # 5. Output Report
    print_comprehensive_report(metrics)


if __name__ == "__main__":
    main()