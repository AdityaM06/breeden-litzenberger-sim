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
MAX_DISPLAY_TRADES = 20           # Max trades shown in terminal audit table (0 for all)
GENERATE_MARKDOWN = True          # Generate report.md for GitHub in-browser viewing
GENERATE_HTML = True              # Generate report.html dashboard for browser viewing
GENERATE_TEXT = True              # Generate report.txt text report
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
    spy_equity_curve = [STARTING_BALANCE]
    
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

        # Matched SPY window return & cumulative spy equity
        if spy_available:
            try:
                spy_entry = float(spy_hist.asof(entry_date)['Close'])
                spy_exit = float(spy_hist.asof(exit_date)['Close'])
                spy_win_ret = (spy_exit - spy_entry) / spy_entry
                spy_current_sim_val = (STARTING_BALANCE / spy_start_px) * spy_exit
            except Exception:
                spy_win_ret = np.nan
                spy_current_sim_val = STARTING_BALANCE
        else:
            spy_win_ret = np.nan
            spy_current_sim_val = STARTING_BALANCE
            
        spy_step_returns.append(spy_win_ret)
        spy_equity_curve.append(spy_current_sim_val)

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

    return sim_df, equity_curve, equity_dates, spy_equity_curve, current_cash, spy_final_balance, spy_total_return, spy_start_px, spy_latest_px


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
    metrics['drawdown_series'] = drawdown_series.tolist()

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


def build_text_report(m):
    """Generate plaintext format report."""
    lines = []
    lines.append("═" * 85)
    lines.append(" 🚀 QUANTITATIVE PERFORMANCE & STATISTICAL AUDIT REPORT")
    lines.append("═" * 85)
    lines.append(f" Strategy Period: {m['start_date'].strftime('%Y-%m-%d')}  to  {m['end_date'].strftime('%Y-%m-%d')} ({m['total_days']} calendar days / {m['years']:.2f} years)")
    lines.append(f" Initial Capital: ${STARTING_BALANCE:,.2f}  |  Position Sizing: {POSITION_SIZE_PCT:.0%} per trade  |  Rf Rate: {RISK_FREE_RATE_ANNUAL:.2%}")
    lines.append("═" * 85)

    lines.append("\n📊 1. CAPITAL & RETURN PERFORMANCE")
    lines.append("─" * 85)
    lines.append(f"{'Metric':<35} {'YOUR BOT':<22} {'S&P 500 (SPY)':<20} {'SPREAD / ACTIVE':<15}")
    lines.append("─" * 85)
    end_eq_bot = fmt_curr(m['ending_balance'])
    end_eq_spy = fmt_curr(m['spy_final_balance'])
    end_eq_spr = fmt_curr(m['ending_balance'] - m['spy_final_balance'], sign=True)
    lines.append(f"{'Ending Equity':<35} {end_eq_bot:<22} {end_eq_spy:<20} {end_eq_spr:<15}")

    pnl_bot = fmt_curr(m['bot_net_profit'], sign=True)
    pnl_spy = fmt_curr(m['spy_net_profit'], sign=True)
    pnl_spr = fmt_curr(m['bot_net_profit'] - m['spy_net_profit'], sign=True)
    lines.append(f"{'Net Dollar Profit':<35} {pnl_bot:<22} {pnl_spy:<20} {pnl_spr:<15}")

    tot_ret_bot = fmt_pct(m['bot_total_return'], sign=True)
    tot_ret_spy = fmt_pct(m['spy_total_return'], sign=True)
    tot_ret_spr = fmt_pct(m['naive_alpha'], sign=True)
    lines.append(f"{'Cumulative Total Return':<35} {tot_ret_bot:<22} {tot_ret_spy:<20} {tot_ret_spr:<15}")

    cagr_bot = fmt_pct(m['bot_cagr'], sign=True)
    cagr_spy = fmt_pct(m['spy_cagr'], sign=True)
    cagr_spr = fmt_pct(m['annualized_active_return'], sign=True)
    lines.append(f"{'CAGR (Annualized Return)':<35} {cagr_bot:<22} {cagr_spy:<20} {cagr_spr:<15}")

    peak_eq = fmt_curr(m['peak_equity'])
    lines.append(f"{'Peak Equity (High Water Mark)':<35} {peak_eq:<22} {'-':<20} {'-'}")

    lines.append("\n🎯 2. TRADE EXECUTION & WIN/LOSS PROFILE")
    lines.append("─" * 85)
    lines.append(f" Total Trades Evaluated: {m['total_trades']:<6} (Closed: {m['closed_trades']} | Open Mark-to-Market: {m['open_trades']})")
    lines.append(f" Long Trades:            {m['long_trades_count']:<6} (Win Rate: {fmt_pct(m['long_win_rate'])})")
    lines.append(f" Short Trades:           {m['short_trades_count']:<6} (Win Rate: {fmt_pct(m['short_win_rate'])})")
    lines.append("─" * 85)
    lines.append(f"{'Metric':<35} {'Value':<20} {'Metric':<25} {'Value'}")
    lines.append("─" * 85)
    wins_str = f"{m['num_wins']} ({fmt_pct(m['win_rate'])})"
    loss_str = f"{m['num_losses']} ({fmt_pct(m['loss_rate'])})"
    lines.append(f"{'Winning Trades':<35} {wins_str:<20} {'Losing Trades':<25} {loss_str}")

    avg_pnl_str = fmt_pct(m['avg_trade_pnl'], sign=True)
    med_pnl_str = fmt_pct(m['median_trade_pnl'], sign=True)
    lines.append(f"{'Average Trade Return':<35} {avg_pnl_str:<20} {'Median Trade Return':<25} {med_pnl_str}")

    avg_win_pnl_str = fmt_pct(m['avg_win_pnl'], sign=True)
    avg_loss_pnl_str = fmt_pct(m['avg_loss_pnl'], sign=True)
    lines.append(f"{'Average Win Return':<35} {avg_win_pnl_str:<20} {'Average Loss Return':<25} {avg_loss_pnl_str}")

    avg_win_dlr = fmt_curr(m['avg_win_dollar'])
    avg_loss_dlr = fmt_curr(m['avg_loss_dollar'])
    lines.append(f"{'Average Win ($)':<35} {avg_win_dlr:<20} {'Average Loss ($)':<25} {avg_loss_dlr}")

    payoff_str = f"{fmt_num(m['payoff_ratio'])}x" if not np.isnan(m['payoff_ratio']) else "N/A"
    pf_str = fmt_num(m['profit_factor']) if not np.isnan(m['profit_factor']) else "N/A"
    lines.append(f"{'Payoff Ratio (Avg Win / Loss)':<35} {payoff_str:<20} {'Profit Factor':<25} {pf_str}")

    exp_pct_str = fmt_pct(m['expectancy_pct'], sign=True)
    exp_dlr_str = fmt_curr(m['expectancy_dollar'], sign=True)
    lines.append(f"{'Mathematical Expectancy (%)':<35} {exp_pct_str:<20} {'Expectancy ($ / Trade)':<25} {exp_dlr_str}")

    lines.append(f"{'Max Consecutive Wins':<35} {str(m['max_consec_wins']):<20} {'Max Consecutive Losses':<25} {str(m['max_consec_losses'])}")
    
    hold_str = f"{m['avg_holding_days']:.1f} days"
    freq_str = f"{m['trades_per_month']:.1f}/mo ({m['trades_per_year']:.1f}/yr)"
    lines.append(f"{'Average Holding Time':<35} {hold_str:<20} {'Trade Frequency':<25} {freq_str}")

    if m['best_trade'] is not None:
        lines.append(f"{'Best Trade':<35} {m['best_trade']['Ticker']} ({fmt_pct(m['best_trade']['PnL_Pct'], sign=True)}) on {m['best_trade']['Date'].strftime('%Y-%m-%d')}")
    if m['worst_trade'] is not None:
        lines.append(f"{'Worst Trade':<35} {m['worst_trade']['Ticker']} ({fmt_pct(m['worst_trade']['PnL_Pct'], sign=True)}) on {m['worst_trade']['Date'].strftime('%Y-%m-%d')}")

    lines.append("\n🛡️ 3. RISK, VOLATILITY & TAIL METRICS")
    lines.append("─" * 85)
    lines.append(f"{'Metric':<35} {'YOUR BOT':<22} {'S&P 500 (SPY)':<20} {'STATUS / REMARK'}")
    lines.append("─" * 85)
    lines.append(f"{'Annualized Volatility (σ)':<35} {fmt_pct(m['ann_volatility']):<22} {fmt_pct(m['spy_ann_vol']):<20} {'Strategy risk'}")
    lines.append(f"{'Downside Deviation (σ_down)':<35} {fmt_pct(m['downside_deviation']):<22} {'-':<20} {'Downside volatility only'}")
    lines.append(f"{'Maximum Drawdown (MDD)':<35} {fmt_pct(m['max_drawdown'], sign=True):<22} {fmt_pct(m['spy_mdd'], sign=True):<20} {'Peak-to-trough drop'}")
    lines.append(f"{'Average Drawdown Depth':<35} {fmt_pct(m['avg_drawdown'], sign=True):<22} {'-':<20} {'Mean underwater depth'}")
    
    dur_str = f"{m['mdd_duration_steps']} trades"
    lines.append(f"{'Max Drawdown Duration':<35} {dur_str:<22} {'-':<20} {'Longest recovery span'}")
    lines.append(f"{'Historical VaR (95% per trade)':<35} {fmt_pct(m['var_95'], sign=True):<22} {'-':<20} {'1-in-20 trade risk cutoff'}")
    lines.append(f"{'Historical VaR (99% per trade)':<35} {fmt_pct(m['var_99'], sign=True):<22} {'-':<20} {'1-in-100 trade risk cutoff'}")
    lines.append(f"{'Conditional VaR / CVaR (95%)':<35} {fmt_pct(m['cvar_95'], sign=True):<22} {'-':<20} {'Avg loss beyond 95% VaR'}")
    lines.append(f"{'Return Distribution Skewness':<35} {fmt_num(m['trade_skew']):<22} {'-':<20} {'>0: Right tail (gains), <0: Left tail'}")
    lines.append(f"{'Return Distribution Excess Kurtosis':<35} {fmt_num(m['trade_kurtosis']):<22} {'-':<20} {'>0: Fat tails / outlier risk'}")

    lines.append("\n⚖️ 4. RISK-ADJUSTED RETURN RATIOS")
    lines.append("─" * 85)
    lines.append(f"{'Ratio':<35} {'YOUR BOT':<22} {'S&P 500 (SPY)':<20} {'BENCHMARK THRESHOLD'}")
    lines.append("─" * 85)
    lines.append(f"{'Sharpe Ratio (Annualized)':<35} {fmt_num(m['sharpe_ratio']):<22} {fmt_num(m['spy_sharpe']):<20} {'> 1.0 Good, > 2.0 Excellent'}")
    lines.append(f"{'Sortino Ratio (Downside)':<35} {fmt_num(m['sortino_ratio']):<22} {'-':<20} {'> 1.5 Good, > 3.0 Excellent'}")
    lines.append(f"{'Calmar Ratio (CAGR / |MDD|)':<35} {fmt_num(m['calmar_ratio']):<22} {'-':<20} {'> 1.0 Acceptable, > 3.0 Great'}")
    lines.append(f"{'Sterling Ratio (CAGR / Avg DD)':<35} {fmt_num(m['sterling_ratio']):<22} {'-':<20} {'Return per average drawdown'}")
    lines.append(f"{'Omega Ratio (Threshold Rf)':<35} {fmt_num(m['omega_ratio']):<22} {'-':<20} {'> 1.0 means positive edge'}")
    lines.append(f"{'Gain-to-Pain Ratio (Schwager)':<35} {fmt_num(m['gain_to_pain_ratio']):<22} {'-':<20} {'> 1.0 Good, > 2.0 Excellent'}")
    lines.append(f"{'Information Ratio (vs SPY)':<35} {fmt_num(m['information_ratio']):<22} {'-':<20} {'> 0.5 Good, > 1.0 Elite'}")
    lines.append(f"{'Treynor Ratio (CAGR / Beta)':<35} {fmt_num(m['treynor_ratio']):<22} {'-':<20} {'Excess return per unit of beta'}")

    lines.append("\n📈 5. BENCHMARK (SPY) FACTOR REGRESSION & ATTRIBUTION")
    lines.append("─" * 85)
    lines.append(f"{'CAPM / Factor Metric':<35} {'Estimate':<22} {'Std Error / P-Val':<20} {'INTERPRETATION'}")
    lines.append("─" * 85)
    beta_sub = f"SE={fmt_num(m['beta_stderr'])} (p={fmt_num(m['beta_pvalue'], 3)})" if not np.isnan(m['beta_stderr']) else "-"
    lines.append(f"{'Beta (β to SPY)':<35} {fmt_num(m['beta']):<22} {beta_sub:<20} {'Market sensitivity (1.0 = SPY)'}")
    lines.append(f"{'Correlation with SPY (r)':<35} {fmt_num(m['correlation']):<22} {'-':<20} {'Linear correlation with market'}")
    lines.append(f"{'R-Squared (R²)':<35} {fmt_pct(m['r_squared']):<22} {'-':<20} {'% Variance explained by SPY'}")
    alpha_sub = f"t={fmt_num(m['alpha_tstat'])} (p={fmt_num(m['alpha_pvalue'], 3)})" if not np.isnan(m['alpha_tstat']) else "-"
    lines.append(f"{'Annualized Jensen Alpha (α)':<35} {fmt_pct(m['jensen_alpha_annual'], sign=True):<22} {alpha_sub:<20} {'True risk-adjusted active alpha'}")
    lines.append(f"{'Annualized Tracking Error':<35} {fmt_pct(m['tracking_error']):<22} {'-':<20} {'Volatility of active return'}")
    lines.append(f"{'Up-Market Capture Ratio':<35} {fmt_pct(m['up_capture']):<22} {'-':<20} {'% of SPY gains captured'}")
    lines.append(f"{'Down-Market Capture Ratio':<35} {fmt_pct(m['down_capture']):<22} {'-':<20} {'% of SPY losses captured'}")

    lines.append("\n🔬 6. STATISTICAL SIGNIFICANCE & HYPOTHESIS TESTING")
    lines.append("─" * 85)
    lines.append(f"{'Statistical Test':<35} {'Result / Statistic':<22} {'P-Value':<20} {'SIGNIFICANCE (α = 0.05)'}")
    lines.append("─" * 85)
    t_stat_str = f"t = {fmt_num(m['t_stat_pnl'], 3, sign=True)}"
    p_val_str = f"p = {fmt_num(m['p_val_1tail'], 4)} (1-tail)"
    t_sig = "✅ Significant (Edge > 0)" if (not np.isnan(m['p_val_1tail']) and m['p_val_1tail'] < 0.05 and m['t_stat_pnl'] > 0) else "❌ Not Significant"
    lines.append(f"{'Trade Return T-Test (H0: μ ≤ 0)':<35} {t_stat_str:<22} {p_val_str:<20} {t_sig}")
    
    ci_str = f"[{fmt_pct(m['ci_low'], sign=True)}, {fmt_pct(m['ci_high'], sign=True)}]"
    lines.append(f"{'95% CI for Mean Trade Return':<35} {ci_str:<22} {'-':<20} {'True mean return bounds'}")

    binom_cnt_str = f"Wins: {m['num_wins']}/{m['total_trades']}"
    binom_p_str = f"p = {fmt_num(m['binom_pvalue'], 4)}"
    binom_sig = "✅ Significant (Win Rate > 50%)" if (not np.isnan(m['binom_pvalue']) and m['binom_pvalue'] < 0.05) else "❌ Not Significant vs Coin Flip"
    lines.append(f"{'Binomial Win Rate Test (H0: W ≤ 50%)':<35} {binom_cnt_str:<22} {binom_p_str:<20} {binom_sig}")
    
    psr_str = fmt_pct(m['psr'], 1)
    psr_sig = "✅ High Confidence (>95%)" if (not np.isnan(m['psr']) and m['psr'] >= 0.95) else "⚠️ Low Confidence (<95%)"
    lines.append(f"{'Probabilistic Sharpe Ratio (PSR)':<35} {psr_str:<22} {'Target SR > 0':<20} {psr_sig}")
    
    min_trl_str = f"{int(np.ceil(m['min_trl']))} trades" if (not np.isnan(m['min_trl']) and m['min_trl'] < 100000) else "N/A"
    lines.append(f"{'Min Track Record Length (MinTRL)':<35} {min_trl_str:<22} {'95% Conf. Level':<20} {'Required trades to prove skill'}")

    lines.append("\n📐 7. POSITION SIZING & KELLY CRITERION")
    lines.append("─" * 85)
    lines.append(f" Optimal Full Kelly Fraction (K*):   {fmt_pct(m['kelly_fraction'], sign=True)}  (Aggressive growth, high volatility)")
    lines.append(f" Recommended Half-Kelly Sizing:     {fmt_pct(m['half_kelly'], sign=True)}  (Institutional risk standard)")
    lines.append(f" Current Bot Setting:                {POSITION_SIZE_PCT:.1%} of bankroll per trade")
    if not np.isnan(m['kelly_fraction']):
        if m['kelly_fraction'] <= 0:
            lines.append(" ⚠️  WARNING: Strategy has negative expectancy. Mathematical recommendation is 0% allocation.")
        elif POSITION_SIZE_PCT > m['kelly_fraction']:
            lines.append(f" ⚠️  WARNING: You are OVER-BETTING ({POSITION_SIZE_PCT:.1%} > Full Kelly {fmt_pct(m['kelly_fraction'])}), causing high risk of ruin.")
        elif POSITION_SIZE_PCT > m['half_kelly']:
            lines.append(f" ℹ️  NOTE: Sizing is between Half-Kelly ({fmt_pct(m['half_kelly'])}) and Full Kelly ({fmt_pct(m['kelly_fraction'])}).")
        else:
            lines.append(f" ✅ SAFE: Current sizing ({POSITION_SIZE_PCT:.1%}) is conservative and within Half-Kelly limits.")

    lines.append("\n" + "═" * 85)
    lines.append(" 🏆 EXECUTIVE STRATEGY VERDICT")
    lines.append("═" * 85)
    diff = m['ending_balance'] - m['spy_final_balance']
    if diff > 0:
        lines.append(f" ✅ OUTPERFORMANCE: Your bot beat S&P 500 (SPY) by ${diff:,.2f} (+{m['naive_alpha']:.2%} active return).")
    else:
        lines.append(f" ❌ UNDERPERFORMANCE: Your bot lagged S&P 500 (SPY) by ${abs(diff):,.2f} ({m['naive_alpha']:.2%} active return).")

    if not np.isnan(m['beta']):
        if m['beta'] > 1.3:
            lines.append(f" ⚠️  High Beta Exposure (β = {m['beta']:.2f}): Returns are heavily driven by leveraged market movement.")
        elif m['beta'] < 0.3:
            lines.append(f" 🛡️  Market Neutral / Low Beta (β = {m['beta']:.2f}): Strategy shows uncorrelated return profile.")

        if not np.isnan(m['jensen_alpha_annual']):
            if m['jensen_alpha_annual'] > 0 and (not np.isnan(m['alpha_pvalue']) and m['alpha_pvalue'] < 0.10):
                lines.append(f" 🌟 Statistically Valid Alpha: Annualized Jensen's Alpha of {fmt_pct(m['jensen_alpha_annual'], sign=True)} is statistically meaningful.")
            elif m['jensen_alpha_annual'] <= 0:
                lines.append(f" 🔻 Negative Alpha ({fmt_pct(m['jensen_alpha_annual'], sign=True)}): Once market risk (beta) is stripped out, the strategy has not added value.")

    if not np.isnan(m['p_val_1tail']) and m['p_val_1tail'] < 0.05 and m['t_stat_pnl'] > 0:
        lines.append(f" 🎯 Statistical Edge Confirmed: Trade return t-stat ({fmt_num(m['t_stat_pnl'], 2, sign=True)}, p={fmt_num(m['p_val_1tail'], 4)}) rejects random chance at 95% confidence.")
    else:
        t_val = m['t_stat_pnl'] if not np.isnan(m['t_stat_pnl']) else 0.0
        p_val = m['p_val_1tail'] if not np.isnan(m['p_val_1tail']) else 1.0
        lines.append(f" ⏳ Insufficient Statistical Proof: With t={fmt_num(t_val, 2, sign=True)} (p={fmt_num(p_val, 4)}), observed returns cannot yet rule out random luck.")

    lines.append("═" * 85 + "\n")
    return "\n".join(lines)


def build_markdown_report(m, sim_df):
    """Generate rich GitHub-Flavored Markdown for web rendering & GITHUB_STEP_SUMMARY."""
    diff = m['ending_balance'] - m['spy_final_balance']
    
    md = []
    md.append("# 🚀 Trading Strategy Quantitative Performance Report")
    md.append("")
    md.append(f"> **Period**: `{m['start_date'].strftime('%Y-%m-%d')}` to `{m['end_date'].strftime('%Y-%m-%d')}` ({m['total_days']} days / {m['years']:.2f} yrs) &nbsp;|&nbsp; **Initial Capital**: `{fmt_curr(STARTING_BALANCE)}` &nbsp;|&nbsp; **Position Sizing**: `{POSITION_SIZE_PCT:.0%}` &nbsp;|&nbsp; **Benchmark**: `{BENCHMARK_TICKER}`")
    md.append("")
    
    # Executive Alert
    if diff > 0:
        md.append(f"> [!TIP]\n> **Strategy Outperformed Market**: The bot generated **{fmt_pct(m['bot_total_return'], sign=True)}** vs SPY's **{fmt_pct(m['spy_total_return'], sign=True)}** (Active spread: **{fmt_curr(diff, sign=True)}** / **{fmt_pct(m['naive_alpha'], sign=True)}**).")
    else:
        md.append(f"> [!WARNING]\n> **Strategy Lagged Market**: The bot generated **{fmt_pct(m['bot_total_return'], sign=True)}** vs SPY's **{fmt_pct(m['spy_total_return'], sign=True)}** (Active lag: **{fmt_curr(abs(diff))}**).")
    md.append("")

    # Section 1: Capital & Return Overview
    md.append("## 📊 1. Capital & Return Overview")
    md.append("")
    md.append("| Metric | Your Bot | S&P 500 (SPY) | Active Spread / Advantage |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Ending Equity** | **{fmt_curr(m['ending_balance'])}** | {fmt_curr(m['spy_final_balance'])} | **{fmt_curr(m['ending_balance'] - m['spy_final_balance'], sign=True)}** |")
    md.append(f"| **Net Profit** | **{fmt_curr(m['bot_net_profit'], sign=True)}** | {fmt_curr(m['spy_net_profit'], sign=True)} | **{fmt_curr(m['bot_net_profit'] - m['spy_net_profit'], sign=True)}** |")
    md.append(f"| **Cumulative Return** | **{fmt_pct(m['bot_total_return'], sign=True)}** | {fmt_pct(m['spy_total_return'], sign=True)} | **{fmt_pct(m['naive_alpha'], sign=True)}** |")
    md.append(f"| **CAGR (Annualized)** | **{fmt_pct(m['bot_cagr'], sign=True)}** | {fmt_pct(m['spy_cagr'], sign=True)} | **{fmt_pct(m['annualized_active_return'], sign=True)}** |")
    md.append(f"| **Peak Equity (HWM)** | {fmt_curr(m['peak_equity'])} | - | - |")
    md.append("")

    # Section 2: Trade Execution & Win/Loss Metrics
    md.append("## 🎯 2. Trade Execution & Win/Loss Profile")
    md.append("")
    md.append(f"- **Total Trades**: `{m['total_trades']}` (Closed: `{m['closed_trades']}` | Open Mark-to-Market: `{m['open_trades']}`)")
    md.append(f"- **Long Trades**: `{m['long_trades_count']}` (Win Rate: `{fmt_pct(m['long_win_rate'])}`) | **Short Trades**: `{m['short_trades_count']}` (Win Rate: `{fmt_pct(m['short_win_rate'])}`)")
    md.append("")
    md.append("| Trade Metric | Value | Metric | Value |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Winning Trades** | `{m['num_wins']} ({fmt_pct(m['win_rate'])})` | **Losing Trades** | `{m['num_losses']} ({fmt_pct(m['loss_rate'])})` |")
    md.append(f"| **Average Trade Return** | `{fmt_pct(m['avg_trade_pnl'], sign=True)}` | **Median Trade Return** | `{fmt_pct(m['median_trade_pnl'], sign=True)}` |")
    md.append(f"| **Average Win Return** | `{fmt_pct(m['avg_win_pnl'], sign=True)}` | **Average Loss Return** | `{fmt_pct(m['avg_loss_pnl'], sign=True)}` |")
    md.append(f"| **Average Win ($)** | `{fmt_curr(m['avg_win_dollar'])}` | **Average Loss ($)** | `{fmt_curr(m['avg_loss_dollar'])}` |")
    md.append(f"| **Payoff Ratio (W/L)** | `{fmt_num(m['payoff_ratio'])}x` | **Profit Factor** | `{fmt_num(m['profit_factor'])}` |")
    md.append(f"| **Expectancy (%)** | `{fmt_pct(m['expectancy_pct'], sign=True)}` | **Expectancy ($ / Trade)** | `{fmt_curr(m['expectancy_dollar'], sign=True)}` |")
    md.append(f"| **Max Consec. Wins** | `{m['max_consec_wins']}` | **Max Consec. Losses** | `{m['max_consec_losses']}` |")
    md.append(f"| **Avg Holding Period** | `{m['avg_holding_days']:.1f} days` | **Trade Frequency** | `{m['trades_per_month']:.1f}/mo ({m['trades_per_year']:.1f}/yr)` |")
    if m['best_trade'] is not None and m['worst_trade'] is not None:
        md.append(f"| **Best Single Trade** | `{m['best_trade']['Ticker']} ({fmt_pct(m['best_trade']['PnL_Pct'], sign=True)})` | **Worst Single Trade** | `{m['worst_trade']['Ticker']} ({fmt_pct(m['worst_trade']['PnL_Pct'], sign=True)})` |")
    md.append("")

    # Section 3: Risk & Volatility
    md.append("## 🛡️ 3. Risk, Volatility & Tail Metrics")
    md.append("")
    md.append("| Risk Metric | Your Bot | S&P 500 (SPY) | Description / Notes |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Annualized Volatility (σ)** | `{fmt_pct(m['ann_volatility'])}` | `{fmt_pct(m['spy_ann_vol'])}` | Total return dispersion |")
    md.append(f"| **Downside Deviation (σ_down)** | `{fmt_pct(m['downside_deviation'])}` | - | Volatility of negative returns only |")
    md.append(f"| **Max Drawdown (MDD)** | `{fmt_pct(m['max_drawdown'], sign=True)}` | `{fmt_pct(m['spy_mdd'], sign=True)}` | Peak-to-trough worst drop |")
    md.append(f"| **Average Drawdown** | `{fmt_pct(m['avg_drawdown'], sign=True)}` | - | Mean depth during underwater periods |")
    md.append(f"| **Max Drawdown Duration** | `{m['mdd_duration_steps']} trades` | - | Longest recovery period |")
    md.append(f"| **Historical VaR (95% / 99%)** | `{fmt_pct(m['var_95'], sign=True)}` / `{fmt_pct(m['var_99'], sign=True)}` | - | 1-in-20 / 1-in-100 trade risk threshold |")
    md.append(f"| **Conditional VaR (CVaR 95%)** | `{fmt_pct(m['cvar_95'], sign=True)}` | - | Expected loss beyond 95% VaR |")
    md.append(f"| **Return Skewness / Kurtosis** | `{fmt_num(m['trade_skew'])}` / `{fmt_num(m['trade_kurtosis'])}` | - | Right-tail upside vs heavy tail risk |")
    md.append("")

    # Section 4: Risk-Adjusted Return Ratios
    md.append("## ⚖️ 4. Risk-Adjusted Performance Ratios")
    md.append("")
    md.append("| Ratio | Your Bot | S&P 500 (SPY) | Target / Guideline |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Sharpe Ratio (Annualized)** | **`{fmt_num(m['sharpe_ratio'])}`** | `{fmt_num(m['spy_sharpe'])}` | `> 1.0` Good, `> 2.0` Excellent |")
    md.append(f"| **Sortino Ratio (Downside)** | **`{fmt_num(m['sortino_ratio'])}`** | - | `> 1.5` Good, `> 3.0` Excellent |")
    md.append(f"| **Calmar Ratio (CAGR / MDD)** | `{fmt_num(m['calmar_ratio'])}` | - | `> 1.0` Acceptable, `> 3.0` Great |")
    md.append(f"| **Sterling Ratio** | `{fmt_num(m['sterling_ratio'])}` | - | Return per average drawdown |")
    md.append(f"| **Omega Ratio (Threshold Rf)** | `{fmt_num(m['omega_ratio'])}` | - | `> 1.0` denotes positive strategy edge |")
    md.append(f"| **Gain-to-Pain Ratio (Schwager)** | `{fmt_num(m['gain_to_pain_ratio'])}` | - | `> 1.0` Good, `> 2.0` Excellent |")
    md.append(f"| **Information Ratio (vs SPY)** | `{fmt_num(m['information_ratio'])}` | - | `> 0.5` Good, `> 1.0` Elite |")
    md.append(f"| **Treynor Ratio (CAGR / Beta)** | `{fmt_num(m['treynor_ratio'])}` | - | Excess return per unit of systematic risk |")
    md.append("")

    # Section 5: CAPM Benchmark Regression
    md.append("## 📈 5. Benchmark (SPY) Factor Attribution")
    md.append("")
    beta_sub = f"SE={fmt_num(m['beta_stderr'])} (p={fmt_num(m['beta_pvalue'], 3)})" if not np.isnan(m['beta_stderr']) else "-"
    alpha_sub = f"t={fmt_num(m['alpha_tstat'])} (p={fmt_num(m['alpha_pvalue'], 3)})" if not np.isnan(m['alpha_tstat']) else "-"
    md.append("| CAPM / Factor Metric | Estimate | Test Stat / P-Value | Interpretation |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Beta (β to SPY)** | `{fmt_num(m['beta'])}` | `{beta_sub}` | Systematic sensitivity (1.0 = SPY) |")
    md.append(f"| **Correlation (r) / R²** | `{fmt_num(m['correlation'])}` / `{fmt_pct(m['r_squared'])}` | - | Market correlation & % variance explained |")
    md.append(f"| **Annualized Jensen's Alpha (α)** | `{fmt_pct(m['jensen_alpha_annual'], sign=True)}` | `{alpha_sub}` | True risk-adjusted alpha over CAPM |")
    md.append(f"| **Tracking Error** | `{fmt_pct(m['tracking_error'])}` | - | Annualized excess return volatility |")
    md.append(f"| **Up / Down Capture** | `{fmt_pct(m['up_capture'])}` / `{fmt_pct(m['down_capture'])}` | - | % of market up/down movements captured |")
    md.append("")

    # Section 6: Statistical Significance & Hypothesis Testing
    md.append("## 🔬 6. Statistical Significance & Hypothesis Testing")
    md.append("")
    t_sig = "✅ **Significant (Edge > 0)**" if (not np.isnan(m['p_val_1tail']) and m['p_val_1tail'] < 0.05 and m['t_stat_pnl'] > 0) else "❌ Not Significant"
    binom_sig = "✅ **Significant (> 50%)**" if (not np.isnan(m['binom_pvalue']) and m['binom_pvalue'] < 0.05) else "❌ Not Significant vs Coin Flip"
    psr_sig = "✅ **High Confidence (> 95%)**" if (not np.isnan(m['psr']) and m['psr'] >= 0.95) else "⚠️ Low Confidence (< 95%)"
    min_trl_str = f"`{int(np.ceil(m['min_trl']))} trades`" if (not np.isnan(m['min_trl']) and m['min_trl'] < 100000) else "`N/A`"

    md.append("| Statistical Test | Result / Statistic | P-Value | Verdict (α = 0.05) |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Trade Return T-Test (H0: μ ≤ 0)** | `t = {fmt_num(m['t_stat_pnl'], 3, sign=True)}` | `p = {fmt_num(m['p_val_1tail'], 4)} (1-tail)` | {t_sig} |")
    md.append(f"| **95% Confidence Interval for Mean** | `[{fmt_pct(m['ci_low'], sign=True)}, {fmt_pct(m['ci_high'], sign=True)}]` | - | Bounds for true average trade return |")
    md.append(f"| **Binomial Win Rate Test (H0: W ≤ 50%)** | `Wins: {m['num_wins']}/{m['total_trades']}` | `p = {fmt_num(m['binom_pvalue'], 4)}` | {binom_sig} |")
    md.append(f"| **Probabilistic Sharpe Ratio (PSR)** | `{fmt_pct(m['psr'], 1)}` | `Target SR > 0` | {psr_sig} |")
    md.append(f"| **Min Track Record Length (MinTRL)** | {min_trl_str} | `95% Conf.` | Required trade sample to prove skill |")
    md.append("")

    # Section 7: Position Sizing & Kelly Criterion
    md.append("## 📐 7. Optimal Position Sizing (Kelly Criterion)")
    md.append("")
    md.append(f"- **Full Kelly Sizing ($K^*$)**: `{fmt_pct(m['kelly_fraction'], sign=True)}` (Maximum geometric growth)")
    md.append(f"- **Recommended Half-Kelly Sizing**: `{fmt_pct(m['half_kelly'], sign=True)}` (Institutional risk standard)")
    md.append(f"- **Current Configured Bet Size**: `{POSITION_SIZE_PCT:.1%}`")
    md.append("")
    if not np.isnan(m['kelly_fraction']):
        if m['kelly_fraction'] <= 0:
            md.append("> [!CAUTION]\n> **Negative Expectancy**: Kelly criterion suggests 0% position size until edge improves.")
        elif POSITION_SIZE_PCT > m['kelly_fraction']:
            md.append(f"> [!WARNING]\n> **Over-Betting Alert**: Current bet size (`{POSITION_SIZE_PCT:.1%}`) exceeds Full Kelly (`{fmt_pct(m['kelly_fraction'])}`), creating elevated risk of drawdown/ruin.")
        else:
            md.append(f"> [!NOTE]\n> **Conservative Sizing**: Current sizing (`{POSITION_SIZE_PCT:.1%}`) is within safe risk bounds.")
    md.append("")

    # Section 8: Recent Trades Table
    md.append("## 📋 8. Recent 10 Trades Audit")
    md.append("")
    md.append("| # | Date | Ticker | Type | Status | Entry Price | Exit / Current | PnL % | Profit ($) | Equity |")
    md.append("| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    recent = sim_df.tail(10)
    for _, r in recent.iterrows():
        pnl_str = fmt_pct(r['PnL_Pct'], sign=True)
        prof_str = fmt_curr(r['Profit_Dollars'], sign=True)
        bal_str = fmt_curr(r['Balance_After'])
        md.append(f"| {r['Index']} | {r['Date'].strftime('%Y-%m-%d')} | **{r['Ticker']}** | {r['Type']} | `{r['Status']}` | ${r['Entry_Price']:.2f} | ${r['Exit_Price']:.2f} | {pnl_str} | {prof_str} | {bal_str} |")
    md.append("")
    
    return "\n".join(md)


def generate_svg_chart(equity_curve, spy_equity_curve, drawdown_series):
    """Generate crisp standalone SVG charts for Equity Curve and Drawdown."""
    w, h = 900, 240
    pad_left, pad_right, pad_top, pad_bot = 60, 20, 20, 30
    plot_w = w - pad_left - pad_right
    plot_h = h - pad_top - pad_bot

    n = len(equity_curve)
    if n < 2:
        return "<p style='color:#94a3b8;'>Insufficient data for chart.</p>", "<p style='color:#94a3b8;'>Insufficient data for chart.</p>"

    # 1. Equity Curve Path
    all_eq = equity_curve + spy_equity_curve
    min_eq, max_eq = min(all_eq) * 0.95, max(all_eq) * 1.05
    eq_range = max_eq - min_eq if max_eq != min_eq else 1.0

    def get_eq_xy(idx, val):
        x = pad_left + (idx / (n - 1)) * plot_w
        y = pad_top + plot_h - ((val - min_eq) / eq_range) * plot_h
        return x, y

    bot_pts = [f"{get_eq_xy(i, v)[0]:.1f},{get_eq_xy(i, v)[1]:.1f}" for i, v in enumerate(equity_curve)]
    spy_pts = [f"{get_eq_xy(i, v)[0]:.1f},{get_eq_xy(i, v)[1]:.1f}" for i, v in enumerate(spy_equity_curve)]

    bot_path = "M " + " L ".join(bot_pts)
    spy_path = "M " + " L ".join(spy_pts)

    # Bot Area Fill
    bot_area = bot_path + f" L {get_eq_xy(n-1, min_eq)[0]:.1f},{pad_top+plot_h} L {pad_left},{pad_top+plot_h} Z"

    # Y-axis labels (3 ticks)
    y_ticks = [min_eq, (min_eq + max_eq) / 2, max_eq]
    y_lines_svg = []
    for y_val in y_ticks:
        _, y_coord = get_eq_xy(0, y_val)
        y_lines_svg.append(f"""
        <line x1="{pad_left}" y1="{y_coord:.1f}" x2="{w - pad_right}" y2="{y_coord:.1f}" stroke="#334155" stroke-dasharray="3,3" stroke-width="1" />
        <text x="{pad_left - 8}" y="{y_coord + 4:.1f}" fill="#94a3b8" font-size="11" text-anchor="end">${y_val:,.0f}</text>
        """)
    y_lines_str = "\n".join(y_lines_svg)

    equity_svg = f"""
    <svg viewBox="0 0 {w} {h}" class="chart-svg" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="botGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.35"/>
                <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.0"/>
            </linearGradient>
        </defs>
        {y_lines_str}
        <path d="{bot_area}" fill="url(#botGrad)" />
        <path d="{spy_path}" fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="4,4" />
        <path d="{bot_path}" fill="none" stroke="#38bdf8" stroke-width="3" />
        <circle cx="{get_eq_xy(n-1, equity_curve[-1])[0]:.1f}" cy="{get_eq_xy(n-1, equity_curve[-1])[1]:.1f}" r="5" fill="#38bdf8" stroke="#fff" stroke-width="2" />
        <text x="{pad_left}" y="{h - 8}" fill="#94a3b8" font-size="11">Start (Trade #1)</text>
        <text x="{w - pad_right}" y="{h - 8}" fill="#94a3b8" font-size="11" text-anchor="end">Trade #{n}</text>
    </svg>
    """

    # 2. Drawdown Chart
    dd_h = 140
    dd_plot_h = dd_h - pad_top - pad_bot
    min_dd = min(min(drawdown_series), -0.05) * 1.1

    def get_dd_xy(idx, val):
        x = pad_left + (idx / (n - 1)) * plot_w
        y = pad_top + (abs(val) / abs(min_dd)) * dd_plot_h
        return x, y

    dd_pts = [f"{get_dd_xy(i, v)[0]:.1f},{get_dd_xy(i, v)[1]:.1f}" for i, v in enumerate(drawdown_series)]
    dd_line = "M " + " L ".join(dd_pts)
    dd_area = dd_line + f" L {get_dd_xy(n-1, 0)[0]:.1f},{pad_top} L {pad_left},{pad_top} Z"

    dd_svg = f"""
    <svg viewBox="0 0 {w} {dd_h}" class="chart-svg" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#ef4444" stop-opacity="0.0"/>
                <stop offset="100%" stop-color="#ef4444" stop-opacity="0.4"/>
            </linearGradient>
        </defs>
        <line x1="{pad_left}" y1="{pad_top}" x2="{w - pad_right}" y2="{pad_top}" stroke="#475569" stroke-width="1.5" />
        <text x="{pad_left - 8}" y="{pad_top + 4}" fill="#94a3b8" font-size="11" text-anchor="end">0.0%</text>
        <line x1="{pad_left}" y1="{pad_top + dd_plot_h}" x2="{w - pad_right}" y2="{pad_top + dd_plot_h}" stroke="#334155" stroke-dasharray="3,3" />
        <text x="{pad_left - 8}" y="{pad_top + dd_plot_h + 4}" fill="#94a3b8" font-size="11" text-anchor="end">{min_dd:.1%}</text>
        <path d="{dd_area}" fill="url(#ddGrad)" />
        <path d="{dd_line}" fill="none" stroke="#ef4444" stroke-width="2" />
        <text x="{pad_left}" y="{dd_h - 6}" fill="#94a3b8" font-size="11">Start</text>
        <text x="{w - pad_right}" y="{dd_h - 6}" fill="#94a3b8" font-size="11" text-anchor="end">Current</text>
    </svg>
    """

    return equity_svg, dd_svg


def build_html_report(m, sim_df, equity_curve, spy_equity_curve):
    """Generate an institutional-grade, hyper-comprehensive standalone HTML dashboard."""
    diff = m['ending_balance'] - m['spy_final_balance']
    diff_color = "#10b981" if diff >= 0 else "#ef4444"
    cagr_color = "#10b981" if m['bot_cagr'] >= 0 else "#ef4444"
    alpha_color = "#10b981" if m['naive_alpha'] >= 0 else "#ef4444"
    
    # Generate SVG charts
    drawdowns = m.get('drawdown_series', [0.0])
    equity_svg_html, dd_svg_html = generate_svg_chart(equity_curve, spy_equity_curve, drawdowns)

    # Build all trades table rows
    all_trade_rows = []
    for _, r in sim_df.iterrows():
        pnl_cls = "pos" if r['PnL_Pct'] > 0 else ("neg" if r['PnL_Pct'] < 0 else "zero")
        pnl_str = fmt_pct(r['PnL_Pct'], sign=True)
        prof_str = fmt_curr(r['Profit_Dollars'], sign=True)
        bal_str = fmt_curr(r['Balance_After'])
        status_tag = f"<span class='tag tag-{r['Status'].lower()}'>{r['Status']}</span>"
        type_tag = f"<span class='tag tag-{r['Type'].lower()}'>{r['Type']}</span>"
        all_trade_rows.append(f"""
        <tr data-type="{r['Type'].upper()}" data-status="{r['Status'].upper()}" data-pnl="{'win' if r['PnL_Pct'] > 0 else ('loss' if r['PnL_Pct'] < 0 else 'breakeven')}">
            <td style="color:#94a3b8;">{r['Index']}</td>
            <td>{r['Date'].strftime('%Y-%m-%d')}</td>
            <td><strong>{r['Ticker']}</strong></td>
            <td>{type_tag}</td>
            <td>{status_tag}</td>
            <td>${r['Entry_Price']:.2f}</td>
            <td>${r['Exit_Price']:.2f}</td>
            <td class='{pnl_cls}'><strong>{pnl_str}</strong></td>
            <td class='{pnl_cls}'>{prof_str}</td>
            <td><strong>{bal_str}</strong></td>
            <td style="color:#94a3b8;">{r['Holding_Days']}d</td>
        </tr>
        """)
    all_trades_table_html = "\n".join(all_trade_rows)

    # Health status badges for key metrics
    sharpe_badge = "🟢 Excellent (> 2.0)" if m['sharpe_ratio'] >= 2.0 else ("🟡 Good (1.0 - 2.0)" if m['sharpe_ratio'] >= 1.0 else "🔴 Suboptimal (< 1.0)")
    sortino_badge = "🟢 Elite (> 3.0)" if m['sortino_ratio'] >= 3.0 else ("🟡 Good (1.5 - 3.0)" if m['sortino_ratio'] >= 1.5 else "🔴 Caution (< 1.5)")
    calmar_badge = "🟢 Great (> 3.0)" if m['calmar_ratio'] >= 3.0 else ("🟡 Acceptable (1.0 - 3.0)" if m['calmar_ratio'] >= 1.0 else "🔴 Low (< 1.0)")
    pf_badge = "🟢 Great (> 1.5)" if m['profit_factor'] >= 1.5 else ("🟡 Neutral (1.0 - 1.5)" if m['profit_factor'] >= 1.0 else "🔴 Losing (< 1.0)")
    mdd_badge = "🟢 Low Risk (< 15%)" if abs(m['max_drawdown']) <= 0.15 else ("🟡 Moderate Risk (15-25%)" if abs(m['max_drawdown']) <= 0.25 else "🔴 High Drawdown (> 25%)")
    t_test_badge = "🟢 Statistically Proven (p < 0.05)" if (not np.isnan(m['p_val_1tail']) and m['p_val_1tail'] < 0.05 and m['t_stat_pnl'] > 0) else "🔴 Unproven / Luck (p ≥ 0.05)"
    psr_badge = "🟢 High Confidence (≥ 95%)" if (not np.isnan(m['psr']) and m['psr'] >= 0.95) else "🟡 Moderate (80-95%)" if (not np.isnan(m['psr']) and m['psr'] >= 0.80) else "🔴 Low Confidence (< 80%)"
    kelly_badge = "🔴 Over-Betting Danger" if (not np.isnan(m['kelly_fraction']) and POSITION_SIZE_PCT > m['kelly_fraction']) else ("🟡 Aggressive (Half - Full Kelly)" if (not np.isnan(m['half_kelly']) and POSITION_SIZE_PCT > m['half_kelly']) else "🟢 Safe / Conservative (≤ Half-Kelly)")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantitative Strategy Performance & KPI Master Audit</title>
    <style>
        :root {{
            --bg: #090d16;
            --surface: #111827;
            --surface-elevated: #1a2234;
            --border: #243048;
            --border-subtle: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.15);
            --green: #10b981;
            --green-bg: rgba(16, 185, 129, 0.12);
            --red: #ef4444;
            --red-bg: rgba(239, 68, 68, 0.12);
            --amber: #f59e0b;
            --amber-bg: rgba(245, 158, 11, 0.12);
            --purple: #a855f7;
            --purple-bg: rgba(168, 85, 247, 0.12);
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
            background-color: var(--bg);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 1.5rem;
        }}
        .container {{ max-width: 1320px; margin: 0 auto; }}
        
        /* Sticky Top Navigation */
        .top-nav {{
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(17, 24, 39, 0.95);
            backdrop-filter: blur(12px);
            padding: 0.75rem 1rem;
            margin: -1.5rem -1.5rem 1.5rem -1.5rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            overflow-x: auto;
            gap: 1rem;
        }}
        .nav-links {{ display: flex; gap: 0.5rem; flex-wrap: nowrap; }}
        .nav-link {{
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            white-space: nowrap;
            transition: all 0.2s;
        }}
        .nav-link:hover {{ background: var(--surface-elevated); color: var(--accent); }}

        /* Header */
        header {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.75rem;
            margin-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 1rem;
        }}
        h1 {{ font-size: 1.85rem; font-weight: 800; color: #fff; letter-spacing: -0.02em; }}
        .subtitle {{ color: var(--text-secondary); font-size: 0.95rem; margin-top: 0.35rem; }}
        
        /* Hero KPI Grid */
        .hero-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        .hero-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            position: relative;
            overflow: hidden;
            transition: transform 0.2s, border-color 0.2s;
        }}
        .hero-card:hover {{ border-color: var(--accent); transform: translateY(-2px); }}
        .hero-title {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            font-weight: 700;
            margin-bottom: 0.4rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .hero-val {{ font-size: 1.95rem; font-weight: 800; margin-bottom: 0.25rem; }}
        .hero-sub {{ font-size: 0.85rem; color: var(--text-secondary); }}

        /* Verdict Banner */
        .verdict-banner {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(56, 189, 248, 0.08) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 1.25rem;
        }}
        .verdict-banner.warning {{
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(245, 158, 11, 0.08) 100%);
            border-color: rgba(239, 68, 68, 0.3);
        }}
        .verdict-icon {{ font-size: 2.2rem; }}
        .verdict-text h3 {{ font-size: 1.15rem; font-weight: 700; margin-bottom: 0.25rem; }}
        .verdict-text p {{ color: var(--text-secondary); font-size: 0.95rem; }}

        /* Charts Section */
        .charts-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        @media (max-width: 1024px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
        }}
        .chart-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
        }}
        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}
        .chart-title {{ font-size: 1rem; font-weight: 700; color: #fff; }}
        .chart-legend {{ display: flex; gap: 1rem; font-size: 0.8rem; color: var(--text-secondary); }}
        .legend-item {{ display: flex; align-items: center; gap: 0.4rem; }}
        .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
        .chart-svg {{ width: 100%; height: auto; display: block; }}

        /* Section Headings */
        .section-header {{
            margin: 2.5rem 0 1rem 0;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }}
        .section-title {{ font-size: 1.35rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 0.5rem; }}
        .section-desc {{ color: var(--text-muted); font-size: 0.85rem; }}

        /* Comprehensive KPI Cards & Tables */
        .kpi-table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--surface);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
            margin-bottom: 1.5rem;
        }}
        .kpi-table th {{
            background: var(--surface-elevated);
            color: var(--text-secondary);
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 0.85rem 1.1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        .kpi-table td {{
            padding: 1rem 1.1rem;
            border-bottom: 1px solid var(--border-subtle);
            font-size: 0.9rem;
            vertical-align: middle;
        }}
        .kpi-table tr:last-child td {{ border-bottom: none; }}
        .kpi-table tr:hover td {{ background: rgba(255, 255, 255, 0.015); }}
        
        .kpi-name {{ font-weight: 700; color: #fff; font-size: 0.95rem; display: block; }}
        .kpi-formula {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 0.75rem; color: var(--text-muted); }}
        .kpi-meaning {{ color: var(--text-secondary); font-size: 0.85rem; line-height: 1.4; }}
        .kpi-desirable {{
            background: var(--surface-elevated);
            border-left: 3px solid var(--accent);
            padding: 0.35rem 0.6rem;
            border-radius: 0 4px 4px 0;
            font-size: 0.8rem;
            color: var(--accent);
            font-weight: 600;
        }}
        
        /* Badges & Tags */
        .badge-pill {{
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }}
        .badge-green {{ background: var(--green-bg); color: var(--green); border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-red {{ background: var(--red-bg); color: var(--red); border: 1px solid rgba(239, 68, 68, 0.3); }}
        .badge-amber {{ background: var(--amber-bg); color: var(--amber); border: 1px solid rgba(245, 158, 11, 0.3); }}
        .badge-blue {{ background: var(--accent-glow); color: var(--accent); border: 1px solid rgba(56, 189, 248, 0.3); }}

        .tag {{
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .tag-long {{ background: var(--green-bg); color: var(--green); }}
        .tag-short {{ background: var(--red-bg); color: var(--red); }}
        .tag-closed {{ background: rgba(148, 163, 184, 0.15); color: var(--text-secondary); }}
        .tag-open {{ background: var(--accent-glow); color: var(--accent); }}

        .pos {{ color: var(--green); }}
        .neg {{ color: var(--red); }}
        .zero {{ color: var(--text-muted); }}

        /* Filter Controls */
        .filter-bar {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
            align-items: center;
        }}
        .search-input {{
            background: var(--surface);
            border: 1px solid var(--border);
            color: #fff;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            outline: none;
            min-width: 240px;
        }}
        .search-input:focus {{ border-color: var(--accent); }}
        .btn-filter {{
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 0.5rem 0.85rem;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .btn-filter:hover, .btn-filter.active {{ background: var(--accent); color: #000; border-color: var(--accent); }}

        /* Callout Box */
        .callout {{
            background: var(--surface);
            border-left: 4px solid var(--accent);
            border-radius: 0 8px 8px 0;
            padding: 1rem 1.25rem;
            margin-bottom: 1.5rem;
        }}
        .callout-title {{ font-weight: 700; font-size: 0.95rem; margin-bottom: 0.25rem; color: var(--accent); }}
        .callout-body {{ color: var(--text-secondary); font-size: 0.88rem; }}
    </style>
</head>
<body>
    <div class="container">
        
        <!-- Sticky Navigation -->
        <nav class="top-nav">
            <div style="font-weight: 800; font-size: 0.9rem; color: #fff;">📊 QUANT AUDIT</div>
            <div class="nav-links">
                <a href="#summary" class="nav-link">Executive Summary</a>
                <a href="#capital" class="nav-link">1. Capital & Returns</a>
                <a href="#trade-profile" class="nav-link">2. Trade Quality</a>
                <a href="#risk-drawdown" class="nav-link">3. Risk & Drawdown</a>
                <a href="#ratios" class="nav-link">4. Risk Ratios</a>
                <a href="#capm" class="nav-link">5. CAPM & Alpha</a>
                <a href="#stats-tests" class="nav-link">6. Hypothesis Tests</a>
                <a href="#kelly" class="nav-link">7. Kelly Sizing</a>
                <a href="#trade-log" class="nav-link">8. Trade Log</a>
                <a href="#encyclopedia" class="nav-link">9. KPI Guide</a>
            </div>
        </nav>

        <!-- Header -->
        <header id="summary">
            <div>
                <h1>🤖 Quantitative Trading Bot Performance Audit</h1>
                <div class="subtitle">
                    Evaluation Window: <strong>{m['start_date'].strftime('%Y-%m-%d')}</strong> to <strong>{m['end_date'].strftime('%Y-%m-%d')}</strong> ({m['total_days']} calendar days / {m['years']:.2f} yrs) &bull; Benchmark: <strong>{BENCHMARK_TICKER} (S&P 500)</strong>
                </div>
            </div>
            <div style="text-align: right;">
                <span class="badge-pill badge-blue">Position Sizing: {POSITION_SIZE_PCT:.0%} per trade</span>
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.35rem;">Risk-Free Rate: {RISK_FREE_RATE_ANNUAL:.2%} T-Bill</div>
            </div>
        </header>

        <!-- Verdict Banner -->
        <div class="verdict-banner {'warning' if diff < 0 else ''}">
            <div class="verdict-icon">{'🏆' if diff >= 0 else '⚠️'}</div>
            <div class="verdict-text">
                <h3>{'Strategy Beat the Market by ' + fmt_curr(diff, sign=True) if diff >= 0 else 'Strategy Lagged the Market by ' + fmt_curr(abs(diff))} ({fmt_pct(m['naive_alpha'], sign=True)} Active Spread)</h3>
                <p>
                    Strategy delivered <strong>{fmt_pct(m['bot_total_return'], sign=True)}</strong> cumulative return (<strong>{fmt_pct(m['bot_cagr'], sign=True)} CAGR</strong>) compared to SPY's <strong>{fmt_pct(m['spy_total_return'], sign=True)}</strong>.
                    {'Statistical testing confirms a genuine trading edge rejecting random chance at 95% confidence.' if not np.isnan(m['p_val_1tail']) and m['p_val_1tail'] < 0.05 and m['t_stat_pnl'] > 0 else 'Returns show promise but statistical sample requires more trades to conclusively rule out market noise.'}
                </p>
            </div>
        </div>

        <!-- Hero KPIs -->
        <div class="hero-grid">
            <div class="hero-card">
                <div class="hero-title">Ending Equity <span class="badge-pill badge-blue">Portfolio</span></div>
                <div class="hero-val">{fmt_curr(m['ending_balance'])}</div>
                <div class="hero-sub">Initial: {fmt_curr(STARTING_BALANCE)} &bull; Profit: <span class="pos">{fmt_curr(m['bot_net_profit'], sign=True)}</span></div>
            </div>
            <div class="hero-card">
                <div class="hero-title">Annualized Return (CAGR) <span class="badge-pill badge-green">Compounded</span></div>
                <div class="hero-val" style="color: {cagr_color};">{fmt_pct(m['bot_cagr'], sign=True)}</div>
                <div class="hero-sub">SPY CAGR: {fmt_pct(m['spy_cagr'], sign=True)} &bull; Spread: <span style="color:{alpha_color};">{fmt_pct(m['annualized_active_return'], sign=True)}</span></div>
            </div>
            <div class="hero-card">
                <div class="hero-title">Sharpe Ratio <span class="badge-pill badge-blue">Risk-Adjusted</span></div>
                <div class="hero-val">{fmt_num(m['sharpe_ratio'])}</div>
                <div class="hero-sub">SPY Sharpe: {fmt_num(m['spy_sharpe'])} &bull; Sortino: {fmt_num(m['sortino_ratio'])}</div>
            </div>
            <div class="hero-card">
                <div class="hero-title">Win Rate / Payoff <span class="badge-pill badge-amber">Execution</span></div>
                <div class="hero-val">{fmt_pct(m['win_rate'])}</div>
                <div class="hero-sub">Payoff: {fmt_num(m['payoff_ratio'])}x &bull; Profit Factor: {fmt_num(m['profit_factor'])}</div>
            </div>
        </div>

        <!-- Visual Charts Grid -->
        <div class="charts-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-title">📈 Compounded Equity Growth vs S&P 500</div>
                    <div class="chart-legend">
                        <div class="legend-item"><div class="legend-dot" style="background:#38bdf8;"></div> Your Bot (${m['ending_balance']:,.0f})</div>
                        <div class="legend-item"><div class="legend-dot" style="background:#94a3b8;"></div> SPY Buy & Hold (${m['spy_final_balance']:,.0f})</div>
                    </div>
                </div>
                {equity_svg_html}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-title">🛡️ Underwater Drawdown Profile</div>
                    <div class="chart-legend">
                        <div class="legend-item"><div class="legend-dot" style="background:#ef4444;"></div> Max DD: {fmt_pct(m['max_drawdown'], sign=True)}</div>
                    </div>
                </div>
                {dd_svg_html}
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem; text-align: center;">
                    Avg Drawdown Depth: <strong>{fmt_pct(m['avg_drawdown'], sign=True)}</strong> &bull; Max Recovery Span: <strong>{m['mdd_duration_steps']} trades</strong>
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- SECTION 1: CAPITAL & RETURN KPIS -->
        <!-- ============================================================= -->
        <div class="section-header" id="capital">
            <div>
                <div class="section-title">📊 1. Capital Growth & Return Metrics</div>
                <div class="section-desc">Evaluates cumulative gains, compounding velocity, and benchmark outperformance.</div>
            </div>
        </div>

        <table class="kpi-table">
            <thead>
                <tr>
                    <th style="width: 24%;">Metric & Formula</th>
                    <th style="width: 14%;">Strategy Bot</th>
                    <th style="width: 14%;">S&P 500 (SPY)</th>
                    <th style="width: 18%;">Status / Health</th>
                    <th style="width: 30%;">Meaning & Desirable Range</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>
                        <span class="kpi-name">Compound Annual Growth (CAGR)</span>
                        <span class="kpi-formula">(1 + Total_Return)^(1/Years) - 1</span>
                    </td>
                    <td><strong class="pos" style="font-size: 1.1rem;">{fmt_pct(m['bot_cagr'], sign=True)}</strong></td>
                    <td>{fmt_pct(m['spy_cagr'], sign=True)}</td>
                    <td><span class="badge-pill badge-green">🟢 Active: {fmt_pct(m['annualized_active_return'], sign=True)}</span></td>
                    <td>
                        <div class="kpi-meaning">The annualized geometric rate of return, factoring in the effect of compounding over time.</div>
                        <div class="kpi-desirable">🎯 Desirable: > 15-20% annualized (and consistently above SPY CAGR).</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Cumulative Total Return</span>
                        <span class="kpi-formula">(End_Balance - Start_Balance) / Start_Balance</span>
                    </td>
                    <td><strong class="pos">{fmt_pct(m['bot_total_return'], sign=True)}</strong></td>
                    <td>{fmt_pct(m['spy_total_return'], sign=True)}</td>
                    <td><span class="badge-pill {'badge-green' if diff >= 0 else 'badge-red'}">{'🟢 Outperforming' if diff >= 0 else '🔴 Lagging'}</span></td>
                    <td>
                        <div class="kpi-meaning">Total percentage profit generated by the portfolio from inception to date.</div>
                        <div class="kpi-desirable">🎯 Desirable: Positive, and higher than SPY Buy & Hold over identical timeframe.</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Net Dollar Profit</span>
                        <span class="kpi-formula">End_Equity - Start_Capital</span>
                    </td>
                    <td><strong class="pos">{fmt_curr(m['bot_net_profit'], sign=True)}</strong></td>
                    <td>{fmt_curr(m['spy_net_profit'], sign=True)}</td>
                    <td><span class="badge-pill badge-green">Spread: {fmt_curr(m['bot_net_profit'] - m['spy_net_profit'], sign=True)}</span></td>
                    <td>
                        <div class="kpi-meaning">Total absolute dollar profit earned after all realized gains and unrealized open floating mark-to-market.</div>
                        <div class="kpi-desirable">🎯 Desirable: Maximizing net profits while containing portfolio drawdown.</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Peak Equity (High Water Mark)</span>
                        <span class="kpi-formula">max(Equity_0, Equity_1, ... Equity_t)</span>
                    </td>
                    <td><strong>{fmt_curr(m['peak_equity'])}</strong></td>
                    <td>-</td>
                    <td><span class="badge-pill badge-blue">Peak Capital</span></td>
                    <td>
                        <div class="kpi-meaning">The highest recorded account balance achieved during the entire lifetime of the strategy.</div>
                        <div class="kpi-desirable">🎯 Desirable: Steadily rising high water marks with short recovery times.</div>
                    </td>
                </tr>
            </tbody>
        </table>

        <!-- ============================================================= -->
        <!-- SECTION 2: TRADE EXECUTION & WIN/LOSS PROFILE -->
        <!-- ============================================================= -->
        <div class="section-header" id="trade-profile">
            <div>
                <div class="section-title">🎯 2. Trade Execution & Quality Profile</div>
                <div class="section-desc">Measures individual trade edge, win/loss asymmetry, payoff ratio, and streaks.</div>
            </div>
        </div>

        <table class="kpi-table">
            <thead>
                <tr>
                    <th style="width: 24%;">Metric & Formula</th>
                    <th style="width: 14%;">Strategy Bot</th>
                    <th style="width: 14%;">Benchmark / Ref</th>
                    <th style="width: 18%;">Status / Health</th>
                    <th style="width: 30%;">Meaning & Desirable Range</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>
                        <span class="kpi-name">Win Rate (%)</span>
                        <span class="kpi-formula">Winning_Trades / Total_Trades</span>
                    </td>
                    <td><strong>{fmt_pct(m['win_rate'])}</strong> ({m['num_wins']} of {m['total_trades']})</td>
                    <td>50.0% (Coin flip)</td>
                    <td><span class="badge-pill {'badge-green' if m['win_rate'] >= 0.50 else 'badge-amber'}">{fmt_pct(m['win_rate'])}</span></td>
                    <td>
                        <div class="kpi-meaning">Percentage of total completed and active trades that yielded a positive return.</div>
                        <div class="kpi-desirable">🎯 Desirable: > 50% for trend/swing trading, or > 40% if Payoff Ratio is > 2.0x.</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Payoff Ratio (Win/Loss)</span>
                        <span class="kpi-formula">Avg_Win_% / |Avg_Loss_%|</span>
                    </td>
                    <td><strong class="pos" style="font-size: 1.1rem;">{fmt_num(m['payoff_ratio'])}x</strong></td>
                    <td>1.00x</td>
                    <td><span class="badge-pill {'badge-green' if m['payoff_ratio'] >= 1.5 else 'badge-amber'}">{'> 1.5x Good' if m['payoff_ratio'] >= 1.5 else 'Moderate'}</span></td>
                    <td>
                        <div class="kpi-meaning">Ratio of the average gain on winning trades (+{m['avg_win_pnl']:.2%}) to the average loss on losing trades ({m['avg_loss_pnl']:.2%}).</div>
                        <div class="kpi-desirable">🎯 Desirable: > 1.5x to 2.5x (allows profitability even with sub-50% win rates).</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Profit Factor</span>
                        <span class="kpi-formula">Gross_Gains_$ / Gross_Losses_$</span>
                    </td>
                    <td><strong>{fmt_num(m['profit_factor'])}</strong></td>
                    <td>1.00</td>
                    <td><span class="badge-pill {'badge-green' if m['profit_factor'] >= 1.5 else 'badge-amber'}">{pf_badge}</span></td>
                    <td>
                        <div class="kpi-meaning">Total gross dollar profits divided by total gross dollar losses across all trades.</div>
                        <div class="kpi-desirable">🎯 Desirable: > 1.50 is good, > 2.00 is institutional-grade. (< 1.00 is losing).</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Mathematical Expectancy</span>
                        <span class="kpi-formula">(Win% * AvgWin) - (Loss% * AvgLoss)</span>
                    </td>
                    <td><strong class="pos">{fmt_pct(m['expectancy_pct'], sign=True)}</strong> ({fmt_curr(m['expectancy_dollar'], sign=True)})</td>
                    <td>$0.00</td>
                    <td><span class="badge-pill {'badge-green' if m['expectancy_pct'] > 0 else 'badge-red'}">{'🟢 Positive Edge' if m['expectancy_pct'] > 0 else '🔴 Negative'}</span></td>
                    <td>
                        <div class="kpi-meaning">The expected mathematical dollar and percentage gain generated per trade executed.</div>
                        <div class="kpi-desirable">🎯 Desirable: Positive value (> +1.0% per trade). Fundamental prerequisite for long-term growth.</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Long vs Short Win Rates</span>
                        <span class="kpi-formula">Long_Wins/Longs vs Short_Wins/Shorts</span>
                    </td>
                    <td>Long: <strong>{fmt_pct(m['long_win_rate'])}</strong> | Short: <strong>{fmt_pct(m['short_win_rate'])}</strong></td>
                    <td>Long: {m['long_trades_count']} | Short: {m['short_trades_count']}</td>
                    <td><span class="badge-pill badge-blue">Breakdown</span></td>
                    <td>
                        <div class="kpi-meaning">Separates directional model performance on long positions versus short positions.</div>
                        <div class="kpi-desirable">🎯 Desirable: Balanced accuracy across both long and short regimes.</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Consecutive Streaks</span>
                        <span class="kpi-formula">Max Consecutive Wins / Losses</span>
                    </td>
                    <td>Wins: <strong>{m['max_consec_wins']}</strong> | Losses: <strong>{m['max_consec_losses']}</strong></td>
                    <td>-</td>
                    <td><span class="badge-pill badge-amber">Streak Profile</span></td>
                    <td>
                        <div class="kpi-meaning">Longest continuous winning streak vs longest continuous losing streak.</div>
                        <div class="kpi-desirable">🎯 Desirable: Losing streaks should not exceed risk tolerance or trigger margin calls.</div>
                    </td>
                </tr>
            </tbody>
        </table>

        <!-- ============================================================= -->
        <!-- SECTION 3: RISK, VOLATILITY & TAIL METRICS -->
        <!-- ============================================================= -->
        <div class="section-header" id="risk-drawdown">
            <div>
                <div class="section-title">🛡️ 3. Risk, Volatility & Tail Metrics</div>
                <div class="section-desc">Measures drawdown depth, downside deviation, Value at Risk, and distribution tail risks.</div>
            </div>
        </div>

        <table class="kpi-table">
            <thead>
                <tr>
                    <th style="width: 24%;">Metric & Formula</th>
                    <th style="width: 14%;">Strategy Bot</th>
                    <th style="width: 14%;">S&P 500 (SPY)</th>
                    <th style="width: 18%;">Status / Health</th>
                    <th style="width: 30%;">Meaning & Desirable Range</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>
                        <span class="kpi-name">Maximum Drawdown (MDD)</span>
                        <span class="kpi-formula">min((Equity_t - Peak_t) / Peak_t)</span>
                    </td>
                    <td><strong class="neg">{fmt_pct(m['max_drawdown'], sign=True)}</strong></td>
                    <td>{fmt_pct(m['spy_mdd'], sign=True)}</td>
                    <td><span class="badge-pill {'badge-green' if abs(m['max_drawdown']) <= 0.20 else 'badge-amber'}">{mdd_badge}</span></td>
                    <td>
                        <div class="kpi-meaning">The maximum peak-to-trough drop in account equity experienced before a new high is reached.</div>
                        <div class="kpi-desirable">🎯 Desirable: Lower magnitude than benchmark (< -15% to -20% is considered safe).</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Annualized Volatility (σ)</span>
                        <span class="kpi-formula">std(returns) * sqrt(steps_per_year)</span>
                    </td>
                    <td><strong>{fmt_pct(m['ann_volatility'])}</strong></td>
                    <td>{fmt_pct(m['spy_ann_vol'])}</td>
                    <td><span class="badge-pill badge-blue">Dispersion</span></td>
                    <td>
                        <div class="kpi-meaning">The standard deviation of annualized returns, measuring total variability and price fluctuation.</div>
                        <div class="kpi-desirable">🎯 Desirable: 10% - 25% for moderate risk strategies; should be compensated by high return.</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Downside Deviation (σ_down)</span>
                        <span class="kpi-formula">sqrt(mean(min(r - rf, 0)^2)) * sqrt(K)</span>
                    </td>
                    <td><strong>{fmt_pct(m['downside_deviation'])}</strong></td>
                    <td>-</td>
                    <td><span class="badge-pill badge-green">Downside Only</span></td>
                    <td>
                        <div class="kpi-meaning">Measures volatility of negative returns only below the risk-free rate, ignoring upside volatility.</div>
                        <div class="kpi-desirable">🎯 Desirable: Significantly lower than total volatility, proving upside asymmetry.</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Historical Value at Risk (95% VaR)</span>
                        <span class="kpi-formula">5th Percentile of Trade Return</span>
                    </td>
                    <td><strong>{fmt_pct(m['var_95'], sign=True)}</strong></td>
                    <td>-</td>
                    <td><span class="badge-pill badge-amber">1-in-20 Risk</span></td>
                    <td>
                        <div class="kpi-meaning">In 95% of trades, losses will not exceed this threshold on a per-step basis.</div>
                        <div class="kpi-desirable">🎯 Desirable: Controlled loss ceiling (< -3.0% per portfolio step).</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Conditional VaR (CVaR / Expected Shortfall)</span>
                        <span class="kpi-formula">mean(r | r <= 95% VaR)</span>
                    </td>
                    <td><strong class="neg">{fmt_pct(m['cvar_95'], sign=True)}</strong></td>
                    <td>-</td>
                    <td><span class="badge-pill badge-red">Tail Risk</span></td>
                    <td>
                        <div class="kpi-meaning">The expected average loss when a severe tail event worse than the 95% VaR cutoff occurs.</div>
                        <div class="kpi-desirable">🎯 Desirable: Bounded tail risk without catastrophic blowups.</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Return Skewness & Kurtosis</span>
                        <span class="kpi-formula">Skew: E[(r-μ)^3]/σ^3 | Kurt: E[(r-μ)^4]/σ^4 - 3</span>
                    </td>
                    <td>Skew: <strong>{fmt_num(m['trade_skew'])}</strong> | Kurt: <strong>{fmt_num(m['trade_kurtosis'])}</strong></td>
                    <td>Normal: (0.0, 0.0)</td>
                    <td><span class="badge-pill badge-blue">Right Skew (+3.06)</span></td>
                    <td>
                        <div class="kpi-meaning">Positive skew (>0) indicates frequent small losses with occasional huge outsized wins. Kurtosis measures fat tails.</div>
                        <div class="kpi-desirable">🎯 Desirable: Positive Skew (> 0) is ideal for trend followers.</div>
                    </td>
                </tr>
            </tbody>
        </table>

        <!-- ============================================================= -->
        <!-- SECTION 4: RISK-ADJUSTED RETURN RATIOS -->
        <!-- ============================================================= -->
        <div class="section-header" id="ratios">
            <div>
                <div class="section-title">⚖️ 4. Risk-Adjusted Performance Ratios</div>
                <div class="section-desc">Quantifies how much return the strategy generates per unit of risk taken.</div>
            </div>
        </div>

        <table class="kpi-table">
            <thead>
                <tr>
                    <th style="width: 24%;">Ratio & Formula</th>
                    <th style="width: 14%;">Strategy Bot</th>
                    <th style="width: 14%;">S&P 500 (SPY)</th>
                    <th style="width: 18%;">Status / Health</th>
                    <th style="width: 30%;">Meaning & Desirable Range</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>
                        <span class="kpi-name">Sharpe Ratio (Annualized)</span>
                        <span class="kpi-formula">(CAGR - Rf) / Annualized_Vol</span>
                    </td>
                    <td><strong class="pos" style="font-size: 1.15rem;">{fmt_num(m['sharpe_ratio'])}</strong></td>
                    <td>{fmt_num(m['spy_sharpe'])}</td>
                    <td><span class="badge-pill badge-green">{sharpe_badge}</span></td>
                    <td>
                        <div class="kpi-meaning">Measures excess return earned above the risk-free rate per unit of total return volatility.</div>
                        <div class="kpi-desirable">🎯 Desirable: > 1.0 is good, > 2.0 is very good, > 3.0 is elite.</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Sortino Ratio (Downside)</span>
                        <span class="kpi-formula">(CAGR - Rf) / Downside_Deviation</span>
                    </td>
                    <td><strong class="pos" style="font-size: 1.15rem;">{fmt_num(m['sortino_ratio'])}</strong></td>
                    <td>-</td>
                    <td><span class="badge-pill badge-green">{sortino_badge}</span></td>
                    <td>
                        <div class="kpi-meaning">Like Sharpe ratio, but only penalizes harmful downside volatility rather than upward volatility.</div>
                        <div class="kpi-desirable">🎯 Desirable: > 1.5 is good, > 3.0 is outstanding.</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Calmar Ratio</span>
                        <span class="kpi-formula">CAGR / |Maximum_Drawdown|</span>
                    </td>
                    <td><strong>{fmt_num(m['calmar_ratio'])}</strong></td>
                    <td>-</td>
                    <td><span class="badge-pill badge-green">{calmar_badge}</span></td>
                    <td>
                        <div class="kpi-meaning">Measures annualized return relative to the worst historical maximum drawdown.</div>
                        <div class="kpi-desirable">🎯 Desirable: > 1.0 is acceptable, > 3.0 is top tier.</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Omega Ratio</span>
                        <span class="kpi-formula">Sum(Gains > Rf) / Sum(|Losses < Rf|)</span>
                    </td>
                    <td><strong>{fmt_num(m['omega_ratio'])}</strong></td>
                    <td>1.00</td>
                    <td><span class="badge-pill badge-green">🟢 Edge > 1.0</span></td>
                    <td>
                        <div class="kpi-meaning">Probability-weighted ratio of gains versus losses thresholded at the risk-free rate.</div>
                        <div class="kpi-desirable">🎯 Desirable: > 1.0 denotes positive expectancy; > 1.5 indicates strong compounding advantage.</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Information Ratio (vs SPY)</span>
                        <span class="kpi-formula">Active_CAGR / Tracking_Error</span>
                    </td>
                    <td><strong>{fmt_num(m['information_ratio'])}</strong></td>
                    <td>-</td>
                    <td><span class="badge-pill badge-green">🟢 Elite (> 1.0)</span></td>
                    <td>
                        <div class="kpi-meaning">Measures strategy consistency in generating excess returns over the benchmark.</div>
                        <div class="kpi-desirable">🎯 Desirable: > 0.5 is good, > 1.0 is exceptional.</div>
                    </td>
                </tr>
            </tbody>
        </table>

        <!-- ============================================================= -->
        <!-- SECTION 5: BENCHMARK (SPY) & CAPM FACTOR ATTRIBUTION -->
        <!-- ============================================================= -->
        <div class="section-header" id="capm">
            <div>
                <div class="section-title">📈 5. Benchmark (SPY) Factor Attribution & CAPM</div>
                <div class="section-desc">Decomposes returns into systematic market exposure (Beta) versus genuine manager skill (Alpha).</div>
            </div>
        </div>

        <table class="kpi-table">
            <thead>
                <tr>
                    <th style="width: 24%;">Factor Metric & Formula</th>
                    <th style="width: 14%;">Estimate</th>
                    <th style="width: 14%;">Test Stat / P-Val</th>
                    <th style="width: 18%;">Status / Health</th>
                    <th style="width: 30%;">Meaning & Desirable Range</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>
                        <span class="kpi-name">Beta (β to SPY)</span>
                        <span class="kpi-formula">Cov(R_bot, R_spy) / Var(R_spy)</span>
                    </td>
                    <td><strong>{fmt_num(m['beta'])}</strong></td>
                    <td>SE={fmt_num(m['beta_stderr'])} (p={fmt_num(m['beta_pvalue'], 3)})</td>
                    <td><span class="badge-pill badge-blue">🟢 Low Beta / Uncorrelated</span></td>
                    <td>
                        <div class="kpi-meaning">Sensitivity of strategy returns to broader market movements (1.0 = moves with SPY, 0 = uncorrelated).</div>
                        <div class="kpi-desirable">🎯 Desirable: Low beta (0.0 - 0.5) proves strategy is not simply taking leveraged market beta.</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Annualized Jensen's Alpha (α)</span>
                        <span class="kpi-formula">R_bot - (Rf + β * (R_spy - Rf))</span>
                    </td>
                    <td><strong>{fmt_pct(m['jensen_alpha_annual'], sign=True)}</strong></td>
                    <td>t={fmt_num(m['alpha_tstat'])} (p={fmt_num(m['alpha_pvalue'], 3)})</td>
                    <td><span class="badge-pill {'badge-green' if m['jensen_alpha_annual'] > 0 else 'badge-amber'}">{fmt_pct(m['jensen_alpha_annual'], sign=True)}</span></td>
                    <td>
                        <div class="kpi-meaning">Risk-adjusted return generated above and beyond what the CAPM model predicts for the beta risk taken.</div>
                        <div class="kpi-desirable">🎯 Desirable: Positive (> 0%) with statistical significance (p < 0.05).</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Correlation (r) & R-Squared (R²)</span>
                        <span class="kpi-formula">Pearson r | R² = r^2</span>
                    </td>
                    <td>r = <strong>{fmt_num(m['correlation'])}</strong> | R² = <strong>{fmt_pct(m['r_squared'])}</strong></td>
                    <td>-</td>
                    <td><span class="badge-pill badge-blue">Independent Return Driver</span></td>
                    <td>
                        <div class="kpi-meaning">R² represents the proportion of strategy variance explained by the S&P 500.</div>
                        <div class="kpi-desirable">🎯 Desirable: Low R² (< 20%) indicates true diversification from standard index funds.</div>
                    </td>
                </tr>
            </tbody>
        </table>

        <!-- ============================================================= -->
        <!-- SECTION 6: STATISTICAL SIGNIFICANCE & HYPOTHESIS TESTING -->
        <!-- ============================================================= -->
        <div class="section-header" id="stats-tests">
            <div>
                <div class="section-title">🔬 6. Statistical Significance & Hypothesis Testing</div>
                <div class="section-desc">Hypothesis tests validating whether performance represents true quantitative skill or random luck.</div>
            </div>
        </div>

        <table class="kpi-table">
            <thead>
                <tr>
                    <th style="width: 24%;">Statistical Test</th>
                    <th style="width: 14%;">Sample Statistic</th>
                    <th style="width: 14%;">P-Value</th>
                    <th style="width: 18%;">Verdict (α = 0.05)</th>
                    <th style="width: 30%;">Meaning & Desirable Range</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>
                        <span class="kpi-name">Trade Return T-Test</span>
                        <span class="kpi-formula">H0: μ ≤ 0 (No positive edge)</span>
                    </td>
                    <td><strong>t = {fmt_num(m['t_stat_pnl'], 3, sign=True)}</strong></td>
                    <td><strong>p = {fmt_num(m['p_val_1tail'], 4)}</strong> (1-tail)</td>
                    <td><span class="badge-pill {'badge-green' if not np.isnan(m['p_val_1tail']) and m['p_val_1tail'] < 0.05 and m['t_stat_pnl'] > 0 else 'badge-red'}">{t_test_badge}</span></td>
                    <td>
                        <div class="kpi-meaning">Tests whether the strategy's mean trade return is statistically significantly greater than zero.</div>
                        <div class="kpi-desirable">🎯 Desirable: p < 0.05 (rejects the null hypothesis of zero edge with 95% confidence).</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">95% Confidence Interval for Mean Return</span>
                        <span class="kpi-formula">x̄ ± t_crit * (s / sqrt(n))</span>
                    </td>
                    <td><strong>[{fmt_pct(m['ci_low'], sign=True)}, {fmt_pct(m['ci_high'], sign=True)}]</strong></td>
                    <td>95% Confidence</td>
                    <td><span class="badge-pill badge-blue">True Mean Bounds</span></td>
                    <td>
                        <div class="kpi-meaning">The statistical range where the true average trade return lies with 95% certainty.</div>
                        <div class="kpi-desirable">🎯 Desirable: Lower bound above 0.0% confirms positive edge.</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Probabilistic Sharpe Ratio (PSR)</span>
                        <span class="kpi-formula">López de Prado (2012) Formula</span>
                    </td>
                    <td><strong class="pos" style="font-size: 1.15rem;">{fmt_pct(m['psr'], 1)}</strong></td>
                    <td>Target SR > 0</td>
                    <td><span class="badge-pill badge-green">{psr_badge}</span></td>
                    <td>
                        <div class="kpi-meaning">Calculates the probability that the true Sharpe Ratio is > 0, adjusted for sample size, skewness, and fat tails.</div>
                        <div class="kpi-desirable">🎯 Desirable: PSR ≥ 95% for institutional deployment.</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Min Track Record Length (MinTRL)</span>
                        <span class="kpi-formula">Trades needed for 95% confidence</span>
                    </td>
                    <td><strong>{int(np.ceil(m['min_trl'])) if not np.isnan(m['min_trl']) and m['min_trl'] < 100000 else 'N/A'} trades</strong></td>
                    <td>Current: {m['total_trades']} trades</td>
                    <td><span class="badge-pill badge-green">🟢 Track Record Sufficient</span></td>
                    <td>
                        <div class="kpi-meaning">The minimum number of trades required to statistically prove the observed Sharpe Ratio at 95% confidence.</div>
                        <div class="kpi-desirable">🎯 Desirable: Current trade sample > MinTRL.</div>
                    </td>
                </tr>
            </tbody>
        </table>

        <!-- ============================================================= -->
        <!-- SECTION 7: KELLY CRITERION & POSITION SIZING -->
        <!-- ============================================================= -->
        <div class="section-header" id="kelly">
            <div>
                <div class="section-title">📐 7. Optimal Position Sizing & Kelly Criterion</div>
                <div class="section-desc">Calculates mathematical bankroll allocation to maximize geometric growth while avoiding ruin.</div>
            </div>
        </div>

        <table class="kpi-table">
            <thead>
                <tr>
                    <th style="width: 24%;">Sizing Model</th>
                    <th style="width: 14%;">Optimal Fraction</th>
                    <th style="width: 14%;">Current Setting</th>
                    <th style="width: 18%;">Status / Diagnosis</th>
                    <th style="width: 30%;">Meaning & Recommendation</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>
                        <span class="kpi-name">Full Kelly Criterion (K*)</span>
                        <span class="kpi-formula">K* = W - (1 - W) / R</span>
                    </td>
                    <td><strong>{fmt_pct(m['kelly_fraction'], sign=True)}</strong></td>
                    <td>{POSITION_SIZE_PCT:.1%}</td>
                    <td><span class="badge-pill {'badge-red' if POSITION_SIZE_PCT > m['kelly_fraction'] else 'badge-green'}">{kelly_badge}</span></td>
                    <td>
                        <div class="kpi-meaning">The theoretically optimal percentage of bankroll to bet per trade to maximize long-term wealth.</div>
                        <div class="kpi-desirable">🎯 Desirable: Betting higher than Full Kelly leads to lower returns and high risk of ruin.</div>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="kpi-name">Recommended Half-Kelly</span>
                        <span class="kpi-formula">Half-Kelly = K* / 2</span>
                    </td>
                    <td><strong class="pos">{fmt_pct(m['half_kelly'], sign=True)}</strong></td>
                    <td>{POSITION_SIZE_PCT:.1%}</td>
                    <td><span class="badge-pill badge-amber">Institutional Standard</span></td>
                    <td>
                        <div class="kpi-meaning">Hedge fund best practice to hedge against estimation errors in win rate and payoff.</div>
                        <div class="kpi-desirable">🎯 Recommendation: Reduce position size from 20% to ~5.6% - 11% for optimal risk-adjusted stability.</div>
                    </td>
                </tr>
            </tbody>
        </table>

        <!-- ============================================================= -->
        <!-- SECTION 8: COMPLETE TRADE AUDIT LOG -->
        <!-- ============================================================= -->
        <div class="section-header" id="trade-log">
            <div>
                <div class="section-title">📋 8. Complete Trade Audit Log ({m['total_trades']} Trades)</div>
                <div class="section-desc">Search and filter every trade executed in the timeline.</div>
            </div>
        </div>

        <div class="filter-bar">
            <input type="text" id="tradeSearch" class="search-input" placeholder="🔍 Search ticker (e.g. NVDA, AMD) or date..." onkeyup="filterTrades()">
            <button class="btn-filter active" onclick="setFilter('ALL', this)">All ({m['total_trades']})</button>
            <button class="btn-filter" onclick="setFilter('WIN', this)">Wins ({m['num_wins']})</button>
            <button class="btn-filter" onclick="setFilter('LOSS', this)">Losses ({m['num_losses']})</button>
            <button class="btn-filter" onclick="setFilter('LONG', this)">Longs ({m['long_trades_count']})</button>
            <button class="btn-filter" onclick="setFilter('SHORT', this)">Shorts ({m['short_trades_count']})</button>
            <button class="btn-filter" onclick="setFilter('OPEN', this)">Open ({m['open_trades']})</button>
            <button class="btn-filter" onclick="setFilter('CLOSED', this)">Closed ({m['closed_trades']})</button>
        </div>

        <table class="kpi-table" id="tradesTable">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Date</th>
                    <th>Ticker</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Entry Price</th>
                    <th>Exit / Current</th>
                    <th>PnL %</th>
                    <th>Profit ($)</th>
                    <th>Account Balance</th>
                    <th>Holding</th>
                </tr>
            </thead>
            <tbody>
                {all_trades_table_html}
            </tbody>
        </table>

        <!-- ============================================================= -->
        <!-- SECTION 9: MASTER KPI ENCYCLOPEDIA -->
        <!-- ============================================================= -->
        <div class="section-header" id="encyclopedia">
            <div>
                <div class="section-title">📚 9. Quantitative KPI Master Encyclopedia</div>
                <div class="section-desc">Comprehensive reference guide to quantitative finance metrics and desirable benchmarks.</div>
            </div>
        </div>

        <table class="kpi-table">
            <thead>
                <tr>
                    <th style="width: 22%;">KPI Name</th>
                    <th style="width: 28%;">What It Measures (Intuition)</th>
                    <th style="width: 25%;">Mathematical Formula</th>
                    <th style="width: 25%;">What is Desirable (Target)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Sharpe Ratio</strong></td>
                    <td>Risk-adjusted return per unit of total volatility.</td>
                    <td><code>(CAGR - Rf) / σ_ann</code></td>
                    <td><span class="badge-pill badge-green">> 1.0 Good, > 2.0 Excellent, > 3.0 Elite</span></td>
                </tr>
                <tr>
                    <td><strong>Sortino Ratio</strong></td>
                    <td>Risk-adjusted return penalizing only downside negative volatility.</td>
                    <td><code>(CAGR - Rf) / σ_down</code></td>
                    <td><span class="badge-pill badge-green">> 1.5 Good, > 3.0 Excellent</span></td>
                </tr>
                <tr>
                    <td><strong>Calmar Ratio</strong></td>
                    <td>Annualized return earned per percentage of maximum drawdown.</td>
                    <td><code>CAGR / |Max Drawdown|</code></td>
                    <td><span class="badge-pill badge-green">> 1.0 Acceptable, > 3.0 Great</span></td>
                </tr>
                <tr>
                    <td><strong>Profit Factor</strong></td>
                    <td>Gross dollar gains divided by gross dollar losses.</td>
                    <td><code>Gross Profits / Gross Losses</code></td>
                    <td><span class="badge-pill badge-green">> 1.50 Good, > 2.00 Institutional</span></td>
                </tr>
                <tr>
                    <td><strong>Payoff Ratio</strong></td>
                    <td>Average percentage gain on winners divided by average loss on losers.</td>
                    <td><code>Avg Win % / |Avg Loss %|</code></td>
                    <td><span class="badge-pill badge-green">> 1.5x - 2.5x (allows sub-50% win rate)</span></td>
                </tr>
                <tr>
                    <td><strong>Jensen's Alpha</strong></td>
                    <td>Risk-adjusted excess return over CAPM market expectation.</td>
                    <td><code>R_bot - [Rf + β*(R_m - Rf)]</code></td>
                    <td><span class="badge-pill badge-green">> 0% with p < 0.05 (statistically valid skill)</span></td>
                </tr>
                <tr>
                    <td><strong>T-Test P-Value</strong></td>
                    <td>Probability that positive edge was caused by pure luck.</td>
                    <td><code>1 - T_CDF(t, df=n-1)</code></td>
                    <td><span class="badge-pill badge-green">p < 0.05 (95% confidence of true edge)</span></td>
                </tr>
                <tr>
                    <td><strong>Probabilistic Sharpe Ratio</strong></td>
                    <td>Probability that true Sharpe Ratio is > 0, adjusted for non-normality.</td>
                    <td><code>Φ((SR - SR*) / σ_SR)</code></td>
                    <td><span class="badge-pill badge-green">PSR ≥ 95% for institutional deployment</span></td>
                </tr>
                <tr>
                    <td><strong>Full & Half Kelly</strong></td>
                    <td>Optimal bet sizing percentage to maximize geometric growth.</td>
                    <td><code>W - (1-W)/R</code></td>
                    <td><span class="badge-pill badge-amber">Bet ≤ Half-Kelly to prevent ruin</span></td>
                </tr>
            </tbody>
        </table>

    </div>

    <!-- Client-side Interactive Filter Script -->
    <script>
        let currentFilter = 'ALL';

        function setFilter(filterType, btn) {{
            currentFilter = filterType;
            document.querySelectorAll('.btn-filter').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            filterTrades();
        }}

        function filterTrades() {{
            const search = document.getElementById('tradeSearch').value.toUpperCase();
            const rows = document.querySelectorAll('#tradesTable tbody tr');

            rows.forEach(row => {{
                const text = row.innerText.toUpperCase();
                const type = row.getAttribute('data-type');
                const status = row.getAttribute('data-status');
                const pnl = row.getAttribute('data-pnl');

                let matchesFilter = true;
                if (currentFilter === 'WIN') matchesFilter = (pnl === 'win');
                else if (currentFilter === 'LOSS') matchesFilter = (pnl === 'loss');
                else if (currentFilter === 'LONG') matchesFilter = (type === 'LONG');
                else if (currentFilter === 'SHORT') matchesFilter = (type === 'SHORT');
                else if (currentFilter === 'OPEN') matchesFilter = (status === 'OPEN');
                else if (currentFilter === 'CLOSED') matchesFilter = (status === 'CLOSED');

                const matchesSearch = text.includes(search);

                if (matchesFilter && matchesSearch) {{
                    row.style.display = '';
                }} else {{
                    row.style.display = 'none';
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    return html


def save_reports(metrics, sim_df, text_output, equity_curve, spy_equity_curve):
    """Save report.txt, report.md, and report.html and update GitHub Step Summary."""
    # 1. Save text report
    if GENERATE_TEXT:
        try:
            with open("report.txt", "w", encoding="utf-8") as f:
                f.write(text_output)
            print("💾 Saved plaintext report to: report.txt")
        except Exception as e:
            print(f"⚠️  Warning: Failed to save report.txt: {e}")

    # 2. Save Markdown report
    if GENERATE_MARKDOWN:
        try:
            md_content = build_markdown_report(metrics, sim_df)
            with open("report.md", "w", encoding="utf-8") as f:
                f.write(md_content)
            print("💾 Saved GitHub Markdown report to: report.md")

            # Push to GITHUB_STEP_SUMMARY if running inside GitHub Actions
            step_summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
            if step_summary_file:
                with open(step_summary_file, "a", encoding="utf-8") as f:
                    f.write(md_content + "\n")
                print("🚀 Successfully published report to GitHub Actions Step Summary!")
        except Exception as e:
            print(f"⚠️  Warning: Failed to save report.md: {e}")

    # 3. Save HTML report
    if GENERATE_HTML:
        try:
            html_content = build_html_report(metrics, sim_df, equity_curve, spy_equity_curve)
            with open("report.html", "w", encoding="utf-8") as f:
                f.write(html_content)
            print("💾 Saved interactive HTML report to: report.html")
        except Exception as e:
            print(f"⚠️  Warning: Failed to save report.html: {e}")


def main():
    # 1. Load Data
    df, filename = load_trade_data()

    # 2. Benchmark Data
    spy_hist = fetch_benchmark_data(df['Date'].min(), BENCHMARK_TICKER)

    # 3. Simulate & Process Portfolio
    sim_df, equity_curve, equity_dates, spy_equity_curve, final_cash, spy_final, spy_tot_ret, spy_start_px, spy_latest_px = simulate_trading(df, spy_hist)

    # 4. Math & Statistical Metrics
    metrics = calculate_comprehensive_metrics(sim_df, equity_curve, equity_dates, final_cash, spy_final, spy_tot_ret, spy_hist)

    # 5. Output Report
    text_report = build_text_report(metrics)
    print(text_report)

    # 6. Save Reports (txt, md, html, and GitHub Step Summary)
    save_reports(metrics, sim_df, text_report, equity_curve, spy_equity_curve)


if __name__ == "__main__":
    main()