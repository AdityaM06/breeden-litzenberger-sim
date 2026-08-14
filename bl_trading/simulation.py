import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

from .config import TradingConfig, SimulationMode
from .data import fetch_current_prices, fetch_historical_prices


@dataclass
class SimulationResult:
    """Complete output from a simulation run."""
    trades: pd.DataFrame           # Enriched trade log with sim columns
    equity_curve: pd.DataFrame     # Daily equity values (date-indexed) for CONCURRENT mode, or per-trade for SEQUENTIAL
    benchmark_curve: pd.DataFrame  # Benchmark equity values aligned to same dates
    final_equity: float
    total_return: float
    starting_capital: float
    spy_final_balance: float
    spy_total_return: float
    config: TradingConfig
    skipped_trades: list           # Trades skipped due to capital/slot constraints (concurrent mode only)
    mode: str                      # 'concurrent' or 'sequential'


def run_simulation(df: pd.DataFrame, config: TradingConfig, spy_hist: pd.DataFrame) -> SimulationResult:
    """
    Main entry point for portfolio simulation. Dispatches to the chosen mode.
    """
    if df.empty:
        raise ValueError("Trade dataframe is empty.")

    if config.simulation_mode == SimulationMode.SEQUENTIAL:
        return _simulate_sequential(df, config, spy_hist)
    else:
        return _simulate_concurrent(df, config, spy_hist)


def _simulate_sequential(df: pd.DataFrame, config: TradingConfig, spy_hist: pd.DataFrame) -> SimulationResult:
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
    live_prices = {}
    if open_tickers:
        try:
            live_prices = fetch_current_prices(open_tickers)
        except Exception:
            pass

    # State variables
    current_cash = config.starting_capital
    equity_curve = [config.starting_capital]
    equity_dates = [df['Date'].min()]
    spy_equity_curve = [config.starting_capital]
    
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
    print(" 📋 REAL-TIME TRADE LOG & SIMULATION (Sequential Mode)")
    print("=" * 80)
    print(f"{'#':<4} {'DATE':<11} {'TICKER':<6} {'TYPE':<6} {'STATUS':<7} {'ENTRY':<9} {'EXIT/CURR':<10} {'PnL %':<9} {'PROFIT ($)':<12} {'COSTS':<8} {'BALANCE':<12}")
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
            if pd.notna(row.get('PnL')) and row['PnL'] != 0.0:
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
        bet_amount = current_cash * config.position_size_pct
        
        total_costs = 0.0
        if getattr(config, 'enable_costs', False):
            entry_slip = bet_amount * getattr(config, 'slippage_pct', 0.0)
            exit_slip = bet_amount * getattr(config, 'slippage_pct', 0.0)
            comm = 2 * getattr(config, 'commission_per_trade', 0.0)
            borrow = 0.0
            if trade_type == 'SHORT':
                borrow = bet_amount * getattr(config, 'short_borrow_daily_pct', 0.0) * h_days
            total_costs = entry_slip + exit_slip + comm + borrow
            
        profit_dollars = bet_amount * trade_pnl_pct - total_costs
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
                spy_current_sim_val = (config.starting_capital / spy_start_px) * spy_exit
            except Exception:
                spy_win_ret = np.nan
                spy_current_sim_val = config.starting_capital
        else:
            spy_win_ret = np.nan
            spy_current_sim_val = config.starting_capital
            
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
            'Costs': total_costs,
            'Balance_After': current_cash,
            'Holding_Days': h_days,
            'Step_Return': step_ret,
            'SPY_Window_Return': spy_win_ret
        }
        trade_results.append(trade_record)

        # Print visual row
        MAX_DISPLAY_TRADES = getattr(config, 'max_display_trades', 0)
        should_print = (MAX_DISPLAY_TRADES == 0 or total_rows <= MAX_DISPLAY_TRADES or 
                        i < 10 or i >= total_rows - 10)
        if should_print:
            status_tag = "CLOSED" if is_closed else "OPEN*"
            pnl_color_str = f"{trade_pnl_pct:+.2%}"
            print(f"{i+1:<4} {entry_date.strftime('%Y-%m-%d'):<11} {ticker:<6} {trade_type:<6} {status_tag:<7} ${entry_price:<8.2f} ${exit_price:<9.2f} {pnl_color_str:<9} ${profit_dollars:+10.2f} ${total_costs:<7.2f} ${current_cash:11.2f}")
        elif i == 10:
            print(f"... [{total_rows - 20} intermediate trades omitted for brevity] ...")

    print("-" * 80)
    print("(* Note: OPEN positions marked-to-market using latest live market prices)\n")

    sim_df = pd.DataFrame(trade_results)
    
    # SPY benchmark final calculations
    spy_shares = config.starting_capital / spy_start_px
    spy_final_balance = spy_shares * spy_latest_px
    spy_total_return = (spy_final_balance - config.starting_capital) / config.starting_capital
    
    total_return = (current_cash - config.starting_capital) / config.starting_capital
    
    eq_df = pd.DataFrame({'Date': equity_dates, 'Equity': equity_curve}).drop_duplicates('Date', keep='last')
    bm_df = pd.DataFrame({'Date': equity_dates, 'Benchmark_Equity': spy_equity_curve}).drop_duplicates('Date', keep='last')

    return SimulationResult(
        trades=sim_df,
        equity_curve=eq_df,
        benchmark_curve=bm_df,
        final_equity=current_cash,
        total_return=total_return,
        starting_capital=config.starting_capital,
        spy_final_balance=spy_final_balance,
        spy_total_return=spy_total_return,
        config=config,
        skipped_trades=[],
        mode='sequential'
    )


def _simulate_concurrent(df: pd.DataFrame, config: TradingConfig, spy_hist: pd.DataFrame) -> SimulationResult:
    """
    Run a realistic concurrent daily walk-forward simulation:
    - Obeys max capital constraints and max open positions
    - Evaluates concurrent positions
    - Daily mark-to-market accounting
    """
    # 1. Determine simulation date range
    start_date = df['Date'].min()
    today = pd.Timestamp.today().normalize()
    
    if pd.isna(start_date):
        raise ValueError("No valid entry dates found in trade log.")

    # 2. Fetch historical close prices for ALL tickers + benchmark in one batch call
    tickers = df['Ticker'].dropna().unique().tolist()
    hist_prices = pd.DataFrame()
    if tickers:
        try:
            hist_res = fetch_historical_prices(tickers, start_date=start_date, end_date=today)
            if isinstance(hist_res, dict):
                hist_prices = pd.DataFrame(hist_res)
            elif isinstance(hist_res, pd.DataFrame):
                hist_prices = hist_res
        except Exception as e:
            print(f"Warning: Could not fetch historical prices: {e}")
            
    # 3. Fetch current live prices for open positions (mark-to-market)
    open_mask = df['Status'] != 'CLOSED'
    open_tickers = df.loc[open_mask, 'Ticker'].dropna().unique().tolist()
    live_prices = {}
    if open_tickers:
        try:
            live_prices = fetch_current_prices(open_tickers)
        except Exception:
            pass
            
    # Merge live prices into hist_prices
    if not hist_prices.empty:
        if today not in hist_prices.index:
            hist_prices.loc[today] = pd.Series(live_prices)
        else:
            for tk, px in live_prices.items():
                hist_prices.at[today, tk] = px
    
    # Forward fill to handle missing market data
    if not hist_prices.empty:
        hist_prices = hist_prices.ffill()

    # 4. Initialize state
    cash = config.starting_capital
    open_positions = {}
    equity_curve_records = []
    skipped_trades = []
    trade_results = []
    
    enable_costs = getattr(config, 'enable_costs', False)
    slippage_pct = getattr(config, 'slippage_pct', 0.0)
    comm_per_trade = getattr(config, 'commission_per_trade', 0.0)
    borrow_daily_pct = getattr(config, 'short_borrow_daily_pct', 0.0)
    max_open = getattr(config, 'max_open_positions', 10)
    cash_reserve = getattr(config, 'cash_reserve_pct', 0.05)
    
    # 5. Build an event schedule
    events_by_date = {}
    for i, row in df.iterrows():
        entry_date = row['Date']
        is_closed = (row['Status'] == 'CLOSED')
        
        exit_date = row.get('Exit_Date', pd.NaT)
        if pd.isna(exit_date) or not is_closed:
            exit_date = today

        row_dict = row.to_dict()
        row_dict['_index'] = i
        
        if entry_date not in events_by_date:
            events_by_date[entry_date] = {'entries': [], 'exits': []}
        events_by_date[entry_date]['entries'].append(row_dict)
        
        if exit_date not in events_by_date:
            events_by_date[exit_date] = {'entries': [], 'exits': []}
        events_by_date[exit_date]['exits'].append(row_dict)
        
    b_days = pd.bdate_range(start=start_date, end=today)
    
    for current_date in b_days:
        events = events_by_date.get(current_date, {'entries': [], 'exits': []})
        
        # a. Process EXITS scheduled for this day
        for exit_event in events['exits']:
            pos_id = exit_event['_index']
            if pos_id in open_positions:
                pos = open_positions.pop(pos_id)
                ticker = pos['Ticker']
                trade_type = pos['Type'].upper()
                entry_price = pos['Entry_Price']
                bet_amount = pos['Bet_Amount']
                is_closed = (exit_event['Status'] == 'CLOSED')
                
                # Determine exit price
                if is_closed:
                    if pd.notna(exit_event.get('PnL')) and exit_event['PnL'] != 0.0:
                        trade_pnl_pct = float(exit_event['PnL'])
                        if pd.notna(exit_event.get('Exit_Price')) and float(exit_event['Exit_Price']) > 0:
                            exit_price = float(exit_event['Exit_Price'])
                        else:
                            exit_price = entry_price * (1 + trade_pnl_pct if trade_type == 'LONG' else 1 - trade_pnl_pct)
                    elif pd.notna(exit_event.get('Exit_Price')) and float(exit_event['Exit_Price']) > 0:
                        exit_price = float(exit_event['Exit_Price'])
                    else:
                        exit_price = entry_price
                else:
                    exit_price = hist_prices.get(ticker, pd.Series()).get(current_date, entry_price)
                
                if pd.isna(exit_price):
                    exit_price = entry_price
                    
                if trade_type == 'LONG':
                    trade_pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
                else:
                    trade_pnl_pct = (entry_price - exit_price) / entry_price if entry_price > 0 else 0.0
                    
                gross_profit = bet_amount * trade_pnl_pct
                h_days = max((current_date - pos['Entry_Date']).days, 1)
                
                # Costs
                exit_slip = bet_amount * slippage_pct if enable_costs else 0.0
                pos['Entry_Cost'] += exit_slip
                
                net_profit = gross_profit - pos['Entry_Cost']
                cash += (bet_amount + net_profit)
                
                trade_results.append({
                    'Index': pos_id + 1,
                    'Date': pos['Entry_Date'],
                    'Exit_Date': current_date,
                    'Ticker': ticker,
                    'Type': trade_type,
                    'Status': 'CLOSED' if is_closed else 'OPEN',
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'PnL_Pct': trade_pnl_pct,
                    'Profit_Dollars': net_profit,
                    'Costs': pos['Entry_Cost'],
                    'Holding_Days': h_days,
                    'Balance_After': total_equity if 'total_equity' in locals() else cash,
                    'Step_Return': trade_pnl_pct,
                    'SPY_Window_Return': np.nan
                })

        # Pre-calculate portfolio value to determine available capital for entries
        current_portfolio_value = 0.0
        for pos_id, pos in open_positions.items():
            current_price = hist_prices.get(pos['Ticker'], pd.Series()).get(current_date, pos['Entry_Price'])
            if pd.isna(current_price): current_price = pos['Entry_Price']
            if pos['Type'] == 'LONG':
                unrealized_pct = (current_price - pos['Entry_Price']) / pos['Entry_Price'] if pos['Entry_Price'] > 0 else 0.0
            else:
                unrealized_pct = (pos['Entry_Price'] - current_price) / pos['Entry_Price'] if pos['Entry_Price'] > 0 else 0.0
            current_portfolio_value += pos['Bet_Amount'] * (1 + unrealized_pct)
            
        total_equity = cash + current_portfolio_value
        
        # b. Process ENTRIES scheduled for this day
        for entry_event in events['entries']:
            if len(open_positions) >= max_open:
                skipped_trades.append({**entry_event, 'Skip_Reason': 'Max positions reached'})
                continue
                
            locked_capital = sum(p['Bet_Amount'] for p in open_positions.values())
            available_capital = total_equity * (1 - cash_reserve) - locked_capital
            
            position_capital = min(total_equity * config.position_size_pct, available_capital)
            
            if position_capital <= 0:
                skipped_trades.append({**entry_event, 'Skip_Reason': 'Insufficient capital'})
                continue
                
            entry_slip = position_capital * slippage_pct if enable_costs else 0.0
            comm = 2 * comm_per_trade if enable_costs else 0.0
            entry_cost = entry_slip + comm
            
            if cash < (position_capital + entry_cost):
                position_capital = cash - entry_cost
                if position_capital <= 0:
                    skipped_trades.append({**entry_event, 'Skip_Reason': 'Insufficient cash for entry'})
                    continue
            
            cash -= (position_capital + entry_cost)
            open_positions[entry_event['_index']] = {
                'Ticker': entry_event['Ticker'],
                'Type': str(entry_event['Type']).upper().strip(),
                'Entry_Date': current_date,
                'Entry_Price': float(entry_event['Entry_Price']),
                'Bet_Amount': position_capital,
                'Entry_Cost': entry_cost
            }
            
        # c. Daily deduction for short borrowing costs
        if enable_costs and borrow_daily_pct > 0:
            for pos_id, pos in open_positions.items():
                if pos['Type'] == 'SHORT':
                    daily_borrow = pos['Bet_Amount'] * borrow_daily_pct
                    cash -= daily_borrow
                    pos['Entry_Cost'] += daily_borrow

        # d. Calculate EOD total equity
        current_portfolio_value = 0.0
        for pos_id, pos in open_positions.items():
            current_price = hist_prices.get(pos['Ticker'], pd.Series()).get(current_date, pos['Entry_Price'])
            if pd.isna(current_price): current_price = pos['Entry_Price']
            if pos['Type'] == 'LONG':
                unrealized_pct = (current_price - pos['Entry_Price']) / pos['Entry_Price'] if pos['Entry_Price'] > 0 else 0.0
            else:
                unrealized_pct = (pos['Entry_Price'] - current_price) / pos['Entry_Price'] if pos['Entry_Price'] > 0 else 0.0
            current_portfolio_value += pos['Bet_Amount'] * (1 + unrealized_pct)
            
        total_equity = cash + current_portfolio_value
        
        # Benchmark logic
        spy_eq = config.starting_capital
        if not spy_hist.empty:
            spy_start = spy_hist.asof(start_date)['Close'] if pd.notna(spy_hist.asof(start_date)['Close']) else spy_hist['Close'].iloc[0]
            spy_curr = spy_hist.asof(current_date)['Close']
            if pd.notna(spy_start) and pd.notna(spy_curr):
                spy_eq = (config.starting_capital / float(spy_start)) * float(spy_curr)

        # e. Record equity_curve entry
        equity_curve_records.append({
            'Date': current_date,
            'Cash': cash,
            'Equity': total_equity,
            'Positions_Open': len(open_positions),
            'Benchmark_Equity': spy_eq
        })

    # Wrap up any remaining open positions to populate trade_results for them
    for pos_id, pos in open_positions.items():
        ticker = pos['Ticker']
        trade_type = pos['Type']
        entry_price = pos['Entry_Price']
        bet_amount = pos['Bet_Amount']
        
        current_price = hist_prices.get(ticker, pd.Series()).get(today, entry_price)
        if pd.isna(current_price): current_price = entry_price
        
        if trade_type == 'LONG':
            trade_pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
        else:
            trade_pnl_pct = (entry_price - current_price) / entry_price if entry_price > 0 else 0.0
            
        gross_profit = bet_amount * trade_pnl_pct
        h_days = max((today - pos['Entry_Date']).days, 1)
        net_profit = gross_profit - pos['Entry_Cost']
        
        trade_results.append({
            'Index': pos_id + 1,
            'Date': pos['Entry_Date'],
            'Exit_Date': today,
            'Ticker': ticker,
            'Type': trade_type,
            'Status': 'OPEN',
            'Entry_Price': entry_price,
            'Exit_Price': current_price,
            'PnL_Pct': trade_pnl_pct,
            'Profit_Dollars': net_profit,
            'Costs': pos['Entry_Cost'],
            'Holding_Days': h_days,
            'Balance_After': total_equity if 'total_equity' in locals() else cash,
            'Step_Return': trade_pnl_pct,
            'SPY_Window_Return': np.nan
        })

    print("\n" + "=" * 80)
    print(" 📋 CONCURRENT SIMULATION SUMMARY")
    print("=" * 80)
    print(f"Total Trades Evaluated: {len(df)}")
    print(f"Executed: {len(trade_results)}")
    print(f"Skipped: {len(skipped_trades)}")
    print("-" * 80)
    
    sim_df = pd.DataFrame(trade_results)
    if not sim_df.empty:
        sim_df = sim_df.sort_values('Date').reset_index(drop=True)
        
    eq_df = pd.DataFrame(equity_curve_records)
    if not eq_df.empty:
        bm_df = eq_df[['Date', 'Benchmark_Equity']].copy()
    else:
        bm_df = pd.DataFrame(columns=['Date', 'Benchmark_Equity'])
    
    # SPY final calcs
    if not spy_hist.empty:
        spy_start_px = float(spy_hist.asof(start_date)['Close']) if pd.notna(spy_hist.asof(start_date)['Close']) else float(spy_hist['Close'].iloc[0])
        spy_latest_px = float(spy_hist['Close'].iloc[-1])
        spy_shares = config.starting_capital / spy_start_px
        spy_final_balance = spy_shares * spy_latest_px
        spy_total_return = (spy_final_balance - config.starting_capital) / config.starting_capital
    else:
        spy_final_balance = config.starting_capital
        spy_total_return = 0.0
        
    final_equity = eq_df['Equity'].iloc[-1] if not eq_df.empty else config.starting_capital
    total_ret = (final_equity - config.starting_capital) / config.starting_capital

    return SimulationResult(
        trades=sim_df,
        equity_curve=eq_df,
        benchmark_curve=bm_df,
        final_equity=final_equity,
        total_return=total_ret,
        starting_capital=config.starting_capital,
        spy_final_balance=spy_final_balance,
        spy_total_return=spy_total_return,
        config=config,
        skipped_trades=skipped_trades,
        mode='concurrent'
    )
