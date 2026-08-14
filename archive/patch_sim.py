import re

with open('status.py', 'r') as f:
    content = f.read()

# Replace fetch_open_prices_batch with fetch_historical_prices
def_fetch = re.search(r"def fetch_open_prices_batch.*?def simulate_trading", content, flags=re.DOTALL)
if def_fetch:
    new_fetch = """def fetch_historical_prices(tickers, start_date):
    \"\"\"Batch fetch historical market prices for all traded tickers.\"\"\"
    if not tickers:
        return pd.DataFrame()
    
    unique_tickers = list(set(tickers))
    print(f"⚡ Batch fetching historical prices for {len(unique_tickers)} tickers...")
    
    try:
        # Download historical data
        data = yf.download(unique_tickers, start=start_date, progress=False, threads=True)
        prices = pd.DataFrame(index=data.index)
        if len(unique_tickers) == 1:
            if 'Close' in data.columns:
                prices[unique_tickers[0]] = data['Close']
        else:
            if 'Close' in data.columns:
                for ticker in unique_tickers:
                    if ticker in data['Close'].columns:
                        prices[ticker] = data['Close'][ticker]
        prices = prices.ffill()
        prices.index = prices.index.tz_localize(None).normalize()
        return prices
    except Exception as e:
        print(f"⚠️  Batch download failed ({e})")
        return pd.DataFrame()

def simulate_trading"""
    content = content.replace(def_fetch.group(0), new_fetch)
else:
    print("Could not find fetch_open_prices_batch")

# Replace simulate_trading
def_sim = re.search(r"def simulate_trading.*?def calculate_comprehensive_metrics", content, flags=re.DOTALL)
if def_sim:
    new_sim = """def simulate_trading(df, spy_hist):
    \"\"\"
    Run realistic concurrent portfolio accounting:
    - Group trades by entry date (batch)
    - Position sizing = min(POSITION_SIZE_PCT, available_capital / n_new_trades)
    - Capital deployment cap at 90% (10% cash reserve)
    - Max concurrent positions
    - Daily mark-to-market for equity curve
    \"\"\"
    today = pd.Timestamp.today().normalize()
    start_date = df['Date'].min()

    # Get historical prices for all traded tickers
    all_tickers = df['Ticker'].unique().tolist()
    hist_prices = fetch_historical_prices(all_tickers, start_date - pd.Timedelta(days=5))

    # Pre-process df to ensure Exit_Date is set
    df['Real_Exit_Date'] = df['Exit_Date']
    for idx, row in df.iterrows():
        if row['Status'] == 'CLOSED' and pd.isna(row['Exit_Date']):
            df.at[idx, 'Real_Exit_Date'] = row['Date'] + pd.Timedelta(days=1)
        elif row['Status'] != 'CLOSED':
            df.at[idx, 'Real_Exit_Date'] = today

    # Create daily calendar grid
    all_dates = pd.concat([df['Date'], df['Real_Exit_Date']]).dropna().dt.normalize().unique()
    calendar_days = sorted(list(all_dates))
    
    spy_available = not spy_hist.empty
    if spy_available:
        spy_days = spy_hist.index.normalize().tolist()
        start_d = calendar_days[0]
        end_d = today
        calendar_days = sorted(list(set(calendar_days + [d for d in spy_days if start_d <= d <= end_d])))
        try:
            spy_start_px = float(spy_hist.asof(calendar_days[0])['Close'])
            spy_latest_px = float(spy_hist['Close'].iloc[-1])
        except Exception:
            spy_start_px = 100.0
            spy_latest_px = 100.0
    else:
        spy_start_px = 100.0
        spy_latest_px = 100.0

    # State variables
    current_cash = STARTING_BALANCE
    equity_curve = []
    equity_dates = []
    spy_equity_curve = []
    
    open_positions = [] # list of dicts
    trade_results = []
    
    print("\\n" + "=" * 90)
    print(" 📋 REAL-TIME PORTFOLIO SIMULATION (Concurrent Batch Allocation)")
    print("=" * 90)
    
    for current_day in calendar_days:
        current_positions_value = 0.0
        still_open = []
        
        # 1. Update prices for all open positions and close those exiting today
        for pos in open_positions:
            idx = pos['row_idx']
            trade = df.loc[idx]
            ticker = trade['Ticker']
            
            # Look up price for today
            if not hist_prices.empty and current_day in hist_prices.index and pd.notna(hist_prices.at[current_day, ticker]):
                curr_px = hist_prices.at[current_day, ticker]
            else:
                curr_px = pos['last_px']
                
            is_closing_today = False
            if trade['Status'] == 'CLOSED' and trade['Real_Exit_Date'] == current_day:
                is_closing_today = True
                if 'Exit_Price' in trade and pd.notna(trade['Exit_Price']) and float(trade['Exit_Price']) > 0:
                    curr_px = float(trade['Exit_Price'])
                elif pd.notna(trade['PnL']) and trade['PnL'] != 0.0:
                    pnl_pct = float(trade['PnL'])
                    curr_px = trade['Entry_Price'] * (1 + pnl_pct if pos['type'] == 'LONG' else 1 - pnl_pct)
                    
            pos['last_px'] = curr_px
            
            # Value the position
            if pos['type'] == 'LONG':
                pos_value = pos['shares'] * curr_px
            else:
                pos_value = pos['allocated'] + (pos['entry_px'] - curr_px) * pos['shares']
                
            # Short borrow cost deduction (daily)
            if pos['type'] == 'SHORT':
                borrow_cost = pos['allocated'] * (SHORT_BORROW_COST_ANNUAL / 365.25)
                pos_value -= borrow_cost
                pos['borrow_fees_paid'] += borrow_cost
                
            if is_closing_today:
                current_cash += pos_value
                
                # Record the completed trade
                profit_dollars = pos_value - pos['allocated']
                trade_pnl_pct = profit_dollars / pos['allocated'] if pos['allocated'] > 0 else 0.0
                h_days = max((current_day - trade['Date']).days, 1)
                
                trade_record = {
                    'Index': idx + 1,
                    'Date': trade['Date'],
                    'Exit_Date': current_day,
                    'Ticker': ticker,
                    'Type': pos['type'],
                    'Status': 'CLOSED',
                    'Entry_Price': pos['entry_px'],
                    'Exit_Price': curr_px,
                    'PnL_Pct': trade_pnl_pct,
                    'Profit_Dollars': profit_dollars,
                    'Balance_After': current_cash + current_positions_value, # Approx at close
                    'Holding_Days': h_days,
                }
                trade_results.append(trade_record)
            else:
                current_positions_value += pos_value
                still_open.append(pos)
                
        open_positions = still_open
        
        # 2. Process NEW trades for today
        new_trades = df[df['Date'] == current_day]
        if not new_trades.empty:
            total_equity_before_alloc = current_cash + current_positions_value
            max_deployable = total_equity_before_alloc * (1.0 - CASH_RESERVE_PCT)
            available_cap = min(current_cash, max(0.0, max_deployable - current_positions_value))
            
            # Only consider trades we have room for up to MAX_POSITIONS
            slots_available = MAX_POSITIONS - len(open_positions)
            trades_to_process = new_trades.head(slots_available)
            n_new = len(trades_to_process)
            
            for idx, trade in new_trades.iterrows():
                if idx not in trades_to_process.index:
                    df.at[idx, 'Status'] = 'SKIPPED_MAX_POS'
                    continue
                
                allocation_pct = min(POSITION_SIZE_PCT, (available_cap / n_new) / total_equity_before_alloc if total_equity_before_alloc > 0 else 0)
                allocation_amount = allocation_pct * total_equity_before_alloc
                
                if allocation_amount <= 0 or current_cash < allocation_amount:
                    df.at[idx, 'Status'] = 'SKIPPED_NO_CAPITAL'
                    continue
                    
                entry_px = float(trade['Entry_Price'])
                shares = allocation_amount / entry_px if entry_px > 0 else 0.0
                
                current_cash -= allocation_amount
                pos = {
                    'row_idx': idx,
                    'shares': shares,
                    'allocated': allocation_amount,
                    'type': str(trade['Type']).upper().strip(),
                    'entry_px': entry_px,
                    'last_px': entry_px,
                    'borrow_fees_paid': 0.0
                }
                open_positions.append(pos)
                current_positions_value += allocation_amount
                
        # 3. Record daily equity
        total_equity_end = current_cash + current_positions_value
        equity_curve.append(total_equity_end)
        equity_dates.append(current_day)
        
        if spy_available:
            try:
                spy_px = float(spy_hist.asof(current_day)['Close'])
                spy_eq = (STARTING_BALANCE / spy_start_px) * spy_px
            except Exception:
                spy_eq = spy_equity_curve[-1] if spy_equity_curve else STARTING_BALANCE
            spy_equity_curve.append(spy_eq)
            
    # Record open trades that never closed
    for pos in open_positions:
        idx = pos['row_idx']
        trade = df.loc[idx]
        ticker = trade['Ticker']
        pos_value = pos['shares'] * pos['last_px'] if pos['type'] == 'LONG' else pos['allocated'] + (pos['entry_px'] - pos['last_px']) * pos['shares'] - pos['borrow_fees_paid']
        profit_dollars = pos_value - pos['allocated']
        trade_pnl_pct = profit_dollars / pos['allocated'] if pos['allocated'] > 0 else 0.0
        h_days = max((today - trade['Date']).days, 1)
        
        trade_record = {
            'Index': idx + 1,
            'Date': trade['Date'],
            'Exit_Date': today,
            'Ticker': ticker,
            'Type': pos['type'],
            'Status': 'OPEN',
            'Entry_Price': pos['entry_px'],
            'Exit_Price': pos['last_px'],
            'PnL_Pct': trade_pnl_pct,
            'Profit_Dollars': profit_dollars,
            'Balance_After': total_equity_end,
            'Holding_Days': h_days,
        }
        trade_results.append(trade_record)

    sim_df = pd.DataFrame(trade_results)
    if not sim_df.empty:
        sim_df = sim_df.sort_values(by=['Exit_Date', 'Date']).reset_index(drop=True)

    print(f"✅ Simulation Complete. Final Equity: ${total_equity_end:,.2f}")
    
    spy_shares = STARTING_BALANCE / spy_start_px
    spy_final_balance = spy_shares * spy_latest_px
    spy_total_return = (spy_final_balance - STARTING_BALANCE) / STARTING_BALANCE

    return sim_df, equity_curve, equity_dates, spy_equity_curve, current_cash, spy_final_balance, spy_total_return, spy_start_px, spy_latest_px

def calculate_comprehensive_metrics"""
    content = content.replace(def_sim.group(0), new_sim)
else:
    print("Could not find simulate_trading")

with open('status.py', 'w') as f:
    f.write(content)
print("Applied simulate_trading patch.")
