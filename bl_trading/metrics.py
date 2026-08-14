import numpy as np
import pandas as pd
import scipy.stats as stats
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .config import TradingConfig, SimulationMode
from .simulation import SimulationResult

def fmt_pct(v, decimals=2, sign=False) -> str:
    """Format floating point number as percentage."""
    if v is None or (isinstance(v, (int, float, np.number)) and np.isnan(v)):
        return "N/A"
    try:
        fmt = f"+.{decimals}%" if sign else f".{decimals}%"
        return f"{float(v):{fmt}}"
    except Exception:
        return "N/A"

def fmt_curr(v, sign=False) -> str:
    """Format floating point number as currency."""
    if v is None or (isinstance(v, (int, float, np.number)) and np.isnan(v)):
        return "N/A"
    try:
        return f"${float(v):+,.2f}" if sign else f"${float(v):,.2f}"
    except Exception:
        return "N/A"

def fmt_num(v, decimals=2, sign=False) -> str:
    """Format floating point number as decimal string."""
    if v is None or (isinstance(v, (int, float, np.number)) and np.isnan(v)):
        return "N/A"
    try:
        fmt = f"+.{decimals}f" if sign else f".{decimals}f"
        return f"{float(v):{fmt}}"
    except Exception:
        return "N/A"

@dataclass
class PerformanceMetrics:
    # Time/Returns
    start_date: pd.Timestamp = pd.NaT
    end_date: pd.Timestamp = pd.NaT
    total_days: int = 0
    years: float = np.nan
    bot_total_return: float = np.nan
    bot_net_profit: float = np.nan
    bot_cagr: float = np.nan
    spy_total_return: float = np.nan
    spy_net_profit: float = np.nan
    spy_cagr: float = np.nan
    naive_alpha: float = np.nan
    annualized_active_return: float = np.nan
    ending_balance: float = np.nan
    spy_final_balance: float = np.nan

    # Trade Stats
    total_trades: int = 0
    closed_trades: int = 0
    open_trades: int = 0
    skipped_trades: int = 0
    num_wins: int = 0
    num_losses: int = 0
    num_breakeven: int = 0
    win_rate: float = np.nan
    loss_rate: float = np.nan
    long_trades_count: int = 0
    short_trades_count: int = 0
    long_win_rate: float = np.nan
    short_win_rate: float = np.nan
    avg_trade_pnl: float = np.nan
    median_trade_pnl: float = np.nan
    std_trade_pnl: float = np.nan
    avg_win_pnl: float = np.nan
    avg_loss_pnl: float = np.nan
    avg_win_dollar: float = np.nan
    avg_loss_dollar: float = np.nan
    payoff_ratio: float = np.nan
    gross_profit: float = np.nan
    gross_loss: float = np.nan
    profit_factor: float = np.nan
    expectancy_pct: float = np.nan
    expectancy_dollar: float = np.nan
    best_trade: Optional[Dict[str, Any]] = None
    worst_trade: Optional[Dict[str, Any]] = None
    max_consec_wins: int = 0
    max_consec_losses: int = 0
    avg_holding_days: float = np.nan
    avg_win_holding_days: float = np.nan
    avg_loss_holding_days: float = np.nan
    trades_per_month: float = np.nan
    trades_per_year: float = np.nan

    # Risk
    peak_equity: float = np.nan
    max_drawdown: float = np.nan
    avg_drawdown: float = np.nan
    mdd_duration_steps: int = 0
    ann_volatility: float = np.nan
    downside_deviation: float = np.nan
    spy_ann_vol: float = np.nan
    spy_mdd: float = np.nan
    var_95: float = np.nan
    var_99: float = np.nan
    cvar_95: float = np.nan
    cvar_99: float = np.nan
    trade_skew: float = np.nan
    trade_kurtosis: float = np.nan
    drawdown_series: List[float] = field(default_factory=list)

    # Ratios
    sharpe_ratio: float = np.nan
    spy_sharpe: float = np.nan
    sortino_ratio: float = np.nan
    calmar_ratio: float = np.nan
    sterling_ratio: float = np.nan
    omega_ratio: float = np.nan
    gain_to_pain_ratio: float = np.nan

    # CAPM
    beta: float = np.nan
    beta_stderr: float = np.nan
    beta_pvalue: float = np.nan
    correlation: float = np.nan
    r_squared: float = np.nan
    jensen_alpha_annual: float = np.nan
    alpha_tstat: float = np.nan
    alpha_pvalue: float = np.nan
    tracking_error: float = np.nan
    information_ratio: float = np.nan
    treynor_ratio: float = np.nan
    up_capture: float = np.nan
    down_capture: float = np.nan

    # Stats
    t_stat_pnl: float = np.nan
    p_val_1tail: float = np.nan
    p_val_2tail: float = np.nan
    ci_low: float = np.nan
    ci_high: float = np.nan
    binom_pvalue: float = np.nan
    psr: float = np.nan
    min_trl: float = np.nan

    # Kelly
    kelly_fraction: float = np.nan
    half_kelly: float = np.nan

    # Costs
    total_commissions: float = 0.0
    total_slippage_cost: float = 0.0
    total_borrow_cost: float = 0.0
    total_costs: float = 0.0
    cost_drag_pct: float = np.nan

    # Config echo
    simulation_mode: str = ""
    position_size_pct: float = np.nan
    enable_costs: bool = False

def calculate_metrics(sim_result: SimulationResult, config: TradingConfig) -> PerformanceMetrics:
    """Calculate all quantitative trading performance, risk, and statistical metrics."""
    metrics = PerformanceMetrics()
    sim_df = sim_result.trades
    if sim_df.empty:
        return metrics

    equity_curve_df = sim_result.equity_curve
    benchmark_curve_df = sim_result.benchmark_curve
    final_cash = sim_result.final_equity
    spy_final_balance = sim_result.spy_final_balance
    spy_total_return = sim_result.spy_total_return

    # 1. TIME HORIZON & CORE RETURNS
    start_date = pd.to_datetime(sim_df['Date'].min())
    if 'Exit_Date' in sim_df.columns and not sim_df['Exit_Date'].isna().all():
        end_date = pd.to_datetime(sim_df['Exit_Date'].max())
        end_date = max(end_date, pd.Timestamp.today().normalize())
    else:
        end_date = pd.Timestamp.today().normalize()

    total_days = max((end_date - start_date).days, 1)
    years = total_days / 365.25

    bot_total_return = (final_cash - config.starting_capital) / config.starting_capital
    bot_net_profit = final_cash - config.starting_capital
    spy_net_profit = spy_final_balance - config.starting_capital
    
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

    metrics.start_date = start_date
    metrics.end_date = end_date
    metrics.total_days = total_days
    metrics.years = years
    metrics.bot_total_return = bot_total_return
    metrics.bot_net_profit = bot_net_profit
    metrics.bot_cagr = bot_cagr
    metrics.spy_total_return = spy_total_return
    metrics.spy_net_profit = spy_net_profit
    metrics.spy_cagr = spy_cagr
    metrics.naive_alpha = naive_alpha
    metrics.annualized_active_return = annualized_active_return
    metrics.ending_balance = final_cash
    metrics.spy_final_balance = spy_final_balance

    # 2. TRADE COUNTS & WIN/LOSS RATIOS
    total_trades = len(sim_df)
    closed_trades = int((sim_df['Status'] == 'CLOSED').sum()) if 'Status' in sim_df.columns else total_trades
    open_trades = int((sim_df['Status'] == 'OPEN').sum()) if 'Status' in sim_df.columns else 0

    trade_pnls = sim_df['PnL_Pct'] if 'PnL_Pct' in sim_df.columns else pd.Series([0.0]*total_trades)
    
    winning_trades = sim_df[trade_pnls > 0]
    losing_trades = sim_df[trade_pnls < 0]
    breakeven_trades = sim_df[trade_pnls == 0]

    num_wins = len(winning_trades)
    num_losses = len(losing_trades)
    num_breakeven = len(breakeven_trades)

    win_rate = num_wins / total_trades if total_trades > 0 else 0.0
    loss_rate = num_losses / total_trades if total_trades > 0 else 0.0

    if 'Type' in sim_df.columns:
        long_trades = sim_df[sim_df['Type'] == 'LONG']
        short_trades = sim_df[sim_df['Type'] == 'SHORT']
    else:
        long_trades = sim_df
        short_trades = pd.DataFrame()

    long_wins = len(long_trades[long_trades['PnL_Pct'] > 0]) if 'PnL_Pct' in long_trades.columns else 0
    short_wins = len(short_trades[short_trades['PnL_Pct'] > 0]) if 'PnL_Pct' in short_trades.columns else 0
    long_win_rate = long_wins / len(long_trades) if len(long_trades) > 0 else 0.0
    short_win_rate = short_wins / len(short_trades) if len(short_trades) > 0 else 0.0

    avg_trade_pnl = float(trade_pnls.mean()) if total_trades > 0 else 0.0
    median_trade_pnl = float(trade_pnls.median()) if total_trades > 0 else 0.0
    std_trade_pnl = float(trade_pnls.std(ddof=1)) if total_trades > 1 else 0.0

    avg_win_pnl = float(winning_trades['PnL_Pct'].mean()) if num_wins > 0 else 0.0
    avg_loss_pnl = float(losing_trades['PnL_Pct'].mean()) if num_losses > 0 else 0.0

    profit_dollars_col = 'Profit_Dollars' if 'Profit_Dollars' in sim_df.columns else 'PnL'
    avg_win_dollar = float(winning_trades[profit_dollars_col].mean()) if num_wins > 0 and profit_dollars_col in winning_trades.columns else 0.0
    avg_loss_dollar = float(abs(losing_trades[profit_dollars_col].mean())) if num_losses > 0 and profit_dollars_col in losing_trades.columns else 0.0

    payoff_ratio = (avg_win_pnl / abs(avg_loss_pnl)) if (avg_loss_pnl != 0 and not np.isnan(avg_loss_pnl)) else np.nan

    gross_profit = float(winning_trades[profit_dollars_col].sum()) if num_wins > 0 and profit_dollars_col in winning_trades.columns else 0.0
    gross_loss = float(abs(losing_trades[profit_dollars_col].sum())) if num_losses > 0 and profit_dollars_col in losing_trades.columns else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else np.nan)

    expectancy_pct = (win_rate * avg_win_pnl) - (loss_rate * abs(avg_loss_pnl))
    expectancy_dollar = (win_rate * avg_win_dollar) - (loss_rate * avg_loss_dollar)

    best_idx = trade_pnls.idxmax() if total_trades > 0 else None
    worst_idx = trade_pnls.idxmin() if total_trades > 0 else None
    best_trade = sim_df.loc[best_idx].to_dict() if best_idx is not None else None
    worst_trade = sim_df.loc[worst_idx].to_dict() if worst_idx is not None else None

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

    if 'Holding_Days' in sim_df.columns:
        avg_holding_days = float(sim_df['Holding_Days'].mean()) if total_trades > 0 else 0.0
        avg_win_holding_days = float(winning_trades['Holding_Days'].mean()) if num_wins > 0 else 0.0
        avg_loss_holding_days = float(losing_trades['Holding_Days'].mean()) if num_losses > 0 else 0.0
    else:
        avg_holding_days = avg_win_holding_days = avg_loss_holding_days = 0.0

    trades_per_month = (total_trades / (total_days / 30.4375)) if total_days > 0 else 0.0
    trades_per_year = (total_trades / years) if years > 0 else 0.0

    metrics.total_trades = total_trades
    metrics.closed_trades = closed_trades
    metrics.open_trades = open_trades
    metrics.num_wins = num_wins
    metrics.num_losses = num_losses
    metrics.num_breakeven = num_breakeven
    metrics.win_rate = win_rate
    metrics.loss_rate = loss_rate
    metrics.long_trades_count = len(long_trades)
    metrics.short_trades_count = len(short_trades)
    metrics.long_win_rate = long_win_rate
    metrics.short_win_rate = short_win_rate
    metrics.avg_trade_pnl = avg_trade_pnl
    metrics.median_trade_pnl = median_trade_pnl
    metrics.std_trade_pnl = std_trade_pnl
    metrics.avg_win_pnl = avg_win_pnl
    metrics.avg_loss_pnl = avg_loss_pnl
    metrics.avg_win_dollar = avg_win_dollar
    metrics.avg_loss_dollar = avg_loss_dollar
    metrics.payoff_ratio = payoff_ratio
    metrics.gross_profit = gross_profit
    metrics.gross_loss = gross_loss
    metrics.profit_factor = profit_factor
    metrics.expectancy_pct = expectancy_pct
    metrics.expectancy_dollar = expectancy_dollar
    metrics.best_trade = best_trade
    metrics.worst_trade = worst_trade
    metrics.max_consec_wins = max_consec_wins
    metrics.max_consec_losses = max_consec_losses
    metrics.avg_holding_days = avg_holding_days
    metrics.avg_win_holding_days = avg_win_holding_days
    metrics.avg_loss_holding_days = avg_loss_holding_days
    metrics.trades_per_month = trades_per_month
    metrics.trades_per_year = trades_per_year

    # 3. RISK, VOLATILITY & DRAWDOWN METRICS
    # Extract equity series from the DataFrame returned by simulation
    if isinstance(equity_curve_df, pd.DataFrame) and 'Equity' in equity_curve_df.columns:
        if 'Date' in equity_curve_df.columns:
            eq_series = pd.Series(equity_curve_df['Equity'].values, index=pd.to_datetime(equity_curve_df['Date']))
        else:
            eq_series = equity_curve_df['Equity'].reset_index(drop=True)
    elif isinstance(equity_curve_df, pd.DataFrame) and not equity_curve_df.empty:
        # Try first numeric column
        num_cols = equity_curve_df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            eq_series = equity_curve_df[num_cols[0]].reset_index(drop=True)
        else:
            eq_series = pd.Series([config.starting_capital])
    else:
        eq_series = pd.Series([config.starting_capital])

    # Extract benchmark SPY data for volatility/drawdown calculations
    if isinstance(benchmark_curve_df, pd.DataFrame) and 'Benchmark_Equity' in benchmark_curve_df.columns:
        spy_bench_series = benchmark_curve_df['Benchmark_Equity']
    else:
        spy_bench_series = pd.Series(dtype=float)

    peak_series = eq_series.cummax()
    drawdown_series = (eq_series - peak_series) / peak_series
    max_drawdown = float(drawdown_series.min()) if not drawdown_series.empty else 0.0
    avg_drawdown = float(drawdown_series[drawdown_series < 0].mean()) if (drawdown_series < 0).any() else 0.0
    peak_equity = float(peak_series.max()) if not peak_series.empty else config.starting_capital

    if not spy_bench_series.empty and len(spy_bench_series) > 1:
        spy_daily_ret = spy_bench_series.pct_change().dropna()
        spy_ann_vol = float(spy_daily_ret.std(ddof=1) * np.sqrt(252))
        spy_peaks = spy_bench_series.cummax()
        spy_mdd = float(((spy_bench_series - spy_peaks) / spy_peaks).min())
    else:
        spy_ann_vol = np.nan
        spy_mdd = np.nan
        spy_daily_ret = pd.Series(dtype=float)

    if config.simulation_mode == SimulationMode.CONCURRENT:
        mdd_duration_steps = 0
        in_dd = False
        dd_start = None
        if isinstance(eq_series.index, pd.DatetimeIndex):
            for dt, dd in drawdown_series.items():
                if dd < 0:
                    if not in_dd:
                        dd_start = dt
                        in_dd = True
                    curr_dd_days = (dt - dd_start).days
                    mdd_duration_steps = max(mdd_duration_steps, curr_dd_days)
                else:
                    in_dd = False
        else:
            curr_dd_len = 0
            for dd in drawdown_series:
                if dd < 0:
                    curr_dd_len += 1
                    mdd_duration_steps = max(mdd_duration_steps, curr_dd_len)
                else:
                    curr_dd_len = 0

        daily_returns = eq_series.pct_change().dropna()
        ann_volatility = float(daily_returns.std(ddof=1) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0
        
        rf_daily = config.risk_free_rate_daily if hasattr(config, 'risk_free_rate_daily') else (config.risk_free_rate_annual / 252)
        downside_diffs = np.minimum(daily_returns - rf_daily, 0.0)
        downside_deviation = float(np.sqrt(np.mean(downside_diffs ** 2)) * np.sqrt(252)) if len(daily_returns) > 0 else 0.0
        
        var_95 = float(np.percentile(daily_returns, 5)) if len(daily_returns) >= 20 else np.nan
        var_99 = float(np.percentile(daily_returns, 1)) if len(daily_returns) >= 100 else (var_95 if len(daily_returns) >= 20 else np.nan)
        cvar_95 = float(daily_returns[daily_returns <= var_95].mean()) if not np.isnan(var_95) and len(daily_returns[daily_returns <= var_95]) > 0 else np.nan
        cvar_99 = float(daily_returns[daily_returns <= var_99].mean()) if not np.isnan(var_99) and len(daily_returns[daily_returns <= var_99]) > 0 else np.nan
        
        step_returns = daily_returns
        rf_per_step = rf_daily
        steps_per_year = 252
    else:
        mdd_duration_steps = 0
        curr_dd_len = 0
        for dd in drawdown_series:
            if dd < 0:
                curr_dd_len += 1
                mdd_duration_steps = max(mdd_duration_steps, curr_dd_len)
            else:
                curr_dd_len = 0
                
        step_returns = sim_df['Step_Return'] if 'Step_Return' in sim_df.columns else pd.Series(dtype=float)
        steps_per_year = total_trades / years if years > 0 else 252
        rf_per_step = (1 + config.risk_free_rate_annual) ** (1 / steps_per_year) - 1 if steps_per_year > 0 else 0.0
        
        ann_volatility = float(step_returns.std(ddof=1) * np.sqrt(steps_per_year)) if len(step_returns) > 1 else 0.0
        downside_diffs = np.minimum(step_returns - rf_per_step, 0.0)
        downside_deviation = float(np.sqrt(np.mean(downside_diffs ** 2)) * np.sqrt(steps_per_year)) if len(step_returns) > 0 else 0.0

        var_95 = float(np.percentile(step_returns, 5)) if len(step_returns) >= 20 else np.nan
        var_99 = float(np.percentile(step_returns, 1)) if len(step_returns) >= 100 else (var_95 if len(step_returns) >= 20 else np.nan)
        cvar_95 = float(step_returns[step_returns <= var_95].mean()) if not np.isnan(var_95) and len(step_returns[step_returns <= var_95]) > 0 else np.nan
        cvar_99 = float(step_returns[step_returns <= var_99].mean()) if not np.isnan(var_99) and len(step_returns[step_returns <= var_99]) > 0 else np.nan

    if len(trade_pnls) >= 4:
        trade_skew = float(stats.skew(trade_pnls))
        trade_kurtosis = float(stats.kurtosis(trade_pnls))
    else:
        trade_skew = np.nan
        trade_kurtosis = np.nan

    metrics.peak_equity = peak_equity
    metrics.max_drawdown = max_drawdown
    metrics.avg_drawdown = avg_drawdown
    metrics.mdd_duration_steps = mdd_duration_steps
    metrics.ann_volatility = ann_volatility
    metrics.downside_deviation = downside_deviation
    metrics.spy_ann_vol = spy_ann_vol
    metrics.spy_mdd = spy_mdd
    metrics.var_95 = var_95
    metrics.var_99 = var_99
    metrics.cvar_95 = cvar_95
    metrics.cvar_99 = cvar_99
    metrics.trade_skew = trade_skew
    metrics.trade_kurtosis = trade_kurtosis
    metrics.drawdown_series = drawdown_series.tolist()

    # 4. RISK-ADJUSTED RATIOS
    sharpe_ratio = ((bot_cagr - config.risk_free_rate_annual) / ann_volatility) if ann_volatility > 0 else np.nan
    spy_sharpe = ((spy_cagr - config.risk_free_rate_annual) / spy_ann_vol) if (not np.isnan(spy_ann_vol) and spy_ann_vol > 0) else np.nan

    sortino_ratio = ((bot_cagr - config.risk_free_rate_annual) / downside_deviation) if downside_deviation > 0 else np.nan
    calmar_ratio = (bot_cagr / abs(max_drawdown)) if (max_drawdown != 0 and not np.isnan(max_drawdown)) else np.nan
    sterling_ratio = (bot_cagr / abs(avg_drawdown)) if (avg_drawdown != 0 and not np.isnan(avg_drawdown)) else np.nan

    pos_excess = step_returns[step_returns > rf_per_step] - rf_per_step
    neg_excess = rf_per_step - step_returns[step_returns < rf_per_step]
    sum_pos = pos_excess.sum()
    sum_neg = neg_excess.sum()
    omega_ratio = float(sum_pos / sum_neg) if sum_neg > 0 else (np.inf if sum_pos > 0 else np.nan)

    sum_step_gains = step_returns.sum()
    sum_step_losses = abs(step_returns[step_returns < 0].sum())
    gain_to_pain_ratio = float(sum_step_gains / sum_step_losses) if sum_step_losses > 0 else np.nan

    metrics.sharpe_ratio = sharpe_ratio
    metrics.spy_sharpe = spy_sharpe
    metrics.sortino_ratio = sortino_ratio
    metrics.calmar_ratio = calmar_ratio
    metrics.sterling_ratio = sterling_ratio
    metrics.omega_ratio = omega_ratio
    metrics.gain_to_pain_ratio = gain_to_pain_ratio

    # 5. CAPM
    if config.simulation_mode == SimulationMode.CONCURRENT:
        if not spy_daily_ret.empty and not daily_returns.empty:
            bot_ret_align = daily_returns.copy()
            if isinstance(bot_ret_align.index, pd.DatetimeIndex):
                bot_ret_align.index = bot_ret_align.index.normalize()
            spy_ret_align = spy_daily_ret.copy()
            if isinstance(spy_ret_align.index, pd.DatetimeIndex):
                spy_ret_align.index = spy_ret_align.index.normalize()
            
            aligned = pd.concat([bot_ret_align, spy_ret_align], axis=1, join='inner').dropna()
            aligned.columns = ['Bot', 'SPY']
            
            if len(aligned) >= 5:
                excess_bot = aligned['Bot'] - rf_daily
                excess_spy = aligned['SPY'] - rf_daily
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(excess_spy, excess_bot)
                beta = float(slope)
                beta_stderr = float(std_err)
                beta_pvalue = float(p_value)
                correlation = float(r_value)
                r_squared = float(r_value ** 2)
                
                jensen_alpha_step = float(intercept)
                jensen_alpha_annual = float((1 + jensen_alpha_step) ** 252 - 1)
                
                n_obs = len(aligned)
                fitted = intercept + slope * excess_spy
                residuals = excess_bot - fitted
                sse = np.sum(residuals ** 2)
                s_err = np.sqrt(sse / (n_obs - 2)) if n_obs > 2 else 0.0
                x_mean = excess_spy.mean()
                ss_x = np.sum((excess_spy - x_mean) ** 2)
                alpha_se = s_err * np.sqrt(1 / n_obs + (x_mean ** 2) / ss_x) if ss_x > 0 else np.nan
                alpha_tstat = (jensen_alpha_step / alpha_se) if (not np.isnan(alpha_se) and alpha_se > 0) else np.nan
                alpha_pvalue = float(2 * (1 - stats.t.cdf(abs(alpha_tstat), df=n_obs - 2))) if not np.isnan(alpha_tstat) else np.nan

                excess_diff = aligned['Bot'] - aligned['SPY']
                tracking_error = float(excess_diff.std(ddof=1) * np.sqrt(252)) if len(excess_diff) > 1 else np.nan
                information_ratio = (annualized_active_return / tracking_error) if (not np.isnan(tracking_error) and tracking_error > 0) else np.nan

                treynor_ratio = ((bot_cagr - config.risk_free_rate_annual) / beta) if (beta != 0 and not np.isnan(beta)) else np.nan

                up_spy = aligned[aligned['SPY'] > 0]
                down_spy = aligned[aligned['SPY'] < 0]
                up_capture = float(up_spy['Bot'].mean() / up_spy['SPY'].mean()) if (len(up_spy) > 0 and up_spy['SPY'].mean() != 0) else np.nan
                down_capture = float(down_spy['Bot'].mean() / down_spy['SPY'].mean()) if (len(down_spy) > 0 and down_spy['SPY'].mean() != 0) else np.nan
            else:
                beta = beta_stderr = beta_pvalue = correlation = r_squared = jensen_alpha_annual = np.nan
                alpha_tstat = alpha_pvalue = tracking_error = information_ratio = treynor_ratio = up_capture = down_capture = np.nan
        else:
            beta = beta_stderr = beta_pvalue = correlation = r_squared = jensen_alpha_annual = np.nan
            alpha_tstat = alpha_pvalue = tracking_error = information_ratio = treynor_ratio = up_capture = down_capture = np.nan
    else:
        if 'Step_Return' in sim_df.columns and 'SPY_Window_Return' in sim_df.columns and 'Holding_Days' in sim_df.columns:
            paired_df = sim_df[['Step_Return', 'SPY_Window_Return', 'Holding_Days']].dropna()
            if len(paired_df) >= 5:
                step_rf_series = (config.risk_free_rate_annual / 365.25) * paired_df['Holding_Days']
                excess_bot = paired_df['Step_Return'] - step_rf_series
                excess_spy = paired_df['SPY_Window_Return'] - step_rf_series

                slope, intercept, r_value, p_value, std_err = stats.linregress(excess_spy, excess_bot)
                beta = float(slope)
                beta_stderr = float(std_err)
                beta_pvalue = float(p_value)
                correlation = float(r_value)
                r_squared = float(r_value ** 2)

                jensen_alpha_step = float(intercept)
                avg_days = paired_df['Holding_Days'].mean()
                steps_yr = 365.25 / avg_days if avg_days > 0 else 252
                jensen_alpha_annual = float((1 + jensen_alpha_step) ** steps_yr - 1)

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

                excess_diff = paired_df['Step_Return'] - paired_df['SPY_Window_Return']
                tracking_error = float(excess_diff.std(ddof=1) * np.sqrt(steps_yr)) if len(excess_diff) > 1 else np.nan
                information_ratio = (annualized_active_return / tracking_error) if (not np.isnan(tracking_error) and tracking_error > 0) else np.nan

                treynor_ratio = ((bot_cagr - config.risk_free_rate_annual) / beta) if (beta != 0 and not np.isnan(beta)) else np.nan

                up_spy = paired_df[paired_df['SPY_Window_Return'] > 0]
                down_spy = paired_df[paired_df['SPY_Window_Return'] < 0]
                up_capture = float(up_spy['Step_Return'].mean() / up_spy['SPY_Window_Return'].mean()) if (len(up_spy) > 0 and up_spy['SPY_Window_Return'].mean() != 0) else np.nan
                down_capture = float(down_spy['Step_Return'].mean() / down_spy['SPY_Window_Return'].mean()) if (len(down_spy) > 0 and down_spy['SPY_Window_Return'].mean() != 0) else np.nan
            else:
                beta = beta_stderr = beta_pvalue = correlation = r_squared = jensen_alpha_annual = np.nan
                alpha_tstat = alpha_pvalue = tracking_error = information_ratio = treynor_ratio = up_capture = down_capture = np.nan
        else:
            beta = beta_stderr = beta_pvalue = correlation = r_squared = jensen_alpha_annual = np.nan
            alpha_tstat = alpha_pvalue = tracking_error = information_ratio = treynor_ratio = up_capture = down_capture = np.nan

    metrics.beta = beta
    metrics.beta_stderr = beta_stderr
    metrics.beta_pvalue = beta_pvalue
    metrics.correlation = correlation
    metrics.r_squared = r_squared
    metrics.jensen_alpha_annual = jensen_alpha_annual
    metrics.alpha_tstat = alpha_tstat
    metrics.alpha_pvalue = alpha_pvalue
    metrics.tracking_error = tracking_error
    metrics.information_ratio = information_ratio
    metrics.treynor_ratio = treynor_ratio
    metrics.up_capture = up_capture
    metrics.down_capture = down_capture

    # 6. STATISTICAL SIGNIFICANCE
    n_sample = len(trade_pnls)
    if n_sample >= 2:
        mean_pnl = trade_pnls.mean()
        se_pnl = std_trade_pnl / np.sqrt(n_sample)
        t_stat_pnl = float(mean_pnl / se_pnl) if se_pnl > 0 else 0.0
        df_deg = n_sample - 1
        p_val_2tail = float(2 * (1 - stats.t.cdf(abs(t_stat_pnl), df=df_deg)))
        p_val_1tail = float(1 - stats.t.cdf(t_stat_pnl, df=df_deg))
        
        ci_res = stats.t.interval(0.95, df=df_deg, loc=mean_pnl, scale=se_pnl)
        ci_low, ci_high = float(ci_res[0]), float(ci_res[1])

        binom_res = stats.binomtest(num_wins, n_sample, p=0.5, alternative='greater')
        binom_pvalue = float(binom_res.pvalue)

        sr_benchmark = 0.0
        if not np.isnan(sharpe_ratio) and n_sample >= 4 and not np.isnan(trade_skew) and not np.isnan(trade_kurtosis):
            sr_step = float((step_returns.mean() - rf_per_step) / step_returns.std(ddof=1)) if step_returns.std(ddof=1) > 0 else 0.0
            denom_variance = 1.0 - (trade_skew * sr_step) + (((trade_kurtosis + 3) - 1.0) / 4.0) * (sr_step ** 2)
            if denom_variance > 0:
                psr_std = np.sqrt(denom_variance / (n_sample - 1))
                psr_z = (sr_step - sr_benchmark) / psr_std
                psr = float(stats.norm.cdf(psr_z))
                
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

    metrics.t_stat_pnl = t_stat_pnl
    metrics.p_val_1tail = p_val_1tail
    metrics.p_val_2tail = p_val_2tail
    metrics.ci_low = ci_low
    metrics.ci_high = ci_high
    metrics.binom_pvalue = binom_pvalue
    metrics.psr = psr
    metrics.min_trl = min_trl

    # 7. KELLY
    if not np.isnan(payoff_ratio) and payoff_ratio > 0:
        kelly_fraction = float(win_rate - ((1.0 - win_rate) / payoff_ratio))
        half_kelly = float(kelly_fraction / 2.0)
    else:
        kelly_fraction = np.nan
        half_kelly = np.nan

    metrics.kelly_fraction = kelly_fraction
    metrics.half_kelly = half_kelly

    # 8. COSTS & CONFIG
    metrics.skipped_trades = getattr(sim_result, 'skipped_trades', 0)
    metrics.total_commissions = getattr(sim_result, 'total_commissions', 0.0)
    metrics.total_slippage_cost = getattr(sim_result, 'total_slippage_cost', 0.0)
    metrics.total_borrow_cost = getattr(sim_result, 'total_borrow_cost', 0.0)
    metrics.total_costs = metrics.total_commissions + metrics.total_slippage_cost + metrics.total_borrow_cost
    metrics.cost_drag_pct = metrics.total_costs / config.starting_capital

    metrics.simulation_mode = config.simulation_mode.value if hasattr(config.simulation_mode, 'value') else str(config.simulation_mode)
    metrics.position_size_pct = config.position_size_pct
    metrics.enable_costs = config.enable_costs

    return metrics
