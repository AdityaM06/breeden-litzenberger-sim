import re

with open('status.py', 'r') as f:
    content = f.read()

def_metrics = re.search(r"def calculate_comprehensive_metrics.*?def build_text_report", content, flags=re.DOTALL)
if def_metrics:
    metrics_chunk = def_metrics.group(0)
    
    # 1. Replace step returns
    old_step_returns = r"    # Step returns volatility\n    step_returns = sim_df\['Step_Return'\]\n    steps_per_year = total_trades / years if years > 0 else 252\n    rf_per_step = \(1 \+ RISK_FREE_RATE_ANNUAL\) \*\* \(1 / steps_per_year\) - 1"
    new_step_returns = """    # Step returns volatility (Daily)
    step_returns = pd.Series(equity_curve).pct_change().dropna()
    daily_bot = step_returns
    steps_per_year = 252
    rf_per_step = (1 + RISK_FREE_RATE_ANNUAL) ** (1 / steps_per_year) - 1"""
    metrics_chunk = re.sub(old_step_returns, new_step_returns, metrics_chunk)
    
    # 2. Replace CAPM Regression
    old_capm = r"    # 5. BENCHMARK \(SPY\) REGRESSION & CAPM FACTOR ANALYSIS\n.*?    # 6. STATISTICAL SIGNIFICANCE & HYPOTHESIS TESTING"
    new_capm = """    # 5. BENCHMARK (SPY) REGRESSION & CAPM FACTOR ANALYSIS
    # -------------------------------------------------------------
    daily_spy = pd.Series(spy_equity_curve).pct_change().dropna()
    min_len = min(len(daily_bot), len(daily_spy))
    bot_ret_aligned = daily_bot.iloc[:min_len]
    spy_ret_aligned = daily_spy.iloc[:min_len]

    if min_len >= 5:
        step_rf_series = pd.Series([rf_per_step] * min_len, index=bot_ret_aligned.index)
        excess_bot = bot_ret_aligned - step_rf_series
        excess_spy = spy_ret_aligned - step_rf_series

        # OLS Linear Regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(excess_spy, excess_bot)
        beta = float(slope)
        beta_stderr = float(std_err)
        beta_pvalue = float(p_value)
        correlation = float(r_value)
        r_squared = float(r_value ** 2)

        # Jensen's Alpha per step and annualized
        jensen_alpha_step = float(intercept)
        steps_yr = 252
        jensen_alpha_annual = float((1 + jensen_alpha_step) ** steps_yr - 1)

        # Alpha standard error and t-test
        n_obs = min_len
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
        excess_diff = bot_ret_aligned - spy_ret_aligned
        tracking_error = float(excess_diff.std(ddof=1) * np.sqrt(steps_yr)) if len(excess_diff) > 1 else np.nan
        information_ratio = (annualized_active_return / tracking_error) if (not np.isnan(tracking_error) and tracking_error > 0) else np.nan

        # Treynor Ratio
        treynor_ratio = ((bot_cagr - RISK_FREE_RATE_ANNUAL) / beta) if (beta != 0 and not np.isnan(beta)) else np.nan

        # Up / Down Market Capture
        up_spy_mask = spy_ret_aligned > 0
        down_spy_mask = spy_ret_aligned < 0
        up_spy_bot = bot_ret_aligned[up_spy_mask]
        up_spy_spy = spy_ret_aligned[up_spy_mask]
        down_spy_bot = bot_ret_aligned[down_spy_mask]
        down_spy_spy = spy_ret_aligned[down_spy_mask]
        
        up_capture = float(up_spy_bot.mean() / up_spy_spy.mean()) if (len(up_spy_spy) > 0 and up_spy_spy.mean() != 0) else np.nan
        down_capture = float(down_spy_bot.mean() / down_spy_spy.mean()) if (len(down_spy_spy) > 0 and down_spy_spy.mean() != 0) else np.nan
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

    # 6. STATISTICAL SIGNIFICANCE & HYPOTHESIS TESTING"""
    
    metrics_chunk = re.sub(old_capm, new_capm, metrics_chunk, flags=re.DOTALL)
    
    content = content.replace(def_metrics.group(0), metrics_chunk)
    with open('status.py', 'w') as f:
        f.write(content)
    print("Metrics patched.")
else:
    print("Could not find calculate_comprehensive_metrics")
