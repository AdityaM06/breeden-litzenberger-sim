# 🚀 Trading Strategy Quantitative Performance Report

> **Period**: `2025-12-25` to `2026-08-14` (232 days / 0.64 yrs) &nbsp;|&nbsp; **Initial Capital**: `$10,000.00` &nbsp;|&nbsp; **Position Sizing**: `20%` &nbsp;|&nbsp; **Benchmark**: `SPY`

> [!TIP]
> **Strategy Outperformed Market**: The bot generated **+125.38%** vs SPY's **+13.27%** (Active spread: **$+11,211.08** / **+112.11%**).

## 📊 1. Capital & Return Overview

| Metric | Your Bot | S&P 500 (SPY) | Active Spread / Advantage |
| :--- | :--- | :--- | :--- |
| **Ending Equity** | **$22,538.38** | $11,327.30 | **$+11,211.08** |
| **Net Profit** | **$+12,538.38** | $+1,327.30 | **$+11,211.08** |
| **Cumulative Return** | **+125.38%** | +13.27% | **+112.11%** |
| **CAGR (Annualized)** | **+259.44%** | +21.68% | **+237.76%** |
| **Peak Equity (HWM)** | $22,538.38 | - | - |

## 🎯 2. Trade Execution & Win/Loss Profile

- **Total Trades**: `174` (Closed: `109` | Open Mark-to-Market: `65`)
- **Long Trades**: `106` (Win Rate: `49.06%`) | **Short Trades**: `68` (Win Rate: `27.94%`)

| Trade Metric | Value | Metric | Value |
| :--- | :--- | :--- | :--- |
| **Winning Trades** | `71 (40.80%)` | **Losing Trades** | `84 (48.28%)` |
| **Average Trade Return** | `+2.63%` | **Median Trade Return** | `+0.00%` |
| **Average Win Return** | `+15.71%` | **Average Loss Return** | `-7.83%` |
| **Average Win ($)** | `$461.81` | **Average Loss ($)** | `$241.07` |
| **Payoff Ratio (W/L)** | `2.01x` | **Profit Factor** | `1.62` |
| **Expectancy (%)** | `+2.63%` | **Expectancy ($ / Trade)** | `$+72.06` |
| **Max Consec. Wins** | `4` | **Max Consec. Losses** | `7` |
| **Avg Holding Period** | `61.7 days` | **Trade Frequency** | `22.8/mo (273.9/yr)` |
| **Best Single Trade** | `AMD (+136.43%)` | **Worst Single Trade** | `SNOW (-53.17%)` |

## 🛡️ 3. Risk, Volatility & Tail Metrics

| Risk Metric | Your Bot | S&P 500 (SPY) | Description / Notes |
| :--- | :--- | :--- | :--- |
| **Annualized Volatility (σ)** | `58.27%` | `17.97%` | Total return dispersion |
| **Downside Deviation (σ_down)** | `25.42%` | - | Volatility of negative returns only |
| **Max Drawdown (MDD)** | `-20.45%` | `-33.72%` | Peak-to-trough worst drop |
| **Average Drawdown** | `-5.99%` | - | Mean depth during underwater periods |
| **Max Drawdown Duration** | `27 trades` | - | Longest recovery period |
| **Historical VaR (95% / 99%)** | `-2.82%` / `-5.15%` | - | 1-in-20 / 1-in-100 trade risk threshold |
| **Conditional VaR (CVaR 95%)** | `-4.73%` | - | Expected loss beyond 95% VaR |
| **Return Skewness / Kurtosis** | `3.06` / `20.24` | - | Right-tail upside vs heavy tail risk |

## ⚖️ 4. Risk-Adjusted Performance Ratios

| Ratio | Your Bot | S&P 500 (SPY) | Target / Guideline |
| :--- | :--- | :--- | :--- |
| **Sharpe Ratio (Annualized)** | **`4.37`** | `0.96` | `> 1.0` Good, `> 2.0` Excellent |
| **Sortino Ratio (Downside)** | **`10.03`** | - | `> 1.5` Good, `> 3.0` Excellent |
| **Calmar Ratio (CAGR / MDD)** | `12.69` | - | `> 1.0` Acceptable, `> 3.0` Great |
| **Sterling Ratio** | `43.34` | - | Return per average drawdown |
| **Omega Ratio (Threshold Rf)** | `1.67` | - | `> 1.0` denotes positive strategy edge |
| **Gain-to-Pain Ratio (Schwager)** | `0.70` | - | `> 1.0` Good, `> 2.0` Excellent |
| **Information Ratio (vs SPY)** | `16.60` | - | `> 0.5` Good, `> 1.0` Elite |
| **Treynor Ratio (CAGR / Beta)** | `31.76` | - | Excess return per unit of systematic risk |

## 📈 5. Benchmark (SPY) Factor Attribution

| CAPM / Factor Metric | Estimate | Test Stat / P-Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **Beta (β to SPY)** | `0.08` | `SE=0.04 (p=0.064)` | Systematic sensitivity (1.0 = SPY) |
| **Correlation (r) / R²** | `0.14` / `1.98%` | - | Market correlation & % variance explained |
| **Annualized Jensen's Alpha (α)** | `-3.12%` | `t=-1.88 (p=0.062)` | True risk-adjusted alpha over CAPM |
| **Tracking Error** | `14.32%` | - | Annualized excess return volatility |
| **Up / Down Capture** | `16.11%` / `303.86%` | - | % of market up/down movements captured |

## 🔬 6. Statistical Significance & Hypothesis Testing

| Statistical Test | Result / Statistic | P-Value | Verdict (α = 0.05) |
| :--- | :--- | :--- | :--- |
| **Trade Return T-Test (H0: μ ≤ 0)** | `t = +1.970` | `p = 0.0252 (1-tail)` | ✅ **Significant (Edge > 0)** |
| **95% Confidence Interval for Mean** | `[-0.01%, +5.26%]` | - | Bounds for true average trade return |
| **Binomial Win Rate Test (H0: W ≤ 50%)** | `Wins: 71/174` | `p = 0.9939` | ❌ Not Significant vs Coin Flip |
| **Probabilistic Sharpe Ratio (PSR)** | `99.0%` | `Target SR > 0` | ✅ **High Confidence (> 95%)** |
| **Min Track Record Length (MinTRL)** | `89 trades` | `95% Conf.` | Required trade sample to prove skill |

## 📐 7. Optimal Position Sizing (Kelly Criterion)

- **Full Kelly Sizing ($K^*$)**: `+11.29%` (Maximum geometric growth)
- **Recommended Half-Kelly Sizing**: `+5.64%` (Institutional risk standard)
- **Current Configured Bet Size**: `20.0%`

> [!WARNING]
> **Over-Betting Alert**: Current bet size (`20.0%`) exceeds Full Kelly (`11.29%`), creating elevated risk of drawdown/ruin.

## 📋 8. Recent 10 Trades Audit

| # | Date | Ticker | Type | Status | Entry Price | Exit / Current | PnL % | Profit ($) | Equity |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 165 | 2026-08-13 | **DIA** | LONG | `OPEN` | $537.91 | $537.91 | +0.00% | $+0.00 | $22,538.38 |
| 166 | 2026-08-13 | **DIA** | LONG | `OPEN` | $537.91 | $537.91 | +0.00% | $+0.00 | $22,538.38 |
| 167 | 2026-08-13 | **TSM** | SHORT | `OPEN` | $430.49 | $430.49 | +0.00% | $+0.00 | $22,538.38 |
| 168 | 2026-08-13 | **TSM** | SHORT | `OPEN` | $430.49 | $430.49 | +0.00% | $+0.00 | $22,538.38 |
| 169 | 2026-08-13 | **MU** | SHORT | `OPEN` | $949.83 | $949.83 | +0.00% | $+0.00 | $22,538.38 |
| 170 | 2026-08-13 | **QCOM** | SHORT | `OPEN` | $164.79 | $164.79 | +0.00% | $+0.00 | $22,538.38 |
| 171 | 2026-08-13 | **QCOM** | SHORT | `OPEN` | $164.79 | $164.79 | +0.00% | $+0.00 | $22,538.38 |
| 172 | 2026-08-13 | **DDOG** | LONG | `OPEN` | $252.24 | $252.24 | -0.00% | $-0.00 | $22,538.38 |
| 173 | 2026-08-13 | **XLF** | LONG | `OPEN` | $58.26 | $58.26 | +0.00% | $+0.00 | $22,538.38 |
| 174 | 2026-08-13 | **DDOG** | LONG | `OPEN` | $252.24 | $252.24 | -0.00% | $-0.00 | $22,538.38 |
