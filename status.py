#!/usr/bin/env python3
"""
Breeden-Litzenberger Trading Strategy — Performance Auditor

Entry point that orchestrates data loading, simulation, metrics, and reporting.
"""
import sys
import warnings
from dataclasses import asdict

warnings.filterwarnings("ignore")

from bl_trading.config import TradingConfig, SimulationMode
from bl_trading.data import load_trade_data, fetch_benchmark_data
from bl_trading.simulation import run_simulation
from bl_trading.metrics import calculate_metrics, fmt_pct
from bl_trading.reports import generate_reports, save_reports


def main():
    config = TradingConfig()

    # 1. Load trade data
    df, filename = load_trade_data(config)

    # 2. Fetch benchmark data
    spy_hist = fetch_benchmark_data(df['Date'].min(), config)

    # 3. Run primary simulation (concurrent by default)
    result = run_simulation(df, config, spy_hist)

    # 4. Calculate metrics
    metrics = calculate_metrics(result, config)

    # 5. Generate and save reports
    # Extract equity values as lists for SVG chart rendering
    if 'Equity' in result.equity_curve.columns:
        eq_list = result.equity_curve['Equity'].tolist()
    else:
        eq_list = [config.starting_capital]
    if 'Benchmark_Equity' in result.benchmark_curve.columns:
        spy_list = result.benchmark_curve['Benchmark_Equity'].tolist()
    else:
        spy_list = [config.starting_capital]

    # Convert metrics to dict for report functions (they use m['key'] access)
    metrics_dict = asdict(metrics)

    reports = generate_reports(metrics_dict, result.trades, eq_list, spy_list, config)
    save_reports(metrics_dict, result.trades, reports.get('text', ''), eq_list, spy_list, config)

    # 6. Optional: run comparison mode
    if config.simulation_mode == SimulationMode.CONCURRENT:
        print("\n" + "=" * 80)
        print(" 🔄 COMPARISON: Running Sequential Mode for baseline comparison...")
        print("=" * 80)
        seq_config = TradingConfig(simulation_mode=SimulationMode.SEQUENTIAL)
        seq_result = run_simulation(df, seq_config, spy_hist)
        seq_metrics = calculate_metrics(seq_result, seq_config)
        print(f"\n  {'Mode':<25} {'Total Return':<18} {'CAGR':<18} {'Sharpe':<12}")
        print(f"  {'-'*73}")
        print(f"  {'CONCURRENT (realistic)':<25} {fmt_pct(metrics.bot_total_return, sign=True):<18} {fmt_pct(metrics.bot_cagr, sign=True):<18} {metrics.sharpe_ratio:<12.2f}")
        print(f"  {'SEQUENTIAL (legacy)':<25} {fmt_pct(seq_metrics.bot_total_return, sign=True):<18} {fmt_pct(seq_metrics.bot_cagr, sign=True):<18} {seq_metrics.sharpe_ratio:<12.2f}")
        diff = seq_metrics.bot_total_return - metrics.bot_total_return
        print(f"\n  ⚠️  Sequential mode inflates returns by {fmt_pct(diff, sign=True)} due to instant PnL realization")
        print(f"  ℹ️  Concurrent mode is the realistic simulation (use for decision-making)")


if __name__ == "__main__":
    main()