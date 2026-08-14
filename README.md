# Breeden-Litzenberger Trading Strategy

An options-implied trading system using **Risk Neutral Density (RND)** analysis to identify high-probability trades, with a comprehensive portfolio simulation and performance auditing framework.

## Strategy Overview

The bot scans a watchlist of stocks, fetches their options chains, fits cubic splines to call prices, and takes the second derivative to extract the **Risk Neutral Density** — the market's implied probability distribution of future prices. When the peak-probability (consensus) price is sufficiently far from the current price and the confidence around that peak is high enough, the bot enters a trade.

## Project Structure

```
tradingTest/
├── bl_trading/                      # Core analysis package
│   ├── __init__.py
│   ├── config.py                    # All tunable parameters (sizing, costs, thresholds)
│   ├── data.py                      # Data loading & market data fetching
│   ├── simulation.py                # Portfolio simulation engine (concurrent + sequential)
│   ├── metrics.py                   # KPI calculations (Sharpe, CAPM, VaR, Kelly, etc.)
│   └── reports.py                   # Report generation (txt, md, html)
├── dynamic_trader_v2.py             # Live trading bot
├── status.py                        # Performance auditor entry point
├── dynamic_trades_2.0.csv           # Trade history
├── requirements.txt                 # Python dependencies
├── archive/                         # Legacy/obsolete files (preserved for reference)
└── .github/workflows/               # CI/CD pipeline
```

## Setup

```bash
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run the Trading Bot
```bash
python dynamic_trader_v2.py
```

### Run the Performance Auditor
```bash
python status.py
```

This generates three reports:
- `report.txt` — plaintext terminal-friendly report
- `report.md` — GitHub-flavored Markdown (pushed to GitHub Actions Step Summary)
- `report.html` — interactive HTML dashboard with SVG charts

## Configuration

All tunable parameters are centralized in `bl_trading/config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `starting_capital` | $10,000 | Initial portfolio value |
| `position_size_pct` | 5% | Fraction of equity per trade |
| `max_open_positions` | 20 | Max simultaneous positions |
| `cash_reserve_pct` | 10% | Cash buffer maintained |
| `stop_loss_pct` | 5% | Stop loss threshold |
| `simulation_mode` | CONCURRENT | Realistic (daily equity) or SEQUENTIAL (legacy) |
| `enable_costs` | True | Include slippage, commissions, borrow costs |

## Simulation Modes

- **CONCURRENT** (default, realistic): Processes trades on a daily calendar timeline with proper capital locking, position limits, and mark-to-market accounting.
- **SEQUENTIAL** (legacy): Instant PnL realization — useful for comparison but inflates returns when trades overlap.

## Roadmap

- **Phase 1** ✅ — Codebase cleanup, modular architecture, corrected simulation
- **Phase 2** — Backtesting with historical options data
- **Phase 3** — AI/optimization of trading parameters (potentially in Go)
