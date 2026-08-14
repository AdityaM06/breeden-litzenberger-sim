"""Centralized configuration for the Breeden-Litzenberger trading simulation.

Every parameter here is a candidate for Phase 3 optimization.
"""
from dataclasses import dataclass
from enum import Enum


class SimulationMode(Enum):
    """Portfolio simulation mode."""
    CONCURRENT = "concurrent"   # Realistic: tracks daily equity, locks capital per position
    SEQUENTIAL = "sequential"   # Legacy: instant PnL realization, no overlap handling


@dataclass
class TradingConfig:
    """All tunable trading parameters."""

    # === Capital ===
    starting_capital: float = 10_000.0
    risk_free_rate_annual: float = 0.045

    # === Position Sizing ===
    position_size_pct: float = 0.02         # 5% of available equity per trade
    max_open_positions: int = 100           # Cap on simultaneous positions
    cash_reserve_pct: float = 0.05          # Keep 10% cash buffer

    # === Entry/Exit Thresholds (used by dynamic_trader_v2) ===
    min_confidence: float = 0.15            # Minimum RND confidence to enter
    min_target_distance_pct: float = 0.02   # Min 2% distance to target
    stop_loss_pct: float = 0.05             # 5% stop loss

    # === Trading Costs ===
    enable_costs: bool = True
    commission_per_trade: float = 0.0       # Most brokers are $0
    slippage_bps: float = 5.0               # 5 basis points bid-ask estimate
    short_borrow_annual_pct: float = 0.03   # 3% annual borrow cost for shorts

    # === Simulation ===
    simulation_mode: SimulationMode = SimulationMode.CONCURRENT
    benchmark_ticker: str = "SPY"
    benchmark_history_years: int = 10

    # === Reporting ===
    trade_file: str = "dynamic_trades_2.0.csv"
    fallback_trade_file: str = "dynamic_trades.csv"
    max_display_trades: int = 100
    generate_text: bool = True
    generate_markdown: bool = True
    generate_html: bool = True

    @property
    def slippage_pct(self) -> float:
        """Convert basis points to percentage."""
        return self.slippage_bps / 10_000

    @property
    def risk_free_rate_daily(self) -> float:
        """Daily risk-free rate."""
        return (1 + self.risk_free_rate_annual) ** (1/252) - 1

    @property
    def short_borrow_daily_pct(self) -> float:
        """Daily short borrow rate."""
        return self.short_borrow_annual_pct / 365.25
