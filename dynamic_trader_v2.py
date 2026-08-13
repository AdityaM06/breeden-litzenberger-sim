import pandas as pd
import yfinance as yf
import numpy as np
from scipy.interpolate import UnivariateSpline
import datetime
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
PROXIMITY_THRESHOLD = 0.015
STOP_LOSS_PCT = 0.05


def get_probability(pdf, k_range, lower_bound, upper_bound):
    idx = np.where((k_range >= lower_bound) & (k_range <= upper_bound))
    if len(idx[0]) == 0:
        return 0.0
    pdf_slice = pdf[idx]
    k_slice = k_range[idx]
    try:
        return np.trapezoid(pdf_slice, k_slice)
    except AttributeError:
        return np.trapz(pdf_slice, k_slice)


def compute_rnd(strikes, prices, smooth_factor=None):
    """
    PURE MATH, no data fetching.

    Given arrays of strikes and mid-prices for calls at a single expiration,
    returns (k_range, pdf, consensus_price, confidence).

    This is deliberately decoupled from yfinance: feed it a historical
    option-chain snapshot from any vendor (CBOE DataShop, ORATS,
    OptionMetrics, Polygon, ThetaData, etc.) and you get the same signal
    logic your live bot uses -- which is what a real backtest needs.
    """
    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(prices, dtype=float)

    order = np.argsort(strikes)
    strikes, prices = strikes[order], prices[order]

    if len(strikes) < 5:
        return None, None, None, None

    if smooth_factor is None:
        smooth_factor = len(strikes) * (1.0 if len(strikes) > 50 else 0.5)

    spline = UnivariateSpline(strikes, prices, k=3, s=smooth_factor)
    k_range = np.linspace(strikes.min(), strikes.max(), 500)
    second_deriv = spline.derivative(n=2)(k_range)
    pdf = np.maximum(second_deriv, 0)

    total_area = get_probability(np.ones_like(pdf), k_range, k_range.min(), k_range.max())
    total_area = np.trapezoid(pdf, k_range) if hasattr(np, 'trapezoid') else np.trapz(pdf, k_range)
    if total_area > 0:
        pdf = pdf / total_area

    peak_idx = np.argmax(pdf)
    consensus_price = k_range[peak_idx]

    range_radius = consensus_price * 0.03
    confidence = get_probability(pdf, k_range, consensus_price - range_radius, consensus_price + range_radius)

    return k_range, pdf, consensus_price, confidence


def fetch_live_option_chain(ticker_symbol, expiration_index=0):
    """
    DATA FETCHING ONLY. Pulls the current live chain from yfinance.
    Returns (current_price, strikes, mid_prices) or (None, None, None).
    """
    stock = yf.Ticker(ticker_symbol)
    try:
        hist = stock.history(period="1d")
        if hist.empty:
            return None, None, None
        current_price = hist['Close'].iloc[-1]
        expirations = stock.options
        if not expirations:
            return None, None, None
        selected_date = expirations[min(expiration_index, len(expirations) - 1)]
    except Exception:
        return None, None, None

    try:
        opt_chain = stock.option_chain(selected_date)
        calls = opt_chain.calls
        calls['mid_price'] = (calls['bid'] + calls['ask']) / 2
        calls = calls[(calls['volume'] > 0) | (calls['openInterest'] > 0)]
        calls = calls[calls['mid_price'] > 0.01]
        if len(calls) < 5:
            return None, None, None
        return current_price, calls['strike'].values, calls['mid_price'].values
    except Exception:
        return None, None, None


def analyze_sentiment(ticker_symbol, expiration_index=0):
    current_price, strikes, mid_prices = fetch_live_option_chain(ticker_symbol, expiration_index)
    if current_price is None:
        return None, None, None
    _, _, consensus_price, confidence = compute_rnd(strikes, mid_prices)
    if consensus_price is None:
        return None, None, None
    return current_price, consensus_price, confidence


class DynamicTrader:
    def __init__(self, filename='dynamic_trades_2.0.csv'):
        self.filename = filename
        if not os.path.exists(self.filename):
            df = pd.DataFrame(columns=[
                'Date', 'Ticker', 'Type', 'Entry_Price', 'Initial_Target',
                'Status', 'Exit_Price', 'Exit_Date', 'PnL', 'Confidence'
            ])
            df.to_csv(self.filename, index=False)
            print(f"Initialized new trade log: {self.filename}")

    def place_trade(self, ticker, signal_threshold=0.02, min_confidence=0.15, allow_short=True):
        try:
            df = pd.read_csv(self.filename)
            if not df.empty:
                existing_trade = df[(df['Ticker'] == ticker) & (df['Status'] == 'OPEN')]
                if not existing_trade.empty:
                    print(f"  --- Skipping {ticker}: Position already OPEN")
                    return
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass

        current, consensus, confidence = analyze_sentiment(ticker, expiration_index=2)
        if current is None:
            return

        diff_pct = (consensus - current) / current
        trade_type = "NEUTRAL"
        if diff_pct > signal_threshold:
            trade_type = "LONG"
        elif diff_pct < -signal_threshold:
            trade_type = "SHORT"

        if trade_type == "SHORT" and not allow_short:
            print(f"  --- SHORT signal on {ticker} ignored (allow_short=False)")
            return

        if trade_type != "NEUTRAL":
            print(f"Ticker: {ticker:<6} | Gap: {diff_pct:>6.2%} | Conf: {confidence:.2%}")
            if confidence < min_confidence:
                print(f"  --- REJECTED: Low Confidence ({confidence:.2%} < {min_confidence:.2%})")
                return
            self._log_trade(ticker, trade_type, current, consensus, confidence)
            print(f"  >>> TRADE OPENED: {trade_type} {ticker} (High Conviction)")

    def _log_trade(self, ticker, trade_type, entry, target, confidence):
        new_row = pd.DataFrame([{
            'Date': datetime.date.today(),
            'Ticker': ticker,
            'Type': trade_type,
            'Entry_Price': entry,
            'Initial_Target': target,
            'Status': 'OPEN',
            'Exit_Price': 0.0,
            'Exit_Date': '',
            'PnL': 0.0,
            'Confidence': confidence
        }])
        new_row.to_csv(self.filename, mode='a', header=False, index=False)

    def update_portfolio(self):
        try:
            df = pd.read_csv(self.filename)
            if df.empty:
                return
        except Exception:
            return

        updates_made = False
        for i, row in df.iterrows():
            if row['Status'] == 'OPEN':
                ticker = row['Ticker']
                entry_price = row['Entry_Price']
                trade_type = row['Type']

                current_price, dynamic_target, _ = analyze_sentiment(ticker, expiration_index=2)
                if current_price is None:
                    print(f"Could not fetch update for {ticker}")
                    continue

                gap_to_target = abs(current_price - dynamic_target) / current_price
                close_trade = False
                exit_reason = ""

                if gap_to_target < PROXIMITY_THRESHOLD:
                    close_trade, exit_reason = True, "Target Hit"
                elif trade_type == 'LONG' and current_price < entry_price * (1 - STOP_LOSS_PCT):
                    close_trade, exit_reason = True, "Stop Loss"
                elif trade_type == 'SHORT' and current_price > entry_price * (1 + STOP_LOSS_PCT):
                    close_trade, exit_reason = True, "Stop Loss"

                if close_trade:
                    updates_made = True
                    df.at[i, 'Status'] = 'CLOSED'
                    df.at[i, 'Exit_Price'] = current_price
                    df.at[i, 'Exit_Date'] = datetime.date.today()
                    if trade_type == 'LONG':
                        pnl = (current_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - current_price) / entry_price
                    df.at[i, 'PnL'] = pnl
                    print(f"CLOSING {ticker} ({exit_reason}) | PnL: {pnl:.2%} | Target Was: ${dynamic_target:.2f}")

        if updates_made:
            df.to_csv(self.filename, index=False)


if __name__ == '__main__':
    bot = DynamicTrader()
    print("\n--- Updating Open Positions ---")
    bot.update_portfolio()

    print("\n--- Scanning for New Trades ---")
    watchlist = [
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'ORCL', 'ADBE', 'CRM', 'IBM',
        'AMD', 'AVGO', 'INTC', 'QCOM', 'MU', 'TSM', 'ARM', 'TXN', 'AMAT',
        'SPY', 'QQQ', 'IWM', 'DIA', 'TLT', 'HYG', 'GLD', 'SLV', 'GDX', 'XLE', 'XLF', 'XLK', 'XBI', 'ARKK', 'SMH',
        'JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'V', 'MA', 'PYPL', 'SQ', 'COIN',
        'WMT', 'TGT', 'COST', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'CMG', 'KO', 'PEP', 'PG', 'DIS', 'ROKU',
        'XOM', 'CVX', 'OXY', 'SLB', 'GE', 'CAT', 'BA', 'LMT', 'RTX', 'F', 'GM', 'UPS',
        'LLY', 'UNH', 'JNJ', 'PFE', 'MRK', 'ABBV',
        'PLTR', 'UBER', 'LYFT', 'SHOP', 'SNOW', 'DDOG', 'DKNG', 'SOFI', 'HOOD', 'MARA', 'RIOT', 'GME', 'AMC',
        'BABA', 'JD', 'PDD', 'NIO',
        'AC.TO', 'CIEN', 'AA', 'BE'
    ]
    print(f"Scanning {len(watchlist)} tickers for high-probability setups...")
    for stock in watchlist:
        bot.place_trade(stock)
    print("\nScan Complete.")