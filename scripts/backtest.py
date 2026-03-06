"""
Simple backtesting harness.

Runs a strategy over historical bars and reports hypothetical P&L.

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --strategy rsi --days 30
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import Settings
from src.data.alpaca_data import AlpacaDataClient
from src.strategies.base_strategy import Action, BaseStrategy
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.utils.log_config import setup_logging, get_logger

STRATEGIES: dict[str, type[BaseStrategy]] = {
    "ma_crossover": MACrossoverStrategy,
    "rsi": RSIStrategy,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple backtest")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default="ma_crossover",
    )
    parser.add_argument("--days", type=int, default=30, help="Lookback days")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols (override watchlist)")
    args = parser.parse_args()

    setup_logging(level="WARNING")
    logger = get_logger("backtest")

    settings = Settings()
    settings.validate()

    symbols = args.symbols.split(",") if args.symbols else settings.watchlist[:5]
    strategy = STRATEGIES[args.strategy]()

    data_client = AlpacaDataClient(settings)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)

    print(f"\n📊 Backtest: {strategy.name}")
    print(f"   Symbols:  {', '.join(symbols)}")
    print(f"   Period:   {start.date()} → {end.date()}")
    print(f"   Timeframe: {settings.bar_timeframe}")
    print("─" * 60)

    # Fetch all data
    all_data: dict[str, object] = {}
    for sym in symbols:
        try:
            df = data_client.get_bars(sym, start=start, end=end, limit=10000)
            all_data[sym] = df
            print(f"   {sym}: {len(df)} bars loaded")
        except Exception as e:
            print(f"   {sym}: FAILED — {e}")

    if not all_data:
        print("\n❌ No data loaded. Exiting.")
        return

    # Simple walk-forward: slide a window and generate signals
    initial_cash = 100_000.0
    cash = initial_cash
    positions: dict[str, dict] = {}  # symbol -> {qty, entry_price}
    trades: list[dict] = []

    # Get all timestamps from the first symbol
    first_df = list(all_data.values())[0]
    if first_df.empty:
        print("\n❌ Empty data. Exiting.")
        return

    timestamps = first_df.index.tolist()
    window_size = 50  # bars to feed to the strategy

    for i in range(window_size, len(timestamps)):
        window_data = {}
        for sym, df in all_data.items():
            # Get bars up to index i
            mask = df.index <= timestamps[i]
            window = df[mask].tail(window_size)
            if not window.empty:
                window_data[sym] = window

        signals = strategy.generate_signals(window_data)

        for sig in signals:
            if sig.action == Action.HOLD:
                continue

            if sig.symbol not in window_data or window_data[sig.symbol].empty:
                continue

            price = float(window_data[sig.symbol]["close"].iloc[-1])

            if sig.action == Action.BUY and sig.symbol not in positions:
                qty = max(1, int((cash * 0.05) / price))
                cost = qty * price
                if cost <= cash:
                    cash -= cost
                    positions[sig.symbol] = {"qty": qty, "entry_price": price}
                    trades.append({
                        "time": str(timestamps[i]),
                        "symbol": sig.symbol,
                        "action": "BUY",
                        "qty": qty,
                        "price": price,
                    })

            elif sig.action == Action.SELL and sig.symbol in positions:
                pos = positions.pop(sig.symbol)
                revenue = pos["qty"] * price
                pnl = revenue - (pos["qty"] * pos["entry_price"])
                cash += revenue
                trades.append({
                    "time": str(timestamps[i]),
                    "symbol": sig.symbol,
                    "action": "SELL",
                    "qty": pos["qty"],
                    "price": price,
                    "pnl": round(pnl, 2),
                })

    # Mark-to-market remaining positions
    portfolio_value = cash
    for sym, pos in positions.items():
        if sym in all_data and not all_data[sym].empty:
            last_price = float(all_data[sym]["close"].iloc[-1])
            portfolio_value += pos["qty"] * last_price

    total_return = (portfolio_value - initial_cash) / initial_cash * 100

    print(f"\n{'─' * 60}")
    print(f"📈 Results")
    print(f"   Initial:     ${initial_cash:,.2f}")
    print(f"   Final:       ${portfolio_value:,.2f}")
    print(f"   Return:      {total_return:+.2f}%")
    print(f"   Trades:      {len(trades)}")
    print(f"   Open Pos:    {len(positions)}")

    if trades:
        print(f"\n   Last 10 trades:")
        for t in trades[-10:]:
            pnl_str = f"  P&L: ${t.get('pnl', 'n/a')}" if 'pnl' in t else ""
            print(f"     {t['time'][:16]}  {t['action']:4s}  {t['qty']:>4}× {t['symbol']:<5} @ ${t['price']:.2f}{pnl_str}")


if __name__ == "__main__":
    main()
