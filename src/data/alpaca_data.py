"""
Alpaca market-data wrapper.

Fetches historical bars and latest quotes via alpaca-py's StockHistoricalDataClient.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from alpaca.data import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.config.settings import Settings
from src.utils.log_config import get_logger

logger = get_logger(__name__)

# Map string timeframe labels → alpaca TimeFrame objects
_TIMEFRAME_MAP: dict[str, TimeFrame] = {
    "1Min": TimeFrame(1, TimeFrameUnit.Minute),
    "5Min": TimeFrame(5, TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
    "1Day": TimeFrame(1, TimeFrameUnit.Day),
}


class AlpacaDataClient:
    """Thin wrapper around Alpaca's stock data API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )

    # ── Historical bars ─────────────────────────────────────

    def get_bars(
        self,
        symbol: str,
        timeframe: str | None = None,
        limit: int = 100,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for a single symbol.

        Returns a DataFrame with columns: open, high, low, close, volume, vwap.
        Index is the bar timestamp (UTC).
        """
        tf = _TIMEFRAME_MAP.get(timeframe or self._settings.bar_timeframe)
        if tf is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        if start is None:
            start = datetime.now(timezone.utc) - timedelta(days=5)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit,
            feed=DataFeed.IEX,
        )
        bars = self._client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            logger.warning("No bars returned", symbol=symbol, timeframe=timeframe)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])

        # If multi-index (symbol, timestamp), drop the symbol level
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")

        return df

    def get_bars_multi(
        self,
        symbols: list[str],
        timeframe: str | None = None,
        limit: int = 100,
    ) -> dict[str, pd.DataFrame]:
        """Fetch bars for multiple symbols.

        Returns {symbol: DataFrame}.
        """
        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                result[sym] = self.get_bars(sym, timeframe=timeframe, limit=limit)
            except Exception:
                logger.exception("Failed to fetch bars", symbol=sym)
                result[sym] = pd.DataFrame()
        return result

    # ── Latest quote ────────────────────────────────────────

    def get_latest_quote(self, symbol: str) -> dict:
        """Return latest bid/ask/last for a symbol."""
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
        quotes = self._client.get_stock_latest_quote(request)
        q = quotes.get(symbol)
        if q is None:
            return {"bid": 0.0, "ask": 0.0, "bid_size": 0, "ask_size": 0}
        return {
            "bid": float(q.bid_price),
            "ask": float(q.ask_price),
            "bid_size": int(q.bid_size),
            "ask_size": int(q.ask_size),
        }

