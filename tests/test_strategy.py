"""Tests for MA Crossover and RSI strategies."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategies.base_strategy import Action
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.rsi_strategy import RSIStrategy, compute_rsi


class TestMACrossover:
    def _make_df(self, prices: list[float]) -> pd.DataFrame:
        """Create a minimal OHLCV DataFrame from close prices."""
        return pd.DataFrame({
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1000] * len(prices),
        })

    def test_bullish_crossover(self):
        """When fast MA crosses above slow MA at the last bar, emit BUY."""
        # Downtrend then a single-bar spike: the crossover must happen
        # between the second-to-last and the very last bar.
        # fast(3)=rolling 3, slow(5)=rolling 5
        # At second-to-last: fast ≤ slow; at last: fast > slow.
        prices = [20, 18, 16, 14, 12, 10, 8, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8]
        # Second-to-last (index 16): close=5, fast_ma=5.0, slow_ma=5.0  → fast <= slow
        # Last           (index 17): close=8, fast_ma=6.0, slow_ma=5.6  → fast > slow  ✓
        strategy = MACrossoverStrategy(fast_period=3, slow_period=5)
        signals = strategy.generate_signals({"TEST": self._make_df(prices)})

        buys = [s for s in signals if s.action == Action.BUY]
        assert len(buys) >= 1
        assert buys[0].symbol == "TEST"

    def test_bearish_crossover(self):
        """When fast MA crosses below slow MA at the last bar, emit SELL."""
        # Uptrend then a single-bar drop: crossover at the last bar.
        prices = [5, 7, 9, 11, 13, 15, 17, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 17]
        # Second-to-last (index 16): close=20, fast_ma=20.0, slow_ma=20.0  → fast >= slow
        # Last           (index 17): close=17, fast_ma=19.0, slow_ma=19.4  → fast < slow  ✓
        strategy = MACrossoverStrategy(fast_period=3, slow_period=5)
        signals = strategy.generate_signals({"TEST": self._make_df(prices)})

        sells = [s for s in signals if s.action == Action.SELL]
        assert len(sells) >= 1

    def test_insufficient_data_returns_nothing(self):
        """With too few bars, no signal should be emitted."""
        prices = [100, 101, 102]
        strategy = MACrossoverStrategy(fast_period=10, slow_period=30)
        signals = strategy.generate_signals({"TEST": self._make_df(prices)})
        assert len(signals) == 0


class TestRSI:
    def test_compute_rsi_range(self):
        """RSI values should always be between 0 and 100."""
        np.random.seed(42)
        prices = pd.Series(np.cumsum(np.random.randn(200)) + 100)
        rsi = compute_rsi(prices, period=14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_oversold_buy(self):
        """Sharply declining prices should trigger RSI oversold → BUY."""
        # Create a steep, sustained decline to push RSI well below 30
        prices = [100] + [100 - i * 2 for i in range(1, 41)]  # 100 → 20
        df = pd.DataFrame({
            "open": prices, "high": prices, "low": prices,
            "close": prices, "volume": [1000] * len(prices),
        })
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)
        signals = strategy.generate_signals({"TEST": df})

        buys = [s for s in signals if s.action == Action.BUY]
        assert len(buys) >= 1

    def test_rsi_overbought_sell(self):
        """Sharply rising prices should trigger RSI overbought → SELL."""
        # Sustained rally with exponential increase to push RSI above 70
        prices = [20] + [20 + i * 2 for i in range(1, 41)]  # 20 → 100
        df = pd.DataFrame({
            "open": prices, "high": prices, "low": prices,
            "close": prices, "volume": [1000] * len(prices),
        })
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)
        signals = strategy.generate_signals({"TEST": df})

        sells = [s for s in signals if s.action == Action.SELL]
        assert len(sells) >= 1
