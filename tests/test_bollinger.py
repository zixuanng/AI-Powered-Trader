"""Tests for BollingerBandStrategy."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategies.base_strategy import Action
from src.strategies.bollinger_strategy import (
    BollingerBandStrategy,
    compute_bollinger_bands,
    compute_percent_b,
)


class TestBollingerBands:
    def _make_df(self, prices: list[float]) -> pd.DataFrame:
        return pd.DataFrame({
            "open": prices, "high": prices, "low": prices,
            "close": prices, "volume": [1000] * len(prices),
        })

    def test_compute_bands_shape(self):
        """Bollinger bands should have the same length as input."""
        prices = pd.Series([float(x) for x in range(50)])
        middle, upper, lower = compute_bollinger_bands(prices, period=20)
        assert len(middle) == len(prices)
        assert len(upper) == len(prices)
        assert len(lower) == len(prices)

    def test_upper_above_lower(self):
        """Upper band should always be above lower band (where defined)."""
        prices = pd.Series([50 + i * 0.5 for i in range(50)])
        middle, upper, lower = compute_bollinger_bands(prices, period=20)
        valid = pd.DataFrame({"upper": upper, "lower": lower}).dropna()
        assert (valid["upper"] >= valid["lower"]).all()

    def test_buy_at_lower_band(self):
        """Price dropping below lower band → BUY signal."""
        # Stable price then a sharp drop
        prices = [100.0] * 25 + [100, 99, 97, 94, 90, 85]
        strategy = BollingerBandStrategy(period=20, num_std=2.0)
        signals = strategy.generate_signals({"TEST": self._make_df(prices)})

        buys = [s for s in signals if s.action == Action.BUY]
        assert len(buys) >= 1

    def test_sell_at_upper_band(self):
        """Price spiking above upper band → SELL signal."""
        prices = [100.0] * 25 + [100, 101, 103, 106, 110, 115]
        strategy = BollingerBandStrategy(period=20, num_std=2.0)
        signals = strategy.generate_signals({"TEST": self._make_df(prices)})

        sells = [s for s in signals if s.action == Action.SELL]
        assert len(sells) >= 1

    def test_hold_in_middle(self):
        """Stable price → HOLD signal."""
        prices = [100.0] * 30
        strategy = BollingerBandStrategy(period=20, num_std=2.0)
        signals = strategy.generate_signals({"TEST": self._make_df(prices)})

        # Should get a HOLD (no buy or sell)
        actionable = [s for s in signals if s.action != Action.HOLD]
        assert len(actionable) == 0
