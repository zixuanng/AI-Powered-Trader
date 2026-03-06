"""Tests for EnsembleStrategy."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategies.base_strategy import Action, BaseStrategy, Signal
from src.strategies.ensemble_strategy import EnsembleStrategy, StrategyWeight


class _StubStrategy(BaseStrategy):
    """A test stub that returns a fixed signal."""
    name = "Stub"

    def __init__(self, action: Action, strength: float = 0.7):
        self._action = action
        self._strength = strength

    def generate_signals(self, data):
        return [
            Signal(symbol=sym, action=self._action, strength=self._strength, reason=f"Stub {self._action.value}")
            for sym in data
        ]


class TestEnsembleMajority:
    def _data(self):
        df = pd.DataFrame({"close": [100.0] * 10, "open": [100.0] * 10,
                           "high": [100.0] * 10, "low": [100.0] * 10,
                           "volume": [1000] * 10})
        return {"TEST": df}

    def test_majority_buy(self):
        """2 out of 3 BUY → ensemble emits BUY."""
        strategies = [
            StrategyWeight(_StubStrategy(Action.BUY)),
            StrategyWeight(_StubStrategy(Action.BUY)),
            StrategyWeight(_StubStrategy(Action.HOLD)),
        ]
        ens = EnsembleStrategy(strategies, mode="majority", threshold=0.5)
        signals = ens.generate_signals(self._data())

        assert len(signals) == 1
        assert signals[0].action == Action.BUY

    def test_majority_sell(self):
        """2 out of 3 SELL → ensemble emits SELL."""
        strategies = [
            StrategyWeight(_StubStrategy(Action.SELL)),
            StrategyWeight(_StubStrategy(Action.SELL)),
            StrategyWeight(_StubStrategy(Action.HOLD)),
        ]
        ens = EnsembleStrategy(strategies, mode="majority", threshold=0.5)
        signals = ens.generate_signals(self._data())

        assert len(signals) == 1
        assert signals[0].action == Action.SELL

    def test_majority_hold_no_agreement(self):
        """1 BUY, 1 SELL, 1 HOLD → no majority → HOLD."""
        strategies = [
            StrategyWeight(_StubStrategy(Action.BUY)),
            StrategyWeight(_StubStrategy(Action.SELL)),
            StrategyWeight(_StubStrategy(Action.HOLD)),
        ]
        ens = EnsembleStrategy(strategies, mode="majority", threshold=0.5)
        signals = ens.generate_signals(self._data())

        assert len(signals) == 1
        assert signals[0].action == Action.HOLD

    def test_conflict_resolution(self):
        """2 BUY + 2 SELL (4 strategies) → conflict → HOLD."""
        strategies = [
            StrategyWeight(_StubStrategy(Action.BUY)),
            StrategyWeight(_StubStrategy(Action.BUY)),
            StrategyWeight(_StubStrategy(Action.SELL)),
            StrategyWeight(_StubStrategy(Action.SELL)),
        ]
        ens = EnsembleStrategy(strategies, mode="majority", threshold=0.5)
        signals = ens.generate_signals(self._data())

        assert len(signals) == 1
        assert signals[0].action == Action.HOLD


class TestEnsembleWeighted:
    def _data(self):
        df = pd.DataFrame({"close": [100.0] * 10, "open": [100.0] * 10,
                           "high": [100.0] * 10, "low": [100.0] * 10,
                           "volume": [1000] * 10})
        return {"TEST": df}

    def test_weighted_buy(self):
        """High-weight BUY should dominate."""
        strategies = [
            StrategyWeight(_StubStrategy(Action.BUY, 0.9), weight=3.0),
            StrategyWeight(_StubStrategy(Action.SELL, 0.5), weight=1.0),
        ]
        ens = EnsembleStrategy(strategies, mode="weighted", threshold=0.3)
        signals = ens.generate_signals(self._data())

        assert len(signals) == 1
        assert signals[0].action == Action.BUY

    def test_weighted_hold_below_threshold(self):
        """Weak signals below threshold → HOLD."""
        strategies = [
            StrategyWeight(_StubStrategy(Action.BUY, 0.2), weight=1.0),
            StrategyWeight(_StubStrategy(Action.HOLD, 0.0), weight=1.0),
        ]
        ens = EnsembleStrategy(strategies, mode="weighted", threshold=0.5)
        signals = ens.generate_signals(self._data())

        assert len(signals) == 1
        assert signals[0].action == Action.HOLD
