"""
Moving-average crossover strategy.

Emits BUY when the fast MA crosses above the slow MA,
SELL when the fast MA crosses below the slow MA,
HOLD otherwise.
"""
from __future__ import annotations

import pandas as pd

from src.strategies.base_strategy import Action, BaseStrategy, Signal
from src.utils.log_config import get_logger

logger = get_logger(__name__)


class MACrossoverStrategy(BaseStrategy):
    """Simple dual moving-average crossover."""

    name = "MA_Crossover"

    def __init__(self, fast_period: int = 10, slow_period: int = 30) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(
        self, data: dict[str, pd.DataFrame]
    ) -> list[Signal]:
        signals: list[Signal] = []

        for symbol, df in data.items():
            if df.empty or len(df) < self.slow_period + 1:
                logger.debug(
                    "Insufficient data for MA crossover",
                    symbol=symbol,
                    rows=len(df),
                )
                continue

            df = df.copy()
            df["fast_ma"] = df["close"].rolling(self.fast_period).mean()
            df["slow_ma"] = df["close"].rolling(self.slow_period).mean()

            # Need at least two rows with valid MAs to detect a crossover
            valid = df.dropna(subset=["fast_ma", "slow_ma"])
            if len(valid) < 2:
                continue

            prev = valid.iloc[-2]
            curr = valid.iloc[-1]

            # Bullish crossover: fast crosses above slow
            if prev["fast_ma"] <= prev["slow_ma"] and curr["fast_ma"] > curr["slow_ma"]:
                sig = Signal(
                    symbol=symbol,
                    action=Action.BUY,
                    strength=0.7,
                    reason=(
                        f"MA crossover BUY: fast({self.fast_period})={curr['fast_ma']:.2f} "
                        f"> slow({self.slow_period})={curr['slow_ma']:.2f}"
                    ),
                )
                signals.append(sig)
                logger.info("Signal generated", **sig.__dict__)

            # Bearish crossover: fast crosses below slow
            elif prev["fast_ma"] >= prev["slow_ma"] and curr["fast_ma"] < curr["slow_ma"]:
                sig = Signal(
                    symbol=symbol,
                    action=Action.SELL,
                    strength=0.7,
                    reason=(
                        f"MA crossover SELL: fast({self.fast_period})={curr['fast_ma']:.2f} "
                        f"< slow({self.slow_period})={curr['slow_ma']:.2f}"
                    ),
                )
                signals.append(sig)
                logger.info("Signal generated", **sig.__dict__)

            else:
                signals.append(
                    Signal(symbol=symbol, action=Action.HOLD, strength=0.0, reason="No crossover")
                )

        return signals
