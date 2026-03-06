"""
RSI (Relative Strength Index) strategy.

BUY when RSI < oversold threshold (default 30),
SELL when RSI > overbought threshold (default 70),
HOLD otherwise.
"""
from __future__ import annotations

import pandas as pd

from src.strategies.base_strategy import Action, BaseStrategy, Signal
from src.utils.log_config import get_logger

logger = get_logger(__name__)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using the standard Wilder smoothing method."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # When avg_loss is 0 (pure uptrend), RS → ∞ → RSI = 100
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Fill NaN/inf (zero-loss case) with 100 (all gains) or 0 (all losses)
    rsi = rsi.fillna(100.0)
    return rsi


class RSIStrategy(BaseStrategy):
    """RSI mean-reversion strategy."""

    name = "RSI"

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(
        self, data: dict[str, pd.DataFrame]
    ) -> list[Signal]:
        signals: list[Signal] = []

        for symbol, df in data.items():
            if df.empty or len(df) < self.period + 2:
                logger.debug("Insufficient data for RSI", symbol=symbol, rows=len(df))
                continue

            rsi = compute_rsi(df["close"], self.period)
            current_rsi = rsi.iloc[-1]

            if pd.isna(current_rsi):
                continue

            if current_rsi < self.oversold:
                sig = Signal(
                    symbol=symbol,
                    action=Action.BUY,
                    strength=min(1.0, (self.oversold - current_rsi) / self.oversold),
                    reason=f"RSI oversold: {current_rsi:.1f} < {self.oversold}",
                )
                signals.append(sig)
                logger.info("Signal generated", **sig.__dict__)

            elif current_rsi > self.overbought:
                sig = Signal(
                    symbol=symbol,
                    action=Action.SELL,
                    strength=min(1.0, (current_rsi - self.overbought) / (100 - self.overbought)),
                    reason=f"RSI overbought: {current_rsi:.1f} > {self.overbought}",
                )
                signals.append(sig)
                logger.info("Signal generated", **sig.__dict__)

            else:
                signals.append(
                    Signal(
                        symbol=symbol,
                        action=Action.HOLD,
                        strength=0.0,
                        reason=f"RSI neutral: {current_rsi:.1f}",
                    )
                )

        return signals
