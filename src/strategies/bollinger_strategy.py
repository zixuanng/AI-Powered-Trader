"""
Bollinger Band mean-reversion strategy.

BUY when price touches/crosses the lower band (oversold),
SELL when price touches/crosses the upper band (overbought),
HOLD otherwise.
"""
from __future__ import annotations

import pandas as pd

from src.strategies.base_strategy import Action, BaseStrategy, Signal
from src.utils.log_config import get_logger

logger = get_logger(__name__)


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands (middle, upper, lower)."""
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def compute_percent_b(
    close: pd.Series, upper: pd.Series, lower: pd.Series
) -> pd.Series:
    """Compute %B: (close - lower) / (upper - lower).

    %B < 0 means price is below the lower band.
    %B > 1 means price is above the upper band.
    """
    bandwidth = upper - lower
    return (close - lower) / bandwidth.replace(0, float("nan"))


class BollingerBandStrategy(BaseStrategy):
    """Bollinger Band strategy — trade reversions from band extremes."""

    name = "Bollinger_Bands"

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        entry_threshold: float = 0.0,
        exit_threshold: float = 1.0,
    ) -> None:
        self.period = period
        self.num_std = num_std
        self.entry_threshold = entry_threshold  # buy when %B < this
        self.exit_threshold = exit_threshold    # sell when %B > this

    def generate_signals(
        self, data: dict[str, pd.DataFrame]
    ) -> list[Signal]:
        signals: list[Signal] = []

        for symbol, df in data.items():
            if df.empty or len(df) < self.period + 1:
                logger.debug(
                    "Insufficient data for Bollinger",
                    symbol=symbol,
                    rows=len(df),
                )
                continue

            close = df["close"]
            middle, upper, lower = compute_bollinger_bands(
                close, self.period, self.num_std
            )
            pct_b = compute_percent_b(close, upper, lower)

            current_pct_b = pct_b.iloc[-1]
            current_price = float(close.iloc[-1])
            current_middle = float(middle.iloc[-1]) if pd.notna(middle.iloc[-1]) else 0

            if pd.isna(current_pct_b):
                continue

            if current_pct_b <= self.entry_threshold:
                # Price at or below lower band — potential bounce
                strength = min(1.0, abs(current_pct_b) * 0.5 + 0.3)
                sig = Signal(
                    symbol=symbol,
                    action=Action.BUY,
                    strength=strength,
                    reason=(
                        f"Bollinger BUY: %B={current_pct_b:.3f} ≤ {self.entry_threshold} "
                        f"(price={current_price:.2f}, lower={float(lower.iloc[-1]):.2f})"
                    ),
                    metadata={"pct_b": round(current_pct_b, 4), "middle": round(current_middle, 2)},
                )
                signals.append(sig)
                logger.info("Signal generated", **sig.__dict__)

            elif current_pct_b >= self.exit_threshold:
                # Price at or above upper band — potential pullback
                strength = min(1.0, (current_pct_b - 1.0) * 0.5 + 0.3)
                sig = Signal(
                    symbol=symbol,
                    action=Action.SELL,
                    strength=strength,
                    reason=(
                        f"Bollinger SELL: %B={current_pct_b:.3f} ≥ {self.exit_threshold} "
                        f"(price={current_price:.2f}, upper={float(upper.iloc[-1]):.2f})"
                    ),
                    metadata={"pct_b": round(current_pct_b, 4), "middle": round(current_middle, 2)},
                )
                signals.append(sig)
                logger.info("Signal generated", **sig.__dict__)

            else:
                signals.append(
                    Signal(
                        symbol=symbol,
                        action=Action.HOLD,
                        strength=0.0,
                        reason=f"Bollinger neutral: %B={current_pct_b:.3f}",
                    )
                )

        return signals
