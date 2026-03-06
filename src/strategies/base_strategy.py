"""
Base strategy interface and Signal data class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """A trading signal emitted by a strategy."""
    symbol: str
    action: Action
    strength: float = 1.0       # 0.0 – 1.0, higher = more confident
    reason: str = ""
    metadata: dict[str, Any] | None = None


class BaseStrategy(ABC):
    """Abstract base for all strategies.

    Subclasses must implement ``generate_signals``.
    """

    name: str = "BaseStrategy"

    @abstractmethod
    def generate_signals(
        self, data: dict[str, pd.DataFrame]
    ) -> list[Signal]:
        """Given {symbol: OHLCV DataFrame}, return a list of Signals.

        The engine will call this once per loop iteration.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
