"""
Ensemble strategy — combines signals from multiple sub-strategies.

Supports voting modes:
  - majority: signal fires if ≥ threshold strategies agree
  - weighted: weighted average of signal strengths determines action
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.strategies.base_strategy import Action, BaseStrategy, Signal
from src.utils.log_config import get_logger

logger = get_logger(__name__)


@dataclass
class StrategyWeight:
    """A strategy with its associated weight for ensemble voting."""
    strategy: BaseStrategy
    weight: float = 1.0


class EnsembleStrategy(BaseStrategy):
    """Combine multiple strategies via voting or weighted averaging."""

    name = "Ensemble"

    def __init__(
        self,
        strategies: list[StrategyWeight],
        mode: Literal["majority", "weighted"] = "majority",
        threshold: float = 0.5,
    ) -> None:
        """
        Args:
            strategies: list of StrategyWeight (strategy + weight)
            mode: 'majority' or 'weighted'
            threshold: for majority, fraction of strategies that must agree;
                       for weighted, minimum weighted score to trigger action.
        """
        if not strategies:
            raise ValueError("Ensemble requires at least one sub-strategy")

        self.strategies = strategies
        self.mode = mode
        self.threshold = threshold
        self.name = f"Ensemble({mode},{len(strategies)})"

    def generate_signals(
        self, data: dict[str, pd.DataFrame]
    ) -> list[Signal]:
        # 1. Collect signals from all sub-strategies
        all_signals: dict[str, list[tuple[Signal, float]]] = {}  # symbol → [(signal, weight)]

        for sw in self.strategies:
            try:
                sub_signals = sw.strategy.generate_signals(data)
            except Exception:
                logger.exception("Sub-strategy failed", strategy=sw.strategy.name)
                continue

            for sig in sub_signals:
                all_signals.setdefault(sig.symbol, []).append((sig, sw.weight))

        # 2. Aggregate per symbol
        result: list[Signal] = []
        for symbol, entries in all_signals.items():
            if self.mode == "majority":
                signal = self._majority_vote(symbol, entries)
            else:
                signal = self._weighted_vote(symbol, entries)
            result.append(signal)

        return result

    def _majority_vote(
        self, symbol: str, entries: list[tuple[Signal, float]]
    ) -> Signal:
        """Simple majority voting (each strategy gets one vote)."""
        votes: Counter[Action] = Counter()
        reasons: list[str] = []

        for sig, _ in entries:
            votes[sig.action] += 1
            if sig.action != Action.HOLD:
                reasons.append(f"{sig.reason}")

        total = len(entries)
        required = max(1, int(total * self.threshold))

        # Check BUY and SELL; if both pass threshold → conflict → HOLD
        buy_count = votes.get(Action.BUY, 0)
        sell_count = votes.get(Action.SELL, 0)

        if buy_count >= required and sell_count >= required:
            return Signal(
                symbol=symbol,
                action=Action.HOLD,
                strength=0.0,
                reason=f"Ensemble conflict: {buy_count} BUY vs {sell_count} SELL",
            )

        if buy_count >= required:
            return Signal(
                symbol=symbol,
                action=Action.BUY,
                strength=buy_count / total,
                reason=f"Ensemble BUY ({buy_count}/{total}): " + "; ".join(reasons),
            )

        if sell_count >= required:
            return Signal(
                symbol=symbol,
                action=Action.SELL,
                strength=sell_count / total,
                reason=f"Ensemble SELL ({sell_count}/{total}): " + "; ".join(reasons),
            )

        return Signal(
            symbol=symbol,
            action=Action.HOLD,
            strength=0.0,
            reason=f"Ensemble HOLD: no majority (BUY={buy_count}, SELL={sell_count}, HOLD={votes.get(Action.HOLD, 0)})",
        )

    def _weighted_vote(
        self, symbol: str, entries: list[tuple[Signal, float]]
    ) -> Signal:
        """Weighted voting: compute weighted buy/sell scores."""
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        reasons: list[str] = []

        for sig, weight in entries:
            total_weight += weight
            if sig.action == Action.BUY:
                buy_score += weight * sig.strength
                reasons.append(f"[BUY {weight:.1f}w] {sig.reason}")
            elif sig.action == Action.SELL:
                sell_score += weight * sig.strength
                reasons.append(f"[SELL {weight:.1f}w] {sig.reason}")

        if total_weight == 0:
            return Signal(symbol=symbol, action=Action.HOLD, strength=0.0, reason="No weights")

        buy_pct = buy_score / total_weight
        sell_pct = sell_score / total_weight

        if buy_pct >= self.threshold and buy_pct > sell_pct:
            return Signal(
                symbol=symbol,
                action=Action.BUY,
                strength=min(1.0, buy_pct),
                reason=f"Ensemble weighted BUY ({buy_pct:.2f}): " + "; ".join(reasons),
            )

        if sell_pct >= self.threshold and sell_pct > buy_pct:
            return Signal(
                symbol=symbol,
                action=Action.SELL,
                strength=min(1.0, sell_pct),
                reason=f"Ensemble weighted SELL ({sell_pct:.2f}): " + "; ".join(reasons),
            )

        return Signal(
            symbol=symbol,
            action=Action.HOLD,
            strength=0.0,
            reason=f"Ensemble weighted HOLD (buy={buy_pct:.2f}, sell={sell_pct:.2f})",
        )
