"""
Risk manager — gates every trade through configurable rules.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config.settings import Settings
from src.core.portfolio import Portfolio
from src.utils.log_config import get_logger

logger = get_logger(__name__)


@dataclass
class RiskVerdict:
    """Result of a risk check."""
    approved: bool
    reason: str
    adjusted_qty: float | None = None  # may suggest a smaller qty


class RiskManager:
    """Enforces position-sizing and portfolio-level risk rules."""

    def __init__(self, settings: Settings) -> None:
        self.max_position_pct = settings.max_position_pct
        self.max_total_exposure_pct = settings.max_total_exposure_pct
        self.max_drawdown_pct = settings.max_drawdown_pct

    def check_buy(
        self,
        symbol: str,
        qty: float,
        price: float,
        portfolio: Portfolio,
    ) -> RiskVerdict:
        """Evaluate whether a BUY order should proceed.

        Returns a RiskVerdict with approved=True/False and a reason string.
        """
        # 1. Drawdown halt
        dd = portfolio.drawdown_pct()
        if dd >= self.max_drawdown_pct:
            msg = (
                f"Drawdown halt: current drawdown {dd:.2%} >= "
                f"max {self.max_drawdown_pct:.2%}"
            )
            logger.warning("Risk REJECTED (drawdown)", symbol=symbol, reason=msg)
            return RiskVerdict(approved=False, reason=msg)

        # 2. Total exposure cap
        exposure = portfolio.current_exposure_pct()
        if exposure >= self.max_total_exposure_pct:
            msg = (
                f"Exposure cap: current {exposure:.2%} >= "
                f"max {self.max_total_exposure_pct:.2%}"
            )
            logger.warning("Risk REJECTED (exposure)", symbol=symbol, reason=msg)
            return RiskVerdict(approved=False, reason=msg)

        # 3. Single position size limit
        order_value = qty * price
        max_value = portfolio.equity * self.max_position_pct
        if order_value > max_value:
            # Suggest a reduced quantity
            adjusted = int(max_value / price)
            if adjusted < 1:
                msg = (
                    f"Position too large: ${order_value:,.2f} > "
                    f"max ${max_value:,.2f} and cannot size down to 1 share"
                )
                logger.warning("Risk REJECTED (size)", symbol=symbol, reason=msg)
                return RiskVerdict(approved=False, reason=msg)

            msg = (
                f"Position sized down: {qty} → {adjusted} shares "
                f"(max {self.max_position_pct:.0%} of equity)"
            )
            logger.info("Risk ADJUSTED", symbol=symbol, reason=msg)
            return RiskVerdict(approved=True, reason=msg, adjusted_qty=adjusted)

        # 4. Sufficient buying power
        if order_value > portfolio.buying_power:
            msg = (
                f"Insufficient buying power: need ${order_value:,.2f}, "
                f"have ${portfolio.buying_power:,.2f}"
            )
            logger.warning("Risk REJECTED (buying power)", symbol=symbol, reason=msg)
            return RiskVerdict(approved=False, reason=msg)

        return RiskVerdict(approved=True, reason="All risk checks passed")

    def check_sell(
        self,
        symbol: str,
        qty: float,
        portfolio: Portfolio,
    ) -> RiskVerdict:
        """Evaluate whether a SELL order should proceed."""
        held = portfolio.position_qty(symbol)
        if held <= 0:
            return RiskVerdict(
                approved=False,
                reason=f"No position in {symbol} to sell",
            )
        if qty > held:
            return RiskVerdict(
                approved=True,
                reason=f"Selling all {held} shares (requested {qty})",
                adjusted_qty=held,
            )
        return RiskVerdict(approved=True, reason="Sell approved")
