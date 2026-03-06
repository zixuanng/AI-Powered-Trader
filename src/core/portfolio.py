"""
Portfolio state management.

Syncs with Alpaca and provides local convenience methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.core.alpaca_wrapper import AlpacaTradingWrapper
from src.utils.log_config import get_logger

logger = get_logger(__name__)


@dataclass
class Portfolio:
    """Local mirror of the Alpaca paper-trading account state."""

    cash: float = 0.0
    equity: float = 0.0
    buying_power: float = 0.0
    portfolio_value: float = 0.0
    last_equity: float = 0.0
    positions: dict[str, dict] = field(default_factory=dict)
    open_orders: list[dict] = field(default_factory=list)

    # ── Sync ────────────────────────────────────────────────

    def sync(self, wrapper: AlpacaTradingWrapper) -> None:
        """Pull the latest state from Alpaca."""
        acct = wrapper.get_account()
        self.cash = acct["cash"]
        self.equity = acct["equity"]
        self.buying_power = acct["buying_power"]
        self.portfolio_value = acct["portfolio_value"]
        self.last_equity = acct["last_equity"]

        raw_positions = wrapper.get_positions()
        self.positions = {p["symbol"]: p for p in raw_positions}

        self.open_orders = wrapper.get_open_orders()

        logger.info(
            "Portfolio synced",
            cash=self.cash,
            equity=self.equity,
            positions=len(self.positions),
            open_orders=len(self.open_orders),
        )

    # ── Queries ─────────────────────────────────────────────

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def get_position(self, symbol: str) -> dict | None:
        return self.positions.get(symbol)

    def position_qty(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return float(pos["qty"]) if pos else 0.0

    def total_market_value(self) -> float:
        """Sum of absolute market values of all positions."""
        return sum(abs(p["market_value"]) for p in self.positions.values())

    def unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(p["unrealized_pl"] for p in self.positions.values())

    def current_exposure_pct(self) -> float:
        """Current total exposure as a fraction of equity."""
        if self.equity <= 0:
            return 0.0
        return self.total_market_value() / self.equity

    def drawdown_pct(self) -> float:
        """Current drawdown from last_equity (prior close).

        Returns a positive number representing percent lost.
        """
        if self.last_equity <= 0:
            return 0.0
        return max(0.0, (self.last_equity - self.equity) / self.last_equity)

    def summary(self) -> dict:
        """Human-readable summary dict."""
        return {
            "equity": round(self.equity, 2),
            "cash": round(self.cash, 2),
            "positions": len(self.positions),
            "exposure_pct": round(self.current_exposure_pct() * 100, 2),
            "unrealized_pnl": round(self.unrealized_pnl(), 2),
            "drawdown_pct": round(self.drawdown_pct() * 100, 2),
        }
