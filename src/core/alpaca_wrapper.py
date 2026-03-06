"""
Thin wrapper around Alpaca's TradingClient for order management.

Always asserts paper=True for safety.
"""
from __future__ import annotations

from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderStatus
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
)

from src.config.settings import Settings
from src.utils.log_config import get_logger

logger = get_logger(__name__)


class AlpacaTradingWrapper:
    """Safe wrapper around Alpaca TradingClient (paper only)."""

    def __init__(self, settings: Settings) -> None:
        if not settings.is_paper:
            raise RuntimeError("This bot only supports paper trading!")

        self._settings = settings
        self._client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=True,
        )
        logger.info("AlpacaTradingWrapper initialized", paper=True)

    # ── Account ─────────────────────────────────────────────

    def get_account(self) -> dict:
        """Return account summary as a plain dict."""
        acct = self._client.get_account()
        return {
            "cash": float(acct.cash),
            "equity": float(acct.equity),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "last_equity": float(acct.last_equity),
            "status": acct.status,
        }

    # ── Positions ───────────────────────────────────────────

    def get_positions(self) -> list[dict]:
        """Get all open positions as a list of dicts."""
        positions = self._client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": str(p.side),
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
            for p in positions
        ]

    def get_position(self, symbol: str) -> dict | None:
        """Get a single position, or None if not held."""
        try:
            p = self._client.get_open_position(symbol)
            return {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": str(p.side),
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
        except Exception:
            return None

    # ── Orders ──────────────────────────────────────────────

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
    ) -> dict:
        """Submit a market order. side = 'buy' or 'sell'."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(request)
        result = {
            "id": str(order.id),
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": str(order.side),
            "type": str(order.type),
            "status": str(order.status),
        }
        logger.info("Order submitted", **result)
        return result

    def submit_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
    ) -> dict:
        """Submit a limit order."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
        )
        order = self._client.submit_order(request)
        result = {
            "id": str(order.id),
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": str(order.side),
            "type": str(order.type),
            "status": str(order.status),
            "limit_price": str(limit_price),
        }
        logger.info("Limit order submitted", **result)
        return result

    def get_open_orders(self) -> list[dict]:
        """List all open (non-filled) orders."""
        request = GetOrdersRequest(status="open")
        orders = self._client.get_orders(request)
        return [
            {
                "id": str(o.id),
                "symbol": o.symbol,
                "qty": str(o.qty),
                "side": str(o.side),
                "type": str(o.type),
                "status": str(o.status),
            }
            for o in orders
        ]

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        cancelled = self._client.cancel_orders()
        count = len(cancelled) if cancelled else 0
        logger.info("Cancelled all orders", count=count)
        return count

    def close_all_positions(self) -> int:
        """Close (liquidate) all open positions. Returns count closed."""
        closed = self._client.close_all_positions(cancel_orders=True)
        count = len(closed) if closed else 0
        logger.info("Closed all positions", count=count)
        return count

    @property
    def raw_client(self) -> TradingClient:
        """Access the underlying TradingClient (for clock, calendar, etc.)."""
        return self._client
