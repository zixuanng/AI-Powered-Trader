"""
Market time utilities.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from alpaca.trading.client import TradingClient

from src.utils.log_config import get_logger

logger = get_logger(__name__)


def is_market_open(client: TradingClient) -> bool:
    """Check whether the US equity market is currently open."""
    try:
        clock = client.get_clock()
        return clock.is_open
    except Exception:
        logger.exception("Failed to check market clock")
        return False


def next_market_open(client: TradingClient) -> datetime:
    """Return the datetime of the next market open."""
    clock = client.get_clock()
    return clock.next_open


def seconds_until_market_open(client: TradingClient) -> float:
    """Seconds until the next market open (0 if already open)."""
    if is_market_open(client):
        return 0.0
    nxt = next_market_open(client)
    # nxt is timezone-aware; get current time aware too
    now = datetime.now(tz=nxt.tzinfo)
    delta = (nxt - now).total_seconds()
    return max(delta, 0.0)
