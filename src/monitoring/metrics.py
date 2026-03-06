"""
Prometheus metrics for the trading bot.

Exposes counters, gauges, and histograms for observability.
Starts an HTTP server on a configurable port (default 8000).
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

from src.utils.log_config import get_logger

logger = get_logger(__name__)

# ── Counters ─────────────────────────────────────────────

trades_total = Counter(
    "trading_bot_trades_total",
    "Total number of trades executed",
    ["side", "symbol"],
)

signals_total = Counter(
    "trading_bot_signals_total",
    "Total number of signals generated",
    ["action", "strategy"],
)

errors_total = Counter(
    "trading_bot_errors_total",
    "Total number of errors",
    ["component"],
)

# ── Gauges ───────────────────────────────────────────────

portfolio_equity = Gauge(
    "trading_bot_portfolio_equity",
    "Current portfolio equity in USD",
)

portfolio_cash = Gauge(
    "trading_bot_portfolio_cash",
    "Current cash balance in USD",
)

portfolio_drawdown_pct = Gauge(
    "trading_bot_portfolio_drawdown_pct",
    "Current drawdown percentage",
)

portfolio_positions_count = Gauge(
    "trading_bot_portfolio_positions_count",
    "Number of open positions",
)

portfolio_exposure_pct = Gauge(
    "trading_bot_portfolio_exposure_pct",
    "Current exposure as percentage of equity",
)

# ── Histograms ───────────────────────────────────────────

iteration_duration_seconds = Histogram(
    "trading_bot_iteration_duration_seconds",
    "Time spent per engine iteration",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)


# ── Helpers ──────────────────────────────────────────────

@contextmanager
def track_iteration():
    """Context manager to time an engine iteration."""
    start = time.monotonic()
    try:
        yield
    finally:
        duration = time.monotonic() - start
        iteration_duration_seconds.observe(duration)


def update_portfolio_metrics(summary: dict) -> None:
    """Update all portfolio gauges from a portfolio summary dict."""
    portfolio_equity.set(summary.get("equity", 0))
    portfolio_cash.set(summary.get("cash", 0))
    portfolio_drawdown_pct.set(summary.get("drawdown_pct", 0))
    portfolio_positions_count.set(summary.get("positions", 0))
    portfolio_exposure_pct.set(summary.get("exposure_pct", 0))


def record_signal(action: str, strategy: str) -> None:
    """Increment the signals counter."""
    signals_total.labels(action=action, strategy=strategy).inc()


def record_trade(side: str, symbol: str) -> None:
    """Increment the trades counter."""
    trades_total.labels(side=side, symbol=symbol).inc()


def record_error(component: str) -> None:
    """Increment the error counter."""
    errors_total.labels(component=component).inc()


def start_metrics_server(port: int = 8000) -> None:
    """Start the Prometheus HTTP metrics server."""
    try:
        start_http_server(port)
        logger.info("Prometheus metrics server started", port=port)
    except OSError:
        logger.warning("Metrics server port already in use", port=port)
