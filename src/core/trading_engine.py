"""
Trading engine — main orchestrator loop.

Cycle: sync portfolio → fetch data → generate signals → risk check → execute orders.
Now with event bus, state persistence, and Prometheus metrics.
"""
from __future__ import annotations

import math
import signal
import time
from typing import Any

from src.config.settings import Settings
from src.core.alpaca_wrapper import AlpacaTradingWrapper
from src.core.event_bus import EventType, get_event_bus
from src.core.portfolio import Portfolio
from src.core.risk import RiskManager
from src.core.state_store import StateStore
from src.data.alpaca_data import AlpacaDataClient
from src.monitoring.metrics import (
    record_error,
    record_signal,
    record_trade,
    start_metrics_server,
    track_iteration,
    update_portfolio_metrics,
)
from src.notifications.notifier import Notifier
from src.strategies.base_strategy import Action, BaseStrategy, Signal
from src.utils.log_config import get_logger
from src.utils.time_utils import is_market_open, seconds_until_market_open

logger = get_logger(__name__)


class TradingEngine:
    """Event-loop trading engine.

    Coordinates data fetching, strategy execution, risk management,
    and order placement on a configurable interval.
    """

    def __init__(
        self,
        settings: Settings,
        strategy: BaseStrategy,
        wrapper: AlpacaTradingWrapper | None = None,
        data_client: AlpacaDataClient | None = None,
        enable_metrics: bool = True,
        metrics_port: int = 8000,
    ) -> None:
        self.settings = settings
        self.strategy = strategy

        self.wrapper = wrapper or AlpacaTradingWrapper(settings)
        self.data_client = data_client or AlpacaDataClient(settings)
        self.portfolio = Portfolio()
        self.risk = RiskManager(settings)
        self.notifier = Notifier(settings)

        # Phase 3: observability
        self.event_bus = get_event_bus()
        self.state_store = StateStore()
        self._enable_metrics = enable_metrics
        self._metrics_port = metrics_port

        self._running = False
        self._iteration = 0

    # ── Lifecycle ───────────────────────────────────────────

    def start(self, run_once: bool = False) -> None:
        """Start the engine loop. If run_once=True, execute one iteration and return."""
        self._running = True
        self._setup_signal_handlers()

        if self._enable_metrics:
            start_metrics_server(self._metrics_port)

        self.state_store.set_engine_started()
        self.event_bus.publish(EventType.ENGINE_STATUS, {
            "status": "started",
            "strategy": self.strategy.name,
            "watchlist": self.settings.watchlist,
        })

        logger.info(
            "Engine starting",
            strategy=self.strategy.name,
            watchlist=self.settings.watchlist,
            interval=self.settings.loop_interval_seconds,
            run_once=run_once,
        )

        try:
            while self._running:
                self._iteration += 1
                logger.info(f"=== Iteration {self._iteration} ===")

                try:
                    with track_iteration():
                        self._run_iteration()
                except Exception:
                    logger.exception("Iteration failed")
                    record_error("engine_iteration")
                    self.event_bus.publish(EventType.ERROR, {
                        "iteration": self._iteration,
                        "error": "Iteration failed",
                    })
                    self.notifier.on_error(
                        f"Iteration {self._iteration} failed",
                        {"iteration": self._iteration},
                    )

                if run_once:
                    break

                self._wait()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down")
        finally:
            self._running = False
            self.event_bus.publish(EventType.ENGINE_STATUS, {"status": "stopped"})
            logger.info("Engine stopped")

    def stop(self) -> None:
        """Signal the engine to stop after the current iteration."""
        self._running = False

    # ── Core loop body ──────────────────────────────────────

    def _run_iteration(self) -> None:
        # 1. Check market hours
        if not is_market_open(self.wrapper.raw_client):
            wait = seconds_until_market_open(self.wrapper.raw_client)
            logger.info("Market closed", next_open_seconds=round(wait))
            return

        # 2. Sync portfolio
        self.portfolio.sync(self.wrapper)
        summary = self.portfolio.summary()
        self.notifier.on_summary(summary)

        # Persist & publish portfolio snapshot
        self.state_store.save_snapshot(summary)
        update_portfolio_metrics(summary)
        self.event_bus.publish(EventType.PORTFOLIO_SNAPSHOT, summary)

        # 3. Fetch data
        data = self.data_client.get_bars_multi(
            self.settings.watchlist,
            timeframe=self.settings.bar_timeframe,
        )

        # 4. Generate signals
        signals = self.strategy.generate_signals(data)
        actionable = [s for s in signals if s.action != Action.HOLD]

        logger.info(
            "Signals generated",
            total=len(signals),
            actionable=len(actionable),
        )

        # Record signal metrics & events
        for sig in signals:
            record_signal(sig.action.value, self.strategy.name)
            self.event_bus.publish(EventType.SIGNAL, {
                "symbol": sig.symbol,
                "action": sig.action.value,
                "strength": sig.strength,
                "reason": sig.reason,
                "strategy": self.strategy.name,
            })

        # 5. Execute actionable signals
        for sig in actionable:
            self._execute_signal(sig, data)

    def _execute_signal(self, sig: Signal, data: dict) -> None:
        """Risk-check and execute a single signal."""
        df = data.get(sig.symbol)
        if df is None or df.empty:
            return

        current_price = float(df["close"].iloc[-1])

        if sig.action == Action.BUY:
            # Calculate qty: target position = max_position_pct of equity
            target_value = self.portfolio.equity * self.settings.max_position_pct * sig.strength
            qty = max(1, int(target_value / current_price))

            verdict = self.risk.check_buy(sig.symbol, qty, current_price, self.portfolio)
            if not verdict.approved:
                logger.info("Signal rejected by risk", symbol=sig.symbol, reason=verdict.reason)
                return

            final_qty = int(verdict.adjusted_qty) if verdict.adjusted_qty else qty
            order = self.wrapper.submit_market_order(sig.symbol, final_qty, "buy")
            self.notifier.on_order_placed(order)
            record_trade("buy", sig.symbol)

            # Persist & publish
            self.state_store.log_trade({
                "side": "buy", "symbol": sig.symbol, "qty": final_qty,
                "price": current_price, "reason": sig.reason,
            })
            self.event_bus.publish(EventType.ORDER, {
                "side": "buy", "symbol": sig.symbol, "qty": final_qty,
                "price": current_price,
            })

        elif sig.action == Action.SELL:
            held = self.portfolio.position_qty(sig.symbol)
            if held <= 0:
                logger.debug("No position to sell", symbol=sig.symbol)
                return

            # Sell entire position
            qty = held
            verdict = self.risk.check_sell(sig.symbol, qty, self.portfolio)
            if not verdict.approved:
                logger.info("Sell rejected by risk", symbol=sig.symbol, reason=verdict.reason)
                return

            final_qty = verdict.adjusted_qty if verdict.adjusted_qty else qty
            order = self.wrapper.submit_market_order(sig.symbol, final_qty, "sell")
            self.notifier.on_order_placed(order)
            record_trade("sell", sig.symbol)

            # Persist & publish
            self.state_store.log_trade({
                "side": "sell", "symbol": sig.symbol, "qty": final_qty,
                "price": current_price, "reason": sig.reason,
            })
            self.event_bus.publish(EventType.ORDER, {
                "side": "sell", "symbol": sig.symbol, "qty": final_qty,
                "price": current_price,
            })

    # ── Helpers ─────────────────────────────────────────────

    def _wait(self) -> None:
        """Sleep for the configured interval, checking _running periodically."""
        remaining = self.settings.loop_interval_seconds
        while remaining > 0 and self._running:
            sleep_chunk = min(remaining, 5)
            time.sleep(sleep_chunk)
            remaining -= sleep_chunk

    def _setup_signal_handlers(self) -> None:
        """Graceful shutdown on SIGINT / SIGTERM."""
        def _handler(signum: int, frame: Any) -> None:
            logger.info(f"Signal {signum} received, stopping engine")
            self.stop()

        signal.signal(signal.SIGINT, _handler)
        try:
            signal.signal(signal.SIGTERM, _handler)
        except (OSError, AttributeError):
            pass  # SIGTERM not available on Windows in all contexts

