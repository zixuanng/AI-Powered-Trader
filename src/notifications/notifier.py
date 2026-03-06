"""
Notification system — console and Telegram (optional).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import requests

from src.config.settings import Settings
from src.utils.log_config import get_logger

logger = get_logger(__name__)


class BaseNotifier(ABC):
    """Interface for notification backends."""

    @abstractmethod
    def send(self, title: str, body: str, data: dict[str, Any] | None = None) -> None:
        ...


class ConsoleNotifier(BaseNotifier):
    """Prints notifications to the console via structlog."""

    def send(self, title: str, body: str, data: dict[str, Any] | None = None) -> None:
        logger.info(f"[NOTIFY] {title}", body=body, **(data or {}))


class TelegramNotifier(BaseNotifier):
    """Sends notifications via Telegram Bot API.

    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in settings.
    """

    TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        self.chat_id = chat_id

    def send(self, title: str, body: str, data: dict[str, Any] | None = None) -> None:
        text = f"*{title}*\n{body}"
        if data:
            text += f"\n```json\n{json.dumps(data, indent=2, default=str)}\n```"

        try:
            resp = requests.post(
                self.TELEGRAM_API.format(token=self.token),
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
            resp.raise_for_status()
        except Exception:
            logger.exception("Telegram notification failed")


class Notifier:
    """Fan-out notifier that dispatches to all configured backends."""

    def __init__(self, settings: Settings) -> None:
        self._backends: list[BaseNotifier] = [ConsoleNotifier()]

        if settings.telegram_bot_token and settings.telegram_chat_id:
            self._backends.append(
                TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
            )
            logger.info("Telegram notifier enabled")

    def on_order_placed(self, order: dict) -> None:
        self._send(
            "📋 Order Placed",
            f"{order.get('side', '').upper()} {order.get('qty')} × {order.get('symbol')}",
            order,
        )

    def on_order_filled(self, order: dict) -> None:
        self._send(
            "✅ Order Filled",
            f"{order.get('side', '').upper()} {order.get('qty')} × {order.get('symbol')}",
            order,
        )

    def on_signal(self, signal_data: dict) -> None:
        self._send(
            f"📊 Signal: {signal_data.get('action', '')}",
            f"{signal_data.get('symbol', '')} — {signal_data.get('reason', '')}",
            signal_data,
        )

    def on_error(self, error: str, context: dict | None = None) -> None:
        self._send("❌ Error", error, context)

    def on_summary(self, summary: dict) -> None:
        self._send("📈 Portfolio Summary", "", summary)

    def _send(self, title: str, body: str, data: dict[str, Any] | None = None) -> None:
        for backend in self._backends:
            try:
                backend.send(title, body, data)
            except Exception:
                logger.exception("Notifier backend failed", backend=type(backend).__name__)
