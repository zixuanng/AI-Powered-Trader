"""
Application settings, loaded from .env file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Locate .env relative to this file (two dirs up → project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Immutable application configuration."""

    # ── Alpaca ──────────────────────────────────────────────
    alpaca_api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    alpaca_base_url: str = field(
        default_factory=lambda: os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    )

    # ── Groq (Phase 2) ─────────────────────────────────────
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

    # ── Trading ─────────────────────────────────────────────
    watchlist: list[str] = field(
        default_factory=lambda: [
            s.strip()
            for s in os.getenv("WATCHLIST", "AAPL,MSFT,GOOGL,AMZN,TSLA").split(",")
            if s.strip()
        ]
    )
    bar_timeframe: str = field(default_factory=lambda: os.getenv("BAR_TIMEFRAME", "5Min"))
    loop_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("LOOP_INTERVAL_SECONDS", "300"))
    )

    # ── Risk ────────────────────────────────────────────────
    max_position_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_PCT", "0.05"))
    )
    max_total_exposure_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "0.80"))
    )
    max_drawdown_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_DRAWDOWN_PCT", "0.10"))
    )

    # ── Notifications ───────────────────────────────────────
    telegram_bot_token: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", "")
    )
    telegram_chat_id: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", "")
    )

    # ── Derived ─────────────────────────────────────────────
    @property
    def is_paper(self) -> bool:
        """True when the base URL points to Alpaca's paper environment."""
        return "paper" in self.alpaca_base_url.lower()

    def validate(self) -> None:
        """Raise if critical settings are missing."""
        if not self.alpaca_api_key:
            raise ValueError("ALPACA_API_KEY is not set")
        if not self.alpaca_secret_key:
            raise ValueError("ALPACA_SECRET_KEY is not set")
        if not self.is_paper:
            raise ValueError(
                "Safety check failed: ALPACA_BASE_URL does not contain 'paper'. "
                "This bot is designed for paper trading only."
            )


# Singleton for convenience
settings = Settings()
