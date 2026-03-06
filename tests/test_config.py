"""Tests for Settings configuration."""
import os
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_settings_loads_from_env(monkeypatch):
    """Settings should pick up values from environment variables."""
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("WATCHLIST", "AAPL,TSLA")
    monkeypatch.setenv("MAX_POSITION_PCT", "0.10")

    from src.config.settings import Settings
    s = Settings()

    assert s.alpaca_api_key == "test_key"
    assert s.alpaca_secret_key == "test_secret"
    assert s.is_paper is True
    assert "AAPL" in s.watchlist
    assert s.max_position_pct == 0.10


def test_settings_validate_missing_key(monkeypatch):
    """validate() should raise if API key is missing."""
    monkeypatch.setenv("ALPACA_API_KEY", "")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    from src.config.settings import Settings
    s = Settings()

    with pytest.raises(ValueError, match="ALPACA_API_KEY"):
        s.validate()


def test_settings_validate_not_paper(monkeypatch):
    """validate() should raise if URL is not paper."""
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")

    from src.config.settings import Settings
    s = Settings()

    with pytest.raises(ValueError, match="paper"):
        s.validate()
