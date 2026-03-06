"""Tests for Portfolio state management."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.portfolio import Portfolio


class TestPortfolio:
    def _make_portfolio(self) -> Portfolio:
        p = Portfolio()
        p.equity = 100_000
        p.cash = 50_000
        p.buying_power = 50_000
        p.portfolio_value = 100_000
        p.last_equity = 100_000
        p.positions = {
            "AAPL": {
                "symbol": "AAPL",
                "qty": 100,
                "market_value": 25_000,
                "avg_entry_price": 240.0,
                "current_price": 250.0,
                "unrealized_pl": 1000.0,
                "unrealized_plpc": 0.04,
                "side": "long",
            },
            "MSFT": {
                "symbol": "MSFT",
                "qty": 50,
                "market_value": 20_000,
                "avg_entry_price": 380.0,
                "current_price": 400.0,
                "unrealized_pl": 1000.0,
                "unrealized_plpc": 0.05,
                "side": "long",
            },
        }
        return p

    def test_has_position(self):
        p = self._make_portfolio()
        assert p.has_position("AAPL")
        assert not p.has_position("GOOGL")

    def test_position_qty(self):
        p = self._make_portfolio()
        assert p.position_qty("AAPL") == 100
        assert p.position_qty("GOOGL") == 0

    def test_total_market_value(self):
        p = self._make_portfolio()
        assert p.total_market_value() == 45_000  # 25000 + 20000

    def test_unrealized_pnl(self):
        p = self._make_portfolio()
        assert p.unrealized_pnl() == 2000.0

    def test_current_exposure_pct(self):
        p = self._make_portfolio()
        assert abs(p.current_exposure_pct() - 0.45) < 0.01

    def test_drawdown_pct_no_loss(self):
        p = self._make_portfolio()
        assert p.drawdown_pct() == 0.0

    def test_drawdown_pct_with_loss(self):
        p = self._make_portfolio()
        p.equity = 90_000  # 10% drop from 100k
        assert abs(p.drawdown_pct() - 0.10) < 0.001

    def test_summary(self):
        p = self._make_portfolio()
        s = p.summary()
        assert "equity" in s
        assert "cash" in s
        assert "positions" in s
        assert s["positions"] == 2
