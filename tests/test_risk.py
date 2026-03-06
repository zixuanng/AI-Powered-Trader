"""Tests for RiskManager."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.risk import RiskManager, RiskVerdict
from src.core.portfolio import Portfolio
from src.config.settings import Settings


def _make_settings(**overrides):
    """Helper to create a Settings with test defaults."""
    import os
    os.environ.setdefault("ALPACA_API_KEY", "test")
    os.environ.setdefault("ALPACA_SECRET_KEY", "test")
    os.environ.setdefault("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    return Settings(**{
        "max_position_pct": 0.05,
        "max_total_exposure_pct": 0.80,
        "max_drawdown_pct": 0.10,
        **overrides,
    })


def _make_portfolio(equity=100_000, cash=50_000, positions=None, last_equity=100_000):
    """Helper to create a Portfolio with known state."""
    p = Portfolio()
    p.equity = equity
    p.cash = cash
    p.buying_power = cash
    p.portfolio_value = equity
    p.last_equity = last_equity
    p.positions = positions or {}
    return p


class TestRiskManagerBuy:
    def test_buy_approved(self):
        rm = RiskManager(_make_settings())
        p = _make_portfolio()
        v = rm.check_buy("AAPL", 10, 150.0, p)
        assert v.approved

    def test_buy_position_too_large_adjusted(self):
        rm = RiskManager(_make_settings(max_position_pct=0.05))
        p = _make_portfolio(equity=100_000)
        # max value = 5000, order = 100 * 150 = 15000 → should be adjusted
        v = rm.check_buy("AAPL", 100, 150.0, p)
        assert v.approved
        assert v.adjusted_qty is not None
        assert v.adjusted_qty <= 33  # 5000/150 = 33

    def test_buy_rejected_drawdown(self):
        rm = RiskManager(_make_settings(max_drawdown_pct=0.05))
        p = _make_portfolio(equity=90_000, last_equity=100_000)  # 10% drawdown
        v = rm.check_buy("AAPL", 1, 150.0, p)
        assert not v.approved
        assert "drawdown" in v.reason.lower()

    def test_buy_rejected_exposure(self):
        rm = RiskManager(_make_settings(max_total_exposure_pct=0.50))
        positions = {
            "MSFT": {"market_value": 60_000, "qty": 100, "unrealized_pl": 0},
        }
        p = _make_portfolio(equity=100_000, positions=positions)
        v = rm.check_buy("AAPL", 1, 150.0, p)
        assert not v.approved
        assert "exposure" in v.reason.lower()

    def test_buy_rejected_buying_power(self):
        rm = RiskManager(_make_settings(max_position_pct=1.0))
        p = _make_portfolio(equity=100_000, cash=100.0)
        v = rm.check_buy("AAPL", 10, 150.0, p)
        assert not v.approved
        assert "buying power" in v.reason.lower()


class TestRiskManagerSell:
    def test_sell_approved(self):
        rm = RiskManager(_make_settings())
        positions = {"AAPL": {"qty": 10, "market_value": 1500, "unrealized_pl": 0}}
        p = _make_portfolio(positions=positions)
        v = rm.check_sell("AAPL", 10, p)
        assert v.approved

    def test_sell_no_position(self):
        rm = RiskManager(_make_settings())
        p = _make_portfolio()
        v = rm.check_sell("AAPL", 10, p)
        assert not v.approved

    def test_sell_more_than_held(self):
        rm = RiskManager(_make_settings())
        positions = {"AAPL": {"qty": 5, "market_value": 750, "unrealized_pl": 0}}
        p = _make_portfolio(positions=positions)
        v = rm.check_sell("AAPL", 10, p)
        assert v.approved
        assert v.adjusted_qty == 5
