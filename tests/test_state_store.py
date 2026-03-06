"""Tests for StateStore."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.state_store import StateStore


class TestStateStore:
    def test_save_and_load_snapshot(self, tmp_path):
        store = StateStore(data_dir=tmp_path)
        store.save_snapshot({"equity": 100000, "cash": 50000, "drawdown_pct": 2.5})

        # Reload from disk
        store2 = StateStore(data_dir=tmp_path)
        snap = store2.get_latest_snapshot()
        assert snap["equity"] == 100000
        assert snap["cash"] == 50000

    def test_equity_history_grows(self, tmp_path):
        store = StateStore(data_dir=tmp_path)
        store.save_snapshot({"equity": 100000, "cash": 50000})
        store.save_snapshot({"equity": 101000, "cash": 49000})

        history = store.get_equity_history()
        assert len(history) == 2
        assert history[0]["equity"] == 100000
        assert history[1]["equity"] == 101000

    def test_equity_history_capped(self, tmp_path):
        store = StateStore(data_dir=tmp_path)
        for i in range(2100):
            store.save_snapshot({"equity": 100000 + i})

        history = store.get_equity_history()
        assert len(history) == 2000

    def test_log_trade(self, tmp_path):
        store = StateStore(data_dir=tmp_path)
        store.log_trade({"side": "buy", "symbol": "AAPL", "qty": 10, "price": 150.0})
        store.log_trade({"side": "sell", "symbol": "MSFT", "qty": 5, "price": 300.0})

        trades = store.get_trades()
        assert len(trades) == 2
        assert trades[0]["symbol"] == "AAPL"
        assert trades[1]["symbol"] == "MSFT"

    def test_trades_capped(self, tmp_path):
        store = StateStore(data_dir=tmp_path)
        for i in range(1100):
            store.log_trade({"symbol": f"SYM{i}"})

        trades = store.get_trades(limit=1100)
        assert len(trades) == 1000

    def test_persistence_across_instances(self, tmp_path):
        store1 = StateStore(data_dir=tmp_path)
        store1.save_snapshot({"equity": 50000})
        store1.log_trade({"side": "buy", "symbol": "GOOG"})

        store2 = StateStore(data_dir=tmp_path)
        assert store2.get_latest_snapshot()["equity"] == 50000
        assert len(store2.get_trades()) == 1

    def test_set_engine_started(self, tmp_path):
        store = StateStore(data_dir=tmp_path)
        store.set_engine_started()
        state = store.get_state()
        assert state["engine_started_at"] is not None

    def test_corrupted_file_uses_default(self, tmp_path):
        # Write garbage to state file
        state_file = tmp_path / "state.json"
        state_file.write_text("not valid json!!!")

        store = StateStore(data_dir=tmp_path)
        assert store.get_latest_snapshot() == {}
        assert store.get_equity_history() == []
