"""
JSON-file state persistence for portfolio snapshots and trade history.

Saves after each engine iteration so state survives restarts.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.utils.log_config import get_logger

logger = get_logger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class StateStore:
    """Persists portfolio snapshots and trade history to JSON files.

    Files:
        data/state.json   — latest portfolio state + equity history
        data/trades.json  — log of all signals and orders
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._state_path = self._data_dir / "state.json"
        self._trades_path = self._data_dir / "trades.json"

        self._state: dict[str, Any] = self._load_json(self._state_path, default={
            "equity_history": [],
            "latest_snapshot": {},
            "engine_started_at": None,
        })
        self._trades: list[dict[str, Any]] = self._load_json(self._trades_path, default=[])

    # ── Public API ──────────────────────────────────────────

    def save_snapshot(self, portfolio_summary: dict[str, Any]) -> None:
        """Record a portfolio snapshot with timestamp."""
        entry = {
            "timestamp": time.time(),
            **portfolio_summary,
        }
        self._state["latest_snapshot"] = entry
        self._state["equity_history"].append({
            "timestamp": entry["timestamp"],
            "equity": portfolio_summary.get("equity", 0),
            "cash": portfolio_summary.get("cash", 0),
            "drawdown_pct": portfolio_summary.get("drawdown_pct", 0),
        })

        # Keep last 2000 snapshots
        if len(self._state["equity_history"]) > 2000:
            self._state["equity_history"] = self._state["equity_history"][-2000:]

        self._save_json(self._state_path, self._state)

    def log_trade(self, trade_data: dict[str, Any]) -> None:
        """Append a trade (signal or order) to the log."""
        entry = {
            "timestamp": time.time(),
            **trade_data,
        }
        self._trades.append(entry)

        # Keep last 1000 trades
        if len(self._trades) > 1000:
            self._trades = self._trades[-1000:]

        self._save_json(self._trades_path, self._trades)

    def set_engine_started(self) -> None:
        """Mark the engine start time."""
        self._state["engine_started_at"] = time.time()
        self._save_json(self._state_path, self._state)

    # ── Queries ─────────────────────────────────────────────

    def get_latest_snapshot(self) -> dict[str, Any]:
        return self._state.get("latest_snapshot", {})

    def get_equity_history(self) -> list[dict[str, Any]]:
        return self._state.get("equity_history", [])

    def get_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._trades[-limit:]

    def get_state(self) -> dict[str, Any]:
        return self._state.copy()

    # ── Internal ────────────────────────────────────────────

    @staticmethod
    def _load_json(path: Path, default: Any = None) -> Any:
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load state file, using default", path=str(path))
        return default if default is not None else {}

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except OSError:
            logger.exception("Failed to save state file", path=str(path))
