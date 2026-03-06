"""Tests for LLMAgentStrategy with mocked Groq API."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategies.base_strategy import Action
from src.strategies.llm_strategy import LLMAgentStrategy


def _make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "open": [100.0] * 30,
        "high": [101.0] * 30,
        "low": [99.0] * 30,
        "close": [100.0] * 30,
        "volume": [10000.0] * 30,
    })


class TestLLMResponseParsing:
    def _make_strategy(self):
        """Create an LLM strategy with a mocked client."""
        strategy = LLMAgentStrategy(api_key="fake_key")
        return strategy

    def test_parse_valid_json(self):
        """Strategy should parse valid JSON decisions."""
        strategy = self._make_strategy()

        response_json = json.dumps([
            {"symbol": "AAPL", "action": "BUY", "strength": 0.8, "reason": "Bullish trend"},
            {"symbol": "MSFT", "action": "HOLD", "strength": 0.3, "reason": "Neutral"},
        ])

        signals = strategy._parse_response(response_json, ["AAPL", "MSFT"])

        assert len(signals) == 2
        aapl = next(s for s in signals if s.symbol == "AAPL")
        assert aapl.action == Action.BUY
        assert aapl.strength == 0.8

    def test_parse_json_with_markdown_wrapper(self):
        """Strategy should handle JSON wrapped in ```json ... ```."""
        strategy = self._make_strategy()

        response = '```json\n[{"symbol": "AAPL", "action": "SELL", "strength": 0.6, "reason": "Overbought"}]\n```'
        signals = strategy._parse_response(response, ["AAPL"])

        aapl = next(s for s in signals if s.symbol == "AAPL")
        assert aapl.action == Action.SELL

    def test_parse_invalid_json_fallback_hold(self):
        """Invalid JSON → missing symbols get HOLD."""
        strategy = self._make_strategy()

        signals = strategy._parse_response("this is not json!", ["AAPL", "MSFT"])

        assert len(signals) == 2
        assert all(s.action == Action.HOLD for s in signals)

    def test_missing_symbol_gets_hold(self):
        """If LLM omits a symbol, it should get a HOLD signal."""
        strategy = self._make_strategy()

        response_json = json.dumps([
            {"symbol": "AAPL", "action": "BUY", "strength": 0.7, "reason": "Looks good"},
        ])
        signals = strategy._parse_response(response_json, ["AAPL", "MSFT"])

        assert len(signals) == 2
        msft = next(s for s in signals if s.symbol == "MSFT")
        assert msft.action == Action.HOLD

    def test_strength_clamped(self):
        """Strength values should be clamped to [0, 1]."""
        strategy = self._make_strategy()

        response = json.dumps([
            {"symbol": "TEST", "action": "BUY", "strength": 1.5, "reason": "Over-confident"},
        ])
        signals = strategy._parse_response(response, ["TEST"])
        assert signals[0].strength == 1.0


class TestLLMAPICall:
    def test_api_error_fallback(self):
        """On API error, all symbols should get HOLD."""
        strategy = LLMAgentStrategy(api_key="fake_key")
        strategy._client = MagicMock()
        strategy._client.chat.completions.create.side_effect = Exception("API down")

        data = {"AAPL": _make_df(), "MSFT": _make_df()}
        signals = strategy.generate_signals(data)

        assert len(signals) == 2
        assert all(s.action == Action.HOLD for s in signals)

    def test_no_client_returns_hold(self):
        """If no API key, return HOLD for all."""
        strategy = LLMAgentStrategy(api_key="")  # empty key → no client
        strategy._client = None

        data = {"AAPL": _make_df()}
        signals = strategy.generate_signals(data)

        assert len(signals) == 1
        assert signals[0].action == Action.HOLD
