"""
LLM Agent strategy using Groq API.

Sends market context (OHLCV summary, technical indicators, portfolio state)
to a Groq-hosted LLM and parses structured JSON trade decisions.
"""
from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd
from groq import Groq

from src.config.settings import Settings
from src.strategies.base_strategy import Action, BaseStrategy, Signal
from src.strategies.rsi_strategy import compute_rsi
from src.utils.log_config import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are an expert quantitative trading analyst. You analyze stock data and make trading decisions.

For each stock symbol provided, you must return a JSON array of decisions. Each decision has:
- "symbol": the stock ticker
- "action": one of "BUY", "SELL", or "HOLD"
- "strength": a float from 0.0 to 1.0 indicating confidence
- "reason": a brief explanation of your reasoning

Consider:
1. Price trends and momentum
2. Technical indicators (RSI, moving averages)
3. Volume patterns
4. Risk management (don't be overly aggressive)

Return ONLY valid JSON. No markdown, no explanation outside the JSON.
Example: [{"symbol": "AAPL", "action": "BUY", "strength": 0.7, "reason": "RSI oversold with bullish momentum"}]"""


class LLMAgentStrategy(BaseStrategy):
    """Strategy powered by Groq LLM inference."""

    name = "LLM_Agent"

    def __init__(
        self,
        settings: Settings | None = None,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
        max_symbols_per_call: int = 5,
        rate_limit_delay: float = 1.0,
    ) -> None:
        self._api_key = api_key or (settings.groq_api_key if settings else "")
        self._model = model
        self._max_symbols = max_symbols_per_call
        self._rate_limit_delay = rate_limit_delay
        self._client: Groq | None = None

        if self._api_key:
            self._client = Groq(api_key=self._api_key)

    def _build_market_context(self, symbol: str, df: pd.DataFrame) -> str:
        """Build a concise text summary of market data for the LLM."""
        if df.empty or len(df) < 20:
            return f"{symbol}: Insufficient data"

        close = df["close"]
        current = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) >= 2 else current

        # Recent performance
        change_1 = (current / prev - 1) * 100 if prev else 0
        change_5 = (current / float(close.iloc[-5]) - 1) * 100 if len(close) >= 5 else 0
        change_20 = (current / float(close.iloc[-20]) - 1) * 100 if len(close) >= 20 else 0

        # Technical indicators
        sma_10 = float(close.rolling(10).mean().iloc[-1]) if len(close) >= 10 else current
        sma_20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else current
        rsi = compute_rsi(close, 14)
        rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

        # Volume
        vol = df["volume"]
        avg_vol = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float(vol.iloc[-1])
        current_vol = float(vol.iloc[-1])
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        return (
            f"{symbol}: price=${current:.2f}, "
            f"1bar={change_1:+.2f}%, 5bar={change_5:+.2f}%, 20bar={change_20:+.2f}%, "
            f"SMA10=${sma_10:.2f}, SMA20=${sma_20:.2f}, RSI={rsi_val:.1f}, "
            f"vol_ratio={vol_ratio:.2f}"
        )

    def generate_signals(
        self, data: dict[str, pd.DataFrame]
    ) -> list[Signal]:
        if not self._client:
            logger.warning("Groq client not initialized — returning HOLD")
            return [
                Signal(symbol=s, action=Action.HOLD, reason="LLM not configured")
                for s in data
            ]

        signals: list[Signal] = []
        symbols = list(data.keys())

        # Process in batches to respect rate limits
        for i in range(0, len(symbols), self._max_symbols):
            batch = symbols[i : i + self._max_symbols]
            batch_signals = self._query_llm(batch, data)
            signals.extend(batch_signals)

            if i + self._max_symbols < len(symbols):
                time.sleep(self._rate_limit_delay)

        return signals

    def _query_llm(
        self, symbols: list[str], data: dict[str, pd.DataFrame]
    ) -> list[Signal]:
        """Send a batch of symbols to the LLM and parse the response."""
        # Build context
        contexts = []
        for sym in symbols:
            df = data.get(sym, pd.DataFrame())
            contexts.append(self._build_market_context(sym, df))

        user_message = "Analyze these stocks and provide trading decisions:\n\n"
        user_message += "\n".join(contexts)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=1024,
            )

            content = response.choices[0].message.content.strip()
            return self._parse_response(content, symbols)

        except Exception:
            logger.exception("LLM query failed")
            return [
                Signal(symbol=s, action=Action.HOLD, strength=0.0, reason="LLM error — fallback HOLD")
                for s in symbols
            ]

    def _parse_response(self, content: str, expected_symbols: list[str]) -> list[Signal]:
        """Parse the LLM JSON response into Signal objects."""
        signals: list[Signal] = []
        seen: set[str] = set()

        try:
            # Try to extract JSON from the response
            # Handle cases where LLM wraps in ```json ... ```
            cleaned = content
            if "```" in cleaned:
                start = cleaned.find("[")
                end = cleaned.rfind("]") + 1
                if start >= 0 and end > start:
                    cleaned = cleaned[start:end]

            decisions = json.loads(cleaned)

            if not isinstance(decisions, list):
                decisions = [decisions]

            for d in decisions:
                symbol = d.get("symbol", "").upper()
                if symbol not in expected_symbols:
                    continue

                action_str = d.get("action", "HOLD").upper()
                try:
                    action = Action(action_str)
                except ValueError:
                    action = Action.HOLD

                strength = max(0.0, min(1.0, float(d.get("strength", 0.5))))
                reason = d.get("reason", "LLM decision")

                signals.append(Signal(
                    symbol=symbol,
                    action=action,
                    strength=strength,
                    reason=f"LLM: {reason}",
                    metadata={"raw_response": d},
                ))
                seen.add(symbol)
                logger.info("LLM signal", symbol=symbol, action=action_str, strength=strength)

        except (json.JSONDecodeError, TypeError, KeyError):
            logger.exception("Failed to parse LLM response", raw=content[:200])

        # Any symbols not in the response → HOLD
        for sym in expected_symbols:
            if sym not in seen:
                signals.append(
                    Signal(symbol=sym, action=Action.HOLD, strength=0.0, reason="LLM: no decision returned")
                )

        return signals
