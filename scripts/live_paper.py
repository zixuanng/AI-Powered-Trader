"""
Entry point — run the paper trading bot.

Usage:
    python scripts/live_paper.py                      # run continuously (MA crossover)
    python scripts/live_paper.py --once               # run one iteration and exit
    python scripts/live_paper.py --strategy rsi       # use RSI strategy
    python scripts/live_paper.py --strategy ensemble  # use ensemble (MA + RSI + Bollinger)
    python scripts/live_paper.py --strategy llm       # use Groq LLM agent
    python scripts/live_paper.py --strategy xgboost   # use XGBoost model
    python scripts/live_paper.py --strategy lstm      # use LSTM model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import Settings
from src.core.trading_engine import TradingEngine
from src.strategies.base_strategy import BaseStrategy
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.bollinger_strategy import BollingerBandStrategy
from src.strategies.xgboost_strategy import XGBoostStrategy
from src.strategies.lstm_strategy import LSTMStrategy
from src.strategies.llm_strategy import LLMAgentStrategy
from src.strategies.ensemble_strategy import EnsembleStrategy, StrategyWeight
from src.utils.log_config import setup_logging, get_logger

STRATEGY_NAMES = [
    "ma_crossover", "rsi", "bollinger",
    "xgboost", "lstm", "llm", "ensemble",
]


def build_strategy(name: str, settings: Settings) -> BaseStrategy:
    """Construct a strategy by name, injecting dependencies as needed."""
    if name == "ma_crossover":
        return MACrossoverStrategy()
    elif name == "rsi":
        return RSIStrategy()
    elif name == "bollinger":
        return BollingerBandStrategy()
    elif name == "xgboost":
        return XGBoostStrategy()
    elif name == "lstm":
        return LSTMStrategy()
    elif name == "llm":
        return LLMAgentStrategy(settings=settings)
    elif name == "ensemble":
        return EnsembleStrategy(
            strategies=[
                StrategyWeight(MACrossoverStrategy(), weight=1.0),
                StrategyWeight(RSIStrategy(), weight=1.0),
                StrategyWeight(BollingerBandStrategy(), weight=1.0),
            ],
            mode="majority",
            threshold=0.5,
        )
    else:
        raise ValueError(f"Unknown strategy: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Paper Trading Bot")
    parser.add_argument("--once", action="store_true", help="Run one iteration and exit")
    parser.add_argument(
        "--strategy",
        choices=STRATEGY_NAMES,
        default="ma_crossover",
        help="Strategy to use (default: ma_crossover)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    parser.add_argument("--json-log", action="store_true", help="Emit JSON log lines")
    args = parser.parse_args()

    setup_logging(level=args.log_level, json_output=args.json_log)
    logger = get_logger("live_paper")

    settings = Settings()
    settings.validate()

    logger.info(
        "Configuration loaded",
        watchlist=settings.watchlist,
        timeframe=settings.bar_timeframe,
        is_paper=settings.is_paper,
    )

    strategy = build_strategy(args.strategy, settings)
    logger.info("Strategy initialized", strategy=strategy.name)

    engine = TradingEngine(settings=settings, strategy=strategy)
    engine.start(run_once=args.once)


if __name__ == "__main__":
    main()

