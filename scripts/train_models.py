"""
Train XGBoost and LSTM models on historical Alpaca data.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --symbols AAPL,MSFT --days 90 --model xgboost
    python scripts/train_models.py --model lstm --epochs 100
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import Settings
from src.data.alpaca_data import AlpacaDataClient
from src.strategies.xgboost_strategy import XGBoostStrategy
from src.strategies.lstm_strategy import LSTMStrategy
from src.utils.log_config import setup_logging, get_logger

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML models for trading strategies")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols")
    parser.add_argument("--days", type=int, default=60, help="Historical lookback days")
    parser.add_argument(
        "--model",
        choices=["xgboost", "lstm", "both"],
        default="both",
        help="Which model(s) to train",
    )
    parser.add_argument("--epochs", type=int, default=50, help="LSTM training epochs")
    parser.add_argument("--rounds", type=int, default=200, help="XGBoost boosting rounds")
    args = parser.parse_args()

    setup_logging(level="INFO")
    logger = get_logger("train_models")

    settings = Settings()
    settings.validate()

    symbols = args.symbols.split(",") if args.symbols else settings.watchlist[:5]
    data_client = AlpacaDataClient(settings)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)

    print(f"\n🔬 Model Training Pipeline")
    print(f"   Symbols:  {', '.join(symbols)}")
    print(f"   Period:   {start.date()} → {end.date()}")
    print(f"   Models:   {args.model}")
    print("─" * 60)

    # Fetch historical data
    print("\n📥 Fetching historical data...")
    data = {}
    for sym in symbols:
        try:
            df = data_client.get_bars(sym, start=start, end=end, limit=10000)
            data[sym] = df
            print(f"   {sym}: {len(df)} bars")
        except Exception as e:
            print(f"   {sym}: FAILED — {e}")

    if not data:
        print("\n❌ No data fetched. Exiting.")
        return

    _MODELS_DIR.mkdir(exist_ok=True)

    # ── XGBoost ─────────────────────────────────────────
    if args.model in ("xgboost", "both"):
        print("\n🌲 Training XGBoost model...")
        try:
            xgb_strategy = XGBoostStrategy()
            metrics = xgb_strategy.train(data, num_rounds=args.rounds)
            model_path = _MODELS_DIR / "xgboost_model.json"
            xgb_strategy.save_model(model_path)
            print(f"   ✅ XGBoost trained!")
            print(f"      Accuracy:    {metrics['accuracy']:.4f}")
            print(f"      Train/Val:   {metrics['train_size']}/{metrics['val_size']}")
            print(f"      Saved to:    {model_path}")
        except Exception as e:
            print(f"   ❌ XGBoost training failed: {e}")
            logger.exception("XGBoost training failed")

    # ── LSTM ────────────────────────────────────────────
    if args.model in ("lstm", "both"):
        print("\n🧠 Training LSTM model...")
        try:
            lstm_strategy = LSTMStrategy()
            metrics = lstm_strategy.train(data, epochs=args.epochs)
            model_path = _MODELS_DIR / "lstm_model.pt"
            lstm_strategy.save_model(model_path)
            print(f"   ✅ LSTM trained!")
            print(f"      Accuracy:    {metrics['accuracy']:.4f}")
            print(f"      Train/Val:   {metrics['train_size']}/{metrics['val_size']}")
            print(f"      Saved to:    {model_path}")
        except Exception as e:
            print(f"   ❌ LSTM training failed: {e}")
            logger.exception("LSTM training failed")

    print(f"\n{'─' * 60}")
    print("✅ Training complete!")


if __name__ == "__main__":
    main()
