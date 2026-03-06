"""Tests for ML pipeline: features, XGBoost, and LSTM."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategies.base_strategy import Action


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    close = np.cumsum(rng.randn(n)) + 100
    close = np.maximum(close, 1)  # no negative prices
    return pd.DataFrame({
        "open": close + rng.uniform(-0.5, 0.5, n),
        "high": close + rng.uniform(0, 1.0, n),
        "low": close - rng.uniform(0, 1.0, n),
        "close": close,
        "volume": rng.randint(1000, 100000, n).astype(float),
    })


class TestFeatures:
    def test_build_features_shape(self):
        from src.ml.features import build_features
        df = _make_ohlcv(200)
        features = build_features(df, target_horizon=5, include_target=True)
        assert len(features) > 0
        assert "target" in features.columns

    def test_build_features_no_target(self):
        from src.ml.features import build_features
        df = _make_ohlcv(200)
        features = build_features(df, include_target=False)
        assert "target" not in features.columns
        assert len(features.columns) > 15  # should have many features

    def test_build_features_no_nans(self):
        from src.ml.features import build_features
        df = _make_ohlcv(200)
        features = build_features(df, include_target=False)
        assert not features.isna().any().any()

    def test_feature_columns_list(self):
        from src.ml.features import get_feature_columns
        cols = get_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 10


class TestXGBoostStrategy:
    def test_train_and_predict(self):
        from src.strategies.xgboost_strategy import XGBoostStrategy
        data = {"SYN1": _make_ohlcv(300, seed=1), "SYN2": _make_ohlcv(300, seed=2)}

        strategy = XGBoostStrategy()
        metrics = strategy.train(data, num_rounds=20)

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

        # Predict
        signals = strategy.generate_signals(data)
        assert len(signals) == 2
        for sig in signals:
            assert sig.action in (Action.BUY, Action.SELL, Action.HOLD)

    def test_no_model_returns_hold(self):
        from src.strategies.xgboost_strategy import XGBoostStrategy
        strategy = XGBoostStrategy()  # no model loaded
        signals = strategy.generate_signals({"TEST": _make_ohlcv(100)})
        assert all(s.action == Action.HOLD for s in signals)

    def test_save_load(self, tmp_path):
        from src.strategies.xgboost_strategy import XGBoostStrategy
        data = {"SYN": _make_ohlcv(200)}
        strategy = XGBoostStrategy()
        strategy.train(data, num_rounds=10)

        path = tmp_path / "test_xgb.json"
        strategy.save_model(path)
        assert path.exists()

        loaded = XGBoostStrategy(model_path=path)
        signals = loaded.generate_signals(data)
        assert len(signals) == 1


class TestLSTMStrategy:
    def test_train_and_predict(self):
        from src.strategies.lstm_strategy import LSTMStrategy
        data = {"SYN1": _make_ohlcv(300, seed=10)}

        strategy = LSTMStrategy(window_size=10, hidden_size=16)
        metrics = strategy.train(data, epochs=5, batch_size=16)

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

        signals = strategy.generate_signals(data)
        assert len(signals) == 1
        assert signals[0].action in (Action.BUY, Action.SELL, Action.HOLD)

    def test_no_model_returns_hold(self):
        from src.strategies.lstm_strategy import LSTMStrategy
        strategy = LSTMStrategy()
        signals = strategy.generate_signals({"TEST": _make_ohlcv(100)})
        assert all(s.action == Action.HOLD for s in signals)

    def test_save_load(self, tmp_path):
        from src.strategies.lstm_strategy import LSTMStrategy
        data = {"SYN": _make_ohlcv(200)}
        strategy = LSTMStrategy(window_size=10, hidden_size=16)
        strategy.train(data, epochs=3)

        path = tmp_path / "test_lstm.pt"
        strategy.save_model(path)
        assert path.exists()

        loaded = LSTMStrategy(model_path=path, window_size=10, hidden_size=16)
        signals = loaded.generate_signals(data)
        assert len(signals) == 1

    def test_lstm_forward_shape(self):
        """Verify LSTM model output shape."""
        from src.strategies.lstm_strategy import LSTMModel
        import torch
        model = LSTMModel(input_size=20, hidden_size=32)
        x = torch.randn(4, 10, 20)  # batch=4, seq=10, features=20
        out = model(x)
        assert out.shape == (4,)
        assert (out >= 0).all() and (out <= 1).all()  # sigmoid output
