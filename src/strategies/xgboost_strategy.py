"""
XGBoost-based trading strategy.

Uses a trained XGBoost classifier to predict whether the next N bars
will be up or down, based on engineered features from OHLCV data.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.ml.features import build_features
from src.strategies.base_strategy import Action, BaseStrategy, Signal
from src.utils.log_config import get_logger

logger = get_logger(__name__)

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class XGBoostStrategy(BaseStrategy):
    """XGBoost classifier strategy.

    Predicts next-bar direction from engineered features.
    Model must be trained before use (see scripts/train_models.py).
    """

    name = "XGBoost"

    def __init__(
        self,
        model_path: str | Path | None = None,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.4,
        target_horizon: int = 5,
    ) -> None:
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.target_horizon = target_horizon
        self._model: xgb.Booster | None = None

        if model_path:
            self.load_model(model_path)
        else:
            default = _MODELS_DIR / "xgboost_model.json"
            if default.exists():
                self.load_model(default)

    def load_model(self, path: str | Path) -> None:
        """Load a trained XGBoost model."""
        self._model = xgb.Booster()
        self._model.load_model(str(path))
        logger.info("XGBoost model loaded", path=str(path))

    def save_model(self, path: str | Path) -> None:
        """Save the model to disk."""
        if self._model is None:
            raise RuntimeError("No model to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        logger.info("XGBoost model saved", path=str(path))

    def train(
        self,
        data: dict[str, pd.DataFrame],
        params: dict[str, Any] | None = None,
        num_rounds: int = 200,
    ) -> dict[str, float]:
        """Train the XGBoost model on historical data.

        Args:
            data: {symbol: OHLCV DataFrame} — concatenated for training.
            params: XGBoost training parameters.
            num_rounds: number of boosting rounds.

        Returns:
            dict with training metrics.
        """
        all_features = []
        for symbol, df in data.items():
            if df.empty or len(df) < 60:
                continue
            features = build_features(df, target_horizon=self.target_horizon, include_target=True)
            if not features.empty:
                all_features.append(features)

        if not all_features:
            raise ValueError("No valid training data")

        combined = pd.concat(all_features, ignore_index=True)
        X = combined.drop(columns=["target"])
        y = combined["target"]

        # Train/validation split (80/20, chronological)
        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X.columns.tolist())
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=X.columns.tolist())

        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "verbosity": 0,
        }
        if params:
            default_params.update(params)

        self._model = xgb.train(
            default_params,
            dtrain,
            num_boost_round=num_rounds,
            evals=[(dtrain, "train"), (dval, "val")],
            verbose_eval=False,
            early_stopping_rounds=20,
        )
        self._feature_names = X.columns.tolist()

        # Metrics
        val_pred = self._model.predict(dval)
        val_binary = (val_pred > 0.5).astype(int)
        accuracy = float((val_binary == y_val.values).mean())
        logger.info("XGBoost trained", accuracy=round(accuracy, 4), samples=len(X))

        return {"accuracy": accuracy, "train_size": split, "val_size": len(X) - split}

    def generate_signals(
        self, data: dict[str, pd.DataFrame]
    ) -> list[Signal]:
        if self._model is None:
            logger.warning("XGBoost model not loaded — returning HOLD for all symbols")
            return [
                Signal(symbol=s, action=Action.HOLD, reason="Model not loaded")
                for s in data
            ]

        signals: list[Signal] = []

        for symbol, df in data.items():
            if df.empty or len(df) < 60:
                signals.append(
                    Signal(symbol=symbol, action=Action.HOLD, reason="Insufficient data")
                )
                continue

            try:
                features = build_features(df, include_target=False)
                if features.empty:
                    signals.append(
                        Signal(symbol=symbol, action=Action.HOLD, reason="No valid features")
                    )
                    continue

                # Use the last row for prediction
                last_row = features.iloc[[-1]]
                dmatrix = xgb.DMatrix(last_row, feature_names=last_row.columns.tolist())
                prob_up = float(self._model.predict(dmatrix)[0])

                if prob_up >= self.buy_threshold:
                    sig = Signal(
                        symbol=symbol,
                        action=Action.BUY,
                        strength=prob_up,
                        reason=f"XGBoost BUY: P(up)={prob_up:.3f} ≥ {self.buy_threshold}",
                        metadata={"prob_up": round(prob_up, 4)},
                    )
                elif prob_up <= self.sell_threshold:
                    sig = Signal(
                        symbol=symbol,
                        action=Action.SELL,
                        strength=1.0 - prob_up,
                        reason=f"XGBoost SELL: P(up)={prob_up:.3f} ≤ {self.sell_threshold}",
                        metadata={"prob_up": round(prob_up, 4)},
                    )
                else:
                    sig = Signal(
                        symbol=symbol,
                        action=Action.HOLD,
                        strength=0.0,
                        reason=f"XGBoost HOLD: P(up)={prob_up:.3f}",
                    )

                signals.append(sig)

            except Exception:
                logger.exception("XGBoost prediction failed", symbol=symbol)
                signals.append(
                    Signal(symbol=symbol, action=Action.HOLD, reason="Prediction error")
                )

        return signals
