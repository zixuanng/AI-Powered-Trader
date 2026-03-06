"""
LSTM-based trading strategy using PyTorch.

Uses a sliding window of feature vectors to predict next-bar direction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.ml.features import build_features
from src.strategies.base_strategy import Action, BaseStrategy, Signal
from src.utils.log_config import get_logger

logger = get_logger(__name__)

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class LSTMModel(nn.Module):
    """Simple 1-layer LSTM for binary classification."""

    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch, seq_len, features) → output: (batch, 1) probabilities."""
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # take the last time step
        out = self.dropout(last_hidden)
        out = torch.sigmoid(self.fc(out))
        return out.squeeze(-1)


class LSTMStrategy(BaseStrategy):
    """PyTorch LSTM strategy for sequence-based prediction."""

    name = "LSTM"

    def __init__(
        self,
        model_path: str | Path | None = None,
        window_size: int = 20,
        hidden_size: int = 64,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.4,
    ) -> None:
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self._model: LSTMModel | None = None
        self._input_size: int | None = None
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None

        if model_path:
            self.load_model(model_path)
        else:
            default = _MODELS_DIR / "lstm_model.pt"
            if default.exists():
                self.load_model(default)

    def load_model(self, path: str | Path) -> None:
        """Load a trained LSTM model + normalization params."""
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        self._input_size = checkpoint["input_size"]
        self._model = LSTMModel(self._input_size, self.hidden_size)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()
        self._feature_means = checkpoint.get("feature_means")
        self._feature_stds = checkpoint.get("feature_stds")
        logger.info("LSTM model loaded", path=str(path))

    def save_model(self, path: str | Path) -> None:
        """Save model + normalization params."""
        if self._model is None:
            raise RuntimeError("No model to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "input_size": self._input_size,
            "feature_means": self._feature_means,
            "feature_stds": self._feature_stds,
        }, str(path))
        logger.info("LSTM model saved", path=str(path))

    def train(
        self,
        data: dict[str, pd.DataFrame],
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
    ) -> dict[str, float]:
        """Train the LSTM on historical data.

        Returns dict with training metrics.
        """
        # Build features from all symbols
        all_features = []
        for symbol, df in data.items():
            if df.empty or len(df) < 60:
                continue
            features = build_features(df, target_horizon=5, include_target=True)
            if not features.empty:
                all_features.append(features)

        if not all_features:
            raise ValueError("No valid training data")

        combined = pd.concat(all_features, ignore_index=True)
        feature_cols = [c for c in combined.columns if c != "target"]
        X_raw = combined[feature_cols].values
        y_raw = combined["target"].values

        # Normalize features
        self._feature_means = X_raw.mean(axis=0)
        self._feature_stds = X_raw.std(axis=0)
        self._feature_stds[self._feature_stds == 0] = 1  # avoid div by zero
        X_norm = (X_raw - self._feature_means) / self._feature_stds

        # Build sequences
        sequences = []
        targets = []
        for i in range(self.window_size, len(X_norm)):
            sequences.append(X_norm[i - self.window_size : i])
            targets.append(y_raw[i])

        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)

        # Train/val split
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self._input_size = X.shape[2]
        self._model = LSTMModel(self._input_size, self.hidden_size)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)

        best_val_loss = float("inf")
        best_state = None

        self._model.train()
        for epoch in range(epochs):
            # Mini-batch training
            indices = torch.randperm(len(X_train_t))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                x_batch = X_train_t[batch_idx]
                y_batch = y_train_t[batch_idx]

                optimizer.zero_grad()
                pred = self._model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            self._model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"LSTM epoch {epoch+1}/{epochs}",
                    train_loss=round(epoch_loss / n_batches, 4),
                    val_loss=round(val_loss, 4),
                )

        # Restore best model
        if best_state:
            self._model.load_state_dict(best_state)
        self._model.eval()

        # Final metrics
        with torch.no_grad():
            val_prob = self._model(X_val_t).numpy()
            val_binary = (val_prob > 0.5).astype(int)
            accuracy = float((val_binary == y_val).mean())

        logger.info("LSTM trained", accuracy=round(accuracy, 4), sequences=len(X))
        return {"accuracy": accuracy, "train_size": split, "val_size": len(X) - split}

    def generate_signals(
        self, data: dict[str, pd.DataFrame]
    ) -> list[Signal]:
        if self._model is None:
            logger.warning("LSTM model not loaded — returning HOLD")
            return [
                Signal(symbol=s, action=Action.HOLD, reason="Model not loaded")
                for s in data
            ]

        self._model.eval()
        signals: list[Signal] = []

        for symbol, df in data.items():
            if df.empty or len(df) < 60:
                signals.append(
                    Signal(symbol=symbol, action=Action.HOLD, reason="Insufficient data")
                )
                continue

            try:
                features = build_features(df, include_target=False)
                if len(features) < self.window_size:
                    signals.append(
                        Signal(symbol=symbol, action=Action.HOLD, reason="Insufficient features")
                    )
                    continue

                # Normalize and build one sequence (last window)
                X_raw = features.values[-self.window_size:]
                X_norm = (X_raw - self._feature_means) / self._feature_stds
                X_tensor = torch.from_numpy(X_norm.astype(np.float32)).unsqueeze(0)

                with torch.no_grad():
                    prob_up = float(self._model(X_tensor).item())

                if prob_up >= self.buy_threshold:
                    sig = Signal(
                        symbol=symbol,
                        action=Action.BUY,
                        strength=prob_up,
                        reason=f"LSTM BUY: P(up)={prob_up:.3f} ≥ {self.buy_threshold}",
                        metadata={"prob_up": round(prob_up, 4)},
                    )
                elif prob_up <= self.sell_threshold:
                    sig = Signal(
                        symbol=symbol,
                        action=Action.SELL,
                        strength=1.0 - prob_up,
                        reason=f"LSTM SELL: P(up)={prob_up:.3f} ≤ {self.sell_threshold}",
                        metadata={"prob_up": round(prob_up, 4)},
                    )
                else:
                    sig = Signal(
                        symbol=symbol,
                        action=Action.HOLD,
                        strength=0.0,
                        reason=f"LSTM HOLD: P(up)={prob_up:.3f}",
                    )

                signals.append(sig)

            except Exception:
                logger.exception("LSTM prediction failed", symbol=symbol)
                signals.append(
                    Signal(symbol=symbol, action=Action.HOLD, reason="Prediction error")
                )

        return signals
