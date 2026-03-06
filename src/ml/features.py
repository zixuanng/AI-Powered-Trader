"""
Feature engineering pipeline for ML strategies.

Transforms raw OHLCV bars into a feature matrix suitable for
XGBoost / LSTM models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.rsi_strategy import compute_rsi
from src.strategies.bollinger_strategy import compute_bollinger_bands, compute_percent_b


def build_features(
    df: pd.DataFrame,
    target_horizon: int = 5,
    include_target: bool = True,
) -> pd.DataFrame:
    """Build a feature DataFrame from OHLCV data.

    Args:
        df: DataFrame with columns [open, high, low, close, volume].
        target_horizon: number of bars ahead for the target return.
        include_target: if True, append 'target' column (1=up, 0=down).

    Returns:
        DataFrame with feature columns (and optionally 'target').
        Rows with NaN features are dropped.
    """
    out = pd.DataFrame(index=df.index)

    close = df["close"]
    volume = df["volume"]
    high = df["high"]
    low = df["low"]

    # ── Price-based features ────────────────────────────────

    # SMA ratios
    for period in [5, 10, 20, 50]:
        sma = close.rolling(period).mean()
        out[f"sma_{period}_ratio"] = close / sma

    # EMA ratios
    for period in [5, 10, 20]:
        ema = close.ewm(span=period, adjust=False).mean()
        out[f"ema_{period}_ratio"] = close / ema

    # Price momentum (returns over different lookbacks)
    for lookback in [1, 3, 5, 10, 20]:
        out[f"return_{lookback}"] = close.pct_change(lookback)

    # High-Low range (normalized)
    out["hl_range"] = (high - low) / close

    # Close position within day's range
    hl_diff = high - low
    out["close_position"] = ((close - low) / hl_diff.replace(0, float("nan")))

    # ── Technical indicators ────────────────────────────────

    # RSI
    out["rsi_14"] = compute_rsi(close, period=14)

    # Bollinger %B
    _, upper, lower = compute_bollinger_bands(close, period=20, num_std=2.0)
    out["bollinger_pct_b"] = compute_percent_b(close, upper, lower)

    # MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd - macd_signal

    # ── Volume features ─────────────────────────────────────

    vol_sma_20 = volume.rolling(20).mean()
    out["volume_ratio"] = volume / vol_sma_20.replace(0, float("nan"))
    out["volume_change"] = volume.pct_change()

    # ── Volatility features ─────────────────────────────────

    out["volatility_10"] = close.pct_change().rolling(10).std()
    out["volatility_20"] = close.pct_change().rolling(20).std()

    # ATR (Average True Range) - simplified
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean() / close  # normalized

    # ── Target ──────────────────────────────────────────────

    if include_target:
        future_return = close.shift(-target_horizon) / close - 1
        out["target"] = (future_return > 0).astype(int)

    # Drop rows with NaN
    out = out.dropna()
    return out


def get_feature_columns() -> list[str]:
    """Return the list of feature column names (excluding target)."""
    # Build a dummy to discover columns
    dummy = pd.DataFrame({
        "open": range(100),
        "high": range(1, 101),
        "low": range(100),
        "close": [50 + i * 0.1 for i in range(100)],
        "volume": [1000] * 100,
    })
    features = build_features(dummy, include_target=False)
    return features.columns.tolist()
