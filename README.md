# AI Paper Trading Bot

A modular, testable, event-loop-based paper trading bot using the Alpaca API.

## Architecture

```
src/
├── config/         # Settings & env parsing
├── data/           # Market data fetching (Alpaca)
├── strategies/     # Signal generation (MA crossover, RSI, etc.)
├── core/           # Trading engine, portfolio, risk, Alpaca wrapper
├── notifications/  # Trade alerts (console, Telegram)
└── utils/          # Logging, time helpers
scripts/            # Entry points (live_paper, reset, backtest)
tests/              # Unit & integration tests
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in your API keys
cp .env.example .env

# 3. Run one iteration (smoke test)
python scripts/live_paper.py --once

# 4. Run continuously
python scripts/live_paper.py

# 5. Reset paper account (close all positions & orders)
python scripts/reset_paper.py
```

## Configuration

All settings are in `.env`. Key parameters:

| Variable | Description | Default |
|----------|-------------|---------|
| `WATCHLIST` | Comma-separated symbols | `AAPL,MSFT,...` |
| `BAR_TIMEFRAME` | Bar interval | `5Min` |
| `MAX_POSITION_PCT` | Max single position (% equity) | `0.05` |
| `MAX_TOTAL_EXPOSURE_PCT` | Max total exposure | `0.80` |
| `MAX_DRAWDOWN_PCT` | Drawdown halt threshold | `0.10` |

## Testing

```bash
python -m pytest tests/ -v
```

## Roadmap

- **Phase 1** ✅ Rule-based strategies (MA crossover, RSI)
- **Phase 2** 🔜 ML models (XGBoost, LSTM)
- **Phase 3** 🔜 LLM agent, WebSocket streaming, dashboard
