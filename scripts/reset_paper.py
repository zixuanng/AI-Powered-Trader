"""
Reset the paper trading account — close all positions and cancel all orders.

Usage:
    python scripts/reset_paper.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import Settings
from src.core.alpaca_wrapper import AlpacaTradingWrapper
from src.utils.log_config import setup_logging, get_logger


def main() -> None:
    setup_logging(level="INFO")
    logger = get_logger("reset_paper")

    settings = Settings()
    settings.validate()

    wrapper = AlpacaTradingWrapper(settings)

    logger.info("Cancelling all open orders...")
    cancelled = wrapper.cancel_all_orders()
    logger.info(f"Cancelled {cancelled} orders")

    logger.info("Closing all positions...")
    closed = wrapper.close_all_positions()
    logger.info(f"Closed {closed} positions")

    acct = wrapper.get_account()
    logger.info("Account state after reset", **acct)
    print("\n✅ Paper account reset complete.")
    print(f"   Equity:  ${acct['equity']:,.2f}")
    print(f"   Cash:    ${acct['cash']:,.2f}")
    print(f"   Status:  {acct['status']}")


if __name__ == "__main__":
    main()
