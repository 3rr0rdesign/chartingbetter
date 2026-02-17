"""
Load and validate bot config from environment.
Never log PRIVATE_KEY or any key material.
"""
import os
from decimal import Decimal
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: Optional[str] = None) -> str:
    v = os.environ.get(key, default)
    if v is None:
        raise ValueError(f"Missing required env: {key}")
    return v.strip()


def _get_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _get_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, str(default)).strip().lower()
    return v in ("1", "true", "yes", "on")


# Required for live trading (optional in backtest/dry)
def get_private_key() -> Optional[str]:
    return os.environ.get("PRIVATE_KEY", "").strip() or None


def get_rpc_url() -> str:
    return _get("RPC_URL", "https://polygon-mainnet.g.alchemy.com/v2/demo")


# Strategy
BET_SIZE_USDC: float = _get_float("BET_SIZE", 10.0)
STREAK_MIN: int = _get_int("STREAK_MIN", 3)
REVERSAL_PROB_BASE: float = _get_float("REVERSAL_PROB_BASE", 0.70)
ADVANTAGE_THRESHOLD: float = _get_float("ADVANTAGE_THRESHOLD", 0.05)
PROFIT_TGT: float = _get_float("PROFIT_TGT", 0.96)
STOP_LOSS: float = _get_float("STOP_LOSS", 0.60)
LIQ_MIN: float = _get_float("LIQ_MIN", 10000.0)

# Safety
DRY_RUN: bool = _get_bool("DRY_RUN", True)
BACKTEST_DAYS: int = _get_int("BACKTEST_DAYS", 7)
MAX_TRADES_PER_HOUR: int = _get_int("MAX_TRADES_PER_HOUR", 5)
MIN_BALANCE_USDC: float = _get_float("MIN_BALANCE_USDC", 50.0)
MAX_GAS_GWEI: int = _get_int("MAX_GAS_GWEI", 50)

# APIs
GAMMA_API_BASE: str = os.environ.get("GAMMA_API_BASE", "https://gamma-api.polymarket.com")
CHAINLINK_WS_URL: str = os.environ.get(
    "CHAINLINK_WS_URL",
    "wss://ws.data.chain.link/mainnet/crypto-usd/btc-usd",
)

# Entry window: only trade in first N seconds of market
ENTRY_WINDOW_SECONDS: int = 60
# Vol spike: skip if |delta| > this (2%)
VOL_SPIKE_DELTA: float = 0.02
