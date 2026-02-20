from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from core.types import MarketSnapshot, RoundState


@dataclass
class RoundData:
    """Container for a single round and its associated snapshots."""

    round: RoundState
    snapshots: List[MarketSnapshot]


@dataclass
class BacktestTradeRow:
    """Backtest-only row for CSV/report. Not used in live trading."""

    ts: datetime
    market_id: str
    side: str
    size: float
    entry_price: float
    prob_win_at_entry: Optional[float]
    edge_at_entry: Optional[float]
    fee: float
    slippage_cost_est: Optional[float]
    pnl_net: float
    outcome: str  # "won" | "lost" | "early_tp" | "early_sl"
    tau_seconds: Optional[float]
    spread_at_entry: Optional[float] = None  # best_ask - best_bid when available
    exit_price: Optional[float] = None
    exit_ts: Optional[datetime] = None

