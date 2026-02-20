from __future__ import annotations

"""
Core domain types for Polymarket BTC 5‑min trading and backtesting.

These are deliberately logic‑free data containers that can be shared across:
1) live data sources (truth feeds, Polymarket APIs)
2) strategy modules (pure signal generation)
3) execution modules (order placement / risk)
4) backtesting / simulation
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Sequence


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OutcomeSide(str, Enum):
    """Polymarket outcomes on BTC up/down markets."""

    UP = "UP"
    DOWN = "DOWN"


@dataclass
class OrderBookLevel:
    """Single price level on one side of the orderbook."""

    price: float
    size: float  # notional size in outcome tokens or USDC-equivalent


@dataclass
class OrderBook:
    """
    L2 orderbook snapshot for a single outcome.

    For Polymarket BTC up/down we usually care about the outcome we might trade
    (e.g. UP or DOWN) but this structure is generic.
    """

    bids: List[OrderBookLevel] = field(default_factory=list)  # descending price
    asks: List[OrderBookLevel] = field(default_factory=list)  # ascending price


@dataclass
class MarketSnapshot:
    """
    Point‑in‑time view of a BTC 5‑min market.

    Used both for live monitoring (dashboard / bot) and for historical
    backtesting (PolyBackTest snapshots).
    """

    ts: datetime
    market_id: str
    outcome_up_price: Optional[float] = None  # 0–1
    outcome_down_price: Optional[float] = None  # 0–1
    btc_price: Optional[float] = None  # reference BTC/USD price at this ts
    # Optional full orderbooks per outcome (keyed by outcome symbol, e.g. "UP"/"DOWN")
    orderbooks: Dict[str, OrderBook] = field(default_factory=dict)


@dataclass
class RoundState:
    """
    High‑level state for a single BTC up/down round (e.g. 5‑min window).
    """

    market_id: str
    slug: Optional[str]
    start_time: datetime
    end_time: datetime
    # BTC reference prices used by Polymarket / PolyBackTest, if known
    btc_price_start: Optional[float] = None
    btc_price_end: Optional[float] = None
    # Resolution info (filled for resolved rounds)
    winner: Optional[OutcomeSide] = None
    resolved_at: Optional[datetime] = None
    # Optional streak info / context
    streak_len: Optional[int] = None
    streak_direction: Optional[OutcomeSide] = None


@dataclass
class Trade:
    """
    An order we intend to place or have placed on Polymarket.
    """

    ts: datetime
    market_id: str
    outcome: OutcomeSide
    side: Side
    price: float  # limit price in [0,1]
    size: float  # size in outcome tokens or USDC-equivalent (depending on executor convention)
    client_order_id: Optional[str] = None
    # For backtests we might tag scenario/strategy ID
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Fill:
    """
    Execution result for a Trade, live or simulated.
    """

    trade: Optional[Trade]
    filled_size: float
    avg_price: float
    fee_paid: float = 0.0  # USDC
    # Optional realized PnL for backtest exits; live PnL is tracked separately
    realized_pnl: Optional[float] = None
    # Optional diagnostics (e.g. size clamp)
    notes: Optional[str] = None
    # Strategy edge / prob at entry (EV and entry-price stats; used by live and backtest)
    edge_at_entry: Optional[float] = None
    prob_win_at_entry: Optional[float] = None
    # Backtest-only: early exit from TP/SL
    exit_price: Optional[float] = None
    exit_ts: Optional[datetime] = None


@dataclass
class StrategySignal:
    """
    Output of a strategy for a given round and snapshot.

    This is intentionally high‑level: the strategy does not place orders;
    it recommends a direction, approximate size, and metadata about the edge.
    """

    ts: datetime
    market_id: str
    desired_outcome: OutcomeSide
    # Probability that this outcome resolves as winner, per the model
    prob_win: float
    # Edge vs. Polymarket implied probability (prob_win − implied), in absolute terms
    edge: float
    # Suggested notional size in USDC (the executor can further clamp/scale)
    suggested_size_usdc: float
    # Raw odds at the time of signal (for logging / backtest)
    poly_odds_reversal: Optional[float] = None
    btc_delta_from_start: Optional[float] = None
    notes: Optional[str] = None
    # For backtesting: which parameter set / strategy variant produced this
    strategy_id: Optional[str] = None


@dataclass
class TruthPrice:
    """
    Result from a truth price feed (Data Streams, PolyBackTest, etc.).
    """

    ts: datetime
    btc_price: float
    # Optional raw fields from upstream (e.g. Data Streams benchmark price, validity window)
    source: str = "unknown"
    valid_from: Optional[datetime] = None
    expires_at: Optional[datetime] = None


@dataclass
class BacktestMetrics:
    """
    Aggregate metrics for a backtest run.
    """

    trades: Sequence[Fill]
    total_pnl: float
    avg_pnl_per_trade: float
    win_rate: float
    max_drawdown: float
    risk_stats: Optional[dict] = None  # worst_loss_trade, best_win_trade, longest_loss_streak, longest_win_streak
    trade_rows: Optional[Sequence] = None  # BacktestTradeRow for CSV/report; only set in backtest
    skip_stats: Optional[dict] = None  # skip_price_cap, skip_price_floor, skip_ev, skip_payout, skip_no_ask, skip_no_bid, skip_spread

