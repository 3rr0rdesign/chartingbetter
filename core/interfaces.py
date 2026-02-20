from __future__ import annotations

"""
Abstract interfaces for core services:

1) TruthFeed        – BTC "truth" price (Data Streams, PolyBackTest, etc.)
2) PolymarketFeed   – live Polymarket markets, odds, orderbooks
3) BacktestDataFeed – historical rounds + snapshots for simulation
4) Executor         – order placement / risk‑aware execution

These are pure interfaces (no logic) so we can:
- plug in real HTTP/WebSocket implementations
- plug in offline replay / PolyBackTest implementations
without changing strategy or higher‑level orchestration code.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple

from .types import (
    Fill,
    MarketSnapshot,
    OrderBook,
    OutcomeSide,
    RoundState,
    StrategySignal,
    Trade,
    TruthPrice,
)


class TruthFeed(ABC):
    """
    Source of the BTC "truth" price used for decision‑making and resolution
    simulation.

    Implementations might include:
    - Chainlink Data Streams client
    - PolyBackTest BTC reference API
    - Fallback exchange price (Binance, etc.)
    """

    @abstractmethod
    def latest(self) -> Optional[TruthPrice]:
        """Return the most recent truth price, or None if unavailable."""

    @abstractmethod
    def price_at(self, ts: datetime) -> Optional[TruthPrice]:
        """
        Return the best available truth price at or immediately before `ts`
        (for backtests or start‑of‑round locking).
        """


class PolymarketFeed(ABC):
    """
    Live Polymarket data source: markets, odds, orderbooks, streak context.
    """

    @abstractmethod
    def active_btc_rounds(self, limit: int = 10) -> List[RoundState]:
        """Return currently active BTC up/down rounds (e.g. 5‑min windows)."""

    @abstractmethod
    def snapshot(self, market_id: str) -> MarketSnapshot:
        """
        Return a lightweight MarketSnapshot for the given market at "now",
        including current outcome prices and (optionally) BTC price / orderbooks.
        """

    @abstractmethod
    def orderbook(self, market_id: str, outcome: OutcomeSide) -> Optional[OrderBook]:
        """
        Return an OrderBook for the requested outcome in the given market,
        or None if detailed depth is unavailable.
        """

    @abstractmethod
    def streak_context(self, market_id: str, lookback: int = 10) -> Tuple[int, Optional[OutcomeSide]]:
        """
        Return (streak_len, streak_direction) for recent resolved BTC rounds
        related to the given market (e.g. last N outcomes in that series).
        """


class BacktestDataFeed(ABC):
    """
    Historical data provider for backtesting and simulation.

    Implementations will typically wrap PolyBackTest APIs.
    """

    @abstractmethod
    def historical_rounds(
        self,
        market_type: str = "5m",
        limit: Optional[int] = None,
    ) -> Sequence[RoundState]:
        """Return a sequence of historical BTC rounds (resolved), newest or oldest first."""

    @abstractmethod
    def snapshots_for_round(self, round_state: RoundState) -> Iterable[MarketSnapshot]:
        """
        Yield MarketSnapshot objects spanning [start_time, end_time] for the
        specified round, including BTC price and Polymarket odds.
        """


class Executor(ABC):
    """
    Order execution and risk‑aware placement.

    Strategy produces StrategySignal objects; the Executor turns them into
    concrete Trades, applies sizing and risk checks, and sends them.
    """

    @abstractmethod
    def submit_from_signal(self, signal: StrategySignal) -> Optional[Fill]:
        """
        Turn a StrategySignal into one or more orders and return a Fill
        representing the executed size. May return None if the signal is
        rejected by risk checks or not filled.
        """

    @abstractmethod
    def submit_trade(self, trade: Trade) -> Optional[Fill]:
        """
        Submit a specific Trade (used for backtests or low‑level control).
        Returns a Fill or None if rejected.
        """
