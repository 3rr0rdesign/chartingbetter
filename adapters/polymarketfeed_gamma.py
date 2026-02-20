from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple

from core.interfaces import PolymarketFeed
from core.types import MarketSnapshot, OrderBook, OrderBookLevel, OutcomeSide, RoundState
from poly_api import (
    _parse_outcomes,
    fetch_active_btc_5min_markets,
    get_streak,
    market_liquidity,
)


def _market_id(m: dict) -> str:
    return str(m.get("id") or m.get("_id") or m.get("marketId") or m.get("slug") or "")


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        # Gamma times are usually ISO8601; treat missing tz as UTC.
        if value.endswith("Z"):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


class GammaPolymarketFeed(PolymarketFeed):
    """
    PolymarketFeed implementation backed by the existing Gamma API helpers
    in poly_api.py. This is intentionally conservative and only exposes the
    fields we already use elsewhere.
    """

    def active_btc_rounds(self, limit: int = 10) -> List[RoundState]:
        markets = fetch_active_btc_5min_markets(limit=limit)
        rounds: List[RoundState] = []
        for m in markets:
            mid = _market_id(m)
            slug = (m.get("slug") or "").strip() or None
            start_raw = m.get("startDate") or m.get("startTime")
            end_raw = m.get("endDate") or m.get("endTime") or m.get("closedTime")
            start_time = _parse_time(start_raw) or datetime.now(timezone.utc)
            end_time = _parse_time(end_raw) or (start_time + (end_time - start_time if end_raw else (5 * 60)))
            # We don't have BTC reference prices here; leave None.
            round_state = RoundState(
                market_id=mid,
                slug=slug,
                start_time=start_time,
                end_time=end_time if isinstance(end_time, datetime) else start_time,
                btc_price_start=None,
                btc_price_end=None,
                winner=None,
                resolved_at=None,
                streak_len=None,
                streak_direction=None,
            )
            rounds.append(round_state)
        return rounds

    def snapshot(self, market_id: str) -> MarketSnapshot:
        # Re-fetch a small batch of active markets and find the one we care about.
        markets = fetch_active_btc_5min_markets(limit=20)
        market = None
        for m in markets:
            if _market_id(m) == market_id:
                market = m
                break

        now = datetime.now(timezone.utc)
        if market is None:
            return MarketSnapshot(ts=now, market_id=market_id)

        outcomes, prices = _parse_outcomes(market)
        up_price = prices[0] if len(prices) > 0 else None
        down_price = prices[1] if len(prices) > 1 else None

        # For now we don't expose full depth; leave orderbooks empty.
        return MarketSnapshot(
            ts=now,
            market_id=_market_id(market),
            outcome_up_price=up_price,
            outcome_down_price=down_price,
            btc_price=None,
            orderbooks={},
        )

    def orderbook(self, market_id: str, outcome: OutcomeSide) -> Optional[OrderBook]:
        """
        Gamma helpers don't expose full depth; as a minimal implementation we
        return None here. Future work can plug in CLOB orderbook endpoints.
        """
        return None

    def streak_context(self, market_id: str, lookback: int = 10) -> Tuple[int, Optional[OutcomeSide]]:
        streak_len, direction = get_streak(limit=lookback)
        out_dir: Optional[OutcomeSide]
        if direction == "UP":
            out_dir = OutcomeSide.UP
        elif direction == "DOWN":
            out_dir = OutcomeSide.DOWN
        else:
            out_dir = None
        return streak_len, out_dir

