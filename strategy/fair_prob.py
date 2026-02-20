from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from datetime import datetime

from core.types import OrderBook


def estimate_sigma(
    prices: List[float],
    times: Optional[Sequence[datetime]] = None,
) -> float:
    """
    Estimate per‑second volatility sigma from a price series using log returns.

    If `times` are provided, scale each log return by sqrt(dt) where dt is in
    seconds, so that variance scales linearly with time. If `times` is None,
    assume dt=1s between points (original behavior).
    """
    if len(prices) < 2:
        return 0.0

    use_times = times is not None and len(times) == len(prices)
    rets = []
    for i, (p0, p1) in enumerate(zip(prices[:-1], prices[1:], strict=False), start=1):
        if p0 <= 0 or p1 <= 0:
            continue
        r = math.log(p1 / p0)
        if use_times:
            dt = (times[i] - times[i - 1]).total_seconds()
            if dt <= 0:
                continue
            r = r / math.sqrt(dt)
        rets.append(r)
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(max(var, 0.0))


def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_finish_above_start(
    start_price: float,
    current_price: float,
    sigma: float,
    tau_seconds: int,
) -> float:
    """
    Under a drift‑less lognormal model:

    log(Final / Current) ~ Normal(0, sigma^2 * tau)

    We want P(Final > Start) = P(log(Final/Current) > log(Start/Current)).
    """
    if start_price <= 0 or current_price <= 0:
        return 0.5
    if tau_seconds <= 0 or sigma <= 0:
        # If no time/vol left, outcome is essentially locked at current.
        if current_price > start_price:
            return 1.0
        if current_price < start_price:
            return 0.0
        return 0.5

    tau = float(tau_seconds)
    threshold = math.log(start_price / current_price)
    denom = sigma * math.sqrt(tau)
    if denom <= 0:
        return 0.5
    z = threshold / denom
    p = 1.0 - _norm_cdf(z)
    return max(0.0, min(1.0, p))


def implied_prob_from_book(
    book_up: Optional[OrderBook],
    book_down: Optional[OrderBook],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Derive implied UP/DOWN probabilities from shallow orderbooks.

    For each outcome:
      mid = (best_bid + best_ask)/2  if both exist
      mid = best_bid or best_ask     if only one side exists
    Returns (implied_up, implied_down) in [0,1] or None if unavailable.
    """

    def mid_from_book(book: Optional[OrderBook]) -> Optional[float]:
        if book is None:
            return None
        bid = book.bids[0].price if book.bids else None
        ask = book.asks[0].price if book.asks else None
        if bid is not None and ask is not None:
            return 0.5 * (bid + ask)
        return bid if bid is not None else ask

    implied_up = mid_from_book(book_up)
    implied_down = mid_from_book(book_down)
    return implied_up, implied_down

