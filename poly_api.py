"""
Polymarket Gamma API client: fetch active 5-min BTC markets, resolved historicals, and streak.
"""
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import GAMMA_API_BASE

logger = logging.getLogger(__name__)

# Slug patterns for 5-min BTC up/down markets
BTC_5MIN_SLUG_PATTERN = re.compile(r"btc-up-down-5min-|btc-updown-5m-", re.I)

# Retry config
RETRIES = 3
RETRY_DELAY = 1.0


def _request(
    path: str,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """GET Gamma API with retries. Returns list (or empty list on error)."""
    url = f"{GAMMA_API_BASE}{path}"
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params or {}, timeout=15)
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, list) else [data]
        except requests.RequestException as e:
            logger.warning("Gamma API attempt %s failed: %s", attempt + 1, e)
            if attempt < RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return []


def _parse_outcomes(market: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    """Parse outcomes and outcomePrices from market (may be JSON strings)."""
    raw_outcomes = market.get("outcomes", "[]")
    raw_prices = market.get("outcomePrices", "[]")
    if isinstance(raw_outcomes, str):
        raw_outcomes = json.loads(raw_outcomes)
    if isinstance(raw_prices, str):
        raw_prices = json.loads(raw_prices)
    outcomes = list(raw_outcomes) if raw_outcomes else []
    prices = [float(p) for p in raw_prices] if raw_prices else []
    return outcomes, prices


def fetch_active_btc_5min_markets(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Poll Gamma API for active crypto markets, filter by 5-min BTC slug.
    Uses active=true&category=crypto&limit&sort=newest.
    """
    params = {
        "active": "true",
        "category": "crypto",
        "limit": max(limit, 20),
        "sort": "newest",
    }
    raw = _request("/markets", params)
    out = []
    for m in raw:
        slug = (m.get("slug") or "").strip()
        if slug and BTC_5MIN_SLUG_PATTERN.search(slug):
            out.append(m)
            if len(out) >= limit:
                break
    return out


def fetch_resolved_btc_5min_markets(limit: int = 20) -> List[Dict[str, Any]]:
    """Fetch resolved markets for historical streak (resolved=true, same slug filter)."""
    params = {
        "closed": "true",
        "limit": max(limit, 50),
    }
    raw = _request("/markets", params)
    out = []
    for m in raw:
        slug = (m.get("slug") or "").strip()
        if slug and BTC_5MIN_SLUG_PATTERN.search(slug):
            out.append(m)
            if len(out) >= limit:
                break
    return out


def get_streak(api_url: Optional[str] = None, limit: int = 10) -> Tuple[int, str]:
    """
    Compute current streak from last N resolved 5-min BTC markets.
    Returns (streak_count, last_direction) where last_direction is "UP" or "DOWN".
    Resolved outcome is inferred from outcomePrices (winner near 1.0) or resolution metadata.
    """
    # Use optional base URL override for testing
    base = (api_url or GAMMA_API_BASE).rstrip("/")
    path = "/markets"
    params = {"closed": "true", "limit": 50}
    try:
        r = requests.get(f"{base}{path}", params=params, timeout=15)
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        logger.warning("get_streak fetch failed: %s", e)
        return 0, ""

    markets = [m for m in (raw if isinstance(raw, list) else [raw]) if m.get("slug") and BTC_5MIN_SLUG_PATTERN.search((m.get("slug") or ""))]
    # Sort by endDate/closedTime descending (most recent first)
    def sort_key(m):
        return m.get("closedTime") or m.get("endDate") or ""

    markets.sort(key=sort_key, reverse=True)
    markets = markets[:limit]

    if not markets:
        return 0, ""

    # Infer resolution: outcomes often ["Up","Down"] or ["Yes","No"]; winner has price ~1
    directions = []
    for m in markets:
        outcomes, prices = _parse_outcomes(m)
        if len(prices) >= 2:
            # First outcome typically Up/Yes, second Down/No
            up_price = prices[0]
            down_price = prices[1]
            if up_price >= 0.99:
                directions.append("UP")
            elif down_price >= 0.99:
                directions.append("DOWN")
            else:
                # Unresolved or ambiguous
                directions.append("")
        else:
            directions.append("")

    # Streak = consecutive same direction at the end (most recent)
    last = ""
    streak = 0
    for d in directions:
        if not d:
            break
        if last and d != last:
            break
        last = d
        streak += 1

    return streak, last


def market_liquidity(market: Dict[str, Any]) -> float:
    """Return liquidity as float (USDC)."""
    liq = market.get("liquidityNum") or market.get("liquidity") or 0
    try:
        return float(liq)
    except (TypeError, ValueError):
        return 0.0


def market_open_price(market: Dict[str, Any]) -> Optional[float]:
    """
    Open price for 5-min window: from metadata if present, else None.
    Caller should use Chainlink price at market start when None.
    """
    # Gamma sometimes has resolutionSource or custom fields; check for open/strike
    open_p = market.get("openPrice") or market.get("strikePrice") or market.get("open_price")
    if open_p is not None:
        try:
            return float(open_p)
        except (TypeError, ValueError):
            pass
    return None


def market_reversal_side(streak_direction: str) -> str:
    """Reversal side: opposite of streak. UP streak -> DOWN reversal."""
    if streak_direction == "UP":
        return "DOWN"
    if streak_direction == "DOWN":
        return "UP"
    return ""


def reversal_outcome_index(market: Dict[str, Any], reversal_side: str) -> int:
    """Index of reversal outcome in outcomes list (0 = first, 1 = second)."""
    outcomes, _ = _parse_outcomes(market)
    reversal_side = (reversal_side or "").upper()
    for i, o in enumerate(outcomes):
        if (o or "").upper().startswith(reversal_side[:1]) or reversal_side in (o or "").upper():
            return i
    # Default: DOWN usually second
    return 1 if reversal_side == "DOWN" else 0


def poly_odds_for_reversal(market: Dict[str, Any], reversal_side: str) -> float:
    """Current Poly odds (price) for the reversal outcome (0..1)."""
    _, prices = _parse_outcomes(market)
    idx = reversal_outcome_index(market, reversal_side)
    if idx < len(prices):
        return float(prices[idx])
    return 0.5


def market_best_bid(market: Dict[str, Any], outcome_index: int) -> float:
    """Best bid for given outcome index; fallback to outcome price."""
    bid = market.get("bestBid")
    if bid is not None:
        try:
            return float(bid)
        except (TypeError, ValueError):
            pass
    _, prices = _parse_outcomes(market)
    if outcome_index < len(prices):
        return float(prices[outcome_index])
    return 0.0
