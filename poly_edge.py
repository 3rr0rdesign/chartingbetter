"""
Manual edge viewer: Chainlink (or spot) vs Polymarket 5-min BTC odds.
No wallet/keys—view only. Polls every 10s and suggests BET UP/DOWN when
delta vs Poly implied prob gives an edge (e.g. >5%). Run: python poly_edge.py
Optional: streamlit run poly_edge.py for browser dashboard.
"""
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

# -----------------------------------------------------------------------------
# Public APIs — no keys
# -----------------------------------------------------------------------------
# Chainlink-style REST (if available); else CoinGecko as spot proxy
CHAINLINK_STYLE_URLS = [
    "https://data.chain.link/eth/mainnet/crypto-usd/btc-usd",
    "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
]
POLY_API = "https://gamma-api.polymarket.com/markets"
BTC_5MIN_SLUG = re.compile(r"btc-up-down-5min-|btc-updown-5m-", re.I)

POLL_INTERVAL = 10
EDGE_THRESHOLD_PCT = 5.0   # Suggest bet when edge > 5%
DELTA_THRESHOLD = 0.005   # Min |delta| to suggest (0.5%)
LAST_MIN_FOCUS = 60       # Seconds before market end to emphasize signals


def get_btc_price() -> Optional[float]:
    """Fetch live BTC/USD from Chainlink-style or CoinGecko. Returns None on failure."""
    for url in CHAINLINK_STYLE_URLS:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue
            data = r.json()
            if "coingecko" in url:
                # {"bitcoin": {"usd": 68450}}
                return float(data.get("bitcoin", {}).get("usd", 0) or 0)
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                if isinstance(item, dict):
                    v = item.get("usd") or item.get("price") or item.get("value")
                    if v is not None:
                        return float(v)
            if isinstance(data, dict):
                v = data.get("usd") or data.get("price")
                if v is not None:
                    return float(v)
        except Exception:
            continue
    return None


def _parse_outcomes(market: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    import json
    raw_o = market.get("outcomes", "[]")
    raw_p = market.get("outcomePrices", "[]")
    if isinstance(raw_o, str):
        raw_o = json.loads(raw_o)
    if isinstance(raw_p, str):
        raw_p = json.loads(raw_p)
    outcomes = list(raw_o) if raw_o else []
    prices = [float(x) for x in raw_p] if raw_p else []
    return outcomes, prices


def get_poly_5min_market() -> Tuple[float, float, Optional[str], Optional[Dict], float]:
    """
    Fetch active 5-min BTC market from Poly Gamma API.
    Returns: (up_odds, open_price, end_time_iso, market_dict, liquidity).
    First outcome = UP/Yes. open_price from market metadata or None (caller uses current as proxy).
    """
    try:
        r = requests.get(
            POLY_API,
            params={"active": "true", "category": "crypto", "limit": 30, "sort": "newest"},
            timeout=15,
        )
        if r.status_code != 200:
            return 0.5, 0.0, None, None, 0.0
        markets = r.json()
        if not isinstance(markets, list):
            markets = [markets]
        for m in markets:
            slug = (m.get("slug") or "").strip()
            if not slug or not BTC_5MIN_SLUG.search(slug):
                continue
            outcomes, prices = _parse_outcomes(m)
            if len(prices) < 2:
                continue
            up_odds = float(prices[0])
            open_price = None
            for key in ("openPrice", "open_price", "strikePrice"):
                v = m.get(key)
                if v is not None:
                    try:
                        open_price = float(v)
                        break
                    except (TypeError, ValueError):
                        pass
            liq = m.get("liquidityNum") or m.get("liquidity") or 0
            try:
                liq = float(liq)
            except (TypeError, ValueError):
                liq = 0.0
            return up_odds, open_price or 0.0, m.get("endDate") or m.get("end_date"), m, liq
    except Exception:
        pass
    return 0.5, 0.0, None, None, 0.0


def suggest_bet(
    current_price: float,
    open_price: float,
    up_odds: float,
    end_time: Optional[str],
    edge_threshold_pct: float = EDGE_THRESHOLD_PCT,
    delta_threshold: float = DELTA_THRESHOLD,
) -> str:
    """
    Suggest BET UP / BET DOWN / HOLD from Chainlink (spot) delta vs Poly odds.
    implied_prob from delta (e.g. |delta|*20 cap 95%); edge = implied - poly_odds (as %).
    """
    if not current_price or not open_price or open_price <= 0:
        return "WAIT - No price data"

    delta = (current_price - open_price) / open_price
    # Implied prob shift: e.g. +0.5% move -> UP more likely; cap 95%
    implied_prob_up = min(0.95, max(0.05, 0.5 + delta * 20.0))  # rough: 2.5% delta -> 1.0
    implied_prob_down = 1.0 - implied_prob_up

    # Edge vs market: if we think UP is 60% but Poly has UP at 50%, edge = +10%
    edge_up_pct = (implied_prob_up - up_odds) * 100
    edge_down_pct = (implied_prob_down - (1.0 - up_odds)) * 100

    time_left_sec = 300
    if end_time:
        try:
            from datetime import timezone
            end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            time_left_sec = max(0, (end - now).total_seconds())
        except Exception:
            pass

    last_min = time_left_sec <= LAST_MIN_FOCUS

    if abs(delta) < delta_threshold:
        return "HOLD - No clear move (|delta| < 0.5%)"

    if delta > delta_threshold and up_odds < 0.90 and edge_up_pct >= edge_threshold_pct:
        return f"BET UP - Delta: +{delta:.2%} | Edge: +{edge_up_pct:.0f}% | Poly UP: {up_odds:.2f}"
    if delta < -delta_threshold and (1.0 - up_odds) < 0.90 and edge_down_pct >= edge_threshold_pct:
        return f"BET DOWN - Delta: {delta:.2%} | Edge: +{edge_down_pct:.0f}% | Poly DOWN: {1 - up_odds:.2f}"

    if last_min:
        if delta > 0:
            return f"HOLD - Slight UP (delta +{delta:.2%}) | Edge: {edge_up_pct:.0f}%"
        return f"HOLD - Slight DOWN (delta {delta:.2%}) | Edge: {edge_down_pct:.0f}%"
    return "HOLD - No clear edge"


def run_console_loop(
    poll_interval: int = POLL_INTERVAL,
    edge_threshold_pct: float = EDGE_THRESHOLD_PCT,
) -> None:
    """Run forever: print live price, Poly UP odds, open, and suggestion every poll_interval sec."""
    print("Poly Edge Scanner: Spot (Chainlink/CoinGecko) vs Poly 5-min BTC | No keys required")
    print(f"Updates every {poll_interval}s. Ctrl+C to stop.\n")
    while True:
        try:
            btc = get_btc_price()
            up_odds, open_price, end_time, market, liq = get_poly_5min_market()
            # If API didn't give open_price, use current as proxy (first tick of window)
            if not open_price and btc:
                open_price = btc
            if not open_price:
                open_price = btc or 68000.0
            suggestion = suggest_bet(btc or open_price, open_price, up_odds, end_time, edge_threshold_pct)

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"--- {ts} ---")
            print(f"BTC: ${(btc or 0):,.0f} | Poly UP: {up_odds:.2f} | Open: ${open_price:,.0f} | Liq: ${liq:,.0f}")
            print(f">> {suggestion}\n")
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}\n")
        time.sleep(poll_interval)


if __name__ == "__main__":
    run_console_loop()
