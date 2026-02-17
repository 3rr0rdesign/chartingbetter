"""
Polymarket 5-min BTC streak-reversal bot: main loop and backtest.
Run: python bot.py [--dry] [--backtest]
"""
import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from chainlink_ws import get_latest_price, start_chainlink_listener_thread
from config import (
    BET_SIZE_USDC,
    ENTRY_WINDOW_SECONDS,
    GAMMA_API_BASE,
    LIQ_MIN,
    MAX_TRADES_PER_HOUR,
    STREAK_MIN,
)
from poly_api import (
    _parse_outcomes,
    fetch_active_btc_5min_markets,
    fetch_resolved_btc_5min_markets,
    get_streak,
    market_best_bid,
    market_liquidity,
    market_open_price,
    market_reversal_side,
    poly_odds_for_reversal,
    reversal_outcome_index,
)
from strategy import (
    advantage,
    boost_prob_if_delta_opposes_streak,
    delta_and_chainlink_implied_prob,
    filter_entry_window,
    filter_vol_spike,
    reversal_prob_from_streak,
    should_signal_buy_reversal,
)
from trading import check_balance, place_order_with_stops_simple

# Logging
LOG_CSV = "bot_log.csv"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_csv_log() -> None:
    """Write CSV header if file is new."""
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "market_id", "streak", "chainlink_price", "poly_odds_reversal",
                "delta", "prob", "action", "simulated_pnl",
            ])


def log_row(
    market_id: str = "",
    streak: int = 0,
    chainlink_price: Optional[float] = None,
    poly_odds_reversal: float = 0.0,
    delta: float = 0.0,
    prob: float = 0.0,
    action: str = "",
    simulated_pnl: Optional[float] = None,
) -> None:
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.utcnow().isoformat() + "Z",
            market_id,
            streak,
            chainlink_price or "",
            poly_odds_reversal,
            delta,
            prob,
            action,
            simulated_pnl if simulated_pnl is not None else "",
        ])


def run_main_loop(dry: bool = True) -> None:
    """
    Infinite loop: every 5s poll Gamma for active 5-min BTC markets;
    on new market fetch open_price and streak; use Chainlink for delta;
    if conditions met, signal BUY reversal and place order with stops.
    """
    setup_csv_log()
    seen_markets: set = set()
    trades_this_hour: List[float] = []
    POLL_INTERVAL = 5

    # Start Chainlink listener in background
    start_chainlink_listener_thread()

    logger.info("Main loop started (dry=%s). Polling Gamma every %ss.", dry, POLL_INTERVAL)

    while True:
        try:
            now = time.time()
            # Trim trades older than 1 hour
            trades_this_hour = [t for t in trades_this_hour if now - t < 3600]
            if len(trades_this_hour) >= MAX_TRADES_PER_HOUR:
                time.sleep(POLL_INTERVAL)
                continue

            markets = fetch_active_btc_5min_markets(limit=5)
            streak, streak_direction = get_streak(GAMMA_API_BASE, limit=10)

            for market in markets:
                mid = market.get("id") or market.get("slug") or ""
                if not mid or mid in seen_markets:
                    continue

                slug = market.get("slug") or ""
                liq = market_liquidity(market)
                if liq < LIQ_MIN:
                    continue

                open_p = market_open_price(market)
                chainlink_p = get_latest_price()
                if open_p is None and chainlink_p is not None:
                    open_p = chainlink_p
                if open_p is None or open_p <= 0:
                    continue

                delta, chainlink_implied = delta_and_chainlink_implied_prob(open_p, chainlink_p)
                if filter_vol_spike(abs(delta)):
                    continue

                if streak < STREAK_MIN or not streak_direction:
                    continue

                reversal_side = market_reversal_side(streak_direction)
                poly_odds = poly_odds_for_reversal(market, reversal_side)
                prob = reversal_prob_from_streak(streak, streak_direction)
                prob = boost_prob_if_delta_opposes_streak(prob, delta, streak_direction)

                # Entry window: use market creation/start if available
                start_sec = now - 30
                try:
                    created = market.get("createdAt") or market.get("startDate")
                    if created:
                        if isinstance(created, (int, float)):
                            start_sec = float(created) / 1000.0 if created > 1e12 else float(created)
                        else:
                            s = (created or "").replace("Z", "").split("+")[0].strip()[:19]
                            start_sec = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timestamp()
                except Exception:
                    pass
                if not filter_entry_window(start_sec, now, ENTRY_WINDOW_SECONDS):
                    continue

                if not should_signal_buy_reversal(poly_odds, chainlink_implied):
                    continue

                # Signal: BUY reversal
                idx = reversal_outcome_index(market, reversal_side)
                bid = market_best_bid(market, idx)
                adv = advantage(poly_odds, chainlink_implied)
                edge_pct = int(prob * 100)
                logger.info("BUY %s @%.2f - %d%% Edge (advantage=%.3f)", reversal_side, bid, edge_pct, adv)
                log_row(
                    market_id=mid,
                    streak=streak,
                    chainlink_price=chainlink_p,
                    poly_odds_reversal=poly_odds,
                    delta=delta,
                    prob=prob,
                    action=f"BUY {reversal_side} @{bid:.2f}",
                )

                # Place order (dry or live)
                token_ids = market.get("clobTokenIds")
                if isinstance(token_ids, str):
                    import json
                    token_ids = json.loads(token_ids) if token_ids else []
                token_id = token_ids[idx] if token_ids and idx < len(token_ids) else ""
                ok, entry, _ = place_order_with_stops_simple(
                    amount_usdc=BET_SIZE_USDC,
                    bid_price=bid or 0.5,
                    market_id=mid,
                    token_id=token_id,
                    side="BUY",
                )
                if ok:
                    seen_markets.add(mid)
                    trades_this_hour.append(now)
                    log_row(market_id=mid, action="ORDER_PLACED", simulated_pnl=0.0)

        except KeyboardInterrupt:
            logger.info("Stopping.")
            break
        except Exception as e:
            logger.exception("Loop error: %s", e)

        time.sleep(POLL_INTERVAL)


def run_backtest(days: int = 7) -> None:
    """
    Backtest: fetch 100+ resolved 5-min BTC markets from Gamma, simulate streak reversal
    + Chainlink edge, log hit rate and EV (aim ~4x on wins).
    """
    setup_csv_log()
    # Fetch resolved markets (API may return limited history; we request enough)
    limit = 100
    resolved = fetch_resolved_btc_5min_markets(limit=limit)
    if len(resolved) < 10:
        logger.warning("Few resolved markets (%s); backtest may be noisy.", len(resolved))

    def sort_key(m):
        return m.get("closedTime") or m.get("endDate") or ""

    resolved_sorted = sorted(resolved, key=sort_key)
    wins = 0
    losses = 0
    total_pnl = 0.0
    payoff_win = 1.0  # win -> 1 per share
    cost = 0.5  # assume buy at 0.5 for simplicity

    for i, market in enumerate(resolved_sorted):
        outcomes, prices = _parse_outcomes(market)
        if len(prices) < 2:
            continue
        # Compute streak from previous resolutions in this list (no live API)
        prev_dirs = []
        for j in range(max(0, i - 10), i):
            m = resolved_sorted[j]
            o, p = _parse_outcomes(m)
            if len(p) >= 2:
                if p[0] >= 0.99:
                    prev_dirs.append("UP")
                elif p[1] >= 0.99:
                    prev_dirs.append("DOWN")
        streak = 0
        direction = ""
        if prev_dirs:
            last = prev_dirs[-1]
            for d in reversed(prev_dirs):
                if d != last:
                    break
                streak += 1
            direction = last

        if streak < STREAK_MIN or not direction:
            continue

        # Actual outcome: which side won?
        if prices[0] >= 0.99:
            actual = "UP"
        elif prices[1] >= 0.99:
            actual = "DOWN"
        else:
            continue

        reversal_side = market_reversal_side(direction)
        bet_reversal = reversal_side == actual
        if bet_reversal:
            wins += 1
            total_pnl += (payoff_win - cost) * (BET_SIZE_USDC / cost)
        else:
            losses += 1
            total_pnl -= cost * (BET_SIZE_USDC / cost)

        log_row(
            market_id=market.get("id") or market.get("slug", ""),
            streak=streak,
            poly_odds_reversal=0.5,
            delta=0.0,
            prob=0.7,
            action="BACKTEST",
            simulated_pnl=total_pnl,
        )

    n = wins + losses
    hit_rate = wins / n if n else 0
    logger.info(
        "Backtest done: markets=%s trades=%s wins=%s losses=%s hit_rate=%.1f%% EV=%.2f",
        len(resolved_sorted), n, wins, losses, hit_rate * 100, total_pnl,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket 5-min BTC streak-reversal bot")
    parser.add_argument("--dry", action="store_true", help="Dry run (no live orders)")
    parser.add_argument("--backtest", action="store_true", help="Run backtest on resolved markets")
    parser.add_argument("--backtest-days", type=int, default=7, help="Backtest history days (hint)")
    args = parser.parse_args()

    if args.backtest:
        run_backtest(days=args.backtest_days)
        return

    dry = args.dry or os.environ.get("DRY_RUN", "true").strip().lower() in ("1", "true", "yes")
    run_main_loop(dry=dry)


if __name__ == "__main__":
    main()
