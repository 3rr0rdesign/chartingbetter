from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from dotenv import load_dotenv

from backtest.engine import FEE_BPS, SLIPPAGE_BPS, run_backtest
from backtest.polybacktest_client import PolyBackTestClient
from backtest.types_bt import BacktestTradeRow
from core.interfaces import BacktestDataFeed
from core.types import Fill, MarketSnapshot, OrderBook, OrderBookLevel, RoundState
from strategy.v1_fair_edge import FairEdgeState, strategy_fn as v1_strategy


def _snapshot_to_dict(s: MarketSnapshot) -> dict[str, Any]:
    ob_dict = {}
    for k, ob in (s.orderbooks or {}).items():
        ob_dict[k] = {
            "bids": [{"price": L.price, "size": L.size} for L in ob.bids],
            "asks": [{"price": L.price, "size": L.size} for L in ob.asks],
        }
    return {
        "ts": s.ts.isoformat(),
        "market_id": s.market_id,
        "outcome_up_price": s.outcome_up_price,
        "outcome_down_price": s.outcome_down_price,
        "btc_price": s.btc_price,
        "orderbooks": ob_dict,
    }


def _snapshots_from_dict(data: list[dict]) -> list[MarketSnapshot]:
    out = []
    for d in data:
        ts = datetime.fromisoformat(d["ts"].replace("Z", "+00:00")) if d.get("ts") else datetime.now(timezone.utc)
        orderbooks = {}
        for k, ob in (d.get("orderbooks") or {}).items():
            orderbooks[k] = OrderBook(
                bids=[OrderBookLevel(price=x["price"], size=x["size"]) for x in ob.get("bids", [])],
                asks=[OrderBookLevel(price=x["price"], size=x["size"]) for x in ob.get("asks", [])],
            )
        out.append(
            MarketSnapshot(
                ts=ts,
                market_id=d.get("market_id", ""),
                outcome_up_price=d.get("outcome_up_price"),
                outcome_down_price=d.get("outcome_down_price"),
                btc_price=d.get("btc_price"),
                orderbooks=orderbooks,
            )
        )
    return out


def _cache_path(cache_dir: str, market_id: str, step_seconds: int, include_orderbook: bool) -> str:
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in market_id)
    ob = "ob" if include_orderbook else "noob"
    return os.path.join(cache_dir, f"{safe_id}_s{step_seconds}_{ob}.json.gz")


def _load_snapshots_from_cache(path: str) -> Optional[list[MarketSnapshot]]:
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        return _snapshots_from_dict(data)
    except Exception:
        return None


def _save_snapshots_to_cache(path: str, snapshots: list[MarketSnapshot]) -> None:
    tmp = path + ".tmp"
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump([_snapshot_to_dict(s) for s in snapshots], f)
        if os.path.exists(path):
            os.remove(path)
        os.rename(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
        raise


def _append_trades_csv(fp, rows: list[BacktestTradeRow], split: str) -> None:
    w = csv.writer(fp)
    for row in rows:
        ts = row.ts.isoformat() if row.ts else ""
        exit_ts_str = row.exit_ts.isoformat() if row.exit_ts else ""
        w.writerow([
            split,
            ts,
            row.market_id,
            row.side,
            row.size,
            row.entry_price,
            row.prob_win_at_entry,
            row.edge_at_entry,
            row.fee,
            row.slippage_cost_est if row.slippage_cost_est is not None else "",
            row.pnl_net,
            row.outcome,
            row.tau_seconds if row.tau_seconds is not None else "",
            row.spread_at_entry if row.spread_at_entry is not None else "",
            row.exit_price if row.exit_price is not None else "",
            exit_ts_str,
        ])


def _bucket_report(rows: list[BacktestTradeRow], label: str) -> dict:
    def _max_dd(rr: list[BacktestTradeRow]) -> float:
        if not rr:
            return 0.0
        sorted_r = sorted(rr, key=lambda x: x.ts)
        eq = 0.0
        peak = 0.0
        dd = 0.0
        for r in sorted_r:
            eq += r.pnl_net
            peak = max(peak, eq)
            dd = min(dd, eq - peak)
        return float(dd)

    def _bucket_stats(rr: list[BacktestTradeRow]) -> dict:
        if not rr:
            return {"trades": 0, "win_rate": 0.0, "mean_pnl": 0.0, "median_pnl": 0.0, "total_pnl": 0.0, "max_dd": 0.0}
        n = len(rr)
        pnls = [float(r.pnl_net) for r in rr]
        wins = sum(1 for p in pnls if p > 0)
        total = sum(pnls)
        pnls_sorted = sorted(pnls)
        median_pnl = pnls_sorted[n // 2] if n % 2 else (pnls_sorted[n // 2 - 1] + pnls_sorted[n // 2]) / 2.0
        return {
            "trades": n,
            "win_rate": float(wins / n),
            "mean_pnl": float(total / n),
            "median_pnl": float(median_pnl),
            "total_pnl": float(total),
            "max_dd": float(_max_dd(rr)),
        }

    entry_buckets = [(0.0, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.01)]
    by_entry = []
    for lo, hi in entry_buckets:
        b = [r for r in rows if lo <= r.entry_price < hi]
        by_entry.append({"bucket": f"{lo:.2f}-{hi:.2f}", **_bucket_stats(b)})
    tau_buckets = [(0, 60), (60, 120), (120, 180), (180, 99999)]
    by_tau = []
    for lo, hi in tau_buckets:
        b = [r for r in rows if r.tau_seconds is not None and lo <= r.tau_seconds < hi]
        by_tau.append({"bucket": f"{lo}-{hi}s", **_bucket_stats(b)})
    spread_buckets = [(0.0, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 1.0)]
    by_spread = []
    for lo, hi in spread_buckets:
        b = [r for r in rows if r.spread_at_entry is not None and lo <= r.spread_at_entry < hi]
        by_spread.append({"bucket": f"{lo:.2f}-{hi:.2f}", **_bucket_stats(b)})
    entry_with_trades = [b for b in by_entry if b["trades"] > 0]
    best_entry_bin = max(entry_with_trades, key=lambda x: x["total_pnl"]) if entry_with_trades else None
    all_buckets = by_entry + by_tau + by_spread
    by_total_pnl = sorted([b for b in all_buckets if b["trades"] > 0], key=lambda x: x["total_pnl"], reverse=True)
    top5 = by_total_pnl[:5]
    worst5 = by_total_pnl[-5:] if len(by_total_pnl) >= 5 else by_total_pnl
    return {
        "label": label,
        "by_entry_price": by_entry,
        "by_tau_seconds": by_tau,
        "by_spread": by_spread,
        "best_entry_bin": best_entry_bin,
        "top5_by_total_pnl": top5,
        "worst5_by_total_pnl": list(reversed(worst5)),
    }


def _spread_cap_recommendation(rows: list[BacktestTradeRow], label: str) -> None:
    """Compute total_pnl and max_dd by spread cap; print best max_spread candidate."""
    candidates = [0.01, 0.015, 0.02, 0.025, 0.03]
    with_spread = [r for r in rows if r.spread_at_entry is not None]
    if not with_spread:
        print(f"{label} spread cap recommendation: no spread_at_entry data")
        return
    sorted_r = sorted(with_spread, key=lambda x: x.ts)
    results = []
    for cap in candidates:
        subset = [r for r in sorted_r if r.spread_at_entry < cap]
        if not subset:
            results.append({"max_spread": cap, "trades": 0, "total_pnl": 0.0, "max_dd": 0.0})
            continue
        total_pnl = sum(r.pnl_net for r in subset)
        eq, peak, dd = 0.0, 0.0, 0.0
        for r in subset:
            eq += r.pnl_net
            peak = max(peak, eq)
            dd = min(dd, eq - peak)
        results.append({"max_spread": cap, "trades": len(subset), "total_pnl": total_pnl, "max_dd": dd})
    best_by_pnl = max(results, key=lambda x: (x["total_pnl"], -abs(x["max_dd"])))
    print(f"{label} spread cap candidates (total_pnl, max_dd):")
    for r in results:
        print(f"  max_spread={r['max_spread']:.3f}: trades={r['trades']} total_pnl={r['total_pnl']:.2f} max_dd={r['max_dd']:.2f}")
    print(f"{label} recommendation: --max-spread {best_by_pnl['max_spread']:.3f} (total_pnl={best_by_pnl['total_pnl']:.2f} max_dd={best_by_pnl['max_dd']:.2f})")


def _print_bucket_report(report: dict, label: str) -> None:
    def _row_fmt(row: dict) -> str:
        return (
            f"  {row['bucket']}: n={row['trades']} win_rate={row['win_rate']:.3f} "
            f"mean_pnl={row.get('mean_pnl', row.get('avg_pnl', 0)):.2f} median_pnl={row.get('median_pnl', 0):.2f} "
            f"total_pnl={row['total_pnl']:.2f} max_dd={row['max_dd']:.2f}"
        )
    print(f"{label} bucket report (entry_price):")
    for row in report["by_entry_price"]:
        if row["trades"] > 0:
            print(_row_fmt(row))
    best_entry = report.get("best_entry_bin")
    if best_entry:
        print(f"{label} recommendation: best entry_price bin by total_pnl: {best_entry['bucket']} (total_pnl={best_entry['total_pnl']:.2f} n={best_entry['trades']})")
    print(f"{label} bucket report (tau_seconds):")
    for row in report["by_tau_seconds"]:
        if row["trades"] > 0:
            print(_row_fmt(row))
    print(f"{label} bucket report (spread_at_entry):")
    for row in report["by_spread"]:
        if row["trades"] > 0:
            print(_row_fmt(row))
    print(f"{label} top 5 buckets by total_pnl:")
    for row in report.get("top5_by_total_pnl", []):
        print(_row_fmt(row))
    print(f"{label} worst 5 buckets by total_pnl:")
    for row in report.get("worst5_by_total_pnl", []):
        print(_row_fmt(row))


class PolyBackTestDataFeed(BacktestDataFeed):
    """Adapter to expose PolyBackTestClient as a BacktestDataFeed."""

    def __init__(
        self,
        client: Optional[PolyBackTestClient] = None,
        step_seconds: int = 5,
    ) -> None:
        self.client = client or PolyBackTestClient()
        self.step_seconds = step_seconds

    def historical_rounds(
        self,
        market_type: str = "5m",
        limit: Optional[int] = None,
    ):
        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=7)).date().isoformat()
        end = now.date().isoformat()
        rounds = self.client.list_rounds(date_from=start, date_to=end, market_type=market_type)
        if limit is not None:
            return rounds[:limit]
        return rounds

    def snapshots_for_round(self, round_state: RoundState, include_orderbook: bool = False):
        snaps = list(self.client.iter_snapshots(
            round_state.market_id,
            step_seconds=self.step_seconds,
            include_orderbook=include_orderbook,
        ))
        if round_state.btc_price_start is None and snaps:
            round_state.btc_price_start = snaps[0].btc_price
            print(f"[WARN] Inferred btc_price_start for round {round_state.market_id} from first snapshot.")
        if round_state.end_time <= round_state.start_time:
            round_state.end_time = round_state.start_time + timedelta(minutes=5)
            print(f"[WARN] Inferred end_time for round {round_state.market_id} as start+5m.")
        return snaps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v1_fair_edge backtest over last 7 days.")
    parser.add_argument("--market_type", type=str, default="5m")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument("--min-edge", type=float, default=0.02)
    parser.add_argument("--base-size", type=float, default=5.0)
    parser.add_argument("--fee-bps", type=float, default=FEE_BPS)
    parser.add_argument("--slippage-bps", type=float, default=SLIPPAGE_BPS)
    parser.add_argument("--tau-min", type=float, default=30.0)
    parser.add_argument("--tau-max", type=float, default=180.0)
    parser.add_argument("--min-z", type=float, default=0.25)
    parser.add_argument("--limit-rounds", type=int, default=None)
    parser.add_argument("--include-orderbook", action="store_true", default=False)
    parser.add_argument("--split", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-seconds", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--spread-bps", type=float, default=0.0)
    parser.add_argument("--gas-usd", type=float, default=0.0)
    parser.add_argument("--min-ev", type=float, default=0.0)
    parser.add_argument("--min-entry-price", type=float, default=0.0, help="Skip if entry_price < this (e.g. 0.55 to avoid 0.45-0.55 bucket)")
    parser.add_argument("--max-entry-price", type=float, default=1.0)
    parser.add_argument("--min-payout", type=float, default=0.0, help="Minimum payout ratio (1/entry_price - 1); e.g. 0.25 = 25%% min payoff")
    parser.add_argument("--max-spread", type=float, default=0.10, help="Skip trade if best_ask - best_bid > this (default 0.10); use 0 to disable")
    parser.add_argument("--execution-mode", type=str, choices=("mark", "orderbook"), default="orderbook", help="mark = use snapshot price + spread/slippage; orderbook = use book (or fallback to mark)")
    parser.add_argument("--debug-fills", type=int, default=0)
    parser.add_argument("--strict", action="store_true", help="Raise if any fill violates --max-entry-price")
    parser.add_argument("--log-file", type=str, default=None, help="Write full debug (skip counts, fill debug) to this file; stdout stays concise")
    parser.add_argument("--dump-trades", type=str, default=None, metavar="FILE", help="Export trades to CSV (ts, market_id, side, size, entry_price, ...)")
    parser.add_argument("--report-json", type=str, default=None, metavar="FILE", help="Write bucket report + risk stats to JSON")
    parser.add_argument("--max-loss-per-day", type=float, default=None, help="Stop trading for the day when net PnL <= -X")
    parser.add_argument("--max-loss-streak-k", type=int, default=0, help="After K consecutive losses, enter cooldown")
    parser.add_argument("--max-loss-streak-cooldown", type=int, default=0, help="Skip N rounds after K consecutive losses")
    parser.add_argument("--max-position-usdc", type=float, default=None, help="Hard cap per-trade size in USDC")
    parser.add_argument("--edge-ref", type=float, default=0.05, help="Reference edge for sizing: size = base_size * clamp(edge/edge_ref, 0, 1)")
    parser.add_argument("--min-size", type=float, default=0.0, help="Skip trade if sized USDC < this")
    parser.add_argument("--take-profit-usdc", type=float, default=None, help="Backtest: close when unrealized PnL >= this (sell at best_bid)")
    parser.add_argument("--stop-loss-usdc", type=float, default=None, help="Backtest: close when unrealized PnL <= -this (sell at best_bid)")
    parser.add_argument("--preset", type=str, choices=("fast", "full"), default="full", help="fast: limit_rounds=300, step_seconds=20, max_workers=1; full: defaults")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache snapshots per market_id+step_seconds+orderbook (gzip JSON); temp then rename")
    parser.add_argument("--no-risk-stats", action="store_true", help="Skip computing risk_stats (worst/best trade, streaks) for faster debug runs")
    args = parser.parse_args()

    if args.preset == "fast":
        args.limit_rounds = 300
        args.step_seconds = 20
        args.max_workers = 1
        print("[PRESET] fast: limit_rounds=300 step_seconds=20 max_workers=1")
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    print("[ARGS]", {
        "min_entry_price": args.min_entry_price,
        "max_entry_price": args.max_entry_price,
        "min_ev": args.min_ev,
        "min_payout": args.min_payout,
        "max_spread": args.max_spread,
        "execution_mode": args.execution_mode,
        "include_orderbook": args.include_orderbook,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "spread_bps": args.spread_bps,
        "tau_min": args.tau_min,
        "tau_max": args.tau_max,
    })

    load_dotenv()
    client = PolyBackTestClient()
    feed = PolyBackTestDataFeed(client, step_seconds=args.step_seconds)

    # Determine date window
    if args.date_from and args.date_to:
        date_from_str = args.date_from
        date_to_str = args.date_to
    else:
        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=args.days)).date()
        end = now.date()
        date_from_str = start.isoformat()
        date_to_str = end.isoformat()

    rounds = client.list_rounds(
        date_from=date_from_str,
        date_to=date_to_str,
        market_type=args.market_type,
        max_rounds=args.limit_rounds,
    )
    # Sort by start_time ascending
    rounds.sort(key=lambda r: r.start_time)
    if args.limit_rounds is not None:
        rounds = rounds[: args.limit_rounds]
    print(f"Loaded {len(rounds)} rounds between {date_from_str} and {date_to_str} (limit_rounds={args.limit_rounds})")

    if not rounds:
        print("No rounds found in the requested window.")
        return

    # Split into train/validation by time (not random)
    split_frac = float(args.split) if hasattr(args, "split") else 0.7
    n_total = len(rounds)
    n_train = int(n_total * split_frac)
    if n_train <= 0:
        n_train = max(1, n_total // 2)
    if n_train >= n_total:
        n_train = max(1, n_total - 1)
    train_rounds = rounds[:n_train]
    val_rounds = rounds[n_train:]
    print(f"Split: TRAIN={len(train_rounds)} rounds, VALID={len(val_rounds)} rounds")

    do_reports = (getattr(args, "limit_rounds", None) is None or args.limit_rounds > 5) or (args.report_json is not None)

    def run_split(label: str, subset: list[RoundState], verbose_log=None, dump_csv_file=None, report_collector=None, cache_dir=None, do_reports_=True):
        if not subset:
            print(f"{label}: no rounds")
            return
        cache_stats = {"hits": 0, "misses": 0, "fetch_time_sum": 0.0}

        state = FairEdgeState()
        debug_stats: dict[str, int] = {}

        def strat(round_state: RoundState, snap: MarketSnapshot):
            cfg = {
                "state": state,
                "min_edge": args.min_edge,
                "base_size": args.base_size,
                "tau_min": args.tau_min,
                "tau_max": args.tau_max,
                "min_z": args.min_z,
                "min_entry_price": getattr(args, "min_entry_price", None),
                "max_entry_price": getattr(args, "max_entry_price", None),
                "min_payout": getattr(args, "min_payout", None),
                "debug": True,
                "debug_stats": debug_stats,
            }
            return v1_strategy(round_state, snap, cfg)

        snapshot_cache: dict[str, list[MarketSnapshot]] = {}
        timing_cache: dict[str, float] = {}

        sem = threading.Semaphore(4 if args.include_orderbook else args.max_workers)

        def worker(round_state: RoundState):
            with sem:
                if cache_dir:
                    path = _cache_path(cache_dir, round_state.market_id, args.step_seconds, args.include_orderbook)
                    cached = _load_snapshots_from_cache(path)
                    if cached is not None:
                        cache_stats["hits"] += 1
                        if round_state.btc_price_start is None and cached:
                            round_state.btc_price_start = cached[0].btc_price
                        if round_state.end_time <= round_state.start_time and cached:
                            round_state.end_time = round_state.start_time + timedelta(minutes=5)
                        return round_state.market_id, cached, 0.0
                t0 = time.time()
                snaps = list(feed.snapshots_for_round(round_state, include_orderbook=args.include_orderbook))
                dt = time.time() - t0
                cache_stats["misses"] += 1
                cache_stats["fetch_time_sum"] += dt
                if cache_dir:
                    path = _cache_path(cache_dir, round_state.market_id, args.step_seconds, args.include_orderbook)
                    try:
                        _save_snapshots_to_cache(path, snaps)
                    except Exception as e:
                        if verbose_log:
                            verbose_log(f"[CACHE] save failed for {round_state.market_id}: {e}")
                return round_state.market_id, snaps, dt

        t_prefetch_start = time.time()
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [ex.submit(worker, r) for r in subset]
            for fut in as_completed(futures):
                mid, snaps, dt = fut.result()
                snapshot_cache[mid] = snaps
                timing_cache[mid] = dt
        t_prefetch_total = time.time() - t_prefetch_start
        total_snaps = sum(len(v) for v in snapshot_cache.values())
        avg_round = t_prefetch_total / len(subset)
        avg_snap = t_prefetch_total / max(total_snaps, 1)
        print(f"Prefetched {len(subset)} rounds in {t_prefetch_total:.2f}s (workers={args.max_workers})")
        print(
            f"Prefetch stats: {total_snaps} snapshots total, "
            f"avg {avg_round:.2f}s/round, ~{avg_snap:.3f}s/snap"
        )
        if cache_dir and (cache_stats["hits"] + cache_stats["misses"]) > 0:
            hits, misses = cache_stats["hits"], cache_stats["misses"]
            rate = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
            time_saved = hits * (cache_stats["fetch_time_sum"] / misses) if misses > 0 else 0.0
            print(f"Cache: hit_rate={rate:.1f}% (hits={hits} misses={misses}) time_saved_est={time_saved:.1f}s")

        # Progress + snapshot timing per round (using prefetched data)
        for i, r in enumerate(subset):
            snaps = snapshot_cache.get(r.market_id, [])
            dt = timing_cache.get(r.market_id, 0.0)
            print(f"[{label} {i+1}/{len(subset)}] market_id={r.market_id} start={r.start_time.isoformat()}")
            if i < 3:
                print(f"  prefetch_snapshots_time={dt:.3f}s")

        # Wrap feed to serve prefetched snapshots to the engine
        class CachedFeed(BacktestDataFeed):
            def historical_rounds(self, market_type: str = "5m", limit: Optional[int] = None):
                return subset if limit is None else subset[:limit]

            def snapshots_for_round(self, round_state: RoundState):
                snaps = snapshot_cache.get(round_state.market_id)
                if snaps is not None:
                    return snaps
                return feed.snapshots_for_round(
                    round_state,
                    include_orderbook=args.include_orderbook,
                )

        cached_feed = CachedFeed()

        metrics = run_backtest(
            subset,
            cached_feed,
            strat,
            spread_bps=args.spread_bps,
            gas_usd=args.gas_usd,
            min_ev=args.min_ev,
            min_entry_price=args.min_entry_price,
            max_entry_price=args.max_entry_price,
            min_payout=args.min_payout,
            max_spread=args.max_spread if args.max_spread > 0 else None,
            execution_mode=args.execution_mode,
            max_position_usdc=args.max_position_usdc,
            edge_ref=args.edge_ref,
            min_size_usdc=args.min_size,
            max_loss_per_day=args.max_loss_per_day,
            max_loss_streak_k=args.max_loss_streak_k,
            max_loss_streak_cooldown_n=args.max_loss_streak_cooldown,
            take_profit_usdc=args.take_profit_usdc,
            stop_loss_usdc=args.stop_loss_usdc,
            fee_bps=args.fee_bps,
            debug_fills=args.debug_fills,
            strict_cap=args.strict,
            no_risk_stats=args.no_risk_stats,
            verbose_log=verbose_log,
        )
        print(f"{label} summary:")
        print(f"  trades       : {len(metrics.trades)}")
        print(f"  win_rate     : {metrics.win_rate:.3f}")
        print(f"  total_pnl    : {metrics.total_pnl:.2f}")
        print(f"  avg_pnl/trade: {metrics.avg_pnl_per_trade:.4f}")
        print(f"  max_drawdown : {metrics.max_drawdown:.2f}")
        if metrics.risk_stats:
            print(f"  worst_loss_trade : {metrics.risk_stats.get('worst_loss_trade')}")
            print(f"  best_win_trade  : {metrics.risk_stats.get('best_win_trade')}")
            print(f"  longest_loss_streak : {metrics.risk_stats.get('longest_loss_streak')}")
            print(f"  longest_win_streak  : {metrics.risk_stats.get('longest_win_streak')}")
        if metrics.skip_stats:
            print(f"  skip_stats: price_cap={metrics.skip_stats.get('skip_price_cap')} price_floor={metrics.skip_stats.get('skip_price_floor')} ev={metrics.skip_stats.get('skip_ev')} payout={metrics.skip_stats.get('skip_payout')} no_ask={metrics.skip_stats.get('skip_no_ask')} no_bid={metrics.skip_stats.get('skip_no_bid')} spread={metrics.skip_stats.get('skip_spread')}")

        trades = metrics.trades
        if trades and args.max_entry_price < 0.99:
            violators = [f for f in trades if f.avg_price > args.max_entry_price]
            if violators:
                print(f"[ERROR] {len(violators)} fill(s) violate max_entry_price={args.max_entry_price}: e.g. avg_price={violators[0].avg_price}")
        if trades:
            wins = [f for f in trades if (f.realized_pnl or 0) > 0]
            losses = [f for f in trades if (f.realized_pnl or 0) <= 0]
            avg_entry_all = sum(f.avg_price for f in trades) / len(trades)
            avg_entry_win = sum(f.avg_price for f in wins) / len(wins) if wins else None
            avg_entry_loss = sum(f.avg_price for f in losses) / len(losses) if losses else None
            print(f"{label} entry-price diagnostics:")
            print(f"  avg_entry_price_all : {avg_entry_all:.4f}")
            if avg_entry_win is not None:
                print(f"  avg_entry_price_win : {avg_entry_win:.4f}")
            if avg_entry_loss is not None:
                print(f"  avg_entry_price_loss: {avg_entry_loss:.4f}")
            with_edge = [f for f in trades if f.edge_at_entry is not None]
            if with_edge:
                edge_win = [f for f in wins if f.edge_at_entry is not None]
                edge_loss = [f for f in losses if f.edge_at_entry is not None]
                avg_edge_all = sum(f.edge_at_entry for f in with_edge) / len(with_edge)
                avg_edge_win = sum(f.edge_at_entry for f in edge_win) / len(edge_win) if edge_win else None
                avg_edge_loss = sum(f.edge_at_entry for f in edge_loss) / len(edge_loss) if edge_loss else None
                print(f"  avg_edge_all : {avg_edge_all:.4f}")
                if avg_edge_win is not None:
                    print(f"  avg_edge_win : {avg_edge_win:.4f}")
                if avg_edge_loss is not None:
                    print(f"  avg_edge_loss: {avg_edge_loss:.4f}")

        if debug_stats:
            print(f"{label} debug stats (reason -> count):")
            for key, count in sorted(debug_stats.items(), key=lambda kv: kv[1], reverse=True):
                print(f"  {key}: {count}")

        trade_rows = metrics.trade_rows if metrics.trade_rows is not None else []
        if dump_csv_file is not None and trade_rows:
            _append_trades_csv(dump_csv_file, trade_rows, label)
        if trade_rows and do_reports_:
            report = _bucket_report(trade_rows, label)
            _print_bucket_report(report, label)
            _spread_cap_recommendation(trade_rows, label)
            if report_collector is not None:
                report_collector[label] = report

    log_file_handle = None
    if args.log_file:
        log_file_handle = open(args.log_file, "w", encoding="utf-8")
        def _verbose_log(msg: str) -> None:
            log_file_handle.write(msg + "\n")
            log_file_handle.flush()
        verbose_log = _verbose_log
    else:
        verbose_log = None
    dump_csv_file = None
    if args.dump_trades:
        dump_csv_file = open(args.dump_trades, "w", newline="", encoding="utf-8")
        dump_csv_file.write("split,ts,market_id,side,size,entry_price,prob_win_at_entry,edge_at_entry,fee,slippage_cost_est,pnl_net,outcome,tau_seconds,spread_at_entry,exit_price,exit_ts\n")
    report_collector = {}
    try:
        run_split("TRAIN", train_rounds, verbose_log=verbose_log, dump_csv_file=dump_csv_file, report_collector=report_collector, cache_dir=args.cache_dir, do_reports_=do_reports)
        run_split("VALID", val_rounds, verbose_log=verbose_log, dump_csv_file=dump_csv_file, report_collector=report_collector, cache_dir=args.cache_dir, do_reports_=do_reports)
    finally:
        if log_file_handle is not None:
            log_file_handle.close()
        if dump_csv_file is not None:
            dump_csv_file.close()
        if args.report_json and report_collector:
            with open(args.report_json, "w", encoding="utf-8") as f:
                json.dump(report_collector, f, indent=2)


if __name__ == "__main__":
    main()

