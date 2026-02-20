from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from core.interfaces import BacktestDataFeed
from core.types import BacktestMetrics, Fill, OutcomeSide, RoundState, StrategySignal, OrderBook, OrderBookLevel, MarketSnapshot
from backtest.types_bt import BacktestTradeRow


StrategyFn = Callable[[RoundState, MarketSnapshot], Optional[StrategySignal]]


SLIPPAGE_BPS = 10.0  # 0.10% price slippage against us
FEE_BPS = 10.0  # 0.10% fee on notional per trade
MAX_SHARES = 200.0


@dataclass
class SkipStats:
    """Aggregated skip counts for a backtest run (price cap, EV, min_payout, etc.)."""

    skip_price_cap: int = 0
    skip_price_floor: int = 0
    skip_ev: int = 0
    skip_payout: int = 0
    skip_no_ask: int = 0
    skip_no_bid: int = 0
    skip_spread: int = 0

    def add(
        self,
        skip_price_cap: int = 0,
        skip_price_floor: int = 0,
        skip_ev: int = 0,
        skip_payout: int = 0,
        skip_no_ask: int = 0,
        skip_no_bid: int = 0,
        skip_spread: int = 0,
    ) -> None:
        self.skip_price_cap += skip_price_cap
        self.skip_price_floor += skip_price_floor
        self.skip_ev += skip_ev
        self.skip_payout += skip_payout
        self.skip_no_ask += skip_no_ask
        self.skip_no_bid += skip_no_bid
        self.skip_spread += skip_spread


def calc_realized_pnl(size_usdc: float, entry_price: float, won: bool, fee_bps: float) -> float:
    """
    Simple binary market payout model with constant fee in bps of notional.
    """
    fee = size_usdc * fee_bps / 10_000.0
    if won:
        pnl = size_usdc * (1.0 / entry_price - 1.0) - fee
    else:
        pnl = -size_usdc - fee
    return pnl


def _entry_price_from_signal(
    signal: StrategySignal,
    snapshot: MarketSnapshot,
) -> Optional[float]:
    if signal.poly_odds_reversal is not None:
        return float(signal.poly_odds_reversal)
    if signal.desired_outcome == OutcomeSide.UP:
        return snapshot.outcome_up_price
    if signal.desired_outcome == OutcomeSide.DOWN:
        return snapshot.outcome_down_price
    return None


def _apply_slippage(price: float, side_buy: bool = True) -> float:
    adj = SLIPPAGE_BPS / 10_000.0
    return price * (1.0 + adj if side_buy else 1.0 - adj)


def _fee_from_notional(notional: float, fee_bps: float = FEE_BPS) -> float:
    return abs(notional) * (fee_bps / 10_000.0)


def fill_from_orderbook(
    size_usdc: float,
    side: OutcomeSide,
    snap: MarketSnapshot,
) -> Optional[tuple[float, float, float]]:
    """
    Simulate an aggressive fill against the orderbook.

    Assumes orderbook sizes are in outcome tokens (shares). We convert USDC
    notional into shares at each price level until we run out of size or book.

    Returns (filled_usdc, avg_price, filled_shares) or None if no book.
    """
    key = "UP" if side == OutcomeSide.UP else "DOWN"
    ob: Optional[OrderBook] = snap.orderbooks.get(key)
    if ob is None or not ob.asks:
        return None

    remaining = size_usdc
    total_spend = 0.0
    total_shares = 0.0

    for level in ob.asks:
        if remaining <= 0:
            break
        level_notional = level.size * level.price
        spend = min(remaining, level_notional)
        shares = spend / level.price if level.price > 0 else 0.0
        total_spend += spend
        total_shares += shares
        remaining -= spend

    if total_shares <= 0 or total_spend <= 0:
        return None

    avg_price = total_spend / total_shares
    filled_usdc = total_spend
    return filled_usdc, avg_price, total_shares


def run_round(
    round_state: RoundState,
    snapshots: Iterable[MarketSnapshot],
    strategy_fn: StrategyFn,
    *,
    spread_bps: float = 0.0,
    gas_usd: float = 0.0,
    min_ev: float = 0.0,
    min_entry_price: float = 0.0,
    max_entry_price: float = 1.0,
    min_payout: float = 0.0,
    max_spread: Optional[float] = 0.10,
    execution_mode: str = "orderbook",
    max_position_usdc: Optional[float] = None,
    edge_ref: float = 0.05,
    min_size_usdc: float = 0.0,
    take_profit_usdc: Optional[float] = None,
    stop_loss_usdc: Optional[float] = None,
    fee_bps: float = FEE_BPS,
    fill_debug_log: Optional[list] = None,
    fill_debug_cap: int = 0,
    fill_console_log_count: Optional[list] = None,
    verbose_log: Optional[Callable[[str], None]] = None,
    skip_stats: Optional[SkipStats] = None,
    skip_cap_debug_count: Optional[list] = None,
) -> Tuple[List[Fill], List[BacktestTradeRow]]:
    """
    Simulate a single round. Returns list of fills.
    Gate: entry_price > max_entry_price -> skip (increment skip_stats.skip_price_cap), do not create Fill.
    """
    fills: List[Fill] = []
    rows: List[BacktestTradeRow] = []
    opened = False
    skipped_price = 0
    skipped_floor = 0
    skipped_ev = 0
    snaps_list = list(sorted(snapshots, key=lambda s: s.ts))

    for i, snap in enumerate(snaps_list):
        if opened:
            continue
        signal = strategy_fn(round_state, snap)
        if signal is None or signal.suggested_size_usdc <= 0:
            continue
        price = _entry_price_from_signal(signal, snap)
        if price is None or price <= 0 or price >= 1 or price < 0.05 or price > 0.95:
            continue

        key_ob = "UP" if signal.desired_outcome == OutcomeSide.UP else "DOWN"
        use_orderbook = execution_mode == "orderbook"
        ob_fill = fill_from_orderbook(signal.suggested_size_usdc, signal.desired_outcome, snap) if use_orderbook else None
        if use_orderbook and ob_fill is None:
            if skip_stats is not None:
                skip_stats.add(skip_no_ask=1)
            continue
        ob = snap.orderbooks.get(key_ob) if snap.orderbooks else None
        best_bid = ob.bids[0].price if ob and ob.bids else None
        best_ask = ob.asks[0].price if ob and ob.asks else None
        if ob_fill is not None:
            size_usdc, entry_price, _ = ob_fill
            if max_spread is not None:
                if best_ask is None:
                    if skip_stats is not None:
                        skip_stats.add(skip_no_ask=1)
                    continue
                if best_bid is None:
                    if skip_stats is not None:
                        skip_stats.add(skip_no_bid=1)
                    continue
                spread_abs = best_ask - best_bid
                if spread_abs > max_spread:
                    if skip_stats is not None:
                        skip_stats.add(skip_spread=1)
                    continue
        else:
            entry_price = _apply_slippage(price, side_buy=True)
            if spread_bps > 0:
                half = (spread_bps / 2.0) / 10_000.0
                entry_price = entry_price * (1.0 + half)
            size_usdc = signal.suggested_size_usdc

        if entry_price > max_entry_price:
            skipped_price += 1
            if skip_cap_debug_count is not None and skip_cap_debug_count[0] < 30:
                key_ob = "UP" if signal.desired_outcome == OutcomeSide.UP else "DOWN"
                ob = snap.orderbooks.get(key_ob) if snap.orderbooks else None
                best_bid = ob.bids[0].price if ob and ob.bids else None
                best_ask = ob.asks[0].price if ob and ob.asks else None
                msg = (
                    f"[SKIP CAP] market_id={round_state.market_id} side={signal.desired_outcome.name} "
                    f"entry_price={entry_price} best_bid={best_bid} best_ask={best_ask} "
                    f"outcome_up_price={snap.outcome_up_price} outcome_down_price={snap.outcome_down_price}"
                )
                (verbose_log or print)(msg)
                skip_cap_debug_count[0] += 1
            continue
        if min_entry_price > 0 and entry_price < min_entry_price:
            skipped_floor += 1
            if skip_stats is not None:
                skip_stats.add(skip_price_floor=1)
            continue

        if min_payout > 0:
            payout = (1.0 / entry_price) - 1.0
            if payout < min_payout:
                if skip_stats is not None:
                    skip_stats.add(skip_payout=1)
                continue

        edge_vs_execution = signal.prob_win - entry_price
        if edge_ref > 0:
            scale = max(0.0, min(1.0, edge_vs_execution / edge_ref))
            size_usdc = signal.suggested_size_usdc * scale
        else:
            size_usdc = signal.suggested_size_usdc
        if min_size_usdc > 0 and size_usdc < min_size_usdc:
            continue

        ev_per_usdc = (signal.prob_win / entry_price) - 1.0
        ev_after_costs = ev_per_usdc - (fee_bps / 10_000.0)
        if size_usdc > 0 and gas_usd > 0:
            ev_after_costs -= gas_usd / size_usdc
        if min_ev > 0 and ev_after_costs < min_ev:
            skipped_ev += 1
            continue

        shares = size_usdc / entry_price if entry_price > 0 else 0.0
        notes = None
        if shares > MAX_SHARES:
            size_usdc = MAX_SHARES * entry_price
            notes = "size_clamped"
        if max_position_usdc is not None and size_usdc > max_position_usdc:
            size_usdc = max_position_usdc
            if notes is None:
                notes = "position_capped"

        notional = size_usdc
        fee = _fee_from_notional(notional, fee_bps)

        if round_state.winner is not None:
            won = signal.desired_outcome == round_state.winner
            pnl = calc_realized_pnl(size_usdc, entry_price, won, fee_bps)
        else:
            pnl = -fee
        pnl -= gas_usd

        tau_seconds = (round_state.end_time - snap.ts).total_seconds()
        implied_at_execution = entry_price

        if entry_price > max_entry_price:
            raise RuntimeError(
                f"Cap violation: entry_price={entry_price} > max_entry_price={max_entry_price} "
                f"market_id={round_state.market_id}"
            )

        if fill_debug_log is not None and fill_debug_cap > 0 and len(fill_debug_log) < fill_debug_cap:
            fill_debug_log.append({
                "market_id": round_state.market_id,
                "side": signal.desired_outcome.name,
                "tau": round(tau_seconds, 1),
                "entry_price": round(entry_price, 4),
                "implied_prob": round(implied_at_execution, 4),
                "prob_model": round(signal.prob_win, 4),
                "edge": round(edge_vs_execution, 4),
                "ev_after_costs": round(ev_after_costs, 6),
            })

        mid_price = (best_bid + best_ask) / 2.0 if (best_bid is not None and best_ask is not None) else None
        slippage_usdc = (shares * max(0.0, entry_price - mid_price)) if (mid_price is not None and shares > 0) else None
        spread_at_entry = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None
        if fill_console_log_count is not None and fill_console_log_count[0] < 5:
            msg = "[FILL DEBUG] " + str({
                "entry_price": entry_price,
                "entry_price_saved_in_fill": entry_price,
                "max_entry_price": max_entry_price,
                "skipped_due_to_price": False,
                "prob_win": round(signal.prob_win, 4),
                "implied_prob": round(implied_at_execution, 4),
                "edge": round(edge_vs_execution, 4),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "market_id": round_state.market_id,
                "side": signal.desired_outcome.name,
                "tau_s": round(tau_seconds, 1),
            })
            (verbose_log or print)(msg)
            fill_console_log_count[0] += 1

        won = round_state.winner is not None and signal.desired_outcome == round_state.winner
        exit_ts_val: Optional[datetime] = None
        exit_price_val: Optional[float] = None
        outcome_val = "won" if won else "lost"
        pnl_final = pnl

        if (take_profit_usdc is not None or stop_loss_usdc is not None) and round_state.winner is None and entry_price > 0:
            for later_snap in snaps_list[i + 1 :]:
                ob_later = later_snap.orderbooks.get(key_ob) if later_snap.orderbooks else None
                best_bid_later = ob_later.bids[0].price if ob_later and ob_later.bids else None
                if best_bid_later is None:
                    continue
                unrealized = size_usdc * (best_bid_later / entry_price - 1.0)
                if take_profit_usdc is not None and unrealized >= take_profit_usdc:
                    pnl_final = size_usdc * (best_bid_later / entry_price - 1.0) - fee - gas_usd
                    exit_ts_val = later_snap.ts
                    exit_price_val = best_bid_later
                    outcome_val = "early_tp"
                    break
                if stop_loss_usdc is not None and unrealized <= -stop_loss_usdc:
                    pnl_final = size_usdc * (best_bid_later / entry_price - 1.0) - fee - gas_usd
                    exit_ts_val = later_snap.ts
                    exit_price_val = best_bid_later
                    outcome_val = "early_sl"
                    break

        fills.append(
            Fill(
                trade=None,
                filled_size=size_usdc,
                avg_price=entry_price,
                fee_paid=fee,
                realized_pnl=pnl_final,
                notes=notes,
                edge_at_entry=edge_vs_execution,
                prob_win_at_entry=signal.prob_win,
                exit_price=exit_price_val,
                exit_ts=exit_ts_val,
            )
        )
        rows.append(
            BacktestTradeRow(
                ts=snap.ts,
                market_id=round_state.market_id,
                side=signal.desired_outcome.name,
                size=size_usdc,
                entry_price=entry_price,
                prob_win_at_entry=signal.prob_win,
                edge_at_entry=edge_vs_execution,
                fee=fee,
                slippage_cost_est=slippage_usdc,
                pnl_net=pnl_final,
                outcome=outcome_val,
                tau_seconds=tau_seconds,
                spread_at_entry=spread_at_entry,
                exit_price=exit_price_val,
                exit_ts=exit_ts_val,
            )
        )
        opened = True

    if skip_stats is not None:
        skip_stats.add(skip_price_cap=skipped_price, skip_price_floor=skipped_floor, skip_ev=skipped_ev)
    return (fills, rows)


def run_backtest(
    rounds: Sequence[RoundState],
    data_feed: BacktestDataFeed,
    strategy_fn: StrategyFn,
    *,
   spread_bps: float = 0.0,              # keep 0 if orderbook exec is used; this is for mark-mode
gas_usd: float = 0.0,                 # CLOB trading usually 0; only add if you truly pay per-trade gas
min_ev: float = 0.002,                # 0.2% per $1 notional after costs gate (tune 0.001–0.005)
min_entry_price: float = 0.0,
max_entry_price: float = 0.65,        # avoid low-payout “sure thing” buys
min_payout: float = 0.25,             # payout >= 25% => entry_price <= 0.80 (redundant w cap but useful)
max_spread: Optional[float] = 0.03,   # 3% max top-of-book spread; skip illiquid moments
execution_mode: str = "orderbook",    # always
    max_position_usdc: Optional[float] = None,
    edge_ref: float = 0.05,
    min_size_usdc: float = 0.0,
    max_loss_per_day: Optional[float] = None,   # stop trading if you’re down $25 in a day
    max_loss_streak_k: int = 0,
    max_loss_streak_cooldown_n: int = 0,
    take_profit_usdc: Optional[float] = None,
    stop_loss_usdc: Optional[float] = None,
    fee_bps: float = FEE_BPS,
    debug_fills: int = 0,
    strict_cap: bool = False,
    no_risk_stats: bool = False,
    verbose_log: Optional[Callable[[str], None]] = None,
) -> BacktestMetrics:
    all_fills: List[Fill] = []
    all_rows: List[BacktestTradeRow] = []
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    fill_debug_log: List[dict] = [] if debug_fills > 0 else None
    fill_console_log_count: List[int] = [0]
    skip_stats = SkipStats()
    skip_cap_debug_count: List[int] = [0]
    cooldown_remaining = 0
    consecutive_losses = 0
    daily_pnl: dict = {}
    day_stopped = set()

    def _no_trade(_r: RoundState, _s: MarketSnapshot) -> None:
        return None

    for r in rounds:
        if cooldown_remaining > 0:
            strat = _no_trade
            cooldown_remaining -= 1
        elif r.start_time.date() in day_stopped:
            strat = _no_trade
        else:
            strat = strategy_fn

        snaps = list(data_feed.snapshots_for_round(r))
        fills, round_rows = run_round(
            r,
            snaps,
            strat,
            spread_bps=spread_bps,
            gas_usd=gas_usd,
            min_ev=min_ev,
            min_entry_price=min_entry_price,
            max_entry_price=max_entry_price,
            min_payout=min_payout,
            max_spread=max_spread,
            execution_mode=execution_mode,
            max_position_usdc=max_position_usdc,
            edge_ref=edge_ref,
            min_size_usdc=min_size_usdc,
            take_profit_usdc=take_profit_usdc,
            stop_loss_usdc=stop_loss_usdc,
            fee_bps=fee_bps,
            fill_debug_log=fill_debug_log,
            fill_debug_cap=debug_fills if debug_fills > 0 else 0,
            fill_console_log_count=fill_console_log_count,
            verbose_log=verbose_log,
            skip_stats=skip_stats,
            skip_cap_debug_count=skip_cap_debug_count,
        )
        for f, row in zip(fills, round_rows):
            pnl = f.realized_pnl or 0.0
            equity += pnl
            peak = max(peak, equity)
            max_dd = min(max_dd, equity - peak)
            day = r.start_time.date()
            daily_pnl[day] = daily_pnl.get(day, 0.0) + pnl
            if row.outcome in ("lost", "early_sl"):
                consecutive_losses += 1
            else:
                consecutive_losses = 0
            if max_loss_streak_k > 0 and consecutive_losses >= max_loss_streak_k:
                cooldown_remaining = max_loss_streak_cooldown_n
        if max_loss_per_day is not None and r.start_time.date() not in day_stopped:
            if daily_pnl.get(r.start_time.date(), 0) <= -max_loss_per_day:
                day_stopped.add(r.start_time.date())
        all_fills.extend(fills)
        all_rows.extend(round_rows)

    total_pnl = sum(f.realized_pnl or 0.0 for f in all_fills)
    trades = len(all_fills)
    avg_pnl = total_pnl / trades if trades else 0.0
    wins = sum(1 for f in all_fills if (f.realized_pnl or 0.0) > 0)
    win_rate = wins / trades if trades else 0.0

    debug_msg = f"[DEBUG] skip_price_cap={skip_stats.skip_price_cap} skip_price_floor={skip_stats.skip_price_floor} skip_ev={skip_stats.skip_ev} skip_payout={skip_stats.skip_payout} skip_no_ask={skip_stats.skip_no_ask} skip_no_bid={skip_stats.skip_no_bid} skip_spread={skip_stats.skip_spread}"
    (verbose_log or print)(debug_msg)

    if strict_cap:
        for f in all_fills:
            if f.avg_price > max_entry_price:
                raise RuntimeError(
                    f"[STRICT] Fill violates cap: avg_price={f.avg_price} > max_entry_price={max_entry_price}"
                )

    if fill_debug_log:
        lines = ["First-fill diagnostics (entry_price vs implied_prob should align for same side):"]
        for i, row in enumerate(fill_debug_log, 1):
            lines.append(
                f"  [{i}] market_id={row['market_id']} side={row['side']} tau={row['tau']}s "
                f"entry_price={row['entry_price']} implied_prob={row['implied_prob']} "
                f"prob_model={row['prob_model']} edge={row['edge']} ev_after_costs={row['ev_after_costs']}"
            )
        (verbose_log or print)("\n".join(lines))

    risk_stats: Optional[dict] = None
    if not no_risk_stats:
        # Avoid max()/min() on empty: no winning trades -> best_win=0; no losses -> worst_loss=0
        wins = [float(f.realized_pnl or 0.0) for f in all_fills if float(f.realized_pnl or 0.0) > 0]
        best_win = max(wins) if wins else 0.0
        losses = [float(f.realized_pnl or 0.0) for f in all_fills if float(f.realized_pnl or 0.0) <= 0]
        worst_loss = min(losses) if losses else 0.0
        loss_streak = 0
        win_streak = 0
        longest_loss_streak = 0
        longest_win_streak = 0
        for f in all_fills:
            pnl = f.realized_pnl or 0.0
            if pnl <= 0:
                loss_streak += 1
                win_streak = 0
                longest_loss_streak = max(longest_loss_streak, loss_streak)
            else:
                win_streak += 1
                loss_streak = 0
                longest_win_streak = max(longest_win_streak, win_streak)
        risk_stats = {
            "worst_loss_trade": worst_loss,
            "best_win_trade": best_win,
            "longest_loss_streak": longest_loss_streak,
            "longest_win_streak": longest_win_streak,
        }
    skip_stats_dict = {
        "skip_price_cap": skip_stats.skip_price_cap,
        "skip_price_floor": skip_stats.skip_price_floor,
        "skip_ev": skip_stats.skip_ev,
        "skip_payout": skip_stats.skip_payout,
        "skip_no_ask": skip_stats.skip_no_ask,
        "skip_no_bid": skip_stats.skip_no_bid,
        "skip_spread": skip_stats.skip_spread,
    }

    return BacktestMetrics(
        trades=all_fills,
        total_pnl=total_pnl,
        avg_pnl_per_trade=avg_pnl,
        win_rate=win_rate,
        max_drawdown=max_dd,
        risk_stats=risk_stats,
        trade_rows=all_rows,
        skip_stats=skip_stats_dict,
    )

