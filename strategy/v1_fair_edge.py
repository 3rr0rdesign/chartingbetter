from __future__ import annotations

"""
v1 fair edge strategy for BTC 5‑min Polymarket rounds.

Assumptions:
- BTC follows a drift‑less lognormal over the remaining time in the round.
- We estimate per‑step sigma from recent BTC prices for this round only.
- We compare model probability of finishing ABOVE the round start price
  against the implied probabilities in Polymarket prices.

This module exposes:
  - FairEdgeState: small helper to maintain per‑round BTC price history.
  - strategy_fn(round_state, snap, cfg) -> Optional[StrategySignal]

The caller is responsible for:
- Holding a single FairEdgeState() instance across calls.
- Passing it in cfg["state"].
- Setting thresholds like min_edge, base_size, min_z, tau bounds, etc.
"""

from dataclasses import dataclass, field
from datetime import datetime
from math import log, sqrt
from typing import Dict, List, Optional

from core.types import MarketSnapshot, OutcomeSide, RoundState, StrategySignal
from strategy.fair_prob import estimate_sigma, prob_finish_above_start


@dataclass
class FairEdgeState:
    """
    Simple per‑round BTC price history tracker.

    Call update(round_id, ts, btc_price) on each snapshot; it maintains a
    sliding window of recent prices for that round and returns the current
    per‑step sigma estimate.
    """

    max_points: int = 300
    _series: Dict[str, List[float]] = field(default_factory=dict)
    _times: Dict[str, List[datetime]] = field(default_factory=dict)

    def update(self, round_id: str, ts: datetime, btc_price: float) -> float:
        if btc_price <= 0:
            return 0.0
        series = self._series.setdefault(round_id, [])
        times = self._times.setdefault(round_id, [])
        series.append(btc_price)
        times.append(ts)
        if len(series) > self.max_points:
            del series[: len(series) - self.max_points]
            del times[: len(times) - self.max_points]
        return estimate_sigma(series, times)


def strategy_fn(
    round_state: RoundState,
    snap: MarketSnapshot,
    cfg: Optional[dict] = None,
) -> Optional[StrategySignal]:
    """
    Fair‑edge strategy:
    - Uses BTC start price vs current price and sigma to get P(UP).
    - Compares with implied UP/DOWN from snapshot prices.
    - Enters only when edge and z‑score exceed thresholds and within tau window.
    """
    cfg = cfg or {}
    state: FairEdgeState = cfg.get("state")
    debug: bool = bool(cfg.get("debug", False))
    debug_stats: dict = cfg.get("debug_stats") or {}

    def bump(reason: str) -> None:
        if not debug:
            return
        debug_stats[reason] = debug_stats.get(reason, 0) + 1

    if state is None:
        bump("no_state")
        return None

    min_edge: float = float(cfg.get("min_edge", 0.02))
    base_size: float = float(cfg.get("base_size", 5.0))
    min_z: float = float(cfg.get("min_z", 0.25))
    tau_min: float = float(cfg.get("tau_min", 30.0))
    tau_max: float = float(cfg.get("tau_max", 180.0))

    # Required fields
    if round_state.btc_price_start is None:
        bump("missing_start")
        return None
    if snap.btc_price is None:
        bump("missing_btc")
        return None
    if snap.outcome_up_price is None or snap.outcome_down_price is None:
        bump("missing_odds")
        return None

    tau_seconds = (round_state.end_time - snap.ts).total_seconds()
    if tau_seconds < tau_min or tau_seconds > tau_max:
        bump("tau_outside")
        return None

    # Update sigma from per‑round state
    sigma = state.update(round_state.market_id, snap.ts, snap.btc_price)
    if sigma <= 0:
        bump("sigma_zero")
        return None

    # Model probability that final BTC > start BTC
    p_up = prob_finish_above_start(
        start_price=round_state.btc_price_start,
        current_price=snap.btc_price,
        sigma=sigma,
        tau_seconds=int(tau_seconds),
    )

    # Distance from start in sigma units (z‑score)
    try:
        z = abs(log(snap.btc_price / round_state.btc_price_start)) / (sigma * sqrt(max(tau_seconds, 1.0)))
    except (ValueError, ZeroDivisionError):
        z = 0.0

    if z < min_z:
        bump("z_low")
        return None

    implied_up = snap.outcome_up_price
    implied_down = snap.outcome_down_price
    # For diagnostics we could infer the missing side, but we already require both non‑None.

    edge_up = p_up - implied_up
    edge_down = (1.0 - p_up) - implied_down

    best_edge = max(edge_up, edge_down)
    if best_edge < min_edge:
        bump("edge_low")
        return None

    if edge_up >= edge_down:
        desired = OutcomeSide.UP
        implied_entry = implied_up
        edge = edge_up
    else:
        desired = OutcomeSide.DOWN
        implied_entry = implied_down
        edge = edge_down

    min_entry_price = cfg.get("min_entry_price")
    max_entry_price = cfg.get("max_entry_price")
    min_payout = cfg.get("min_payout")
    if min_entry_price is not None or max_entry_price is not None or min_payout is not None:
        key_ob = "UP" if desired == OutcomeSide.UP else "DOWN"
        ob = (snap.orderbooks or {}).get(key_ob)
        best_ask = ob.asks[0].price if ob and getattr(ob, "asks", None) else None
        exec_price = float(best_ask) if best_ask is not None else implied_entry
        if exec_price is None or exec_price <= 0:
            bump("no_exec_price")
            return None
        if min_entry_price is not None and min_entry_price > 0 and exec_price < min_entry_price:
            bump("exec_price_below_floor")
            return None
        if max_entry_price is not None and exec_price > max_entry_price:
            bump("exec_price_over_cap")
            return None
        if min_payout is not None and min_payout > 0:
            payout = (1.0 / exec_price) - 1.0
            if payout < min_payout:
                bump("payout_below_min")
                return None

    btc_delta = (snap.btc_price - round_state.btc_price_start) / round_state.btc_price_start

    return StrategySignal(
        ts=snap.ts,
        market_id=round_state.market_id,
        desired_outcome=desired,
        prob_win=p_up if desired == OutcomeSide.UP else (1.0 - p_up),
        edge=edge,
        suggested_size_usdc=base_size,
        poly_odds_reversal=implied_entry,
        btc_delta_from_start=btc_delta,
        notes=f"z={z:.3f}",
        strategy_id="v1_fair_edge",
    )

