from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from core.interfaces import Executor
from core.types import Fill, OutcomeSide, Side, StrategySignal, Trade
from trading import place_order_with_stops_simple


class TradingExecutor(Executor):
    """
    Executor implementation that delegates to the existing trading.py helpers.

    This adapter is intentionally thin and conservative:
    - It uses StrategySignal.suggested_size_usdc as the notional size.
    - It uses StrategySignal.poly_odds_reversal (if present) as the limit price,
      otherwise a neutral 0.5.
    - It does NOT change any underlying trading behavior; it just wraps the
      existing place_order_with_stops_simple() function.
    """

    def submit_from_signal(self, signal: StrategySignal) -> Optional[Fill]:
        price = signal.poly_odds_reversal if signal.poly_odds_reversal is not None else 0.5
        size_usdc = max(0.0, signal.suggested_size_usdc)
        if size_usdc <= 0:
            return None

        trade = Trade(
            ts=signal.ts,
            market_id=signal.market_id,
            outcome=signal.desired_outcome,
            side=Side.BUY,
            price=price,
            size=size_usdc,
            client_order_id=None,
            tags={"strategy_id": signal.strategy_id or ""},
        )
        return self.submit_trade(trade)

    def submit_trade(self, trade: Trade) -> Optional[Fill]:
        # Map Trade into existing trading.py call. Token ID and outcome index
        # are not tracked in the current simplified flow, so we leave token_id blank.
        success, entry_price, _ = place_order_with_stops_simple(
            amount_usdc=trade.size,
            bid_price=trade.price,
            market_id=trade.market_id,
            token_id="",
            side=trade.side.value,
        )
        if not success or entry_price is None:
            return None

        # Fees and realized PnL are handled elsewhere; here we only capture
        # the fact that we "filled" the order at entry_price.
        return Fill(
            trade=trade,
            filled_size=trade.size,
            avg_price=entry_price,
            fee_paid=0.0,
            realized_pnl=None,
        )

