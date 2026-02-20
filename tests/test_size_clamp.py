from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from backtest.engine import run_round
from core.types import MarketSnapshot, OutcomeSide, RoundState, StrategySignal


def _fake_strategy(round_state, snap):
    # Always suggest a huge size to trigger clamp.
    return StrategySignal(
        ts=snap.ts,
        market_id=round_state.market_id,
        desired_outcome=OutcomeSide.UP,
        prob_win=0.6,
        edge=0.1,
        suggested_size_usdc=10_000.0,
        poly_odds_reversal=0.5,
        btc_delta_from_start=0.0,
        notes=None,
        strategy_id="test",
    )


class TestSizeClamp(unittest.TestCase):
    def test_clamp_applied_and_noted(self):
        now = datetime.now(timezone.utc)
        r = RoundState(
            market_id="m1",
            slug=None,
            start_time=now,
            end_time=now + timedelta(minutes=5),
            btc_price_start=100.0,
            btc_price_end=None,
            winner=OutcomeSide.UP,
            resolved_at=None,
            streak_len=None,
            streak_direction=None,
        )
        snap = MarketSnapshot(
            ts=now,
            market_id="m1",
            outcome_up_price=0.5,
            outcome_down_price=0.5,
            btc_price=100.0,
            orderbooks={},
        )
        fills = run_round(r, [snap], _fake_strategy)
        self.assertEqual(len(fills), 1)
        f = fills[0]
        # Clamp should have reduced size compared to suggested 10_000 USDC at p=0.5:
        # MAX_SHARES=200 => max size ~= 200 * 0.5 = 100
        self.assertLess(f.filled_size, 10_000.0)
        self.assertEqual(f.notes, "size_clamped")


if __name__ == "__main__":
    unittest.main()

