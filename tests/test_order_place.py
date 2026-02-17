"""Unit tests: order placement and stops (mocked Web3)."""
import os
import unittest
from unittest.mock import MagicMock

# Ensure DRY_RUN so no real orders
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("RPC_URL", "https://polygon-mainnet.g.alchemy.com/v2/demo")

from trading import place_order_with_stops, place_order_with_stops_simple


class TestPlaceOrderWithStops(unittest.TestCase):
    def test_place_order_with_stops_dry_run(self):
        w3 = MagicMock()
        ok, entry, profit_tgt = place_order_with_stops(
            w3,
            amount=10.0,
            price=0.12,
            side="BUY",
            market_id="test-market",
            token_id="123",
            outcome_index=0,
        )
        self.assertTrue(ok)
        self.assertAlmostEqual(entry, 0.12)
        self.assertAlmostEqual(profit_tgt, 0.12 * 0.96)

    def test_place_order_simple_dry(self):
        ok, entry, _ = place_order_with_stops_simple(
            amount_usdc=10,
            bid_price=0.15,
            market_id="m1",
            token_id="t1",
            side="BUY",
        )
        self.assertTrue(ok)
        self.assertAlmostEqual(entry, 0.15)
