from __future__ import annotations

import unittest

from backtest.engine import calc_realized_pnl


class TestPayoutModel(unittest.TestCase):
    def test_win_at_half_price_approx_size(self):
        size = 10.0
        entry_price = 0.5
        fee_bps = 0.0
        pnl = calc_realized_pnl(size, entry_price, True, fee_bps)
        # 10 * (1/0.5 - 1) = 10
        self.assertAlmostEqual(pnl, size, places=6)

    def test_loss_at_half_price_minus_size(self):
        size = 10.0
        entry_price = 0.5
        fee_bps = 0.0
        pnl = calc_realized_pnl(size, entry_price, False, fee_bps)
        self.assertAlmostEqual(pnl, -size, places=6)

    def test_fee_reduces_pnl(self):
        size = 10.0
        entry_price = 0.5
        fee_bps = 100.0  # 1%
        pnl_win = calc_realized_pnl(size, entry_price, True, fee_bps)
        pnl_lose = calc_realized_pnl(size, entry_price, False, fee_bps)
        fee = size * fee_bps / 10_000.0
        self.assertAlmostEqual(pnl_win, size - fee, places=6)
        self.assertAlmostEqual(pnl_lose, -size - fee, places=6)


if __name__ == "__main__":
    unittest.main()

