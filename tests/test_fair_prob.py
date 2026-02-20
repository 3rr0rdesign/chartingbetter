from __future__ import annotations

import math
import unittest

from strategy.fair_prob import estimate_sigma, prob_finish_above_start


class TestProbFinishAboveStart(unittest.TestCase):
    def test_monotonic_in_current_price(self):
        start = 100.0
        sigma = 0.02
        tau = 60

        p1 = prob_finish_above_start(start, 101.0, sigma, tau)
        p2 = prob_finish_above_start(start, 105.0, sigma, tau)
        p3 = prob_finish_above_start(start, 120.0, sigma, tau)

        self.assertGreater(p2, p1)
        self.assertGreater(p3, p2)

    def test_sigma_pushes_toward_half_near_start(self):
        start = 100.0
        current = 101.0
        tau = 60

        p_low_sigma = prob_finish_above_start(start, current, 0.005, tau)
        p_high_sigma = prob_finish_above_start(start, current, 0.10, tau)

        self.assertGreater(p_low_sigma, 0.5)
        self.assertLess(abs(p_high_sigma - 0.5), abs(p_low_sigma - 0.5))


class TestEstimateSigma(unittest.TestCase):
    def test_constant_price_zero_sigma(self):
        prices = [100.0] * 10
        self.assertAlmostEqual(estimate_sigma(prices), 0.0, places=8)

    def test_increasing_price_positive_sigma(self):
        prices = [100 + i for i in range(10)]
        self.assertGreater(estimate_sigma(prices), 0.0)


if __name__ == "__main__":
    unittest.main()

