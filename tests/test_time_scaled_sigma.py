from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from strategy.fair_prob import estimate_sigma


class TestTimeScaledSigma(unittest.TestCase):
    def test_same_path_different_sampling(self):
        # Simple Brownian-like path: multiplicative small steps
        base_time = datetime.now(timezone.utc)
        prices_1s = [100.0]
        times_1s = [base_time]
        for i in range(1, 11):
            prices_1s.append(prices_1s[-1] * (1.0 + 0.001 * ((-1) ** i)))
            times_1s.append(base_time + timedelta(seconds=i))

        # Sample every 5 seconds
        prices_5s = prices_1s[::5]
        times_5s = times_1s[::5]

        sigma_1s = estimate_sigma(prices_1s, times_1s)
        sigma_5s = estimate_sigma(prices_5s, times_5s)

        # Per-second sigma estimates should be in the same ballpark
        self.assertGreater(sigma_1s, 0.0)
        self.assertGreater(sigma_5s, 0.0)
        ratio = sigma_5s / sigma_1s
        self.assertTrue(0.5 < ratio < 1.5)


if __name__ == "__main__":
    unittest.main()

