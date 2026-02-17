"""Unit tests: Poly API fetch and streak (mock or live)."""
import json
import unittest
from unittest.mock import patch, MagicMock

from poly_api import (
    BTC_5MIN_SLUG_PATTERN,
    _parse_outcomes,
    market_liquidity,
    market_open_price,
    market_reversal_side,
    poly_odds_for_reversal,
    reversal_outcome_index,
    get_streak,
)


class TestParseOutcomes(unittest.TestCase):
    def test_parse_string_outcomes(self):
        m = {"outcomes": '["Up", "Down"]', "outcomePrices": "[0.4, 0.6]"}
        o, p = _parse_outcomes(m)
        self.assertEqual(o, ["Up", "Down"])
        self.assertEqual(p, [0.4, 0.6])

    def test_parse_list_outcomes(self):
        m = {"outcomes": ["Yes", "No"], "outcomePrices": [0.5, 0.5]}
        o, p = _parse_outcomes(m)
        self.assertEqual(o, ["Yes", "No"])
        self.assertEqual(p, [0.5, 0.5])


class TestSlugPattern(unittest.TestCase):
    def test_btc_5min_slugs(self):
        self.assertTrue(BTC_5MIN_SLUG_PATTERN.search("btc-up-down-5min-123"))
        self.assertTrue(BTC_5MIN_SLUG_PATTERN.search("btc-updown-5m-456"))
        self.assertFalse(BTC_5MIN_SLUG_PATTERN.search("btc-weekly-123"))


class TestMarketHelpers(unittest.TestCase):
    def test_market_liquidity(self):
        self.assertEqual(market_liquidity({"liquidityNum": 15000}), 15000.0)
        self.assertEqual(market_liquidity({"liquidity": "10000"}), 10000.0)

    def test_market_open_price(self):
        self.assertEqual(market_open_price({"openPrice": 97000}), 97000.0)
        self.assertIsNone(market_open_price({}))

    def test_market_reversal_side(self):
        self.assertEqual(market_reversal_side("UP"), "DOWN")
        self.assertEqual(market_reversal_side("DOWN"), "UP")
        self.assertEqual(market_reversal_side(""), "")

    def test_reversal_outcome_index(self):
        m = {"outcomes": '["Up", "Down"]', "outcomePrices": "[0.5, 0.5]"}
        self.assertEqual(reversal_outcome_index(m, "DOWN"), 1)
        self.assertEqual(reversal_outcome_index(m, "UP"), 0)

    def test_poly_odds_for_reversal(self):
        m = {"outcomes": '["Up", "Down"]', "outcomePrices": "[0.3, 0.7]"}
        self.assertAlmostEqual(poly_odds_for_reversal(m, "DOWN"), 0.7)
        self.assertAlmostEqual(poly_odds_for_reversal(m, "UP"), 0.3)


class TestGetStreak(unittest.TestCase):
    @patch("poly_api.requests.get")
    def test_get_streak_empty(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [])
        streak, direction = get_streak("https://gamma-api.polymarket.com", limit=10)
        self.assertEqual(streak, 0)
        self.assertEqual(direction, "")

    @patch("poly_api.requests.get")
    def test_get_streak_resolved(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {"slug": "btc-updown-5m-1", "outcomes": "[\"Up\",\"Down\"]", "outcomePrices": "[1,0]"},
                {"slug": "btc-updown-5m-2", "outcomes": "[\"Up\",\"Down\"]", "outcomePrices": "[1,0]"},
            ],
        )
        streak, direction = get_streak("https://gamma-api.polymarket.com", limit=10)
        self.assertGreaterEqual(streak, 1)
        self.assertIn(direction, ("UP", "DOWN", ""))
