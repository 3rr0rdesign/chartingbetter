"""Unit tests: Bayesian and reversal prob calculations."""
import unittest

from strategy import (
    advantage,
    bayesian_update,
    boost_prob_if_delta_opposes_streak,
    reversal_prob_from_streak,
    should_signal_buy_reversal,
)


class TestBayesianUpdate(unittest.TestCase):
    def test_bayesian_update_basic(self):
        prior = 0.5
        likelihood = 0.75
        evidence = 1.0
        post = bayesian_update(prior, evidence, likelihood)
        self.assertGreater(post, prior)
        self.assertLessEqual(post, 1.0)
        self.assertGreaterEqual(post, 0.0)

    def test_bayesian_update_extreme_prior(self):
        self.assertEqual(bayesian_update(0.0, 1.0, 0.75), 0.0)
        self.assertEqual(bayesian_update(1.0, 1.0, 0.75), 1.0)


class TestReversalProb(unittest.TestCase):
    def test_reversal_prob_streak_3_up(self):
        p = reversal_prob_from_streak(3, "UP", base_prob=0.70)
        self.assertGreaterEqual(p, 0.5)
        self.assertLessEqual(p, 1.0)

    def test_reversal_prob_streak_0(self):
        p = reversal_prob_from_streak(0, "UP")
        self.assertEqual(p, 0.5)

    def test_reversal_prob_no_direction(self):
        p = reversal_prob_from_streak(3, "")
        self.assertEqual(p, 0.5)


class TestBoostDelta(unittest.TestCase):
    def test_boost_negative_delta_after_up_streak(self):
        p = boost_prob_if_delta_opposes_streak(0.70, -0.01, "UP", boost=0.10)
        self.assertGreater(p, 0.70)
        self.assertLessEqual(p, 1.0)

    def test_no_boost_same_direction(self):
        p = boost_prob_if_delta_opposes_streak(0.70, 0.01, "UP", boost=0.10)
        self.assertEqual(p, 0.70)


class TestAdvantage(unittest.TestCase):
    def test_advantage_positive(self):
        a = advantage(0.15, 0.25, threshold=0.05)
        self.assertGreater(a, 0.05)

    def test_should_signal_panic_cheap(self):
        self.assertTrue(should_signal_buy_reversal(0.15, 0.25, advantage_threshold=0.05))
        self.assertFalse(should_signal_buy_reversal(0.30, 0.35, advantage_threshold=0.05))
