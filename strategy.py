"""
Streak reversal + Chainlink advantage: Bayesian prob, delta, advantage threshold, filters.
"""
from typing import Optional, Tuple

from config import (
    ADVANTAGE_THRESHOLD,
    REVERSAL_PROB_BASE,
    VOL_SPIKE_DELTA,
)
from chainlink_ws import get_latest_price


def bayesian_update(prior: float, evidence: float, likelihood: float) -> float:
    """
    Simple Bayesian update: P(H|E) = P(E|H)*P(H) / P(E).
    Here: H = reversal wins, E = observed flip in history.
    likelihood = P(flip | reversal regime) ~ historical flip rate (e.g. 0.75).
    evidence = P(flip) marginal; we approximate with likelihood*prior + (1-likelihood)*(1-prior).
    """
    if prior <= 0 or prior >= 1:
        return prior
    # P(E) = P(E|H)*P(H) + P(E|¬H)*(1-P(H))
    p_e = likelihood * prior + (1 - likelihood) * (1 - prior)
    if p_e <= 0:
        return prior
    posterior = (likelihood * prior) / p_e
    return max(0.0, min(1.0, posterior))


def reversal_prob_from_streak(
    streak: int,
    streak_direction: str,
    base_prob: float = REVERSAL_PROB_BASE,
    prior: float = 0.5,
    historical_flip_likelihood: float = 0.75,
) -> float:
    """
    Base reversal probability from streak (e.g. streak>=3 -> 70% reversal).
    Optionally refine with one Bayesian step using historical flip rate.
    """
    if streak < 1 or not streak_direction:
        return 0.5
    # Base: higher streak -> higher reversal prob
    prob = base_prob if streak >= 3 else 0.5 + (base_prob - 0.5) * (streak / 3.0)
    prob = bayesian_update(prior, 1.0, historical_flip_likelihood) * 0.5 + prob * 0.5
    return max(0.0, min(1.0, prob))


def delta_and_chainlink_implied_prob(
    open_price: float,
    current_price: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Delta = (current - open) / open.
    Chainlink-implied prob (reversal DOWN when delta < 0, UP when delta > 0):
    mapped as |delta| * 200 capped at 95% for "price moved against streak" implying reversal.
    Returns (delta, implied_prob).
    """
    if current_price is None:
        current_price = get_latest_price()
    if current_price is None or open_price <= 0:
        return 0.0, 0.5

    delta = (current_price - open_price) / open_price
    # Implied prob: delta * 200 capped at 95% (e.g. 0.475% move -> 95%)
    implied = min(0.95, abs(delta) * 20.0)
    return delta, implied


def boost_prob_if_delta_opposes_streak(
    reversal_prob: float,
    delta: float,
    streak_direction: str,
    boost: float = 0.10,
) -> float:
    """
    If delta opposes streak (e.g. negative after UP streak), boost reversal prob by +10%.
    """
    if not streak_direction:
        return reversal_prob
    opposes = (streak_direction == "UP" and delta < 0) or (streak_direction == "DOWN" and delta > 0)
    if opposes:
        return min(1.0, reversal_prob + boost)
    return reversal_prob


def advantage(
    poly_odds_reversal: float,
    chainlink_implied_prob: float,
    threshold: float = ADVANTAGE_THRESHOLD,
) -> float:
    """
    Advantage = chainlink_implied_prob - poly_odds_reversal.
    Signal when poly_odds_reversal < 0.20 (panic cheap) and advantage > threshold (e.g. 5%).
    """
    return chainlink_implied_prob - poly_odds_reversal


def should_signal_buy_reversal(
    poly_odds_reversal: float,
    chainlink_implied_prob: float,
    advantage_threshold: float = ADVANTAGE_THRESHOLD,
    panic_cheap_max: float = 0.20,
) -> bool:
    """
    BUY reversal when: Poly odds for reversal < 0.20 (panic cheap)
    and Chainlink-implied prob > Poly + advantage_threshold (e.g. +5%).
    """
    if poly_odds_reversal >= panic_cheap_max:
        return False
    adv = advantage(poly_odds_reversal, chainlink_implied_prob, 0)
    return adv >= advantage_threshold


def filter_vol_spike(abs_delta: float, max_delta: float = VOL_SPIKE_DELTA) -> bool:
    """True if we should skip (vol spike)."""
    return abs_delta > max_delta


def filter_entry_window(epoch_start_sec: float, now_sec: float, window_sec: int = 60) -> bool:
    """True if still within entry window (first 60s)."""
    return (now_sec - epoch_start_sec) <= window_sec
