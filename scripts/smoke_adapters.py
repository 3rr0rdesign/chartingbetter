from __future__ import annotations

"""
Lightweight smoke test for the new adapter layer.

Run:
    python -m scripts.smoke_adapters
"""

from datetime import datetime

from adapters.truthfeed_chainlink_ws import ChainlinkWSTruthFeed
from adapters.polymarketfeed_gamma import GammaPolymarketFeed


def main() -> None:
    truth = ChainlinkWSTruthFeed()
    pm = GammaPolymarketFeed()

    print("=== TruthFeed.latest() ===")
    t = truth.latest()
    print(t)

    print("=== PolymarketFeed.active_btc_rounds(1) ===")
    rounds = pm.active_btc_rounds(limit=1)
    print(rounds)

    if rounds:
        r = rounds[0]
        print("=== snapshot for first round ===")
        snap = pm.snapshot(r.market_id)
        print(snap)


if __name__ == "__main__":
    main()

