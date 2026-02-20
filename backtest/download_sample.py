from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from backtest.polybacktest_client import PolyBackTestClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one day of PolyBackTest BTC rounds.")
    parser.add_argument("--days_ago", type=int, default=1, help="How many days ago to start (default: 1)")
    parser.add_argument("--market_type", type=str, default="5m", help="Market type (default: 5m)")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=args.days_ago)).date().isoformat()
    end = (now - timedelta(days=args.days_ago - 1)).date().isoformat()

    client = PolyBackTestClient()
    rounds = client.list_rounds(date_from=start, date_to=end, market_type=args.market_type)
    print(f"Rounds between {start} and {end} (type={args.market_type}): {len(rounds)}")

    # Rough spread estimate: for each round, use first snapshot and measure how
    # far UP/DOWN prices deviate from a fair 0.5 / 0.5 split.
    total_spread = 0.0
    spread_count = 0
    first_snapshot_example = None

    for r in rounds:
        snaps = list(client.iter_snapshots(r.market_id, step_seconds=5))
        if not snaps:
            continue
        s0 = snaps[0]
        if first_snapshot_example is None:
            first_snapshot_example = s0
        if s0.outcome_up_price is not None and s0.outcome_down_price is not None:
            # Treat implied fair price as 0.5; measure average absolute deviation.
            spread = abs(s0.outcome_up_price - 0.5) + abs(s0.outcome_down_price - 0.5)
            total_spread += spread
            spread_count += 1

    if spread_count:
        avg_spread = total_spread / spread_count
    else:
        avg_spread = 0.0

    print(f"Avg initial spread estimate (deviation from 0.5): {avg_spread:.4f} over {spread_count} rounds")
    if first_snapshot_example:
        print("First snapshot example:")
        print(first_snapshot_example)


if __name__ == "__main__":
    main()

