from __future__ import annotations

"""
Quick auth check for PolyBackTest API.

Run:
    POLYBACKTEST_API_KEY=... python -m scripts.check_polybacktest_auth
"""

from backtest.polybacktest_client import PolyBackTestClient


def main() -> None:
    client = PolyBackTestClient()
    data = client._get_json("/v1/markets", params={"limit": 1})
    print("Auth OK. Sample response:")
    print(data[:1] if isinstance(data, list) else data)


if __name__ == "__main__":
    main()

