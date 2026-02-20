"""
Benchmark Polygon RPC endpoints (median and p95 latency).

Run from repo root:
  python -m scripts.bench_rpc

Then set RPC_URL=<fastest one> in .env.
"""
from __future__ import annotations

import statistics
import time

import requests

RPCS = [
    "https://polygon-rpc.com/",
    "https://rpc.ankr.com/polygon",
    "https://polygon-bor-rpc.publicnode.com",
    "https://1rpc.io/matic",
    "https://polygon.drpc.org",
]


def ping(rpc: str, n: int = 15) -> list[float]:
    times_ms: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            r = requests.post(
                rpc,
                json={"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []},
                timeout=10,
            )
            r.raise_for_status()
        except Exception:
            pass
        times_ms.append((time.perf_counter() - t0) * 1000)
    return times_ms


def main() -> None:
    for rpc in RPCS:
        try:
            t = ping(rpc)
            median_ms = round(statistics.median(t), 1)
            p95_idx = max(0, int(len(t) * 0.95) - 1)
            p95_ms = round(sorted(t)[p95_idx], 1)
            print(rpc)
            print("  median_ms:", median_ms)
            print("  p95_ms   :", p95_ms)
        except Exception as e:
            print(rpc, "FAILED:", e)
        print()


if __name__ == "__main__":
    main()
