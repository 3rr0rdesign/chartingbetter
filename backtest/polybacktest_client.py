from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import requests
from dotenv import load_dotenv

from core.types import MarketSnapshot, OutcomeSide, OrderBook, OrderBookLevel, RoundState
from .types_bt import RoundData


def _normalize_price(raw: float) -> float:
    """Ensure price is in [0, 1]. If API returns 0-100 scale, normalize and warn once."""
    p = float(raw)
    if p < 0 or p > 1:
        if 1.0 < p < 1000.0:
            import warnings
            warnings.warn(
                f"Orderbook price scale: raw price {p} outside [0,1]; normalizing by /100. "
                "If this is wrong, fix API scaling."
            )
            p = p / 100.0
        elif p >= 1000.0:
            raise RuntimeError(
                f"Unexpected orderbook price scale: raw={p}. "
                "Expected [0,1] or [0,100]. Add normalization or fix API."
            )
    return max(0.0, min(1.0, p))


def _parse_dt(value: str) -> datetime:
    # ISO8601 with optional Z
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class PolyBackTestClient:
    """
    Thin client for the PolyBackTest REST API.

    This is intentionally conservative and only uses documented patterns from
    https://docs.polybacktest.com/.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        # Ensure .env is loaded so POLYBACKTEST_API_KEY and related settings
        # are available from environment variables.
        load_dotenv()
        self.base_url = (base_url or os.environ.get("POLYBACKTEST_BASE_URL", "https://api.polybacktest.com")).rstrip("/")
        self.api_key = api_key or os.environ.get("POLYBACKTEST_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "POLYBACKTEST_API_KEY is required for PolyBackTestClient. "
                "Set it in your environment or pass api_key= explicitly."
            )
        cache_root = cache_dir or os.path.join(os.getcwd(), ".cache", "polybacktest")
        self.cache_path = Path(cache_root)
        self.cache_path.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=50,
            pool_maxsize=50,
            max_retries=3,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # ------------- HTTP helpers -------------

    def _headers(self) -> dict:
        return {
            "Accept": "application/json",
            "X-API-Key": self.api_key,
        }

    def _cache_file(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_path / f"{digest}.json"

    def _get_json(self, path: str, params: Optional[dict] = None, use_cache: bool = True) -> any:
        url = f"{self.base_url}{path}"
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        cache_file = self._cache_file(key)

        if use_cache and cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        backoff = 1.0
        for attempt in range(4):
            try:
                r = self.session.get(url, headers=self._headers(), params=params or {}, timeout=20)
                if r.status_code in (401, 403):
                    try:
                        body = r.json()
                        detail = body.get("detail") or body
                    except Exception:
                        detail = r.text
                    raise RuntimeError(
                        f"PolyBackTest auth error {r.status_code}: {detail}. "
                        "Check POLYBACKTEST_API_KEY or rotate the key."
                    )
                r.raise_for_status()
                data = r.json()
                cache_file.write_text(json.dumps(data), encoding="utf-8")
                return data
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(backoff)
                backoff *= 2

    # ------------- Public API -------------

    def list_rounds(
        self,
        date_from: str,
        date_to: str,
        market_type: str = "5m",
        max_rounds: Optional[int] = None,
    ) -> List[RoundState]:
        """
        List BTC up/down rounds in a time window.

        PolyBackTest docs show /v1/markets?market_type=5m&limit=..., and example
        responses include fields like start_time, end_time, btc_price_start, winner.
        We also pass date filters when supported.
        """
        if os.environ.get("DEBUG_POLYBACKTEST", "").strip().lower() in ("1", "true", "yes", "on"):
            print("[list_rounds] fetching rounds...")
        out: List[RoundState] = []
        pages = 0
        offset = 0
        limit = 100
        debug_first = os.environ.get("DEBUG_POLYBACKTEST", "").strip().lower() in ("1", "true", "yes", "on")
        base_params = {
            "market_type": market_type,
            "start_time_from": date_from,
            "start_time_to": date_to,
        }

        while True:
            params = dict(base_params)
            params["limit"] = limit
            params["offset"] = offset
            if debug_first and pages == 0:
                print(f"[DEBUG] /v1/markets url={self.base_url+'/v1/markets'} params={params}")

            resp = self._get_json("/v1/markets", params=params)
            markets = resp.get("markets", []) if isinstance(resp, dict) else resp
            if not isinstance(markets, list):
                raise RuntimeError(f"Unexpected markets payload type: {type(markets)}")

            if not markets:
                break

            for m in markets:
                if not isinstance(m, dict):
                    continue
                market_id = str(m.get("market_id") or m.get("id") or "")
                slug = m.get("slug")
                # Parse times as timezone‑aware datetimes.
                start = _parse_dt(m["start_time"]) if m.get("start_time") else datetime.now(timezone.utc)
                end = _parse_dt(m["end_time"]) if m.get("end_time") else start
                resolved_at = _parse_dt(m["resolved_at"]) if m.get("resolved_at") else None
                # BTC reference prices might be strings or numbers.
                btc_start_raw = m.get("btc_price_start")
                btc_end_raw = m.get("btc_price_end")
                btc_start = None
                btc_end = None
                try:
                    if btc_start_raw is not None:
                        btc_start = float(btc_start_raw)
                except (TypeError, ValueError):
                    btc_start = None
                try:
                    if btc_end_raw is not None:
                        btc_end = float(btc_end_raw)
                except (TypeError, ValueError):
                    btc_end = None
                # Winner mapping is case‑insensitive.
                winner_raw = (m.get("winner") or "").strip().upper()
                winner: Optional[OutcomeSide]
                if winner_raw == "UP":
                    winner = OutcomeSide.UP
                elif winner_raw == "DOWN":
                    winner = OutcomeSide.DOWN
                else:
                    winner = None
                out.append(
                    RoundState(
                        market_id=market_id,
                        slug=slug,
                        start_time=start,
                        end_time=end,
                        btc_price_start=btc_start,
                        btc_price_end=btc_end,
                        winner=winner,
                        resolved_at=resolved_at,
                        streak_len=None,
                        streak_direction=None,
                    )
                )
                if max_rounds is not None and len(out) >= max_rounds:
                    print(f"Fetched {len(out)} rounds via pagination (pages={pages+1})")
                    return out

            pages += 1
            if len(markets) < limit:
                break
            offset += limit

        print(f"Fetched {len(out)} rounds via pagination (pages={pages})")
        return out

    def load_round(
        self,
        round_id: str,
        include_orderbook: bool = False,
        max_pages: int = 5,
    ) -> RoundData:
        """
        Load full metadata + snapshots for a single round.
        """
        market = self._get_json(f"/v1/markets/{round_id}", params=None)
        start_raw = market.get("start_time")
        end_raw = market.get("end_time")
        start_param = start_raw
        end_param = end_raw

        round_state = RoundState(
            market_id=str(market.get("market_id") or market.get("id") or ""),
            slug=market.get("slug"),
            start_time=_parse_dt(market["start_time"]) if market.get("start_time") else datetime.now(timezone.utc),
            end_time=_parse_dt(market["end_time"]) if market.get("end_time") else datetime.now(timezone.utc),
            btc_price_start=float(market.get("btc_price_start") or 0.0),
            btc_price_end=float(market.get("btc_price_end") or 0.0) if market.get("btc_price_end") is not None else None,
            winner=None,
            resolved_at=_parse_dt(market["resolved_at"]) if market.get("resolved_at") else None,
            streak_len=None,
            streak_direction=None,
        )

        snaps: List[MarketSnapshot] = []

        offset = 0
        total = None
        pages = 0
        while pages < max_pages:
            snap_params = {
                "limit": 1000,
                "offset": offset,
            }
            if start_param:
                snap_params["start_time_from"] = start_param
            if end_param:
                snap_params["end_time_to"] = end_param
            if include_orderbook:
                snap_params["include_orderbook"] = "true"

            page = self._get_json(f"/v1/markets/{round_id}/snapshots", params=snap_params)
            if isinstance(page, dict):
                page_snaps = page.get("snapshots", [])
                total = page.get("total") or len(page_snaps)
            else:
                page_snaps = page or []
                total = len(page_snaps) if total is None else total

            if not page_snaps:
                break

            debug_snap_limit = 0
            try:
                debug_snap_limit = int(os.environ.get("DEBUG_SNAPSHOTS", "0"))
            except ValueError:
                pass

            for snap_idx, s in enumerate(page_snaps):
                if debug_snap_limit > 0 and len(snaps) == 0:
                    print(f"[SNAP DEBUG] first snapshot keys: {sorted(s.keys())}")
                ts = _parse_dt(s["time"]) if s.get("time") else datetime.now(timezone.utc)
                btc_price = float(s.get("btc_price") or 0.0) if s.get("btc_price") is not None else None
                price_up = s.get("price_up")
                price_down = s.get("price_down")
                orderbooks = {}
                if include_orderbook:
                    # API raw format: orderbook_up/orderbook_down have "bids" and "asks" lists;
                    # each level has "price" (0-1 or 0-100; we normalize) and "size".
                    # Bids must be sorted descending (best bid highest), asks ascending (best ask lowest).
                    ob_up = s.get("orderbook_up") or {}
                    ob_down = s.get("orderbook_down") or {}
                    if ob_up:
                        bids = [OrderBookLevel(price=_normalize_price(b["price"]), size=float(b["size"])) for b in ob_up.get("bids", [])]
                        asks = [OrderBookLevel(price=_normalize_price(a["price"]), size=float(a["size"])) for a in ob_up.get("asks", [])]
                        bids.sort(key=lambda L: L.price, reverse=True)
                        asks.sort(key=lambda L: L.price)
                        ob_u = OrderBook(bids=bids, asks=asks)
                        orderbooks["UP"] = ob_u
                        best_bid = ob_u.bids[0].price if ob_u.bids else None
                        best_ask = ob_u.asks[0].price if ob_u.asks else None
                        if best_bid is not None and best_ask is not None:
                            price_up = 0.5 * (best_bid + best_ask)
                        else:
                            price_up = best_bid if best_bid is not None else best_ask
                    if ob_down:
                        bids = [OrderBookLevel(price=_normalize_price(b["price"]), size=float(b["size"])) for b in ob_down.get("bids", [])]
                        asks = [OrderBookLevel(price=_normalize_price(a["price"]), size=float(a["size"])) for a in ob_down.get("asks", [])]
                        bids.sort(key=lambda L: L.price, reverse=True)
                        asks.sort(key=lambda L: L.price)
                        ob_d = OrderBook(bids=bids, asks=asks)
                        orderbooks["DOWN"] = ob_d
                        best_bid = ob_d.bids[0].price if ob_d.bids else None
                        best_ask = ob_d.asks[0].price if ob_d.asks else None
                        if best_bid is not None and best_ask is not None:
                            price_down = 0.5 * (best_bid + best_ask)
                        else:
                            price_down = best_bid if best_bid is not None else best_ask

                    if debug_snap_limit > 0 and len(snaps) < debug_snap_limit:
                        ob_u = orderbooks.get("UP")
                        bb = ob_u.bids[0].price if ob_u and ob_u.bids else None
                        ba = ob_u.asks[0].price if ob_u and ob_u.asks else None
                        mid = (0.5 * (bb + ba)) if (bb is not None and ba is not None) else None
                        raw_bids = ob_up.get("bids", [])
                        raw_asks = ob_up.get("asks", [])
                        raw_bid_price = raw_bids[0].get("price") if raw_bids else None
                        raw_ask_price = raw_asks[0].get("price") if raw_asks else None
                        if len(snaps) == 0:
                            print(f"[SNAP DEBUG] UP raw first 3 bids: {raw_bids[:3]}")
                            print(f"[SNAP DEBUG] UP raw first 3 asks: {raw_asks[:3]}")
                        print(
                            f"[SNAP DEBUG {len(snaps)+1}] round_id={round_id} ts={ts} "
                            f"raw price_up={s.get('price_up')} price_down={s.get('price_down')} "
                            f"raw_top_bid={raw_bid_price} raw_top_ask={raw_ask_price} "
                            f"normalized best_bid_up={bb} best_ask_up={ba} mid_up={mid}"
                        )
                outcome_up = _normalize_price(price_up) if price_up is not None else None
                outcome_down = _normalize_price(price_down) if price_down is not None else None
                snaps.append(
                    MarketSnapshot(
                        ts=ts,
                        market_id=round_state.market_id,
                        outcome_up_price=outcome_up,
                        outcome_down_price=outcome_down,
                        btc_price=btc_price,
                        orderbooks=orderbooks,
                    )
                )

            offset += len(page_snaps)
            pages += 1
            if total is not None and offset >= total:
                break

        if total is not None and offset < total:
            print(
                f"[WARN] Truncated snapshots for round {round_id}: "
                f"fetched={offset} total={total} max_pages={max_pages}"
            )

        return RoundData(round=round_state, snapshots=snaps)

    def iter_snapshots(
        self,
        round_id: str,
        step_seconds: int = 1,
        include_orderbook: bool = False,
    ) -> Iterator[MarketSnapshot]:
        """
        Convenience iterator over snapshots for a round, optionally down‑sampled
        by `step_seconds`.
        """
        data = self.load_round(round_id, include_orderbook=include_orderbook)
        last_ts: Optional[datetime] = None
        for snap in data.snapshots:
            if last_ts is None:
                yield snap
                last_ts = snap.ts
                continue
            if step_seconds <= 0:
                yield snap
                continue
            if (snap.ts - last_ts).total_seconds() >= step_seconds:
                yield snap
                last_ts = snap.ts

