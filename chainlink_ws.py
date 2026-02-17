"""
Chainlink BTC/USD WebSocket listener for sub-3s prices.
Updates shared state (latest_price) for strategy to compute delta vs market open.
"""
import json
import logging
import threading
import time
from typing import Any, Callable, Optional

import websocket

from config import CHAINLINK_WS_URL

logger = logging.getLogger(__name__)

# Shared state: latest BTC price from Chainlink (thread-safe via single writer)
_latest_price: Optional[float] = None
_lock = threading.Lock()


def get_latest_price() -> Optional[float]:
    with _lock:
        return _latest_price


def set_latest_price(price: Optional[float]) -> None:
    with _lock:
        global _latest_price
        _latest_price = price


def _parse_price_from_payload(data: Any) -> Optional[float]:
    """
    Parse numeric price from Chainlink WS payload.
    Handles common shapes: {"price": N}, {"data": {"price": N}}, {"last": N}, etc.
    """
    if data is None:
        return None
    if isinstance(data, (int, float)):
        return float(data)
    if isinstance(data, dict):
        for key in ("price", "p", "last", "close", "value"):
            if key in data:
                v = data[key]
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, str):
                    try:
                        return float(v)
                    except ValueError:
                        pass
        if "data" in data and isinstance(data["data"], dict):
            return _parse_price_from_payload(data["data"])
        if "report" in data:
            return _parse_price_from_payload(data["report"])
    if isinstance(data, list) and len(data) > 0:
        return _parse_price_from_payload(data[0])
    return None


def _on_message(ws: websocket.WebSocketApp, message: str) -> None:
    try:
        payload = json.loads(message)
        price = _parse_price_from_payload(payload)
        if price is not None and price > 0:
            set_latest_price(price)
    except json.JSONDecodeError:
        pass
    except Exception as e:
        logger.debug("Chainlink parse error: %s", e)


def _on_error(ws: websocket.WebSocketApp, error: Optional[Exception]) -> None:
    if error:
        logger.warning("Chainlink WS error: %s", error)


def _on_close(ws: websocket.WebSocketApp, close_status: Optional[int], close_msg: Optional[str]) -> None:
    logger.info("Chainlink WS closed: %s %s", close_status, close_msg)


def _on_open(ws: websocket.WebSocketApp) -> None:
    logger.info("Chainlink WS connected")


def run_chainlink_listener(
    url: Optional[str] = None,
    on_price_callback: Optional[Callable[[float], None]] = None,
) -> None:
    """
    Run Chainlink WebSocket listener in current thread (blocking).
    Updates get_latest_price() on each message; optionally calls on_price_callback(price).
    """
    ws_url = url or CHAINLINK_WS_URL

    def on_message(ws, message):
        _on_message(ws, message)
        p = get_latest_price()
        if p is not None and on_price_callback:
            try:
                on_price_callback(p)
            except Exception:
                pass

    app = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=_on_error,
        on_close=_on_close,
        on_open=_on_open,
    )
    while True:
        try:
            app.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            logger.warning("Chainlink WS run_forever: %s", e)
        time.sleep(5)


def start_chainlink_listener_thread(
    url: Optional[str] = None,
    on_price_callback: Optional[Callable[[float], None]] = None,
) -> threading.Thread:
    """Start Chainlink listener in a daemon thread."""
    t = threading.Thread(
        target=run_chainlink_listener,
        kwargs={"url": url, "on_price_callback": on_price_callback},
        daemon=True,
    )
    t.start()
    return t
