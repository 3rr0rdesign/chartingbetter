from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from core.interfaces import TruthFeed
from core.types import TruthPrice
from chainlink_ws import get_latest_price


class ChainlinkWSTruthFeed(TruthFeed):
    """
    TruthFeed implementation backed by the existing Chainlink WebSocket listener.

    Note: This adapter only surfaces whatever price the listener last saw. It does
    not manage the listener thread itself; callers should ensure
    start_chainlink_listener_thread() has been called elsewhere if they expect
    fresh data.
    """

    def latest(self) -> Optional[TruthPrice]:
        px = get_latest_price()
        if px is None:
            return None
        now = datetime.now(timezone.utc)
        return TruthPrice(ts=now, btc_price=float(px), source="chainlink_ws")

    def price_at(self, ts: datetime) -> Optional[TruthPrice]:
        """
        For now, just return the latest available price.

        Backtesting / precise timestamp queries should use dedicated
        historical data sources (e.g. PolyBackTest).
        """
        return self.latest()

