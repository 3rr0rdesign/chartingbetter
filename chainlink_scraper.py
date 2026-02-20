"""
Chainlink Data Streams BTC/USD — page scraper (no API key).

Loads the Chainlink stream page and extracts the displayed Mid-price.
Serves GET /truth in the same JSON shape as the Go server so the dashboard can use it.

Caveat: the data.chain.link website is itself delayed by a few seconds vs the live
Data Streams API. So this gives you the same number as the site, not the exact
resolution moment — only the real API does that.
"""

import json
import re
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Optional: try requests first (faster if price is in HTML)
try:
    import requests
    from bs4 import BeautifulSoup
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Playwright for JS-rendered page
try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


PORT = int(__import__("os").environ.get("SCRAPE_PORT", "8788"))
STREAM_URLS = [
    "https://data.chain.link/streams/btc-usd-cexprice-streams",
    "https://data.chain.link/streams/btc-usd",
]
REFRESH_INTERVAL = 2.0  # seconds
PRICE_RE = re.compile(r"\$?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)")

# Shared state (thread-safe: single writer in scraper thread, readers in HTTP handler)
_last_price: float | None = None
_last_ts: int = 0
_lock = threading.Lock()


def _parse_price_from_text(text: str) -> float | None:
    """Extract first plausible BTC price (e.g. 67012.51) from text."""
    if not text:
        return None
    # Match numbers that look like prices: 67,012.51 or 67012.51 (40k–120k range)
    for m in PRICE_RE.finditer(text):
        s = m.group(1).replace(",", "")
        try:
            p = float(s)
            if 10_000 < p < 200_000:  # sane BTC range
                return p
        except ValueError:
            continue
    return None


def _fetch_with_requests() -> float | None:
    if not HAS_REQUESTS:
        return None
    for url in STREAM_URLS:
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0"})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text()
            return _parse_price_from_text(text)
        except Exception:
            continue
    return None


def _fetch_with_playwright() -> float | None:
    if not HAS_PLAYWRIGHT:
        return None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.set_default_timeout(15000)
            for url in STREAM_URLS:
                try:
                    page.goto(url, wait_until="networkidle")
                    page.wait_for_timeout(2000)  # let client-side price render
                    # Try visible price: often in a large heading or span
                    body = page.inner_text("body")
                    price = _parse_price_from_text(body)
                    if price is not None:
                        return price
                    # Try data attributes or specific selectors
                    for sel in [
                        '[data-testid*="price"]',
                        '[class*="price"]',
                    ]:
                        try:
                            loc = page.locator(sel)
                            if loc.count() > 0:
                                t = loc.first.inner_text()
                                p = _parse_price_from_text(t)
                                if p is not None:
                                    return p
                        except Exception:
                            continue
                except Exception:
                    continue
        finally:
            browser.close()
    return None


def _scrape_once() -> float | None:
    p = _fetch_with_requests()
    if p is not None:
        return p
    return _fetch_with_playwright()


def _scraper_loop():
    global _last_price, _last_ts
    while True:
        try:
            p = _scrape_once()
            if p is not None:
                with _lock:
                    _last_price = p
                    _last_ts = int(time.time())
        except Exception:
            pass
        time.sleep(REFRESH_INTERVAL)


class TruthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.rstrip("/") != "/truth":
            self.send_response(404)
            self.end_headers()
            return
        with _lock:
            price = _last_price
            ts = _last_ts
        if price is None:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"no price yet"}')
            return
        # Same shape as Go truth server (18-decimal string, Unix timestamp)
        benchmark18 = str(int(round(price * 1e18)))
        payload = {
            "feedId": "scraper",
            "benchmarkPrice18": benchmark18,
            "observationsTimestamp": ts,
            "validFromTimestamp": ts,
            "expiresAt": ts + 86400,
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, format, *args):
        pass  # quiet


def main():
    if not HAS_PLAYWRIGHT and not HAS_REQUESTS:
        print("Install: pip install playwright requests beautifulsoup4 && playwright install chromium")
        return
    print("Starting Chainlink page scraper (same price as data.chain.link)...")
    print("  URLs:", STREAM_URLS)
    print("  Refresh every", REFRESH_INTERVAL, "s")
    t = threading.Thread(target=_scraper_loop, daemon=True)
    t.start()
    # Prefer one initial scrape before first request
    time.sleep(0.5)
    server = HTTPServer(("", PORT), TruthHandler)
    print("Serving GET /truth on http://localhost:%s" % PORT)
    print("In the dashboard: set Truth API URL to http://localhost:%s and enable Truth API." % PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
