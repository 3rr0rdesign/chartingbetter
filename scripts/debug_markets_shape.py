from backtest.polybacktest_client import PolyBackTestClient

c = PolyBackTestClient()
r = c._get_json("/v1/markets", params={"limit": 3})

print("type(r):", type(r))
if isinstance(r, dict):
    print("keys:", list(r.keys()))
    markets = r.get("markets")
    print("type(markets):", type(markets))
    if markets:
        print("type(markets[0]):", type(markets[0]))
        print("markets[0]:", markets[0])
else:
    print("r:", r)
