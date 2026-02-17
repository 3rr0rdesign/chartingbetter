# Polymarket 5-Min BTC Streak-Reversal Bot

Advanced Python trading bot for Polymarket’s 5-minute BTC up/down markets using a **streak reversal** strategy with a **Chainlink price edge**. Strategy aims for a high win rate by fading streaks of 3+ and using live Chainlink BTC/USD data when Poly odds are “panic cheap.”

## Requirements

- **Python 3.10+**
- Libraries: `websocket-client`, `requests`, `web3`, `python-dotenv`, `pandas`, `scikit-learn` (see `requirements.txt`). Assume `pip install -r requirements.txt` is run by you.

## Setup

1. **Clone / enter project**
   ```bash
   cd polybot
   ```

2. **Create virtualenv and install**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

3. **Configure environment**
   - Copy `.env.example` to `.env`.
   - Set at least:
     - `PRIVATE_KEY` – wallet private key (never log or commit).
     - `RPC_URL` – Polygon RPC (e.g. Alchemy/Infura).
   - Optional: `BET_SIZE`, `STREAK_MIN`, `REVERSAL_PROB_BASE`, `ADVANTAGE_THRESHOLD`, `PROFIT_TGT`, `STOP_LOSS`, `LIQ_MIN`, `DRY_RUN`, `BACKTEST_DAYS`, etc. (see `.env.example`).

4. **Fund wallet**
   - Bridge **USDC.e** to Polygon (e.g. Polymarket deposit flow). Bot uses USDC.e only.
   - Ensure balance &gt; `MIN_BALANCE_USDC` (default 50) for live trades.

5. **Run (dry run first)**
   ```bash
   python bot.py --dry
   ```
   Then, when ready:
   ```bash
   python bot.py
   ```
   Backtest:
   ```bash
   python bot.py --backtest --backtest-days 7
   ```

## Strategy Summary

- **Streak reversal:** Track last 10 resolutions via Poly Gamma API; if streak ≥ 3 (e.g. 3 UP), reversal side = opposite (DOWN). Base reversal prob ≈ 70%; updated with simple Bayesian prior/likelihood from historical flip rate.
- **Chainlink edge:** Live BTC/USD from Chainlink WS; delta = (current − open) / open. If delta opposes streak (e.g. negative after UP streak), boost reversal prob +10%. **Advantage:** If Poly odds for reversal &lt; 0.20 and Chainlink-implied prob &gt; Poly + 5%, signal **BUY reversal**.
- **Entry:** Only in first 60s of market; $10–20 USDC.e limit at current bid when advantage &gt; 0. Skip if liquidity &lt; $10k or |delta| &gt; 2%.
- **Exits:** Limit sell at `0.96 * entry` (profit); stop at `0.60 * entry` (loss).
- **Filters:** No trade if streak &lt; 3 or neutral delta; max 1 trade per market, 3–5/hour.

## Bot Flow

- **Loop (every 5s):** Poll Gamma API for active 5-min BTC markets (slug filter: `btc-up-down-5min-*` / `btc-updown-5m-*`).
- **New market:** Fetch/store `open_price`, start streak from historicals.
- **Chainlink:** WebSocket listener updates latest price; delta and implied prob computed in real time.
- **Signal:** When conditions met, execute via Web3 (approve USDC.e, buy shares via Poly contracts / CLOB).
- **Backtest (`--backtest`):** Simulate 100+ historical resolved markets, log hit rate and EV.

## Config (.env)

| Variable | Description |
|----------|-------------|
| `PRIVATE_KEY` | Wallet key (required for live) |
| `RPC_URL` | Polygon RPC |
| `BET_SIZE` | Bet size in USDC (default 10) |
| `STREAK_MIN` | Min streak for reversal (default 3) |
| `REVERSAL_PROB_BASE` | Base reversal prob (default 0.70) |
| `ADVANTAGE_THRESHOLD` | Min edge vs Poly (default 0.05) |
| `PROFIT_TGT` | Profit target multiplier (default 0.96) |
| `STOP_LOSS` | Stop loss multiplier (default 0.60) |
| `LIQ_MIN` | Min liquidity (default 10000) |
| `DRY_RUN` | If true, no live orders |
| `BACKTEST_DAYS` | Backtest history hint (default 7) |
| `MAX_TRADES_PER_HOUR` | Cap (default 5) |
| `MIN_BALANCE_USDC` | Min balance to trade (default 50) |
| `MAX_GAS_GWEI` | Skip if gas &gt; this (default 50) |

## Logging

- **CSV:** `bot_log.csv` – timestamp, market_id, streak, chainlink_price, poly_odds_reversal, delta, prob, action, simulated_pnl.
- **Console:** Alerts like `BUY DOWN @0.12 - 72% Edge`.

## Safety

- Gas check (&lt; 50 gwei); retry 3× on failures; no trades if balance &lt; 50 USDC.
- No logging of `PRIVATE_KEY` or key material.

## Tests

```bash
python -m pytest tests/ -v
# or
python -m unittest discover -s tests -v
```

- `test_prob_calc.py` – Bayesian and reversal prob.
- `test_api_fetch.py` – Poly API fetch and streak (with mocks).
- `test_order_place.py` – Order placement with stops (dry run).

## Manual edge viewer (no keys)

For a **view-only** dashboard that spots Chainlink/spot vs Poly odds lags and suggests BET UP/DOWN (no wallet or `.env` needed):

```bash
python poly_edge.py
```

- Polls every 10s: live BTC price (Chainlink-style URL or CoinGecko fallback), active 5-min Poly market UP odds and open price.
- Suggests **BET UP** or **BET DOWN** when delta vs implied prob gives an edge &gt; 5% (tunable).
- Example: `BET UP - Delta: +0.4% | Edge: +8% | Odds: 0.82`

**Browser UI:** `pip install streamlit` then:

```bash
streamlit run poly_edge_dash.py
```

**Existing tool (15-min, adaptable to 5-min):** [PolyRec](https://github.com/txbabaxyz/polyrec) — terminal dashboard with Chainlink + Poly orderbook and 70+ indicators. Clone, `pip install -r requirements.txt`, run `python dash.py`. To target 5-min markets, change the slug/query filter in the script (e.g. include `5-minute` or `btc-updown-5m-` in the market query).

## Extensibility

- **RSI / talib:** Add a module that computes RSI from a price series and feed it into the signal (e.g. require RSI &lt; 30 for reversal UP or RSI &gt; 70 for reversal DOWN) in `strategy.py` or `bot.py` without changing core loop.

## Module Overview

| Module | Purpose |
|--------|---------|
| `config.py` | Load .env and config constants |
| `poly_api.py` | Gamma API: markets, historicals, `get_streak()`, liquidity, odds |
| `chainlink_ws.py` | Chainlink listener, `get_latest_price()` |
| `strategy.py` | `bayesian_update()`, reversal prob, delta, advantage, filters |
| `trading.py` | `place_order_with_stops()`, approve USDC.e, gas/balance checks |
| `bot.py` | `main()`, 5s loop, backtest |
| `poly_edge.py` | Manual edge viewer: spot vs Poly odds, BET UP/DOWN suggestions (no keys) |
