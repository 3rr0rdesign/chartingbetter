import streamlit as st
import os
from datetime import datetime, timezone

BG = "#131722"

# ---- Optional: Chainlink reference (slow, cached) ----
CHAINLINK_POLYGON_ADDRESS = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
AGGREGATOR_ABI = [{
    "inputs": [],
    "name": "latestRoundData",
    "outputs": [
        {"name": "roundId", "type": "uint80"},
        {"name": "answer", "type": "int256"},
        {"name": "startedAt", "type": "uint256"},
        {"name": "updatedAt", "type": "uint256"},
        {"name": "answeredInRound", "type": "uint80"},
    ],
    "stateMutability": "view",
    "type": "function",
}]


@st.cache_data(ttl=15)
def get_chainlink_polygon_btc():
    rpc = os.environ.get("RPC_URL", "https://polygon-rpc.com")
    try:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 12}))
        if not w3.is_connected():
            return None, None
        c = w3.eth.contract(
            address=Web3.to_checksum_address(CHAINLINK_POLYGON_ADDRESS),
            abi=AGGREGATOR_ABI
        )
        round_id, answer, _, updated_at, _ = c.functions.latestRoundData().call()
        px = float(answer) / 1e8
        ts = datetime.fromtimestamp(int(updated_at), tz=timezone.utc) if int(updated_at) else datetime.now(timezone.utc)
        return px, ts
    except Exception:
        return None, None


# -------------------------
# Streamlit UI (STATIC — no rerun loop)
# -------------------------
st.set_page_config(layout="wide", page_title="Polymarket 5-min — Chainlink Truth")
st.markdown(f"<style>.stApp{{background-color:{BG};}}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Prediction trade")
    input_price = st.number_input("Input price to beat", value=0.0, step=50.0, min_value=0.0, format="%.2f", help="When Truth API is on: ignored; barrier = Chainlink price at 5-min window start (locked automatically).")

    c1, c2 = st.columns(2)
    with c1:
        horizon_min = st.number_input("Minutes", min_value=0, max_value=10, value=1, step=1)
    with c2:
        horizon_sec = st.number_input("Seconds", min_value=0, max_value=59, value=0, step=1)

    horizon_total = int(horizon_min) * 60 + int(horizon_sec)
    horizon_total = max(1, min(horizon_total, 600))

    st.divider()
    window_min = st.slider("Visible window (minutes)", 1, 30, 5)

    # model knobs
    st.subheader("Model knobs")
    lookback = st.slider("Vol lookback cap (seconds)", 60, 900, 240, step=30)
    lam = st.slider("EWMA lambda (vol smoothing)", 0.80, 0.99, 0.92, step=0.01)
    min_trades = st.slider("Min samples before model", 30, 200, 90, step=10)
    drift_cap_bp = st.slider("Drift cap (bp/sec)", 0.0, 5.0, 1.5, step=0.5)

    st.divider()
    st.subheader("Edge filter (be picky)")
    min_pstay = st.slider("Only signal if P(stay) ≥", 0.50, 0.80, 0.58, step=0.01)
    min_dist_sigmas = st.slider("Only signal if |price-input| ≥ (σ√T)*", 0.0, 3.0, 0.6, step=0.1)
    buffer_bp = st.slider("Barrier buffer (bp)", 0, 20, 5, step=1)

    st.divider()
    show_projection = st.checkbox("Show future projection (mean + band)", True)

    st.divider()
    use_truth_api = st.checkbox("Use Truth Price API (Chainlink Data Streams)", True)
    truth_api_url = st.text_input(
        "Truth API URL",
        value=os.environ.get("TRUTH_API_URL", "http://localhost:8787"),
        help="Go truth server (API key): http://localhost:8787. Scraper (delayed): http://localhost:8788.",
    ).rstrip("/")
    with st.expander("No API key? Get same price as Chainlink’s site"):
        st.markdown(
            "**The Chainlink Data Streams website is delayed by a few seconds too.** "
            "So scraping it or watching the site doesn't give you the exact price Polymarket uses at resolution — only the **Data Streams API** does."
        )
        st.markdown(
            "**Without API access:**  \n"
            "• **Scraper** (`python chainlink_scraper.py`) — same number as data.chain.link, but that page is already delayed.  \n"
            "• **Binance** — real-time but a different source; can flip at the bell.  \n"
            "Neither matches resolution timing. **Only way to get the real resolution price:** request [Chainlink Data Streams](https://docs.chain.link/data-streams) API access and run the Go truth server."
        )

    st.divider()
    cl_px, cl_ts = get_chainlink_polygon_btc()
    if cl_px is not None:
        st.caption("Polygon Data Feed (NOT Polymarket resolution)")
        st.metric("Polygon BTC/USD feed", f"${cl_px:,.2f}")
        st.caption(f"Updated: {cl_ts.strftime('%H:%M:%S UTC') if cl_ts else '—'}")
    else:
        st.caption("Polygon feed: unavailable (RPC/web3)")

st.markdown("### BTC/USD — Polymarket 5‑min helper")
st.caption(
    "Truth API on → price = Chainlink Data Streams (same as Polymarket resolution). "
    "Truth API off → price = Binance (proxy only; resolution can differ at the bell)."
)


# -------------------------
# Browser chart (Plotly.js + Truth API or Binance WS)
# -------------------------
input_price_js = float(input_price)
horizon_js = int(horizon_total)
window_sec_js = int(window_min) * 60
lookback_js = int(lookback)
show_proj_js = "true" if show_projection else "false"
use_truth_js = "true" if use_truth_api else "false"
truth_api_url_js = repr(truth_api_url)  # safe for JS string

lam_js = float(lam)
min_trades_js = int(min_trades)
drift_cap = float(drift_cap_bp) / 10000.0  # bp/sec -> frac/sec
min_pstay_js = float(min_pstay)
min_dist_sigmas_js = float(min_dist_sigmas)
buffer_bp_js = int(buffer_bp)

html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body {{ margin:0; background:{BG}; font-family: Arial, sans-serif; }}
    #wrap {{ width:100%; }}
    #topbar {{
      display:flex; gap:12px; align-items:baseline; flex-wrap:wrap;
      padding: 8px 10px; color:#d1d4dc; font-size:14px;
    }}
    #price {{ font-size:22px; font-weight:700; }}
    .pill {{
      display:inline-block; padding:4px 8px; border-radius:8px;
      background:#1e222d; border:1px solid #2a2e39;
      font-size:12px;
    }}
    #chart {{ width:100%; height:720px; }}
  </style>
</head>
<body>
<div id="wrap">
  <div id="topbar">
    <div id="price">—</div>
    <div id="change" class="pill">—</div>
    <div id="prob" class="pill">Prob: —</div>
    <div id="edge" class="pill">Edge: —</div>
    <div id="resolution" class="pill">Start: — | Now: — | If ended: — | Countdown: —</div>
    <div id="lag" class="pill">Truth age: — | Binance vs Truth: —</div>
    <div id="meta" class="pill">—</div>
  </div>
  <div id="chart"></div>
</div>

<script>
  // ---- Settings from Streamlit ----
  const INPUT_PRICE = {input_price_js};
  const HORIZON_SEC = {horizon_js};
  const WINDOW_SEC  = {window_sec_js};
  const LOOKBACK_UI = {lookback_js};
  const SHOW_PROJ   = {show_proj_js};
  const USE_TRUTH   = {use_truth_js};
  const TRUTH_API   = {truth_api_url_js};
  const ROUND_SEC   = 300;  // 5-min Polymarket window

  if (!USE_TRUTH) {{
    document.getElementById('resolution').style.display = 'none';
    document.getElementById('lag').style.display = 'none';
  }}

  const LAMBDA      = {lam_js};
  const MIN_SAMPLES = {min_trades_js};
  const DRIFT_CAP   = {drift_cap};              // frac/sec (log drift cap too)
  const MIN_PSTAY   = {min_pstay_js};
  const MIN_DIST_S  = {min_dist_sigmas_js};
  const BUFFER_BP   = {buffer_bp_js};

  const GREEN = "#26a69a";
  const RED   = "#f23645";
  const WHITE = "#ffffff";
  const GRID  = "#363a45";
  const BG    = "{BG}";

  // ---- State ----
  let lastSec   = null;
  let lastPrice = null;
  let prevPrice = null;
  let roundStartPrice = null;  // locked at 5-min window start (truth)
  let roundStartTime  = null;  // Date of window start
  let roundEndTime    = null;  // Date of window end
  let lastTruthTs     = null;  // observation timestamp (seconds) for report age
  let binanceLast     = null;  // for lag warning when using truth

  const times = [];
  const prices = [];
  const logReturns = [];

  // ---- Plotly figure (constant traces) ----
  const data = [
    {{
      x: [], y: [],
      type: 'scatter', mode: 'lines',
      name: USE_TRUTH ? 'Chainlink (truth)' : 'Binance (1s)',
      line: {{color: WHITE, width: 3, shape: 'spline'}},
      hovertemplate: '%{{x}}<br>$%{{y:,.2f}}<extra></extra>'
    }},
    {{
      x: [], y: [],
      type: 'scatter', mode: 'markers',
      name: 'Last',
      marker: {{size: 9, color: WHITE}},
      hoverinfo: 'skip',
      showlegend: false
    }},
    {{
      x: [], y: [],
      type: 'scatter', mode: 'lines',
      name: 'Input',
      line: {{width: 2, dash: 'dash', color: WHITE}},
      hoverinfo: 'skip',
      visible: true
    }},
    {{
      x: [], y: [],
      type: 'scatter', mode: 'lines',
      name: 'Proj (mean)',
      line: {{width: 2, dash: 'dot', color: '#9aa4b2'}},
      hoverinfo: 'skip',
      visible: SHOW_PROJ
    }},
    {{
      x: [], y: [],
      type: 'scatter', mode: 'lines',
      name: '+1σ',
      line: {{width: 1, color: '#6b7280'}},
      hoverinfo: 'skip',
      visible: SHOW_PROJ
    }},
    {{
      x: [], y: [],
      type: 'scatter', mode: 'lines',
      name: '-1σ',
      line: {{width: 1, color: '#6b7280'}},
      hoverinfo: 'skip',
      fill: 'tonexty',
      fillcolor: 'rgba(107,114,128,0.15)',
      visible: SHOW_PROJ
    }},
  ];

  const layout = {{
    paper_bgcolor: BG,
    plot_bgcolor: BG,
    margin: {{l: 55, r: 20, t: 10, b: 40}},
    height: 720,
    showlegend: true,
    legend: {{orientation: 'h', y: 1.05, x: 0}},
    hovermode: 'x unified',
    uirevision: 'btc-chart',
    xaxis: {{
      showgrid: true,
      gridcolor: GRID,
      zeroline: false,
      showline: true,
      linecolor: WHITE,
      tickformat: "%H:%M:%S",
      rangeslider: {{visible: false}}
    }},
    yaxis: {{
      showgrid: true,
      gridcolor: GRID,
      zeroline: false,
      showline: true,
      linecolor: WHITE,
      tickformat: ",.0f"
    }},
  }};

  const config = {{
    scrollZoom: true,
    displayModeBar: true,
    displaylogo: false,
  }};

  Plotly.newPlot('chart', data, layout, config);

  // ---- Helpers ----
  function fmt2(x) {{
    return x === null ? "—" : x.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
  }}

  function clampWindow() {{
    while (times.length > WINDOW_SEC) {{
      times.shift();
      prices.shift();
    }}
  }}

  function price18ToFloat(s) {{
    if (s == null || s === undefined) return null;
    const bi = BigInt(s);
    const whole = Number(bi / 1000000000000000000n);
    const frac = Number(bi % 1000000000000000000n) / 1e18;
    return whole + frac;
  }}

  async function fetchTruth() {{
    const r = await fetch(TRUTH_API + "/truth", {{ cache: "no-store" }});
    if (!r.ok) throw new Error("truth fetch failed");
    return await r.json();
  }}

  function get5MinWindow(now) {{
    const t = now.getTime();
    const bucket = Math.floor(t / 1000 / ROUND_SEC) * ROUND_SEC;
    const start = new Date(bucket * 1000);
    const end = new Date((bucket + ROUND_SEC) * 1000);
    return {{ start, end }};
  }}

  function updateResolutionUI(priceNow) {{
    const el = document.getElementById('resolution');
    if (roundStartPrice == null || priceNow == null) {{
      el.textContent = "Start: — | Now: — | If ended: — | Countdown: —";
      return;
    }}
    const win = get5MinWindow(new Date());
    const delta = priceNow - roundStartPrice;
    const ifEnded = priceNow >= roundStartPrice ? "UP" : "DOWN";
    const remaining = Math.max(0, Math.floor((win.end - new Date()) / 1000));
    const countdown = remaining + "s";
    el.innerHTML = "Start: $"+fmt2(roundStartPrice)+" | Now: $"+fmt2(priceNow)+" | &Delta; $"+fmt2(delta)+" | If ended now: <b>"+ifEnded+"</b> | Countdown: "+countdown;
    el.style.borderColor = ifEnded === "UP" ? GREEN : RED;
  }}

  function updateLagUI(truthPrice, truthTsSec, binancePx) {{
    const el = document.getElementById('lag');
    const nowSec = Math.floor(Date.now() / 1000);
    const age = truthTsSec != null ? (nowSec - truthTsSec) + "s" : "—";
    let binanceVs = "—";
    if (binancePx != null && truthPrice != null) {{
      const d = binancePx - truthPrice;
      binanceVs = "$"+fmt2(d)+" vs Truth";
    }}
    el.textContent = "Truth report age: "+age+" | Binance vs Truth: "+binanceVs;
  }}

  function updateTopbar(price) {{
    const priceEl = document.getElementById('price');
    const chgEl   = document.getElementById('change');

    priceEl.textContent = "$" + fmt2(price);

    if (prevPrice !== null && prevPrice > 0) {{
      const pct = ((price - prevPrice)/prevPrice) * 100.0;
      chgEl.textContent = (pct >= 0 ? "+" : "") + pct.toFixed(3) + "%";
      chgEl.style.borderColor = (pct >= 0) ? GREEN : RED;
      chgEl.style.color = (pct >= 0) ? GREEN : RED;
    }}
  }}

  function getEffectiveInputPrice() {{
    if (USE_TRUTH && roundStartPrice != null && roundStartPrice > 0) return roundStartPrice;
    return INPUT_PRICE;
  }}

  function updateInputLine(x0, x1, priceNow) {{
    const K = getEffectiveInputPrice();
    if (!K || K <= 0) {{
      Plotly.restyle('chart', {{x: [[]], y: [[]], visible: false}}, [2]);
      return;
    }}
    const color = (priceNow !== null)
      ? (K < priceNow ? GREEN : (K > priceNow ? RED : WHITE))
      : WHITE;

    Plotly.restyle('chart', {{
      x: [[x0, x1]],
      y: [[K, K]],
      'line.color': color,
      visible: true
    }}, [2]);
  }}

  function updateLastMarker(t, p) {{
    Plotly.restyle('chart', {{x: [[t]], y: [[p]]}}, [1]);
  }}

  function pushReturn(p) {{
    if (prices.length >= 2) {{
      const p0 = prices[prices.length - 2];
      const r  = Math.log(p / p0);
      logReturns.push(r);

      // dynamic lookback: prefer ~2× horizon, bounded by LOOKBACK_UI
      const dyn = Math.min(Math.max(120, 2*HORIZON_SEC), LOOKBACK_UI);
      while (logReturns.length > dyn) logReturns.shift();
    }}
  }}

  // --- Normal CDF ---
  function erf(x) {{
    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);
    const a1=0.254829592, a2=-0.284496736, a3=1.421413741, a4=-1.453152027, a5=1.061405429, p=0.3275911;
    const t = 1.0/(1.0+p*x);
    const y = 1.0-((((a5*t+a4)*t+a3)*t+a2)*t+a1)*t*Math.exp(-x*x);
    return sign*y;
  }}
  function normCdf(z) {{
    return 0.5 * (1 + erf(z / Math.sqrt(2)));
  }}

  // --- Robust stats ---
  function median(arr) {{
    const a = arr.slice().sort((x,y)=>x-y);
    const n = a.length;
    if (n===0) return 0;
    return (n%2===1) ? a[(n-1)/2] : 0.5*(a[n/2-1]+a[n/2]);
  }}
  function mad(arr) {{
    const m = median(arr);
    const dev = arr.map(x=>Math.abs(x-m));
    return median(dev);
  }}
  function winsorize(arr, k=6.0) {{
    const m = median(arr);
    const s = Math.max(1e-9, 1.4826*mad(arr));
    const lo = m - k*s, hi = m + k*s;
    return arr.map(x => Math.min(hi, Math.max(lo, x)));
  }}
  function ewmaSigma(returns) {{
    let v = 0;
    for (const r of returns) {{
      v = LAMBDA*v + (1-LAMBDA)*r*r;
    }}
    return Math.sqrt(Math.max(v, 1e-12));
  }}

  // ---- Barrier survival (GBM in log-space, reflection principle) ----
  // Model: d ln S = a dt + sigma dW
  function probStayAboveGBM(S0, H, T, a, sigma) {{
    if (T <= 0) return (S0 > H) ? 1.0 : 0.0;
    if (sigma <= 0) return (S0 > H) ? 1.0 : 0.0;
    if (H <= 0) return 1.0;
    if (S0 <= H) return 0.0;

    const x = Math.log(S0 / H);
    const srt = sigma * Math.sqrt(T);

    const d1 = (x + a*T) / srt;
    const d2 = (-x + a*T) / srt;

    const expo = Math.exp(-2.0 * a * x / (sigma*sigma));
    const p = normCdf(d1) - expo * normCdf(d2);
    return Math.max(0, Math.min(1, p));
  }}

  function probStayBelowGBM(S0, H, T, a, sigma) {{
    if (T <= 0) return (S0 < H) ? 1.0 : 0.0;
    if (sigma <= 0) return (S0 < H) ? 1.0 : 0.0;
    if (H <= 0) return 0.0;
    if (S0 >= H) return 0.0;

    const x = Math.log(H / S0);
    const srt = sigma * Math.sqrt(T);

    const d1 = (x - a*T) / srt;
    const d2 = (-x - a*T) / srt;

    const expo = Math.exp( 2.0 * a * x / (sigma*sigma));
    const p = normCdf(d1) - expo * normCdf(d2);
    return Math.max(0, Math.min(1, p));
  }}

  function probFinishAboveGBM(S0, K, T, a, sigma) {{
    if (T <= 0) return (S0 > K) ? 1.0 : 0.0;
    if (sigma <= 0) return (S0 > K) ? 1.0 : 0.0;
    if (K <= 0) return 1.0;

    const m = Math.log(S0) + a*T;
    const s = Math.max(1e-12, sigma*Math.sqrt(T));
    const z = (Math.log(K) - m) / s;
    const p = 1 - normCdf(z);
    return Math.max(0, Math.min(1, p));
  }}

  // ---- Core model: returns -> (a, sigma) ----
  function estimateParams() {{
    if (logReturns.length < MIN_SAMPLES) return null;

    const r0 = winsorize(logReturns);

    // drift: median of returns (tiny), cap it
    const aRaw = median(r0);
    const a = Math.max(-DRIFT_CAP, Math.min(DRIFT_CAP, aRaw));

    // vol: EWMA
    let sigma = ewmaSigma(r0);

    // stress vol if tails are heavy (simple proxy)
    let m4 = 0;
    for (const r of r0) m4 += Math.pow(r,4);
    m4 /= r0.length;
    const kurtProxy = m4 / Math.max(1e-12, Math.pow(sigma,4));
    if (kurtProxy > 10) sigma *= 1.30;
    else if (kurtProxy > 6) sigma *= 1.15;

    sigma = Math.max(sigma, 1e-6);

    return {{a, sigma, n: r0.length}};
  }}

  // ---- UI: probability + "edge filter" ----
  function updateProbUI() {{
    const probEl = document.getElementById('prob');
    const edgeEl = document.getElementById('edge');
    const Kraw = getEffectiveInputPrice();

    if (!Kraw || Kraw <= 0 || lastPrice === null) {{
      probEl.textContent = "Prob: set input" + (USE_TRUTH ? " or wait for round start" : "");
      edgeEl.textContent = "Edge: —";
      return;
    }}

    const est = estimateParams();
    if (!est) {{
      probEl.textContent = "Prob: collecting data…";
      edgeEl.textContent = "Edge: —";
      return;
    }}

    const S0 = lastPrice;
    const T  = HORIZON_SEC;

    // buffer: slightly widen barrier to avoid micro-noise (bp)
    const buf = Math.max(1.0, S0 * (BUFFER_BP / 10000.0));

    let modeTxt = "";
    let pStay = 0.0;
    let pFinish = 0.0;
    let edge = "NONE";

    if (Kraw < S0) {{
      const K = Kraw + buf; // harder to stay above
      pStay   = probStayAboveGBM(S0, K, T, est.a, est.sigma);
      pFinish = probFinishAboveGBM(S0, K, T, est.a, est.sigma);
      modeTxt = "Stay ABOVE";
    }} else if (Kraw > S0) {{
      const K = Kraw - buf; // harder to stay below (upper barrier)
      pStay   = probStayBelowGBM(S0, K, T, est.a, est.sigma);
      pFinish = 1.0 - probFinishAboveGBM(S0, K, T, est.a, est.sigma); // P(S_T < K)
      modeTxt = "Stay BELOW";
    }} else {{
      modeTxt = "At barrier";
      pStay = 0.0;
      pFinish = 0.5;
    }}

    // distance filter: avoid taking bets when too close to barrier vs vol
    const dist = Math.abs(S0 - Kraw);
    const volStep = est.sigma * Math.sqrt(T) * S0; // approx $ move scale in T
    const distSig = volStep > 0 ? (dist / volStep) : 0;

    // Decide if we show a "signal"
    const passes = (pStay >= MIN_PSTAY) && (distSig >= MIN_DIST_S);

    if (passes) {{
      edge = (Kraw < S0) ? "UP" : (Kraw > S0) ? "DOWN" : "NONE";
      edgeEl.style.borderColor = (edge === "UP") ? GREEN : (edge === "DOWN") ? RED : "#9aa4b2";
      edgeEl.style.color       = (edge === "UP") ? GREEN : (edge === "DOWN") ? RED : "#d1d4dc";
    }} else {{
      edge = "NO-TRADE";
      edgeEl.style.borderColor = "#9aa4b2";
      edgeEl.style.color       = "#d1d4dc";
    }}

    probEl.textContent =
      modeTxt + " @ " + T + "s: " + Math.round(pStay*100) + "% | Finish: " + Math.round(pFinish*100) +
      "% | dist: " + distSig.toFixed(2) + "σ";

    edgeEl.textContent =
      "Edge: " + edge + " | σ(1s): " + (est.sigma*10000).toFixed(2) + " bp | n=" + est.n;

    probEl.style.borderColor = passes ? ((Kraw < S0) ? GREEN : (Kraw > S0) ? RED : "#9aa4b2") : "#9aa4b2";
    probEl.style.color       = passes ? ((Kraw < S0) ? GREEN : (Kraw > S0) ? RED : "#d1d4dc") : "#d1d4dc";
  }}

  function updateProjection() {{
    if (!SHOW_PROJ || lastPrice === null || times.length === 0) {{
      Plotly.restyle('chart', {{x:[[]], y:[[]], visible:false}}, [3]);
      Plotly.restyle('chart', {{x:[[]], y:[[]], visible:false}}, [4]);
      Plotly.restyle('chart', {{x:[[]], y:[[]], visible:false}}, [5]);
      return;
    }}

    const est = estimateParams();
    if (!est) {{
      Plotly.restyle('chart', {{x:[[]], y:[[]], visible:false}}, [3]);
      Plotly.restyle('chart', {{x:[[]], y:[[]], visible:false}}, [4]);
      Plotly.restyle('chart', {{x:[[]], y:[[]], visible:false}}, [5]);
      return;
    }}

    const base = new Date(times[times.length - 1].getTime());
    const x = [];
    const yMean = [];
    const yUp = [];
    const yLo = [];
    const S0 = lastPrice;

    for (let i=1; i<=HORIZON_SEC; i++) {{
      const ti = new Date(base.getTime() + i*1000);
      x.push(ti);

      // ln S ~ N( ln S0 + a i, sigma^2 i )
      const m = Math.log(S0) + est.a*i;
      const s = est.sigma*Math.sqrt(i);

      // E[S] for lognormal = exp(m + 0.5 s^2)
      const mean = Math.exp(m + 0.5*s*s);
      const up   = Math.exp(m + 1.0*s);
      const lo   = Math.exp(m - 1.0*s);

      yMean.push(mean);
      yUp.push(up);
      yLo.push(lo);
    }}

    Plotly.restyle('chart', {{x:[x], y:[yMean], visible:true}}, [3]);
    Plotly.restyle('chart', {{x:[x], y:[yUp],   visible:true}}, [4]);
    Plotly.restyle('chart', {{x:[x], y:[yLo],   visible:true}}, [5]);
  }}

  function processPriceTick(ts, price, fromTruth, truthTsSec) {{
    const secKey = ts.toISOString().slice(0,19);
    if (lastSec === null || secKey > lastSec) {{
      lastSec = secKey;
      prevPrice = lastPrice;
      lastPrice = price;

      if (fromTruth) {{
        lastTruthTs = truthTsSec;
        const win = get5MinWindow(ts);
        if (roundStartTime === null || roundStartTime.getTime() !== win.start.getTime()) {{
          roundStartPrice = price;
          roundStartTime = win.start;
          roundEndTime = win.end;
        }}
      }}

      times.push(ts);
      prices.push(price);
      clampWindow();
      pushReturn(price);

      Plotly.extendTraces('chart', {{ x: [[ts]], y: [[price]] }}, [0], WINDOW_SEC);
      updateLastMarker(ts, price);
      if (times.length >= 2) {{
        updateInputLine(times[0], times[times.length-1], price);
      }}
      updateTopbar(price);
      updateProbUI();
      updateProjection();
      if (USE_TRUTH) {{
        updateResolutionUI(price);
        updateLagUI(price, lastTruthTs, binanceLast);
      }}
    }}
  }}

  let truthTimer = null;
  async function truthLoop() {{
    try {{
      const t = await fetchTruth();
      const price = price18ToFloat(t.benchmarkPrice18);
      if (price == null || price <= 0) {{ truthTimer = setTimeout(truthLoop, 1000); return; }}
      const ts = new Date(Number(t.observationsTimestamp) * 1000);
      processPriceTick(ts, price, true, t.observationsTimestamp);
      document.getElementById('meta').textContent = "Truth: OK";
    }} catch (e) {{
      document.getElementById('meta').textContent = "Truth: error";
      if (lastPrice != null) {{
        updateResolutionUI(lastPrice);
        updateLagUI(lastPrice, lastTruthTs, binanceLast);
      }}
    }} finally {{
      truthTimer = setTimeout(truthLoop, 1000);
    }}
  }}

  if (USE_TRUTH) {{
    document.getElementById('meta').textContent = "Truth: connecting…";
    truthLoop();
    const wsRef = new WebSocket("wss://stream.binance.com:9443/ws/btcusdt@trade");
    wsRef.onmessage = (ev) => {{
      try {{
        const msg = JSON.parse(ev.data);
        binanceLast = parseFloat(msg.p);
      }} catch (e) {{}}
    }};
  }} else {{
    const ws = new WebSocket("wss://stream.binance.com:9443/ws/btcusdt@trade");
    ws.onopen  = () => {{ document.getElementById('meta').textContent = "WS: connected"; }};
    ws.onerror = () => {{ document.getElementById('meta').textContent = "WS: error"; }};
    ws.onclose = () => {{ document.getElementById('meta').textContent = "WS: closed"; }};
    ws.onmessage = (ev) => {{
      try {{
        const msg = JSON.parse(ev.data);
        const price = parseFloat(msg.p);
        const tms = parseInt(msg.T);
        const ts = new Date(tms);
        processPriceTick(ts, price, false, null);
      }} catch (e) {{}}
    }};
  }}
</script>
</body>
</html>
"""

st.components.v1.html(html, height=770)
