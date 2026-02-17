"""
TradingView-style chart: Green/red direction, prominent current/input hlines (thick white current, dashed colored input), smooth spline, minimal clutter.
1s updates (optimized: single thick spline + sparse overlays). Overbought/oversold price zones (±1% mean). Prob for input/horizon.
Default zoom: Y-axis ±50 from mean for closer view on BTC moves.
Run: streamlit run poly_edge_dash.py
"""
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy import stats

BINANCE_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
CHAINLINK_POLYGON_ADDRESS = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
AGGREGATOR_ABI = [{"inputs": [], "name": "latestRoundData", "outputs": [{"name": "roundId", "type": "uint80"}, {"name": "answer", "type": "int256"}, {"name": "startedAt", "type": "uint256"}, {"name": "updatedAt", "type": "uint256"}, {"name": "answeredInRound", "type": "uint80"}], "stateMutability": "view", "type": "function"}]
POLL_SEC = 1  # 1s updates
HISTORY_POINTS = 300  # ~5min at 1s
GREEN = "#26a69a"
RED = "#f23645"
WHITE = "#ffffff"
LIGHT_GRAY = "#363a45"
ZOOM_RANGE = 50  # ±50 $ on y-axis default for closer zoom


def get_chainlink_polygon_btc():
    rpc = os.environ.get("RPC_URL", "https://polygon-rpc.com")
    try:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(rpc))
        if w3.is_connected():
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(CHAINLINK_POLYGON_ADDRESS),
                abi=AGGREGATOR_ABI,
            )
            _, answer, _, _, _ = contract.functions.latestRoundData().call()
            return float(answer) / 1e8
    except Exception:
        pass
    return None


def get_binance_btc():
    try:
        r = requests.get(BINANCE_URL, timeout=3)
        if r.status_code == 200:
            return float(r.json().get("price", 0))
    except Exception:
        pass
    return None


def get_price():
    price = get_chainlink_polygon_btc()
    if price is None:
        price = get_binance_btc()
    return price


def gbm_prob_above(S0, K, T, mu, sigma):
    if sigma == 0:
        return 100 if S0 >= K else 0
    d2 = (np.log(S0 / K) + (mu - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return stats.norm.cdf(d2) * 100


def predict_prob(df, input_price, horizon_secs):
    if len(df) < 10 or input_price <= 0:
        return 50, "Low data or invalid input"

    recent = df.tail(10)['price'].values
    returns = np.log(recent[1:] / recent[:-1])
    dt = 1 / 3600  # per second to hour
    mu = np.mean(returns) / dt if len(returns) > 0 else -0.001  # default bear if low data
    sigma = np.std(returns) / np.sqrt(dt) if len(returns) > 0 else 0.005  # default vol
    T = horizon_secs / 3600  # to hours
    S0 = recent[-1]

    prob = gbm_prob_above(S0, input_price, T, mu, sigma)
    
    # Conf from linregress for R2/std
    times = np.arange(len(recent))
    slope, _, r_value, _, _ = stats.linregress(times, recent)
    std = np.std(recent)
    r2 = r_value**2
    conf = f"R²={r2:.2f} | Std={std:.2f}"
    
    return prob, conf


def make_chart(df: pd.DataFrame, input_price: float, current_price: float | None, height: int, title: str):
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    t = df["time"].tolist()
    p = df["price"].tolist()
    
    # Single thick smooth spline for price (white, high contrast)
    fig.add_trace(go.Scatter(
        x=t, y=p, mode='lines', name='Price',
        line=dict(color=WHITE, width=4, shape='spline'),  # Thicker white for max visibility
        hovertemplate='<b>Time</b>: %{x}<br><b>Price</b>: $%{y:,.2f}<extra></extra>'
    ))
    
    # Sparse direction overlay (every 5th, thin for subtle color)
    for i in range(1, len(p), 5):
        if i < len(p) - 1:
            color = GREEN if p[i] >= p[i - 1] else RED
            fig.add_trace(go.Scatter(
                x=[t[i-1], t[i]], y=[p[i-1], p[i]],
                mode='lines', line=dict(color=color, width=1.5, shape='spline'),
                showlegend=False, hoverinfo='skip'
            ))
    
    # Prominent current price hline (thick white, bold right label)
    if current_price is not None:
        fig.add_hline(
            y=current_price,
            line_dash="solid",
            line_color=WHITE,
            line_width=4,  # Extra thick
            annotation_text=f"Current: {current_price:,.2f}",
            annotation_font=dict(color=WHITE, size=16, family="Arial Bold"),
            annotation_position="right",
        )
    
    # Input price hline (dashed, colored red below current, green above, bold left label)
    if input_price > 0:
        hline_color = WHITE
        if current_price is not None:
            hline_color = RED if input_price < current_price else GREEN if input_price > current_price else WHITE
        fig.add_hline(
            y=input_price,
            line_dash="dash",
            line_color=hline_color,
            line_width=3,
            annotation_text=f"Input: {input_price:,.0f}",
            annotation_font_color=hline_color,
            annotation_font=dict(size=16, family="Arial Bold"),
            annotation_position="left",
            annotation_xanchor="left",
        )
    
    # Overbought/oversold price zones (±1% from mean, thicker dotted lines, bold labels)
    mean_price = np.mean(p)
    overbought = mean_price * 1.01  # +1%
    oversold = mean_price * 0.99  # -1%
    fig.add_hline(y=overbought, line_dash="dot", line_color=RED, line_width=2, annotation_text="Overbought (Sell)", annotation_position="top right", annotation_font_color=RED, annotation_font=dict(size=14, family="Arial Bold"))
    fig.add_hline(y=oversold, line_dash="dot", line_color=GREEN, line_width=2, annotation_text="Oversold (Buy)", annotation_position="bottom right", annotation_font_color=GREEN, annotation_font=dict(size=14, family="Arial Bold"))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        font=dict(family="Helvetica Neue, Helvetica, Arial", color="#d1d4dc", size=12),  # Larger base font
        title=dict(text=title, font=dict(size=14, color="#d1d4dc"), x=0.005, xanchor="left", y=0.98),  # Bold title
        xaxis=dict(
            title="Time",
            showgrid=True,
            gridcolor=LIGHT_GRAY,  # Lighter grid for contrast
            zeroline=False,
            showline=True,
            linecolor=WHITE,
            linewidth=1.5,
            tickfont=dict(color="#d1d4dc", size=12),  # Larger ticks
            tickformat="%H:%M:%S",
            rangeslider_visible=False,
        ),
        yaxis=dict(
            title="Price ($)",
            side="left",
            showgrid=True,
            gridcolor=LIGHT_GRAY,
            zeroline=False,
            showline=True,
            linecolor=WHITE,
            linewidth=1.5,
            tickfont=dict(color="#d1d4dc", size=12),
            tickformat=",.0f",
            separatethousands=True,
            range=[mean_price - ZOOM_RANGE, mean_price + ZOOM_RANGE] # Default zoom ±50
        ),
        margin=dict(l=80, r=80, t=50, b=50), # Extra margin for bold labels
        height=height,
        showlegend=False,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(showspikes=True, spikecolor=WHITE, spikethickness=2, spikemode="across")
    fig.update_yaxes(showspikes=True, spikecolor=WHITE, spikethickness=2, spikemode="across", fixedrange=False)
    return fig


st.set_page_config(layout="wide", page_title="BTC Chainlink Chart")
st.markdown("""
<style>
.stApp { background-color: #131722; }
.stSidebar { background-color: #1e222d; }
.stSidebar .stNumberInput label { color: #d1d4dc; font-size: 12px; }
[data-testid="stMetricValue"] { font-size: 1.35rem; color: #d1d4dc; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Header: pair + price + change
price = get_chainlink_polygon_btc()
now = datetime.now(timezone.utc)
if "chart_df" not in st.session_state:
    st.session_state.chart_df = pd.DataFrame(columns=["time", "price"])
new_row = pd.DataFrame({"time": [now], "price": [price if price is not None else np.nan]})
st.session_state.chart_df = pd.concat([st.session_state.chart_df, new_row], ignore_index=True)
st.session_state.chart_df = st.session_state.chart_df.tail(HISTORY_POINTS) # Persist full history
df_all = st.session_state.chart_df.dropna(subset=["price"])
if df_all.empty:
    df_all = pd.DataFrame({"time": [now], "price": [price or 0]})

prev = df_all["price"].iloc[-2] if len(df_all) >= 2 else (price or 0)
chg = ((price - prev) / prev * 100) if (price is not None and prev and prev != 0) else None
chg_str = f"+{chg:.2f}%" if chg is not None and chg >= 0 else (f"{chg:.2f}%" if chg is not None else "")
chg_color = "#26a69a" if chg is not None and chg >= 0 else "#f23645"

header_col1, header_col2, header_col3 = st.columns([1, 2, 1])
with header_col1:
    st.markdown("<span style='color:#d1d4dc; font-size:1.1rem; font-weight:600;'>BTC/USD</span>", unsafe_allow_html=True)
with header_col2:
    price_disp = f"${price:,.2f}" if price is not None else "—"
    chg_disp = f" <span style='color:{chg_color}; font-size:1rem; font-weight:bold;'>({chg_str})</span>" if chg_str else ""
    st.markdown(f"<span style='color:#d1d4dc; font-size:1.45rem; font-weight:700;'>{price_disp}</span>{chg_disp}", unsafe_allow_html=True)
with header_col3:
    st.markdown("<span style='color:#d1d4dc; font-size:1rem;'>Chainlink · Polygon · 1s</span>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    input_price = st.number_input("Input price (horizontal line)", value=0.0, step=100.0, min_value=0.0, format="%.2f")
    col1, col2 = st.columns(2)
    with col1:
        minutes = st.number_input("Minutes", min_value=0, max_value=5, value=0, step=1)
    with col2:
        seconds = st.number_input("Seconds", min_value=0, max_value=59, value=0, step=1)
    horizon = minutes * 60 + seconds
    horizon = max(1, min(horizon, 300))
    st.caption("Set input >0 for prob calc (colored hline: red below current, green above).")

# Single Chart: Full history with prominent lines + overbought/oversold zones
fig = make_chart(df_all, input_price, price, height=600, title="BTC Price (Smooth, Zoomable)")
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True, "displaylogo": False, "modeBarButtonsToRemove": ["pan2d", "lasso2d"]})

# Prob Calc & Edge (if input set)
if input_price > 0 and len(df_all) >= 10:
    prob, conf = predict_prob(df_all, input_price, horizon)
    st.metric("Probability of future price staying up or above input price is", f"{prob:.0f}%", conf)
    if prob > 50:
        st.success("Edge: BUY UP (Above Input)")
    elif prob < 50:
        st.error("Edge: BUY DOWN (Below Input)")
    else:
        st.info("HOLD - Neutral")
else:
    st.info("Set input price + wait 10s for prob.")

st.caption(f"Last: ${price:,.2f} · Updates every 1s · Zoom/pan freely (history preserved)" if price else "Waiting for price…")
time.sleep(POLL_SEC)
st.rerun()