"""
Paper Trade AI — Dashboard

Reads from data/state.json and data/trades.json to display:
- Portfolio overview with key metrics
- Equity performance chart
- AI trade signals with rationale
- Recent trade activity feed
- Portfolio allocation breakdown

Usage:
    streamlit run scripts/dashboard.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Project paths ──────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

_DATA_DIR = _ROOT / "data"
_STATE_FILE = _DATA_DIR / "state.json"
_TRADES_FILE = _DATA_DIR / "trades.json"


# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Paper Trade AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Design System CSS ──────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global Reset ────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #C9D1D9;
    }

    .stApp {
        background-color: #0D1117;
    }

    /* Hide default Streamlit sidebar toggle & header */
    [data-testid="stSidebar"] { display: none; }
    header[data-testid="stHeader"] { background-color: #0D1117; }
    .stDeployButton { display: none; }
    div[data-testid="stToolbar"] { display: none; }

    /* ── Top Navigation Bar ──────────────────────── */
    .top-nav {
        background-color: #161B22;
        border-bottom: 1px solid #21262D;
        padding: 14px 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: -1rem -1rem 1.5rem -1rem;
        border-radius: 0;
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .nav-logo {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #2DD4A0 0%, #1A9F7A 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
    }
    .nav-title {
        font-size: 17px;
        font-weight: 600;
        color: #F0F6FC;
        letter-spacing: -0.3px;
    }
    .nav-links {
        display: flex;
        gap: 28px;
        margin-left: 40px;
    }
    .nav-link {
        font-size: 14px;
        color: #8B949E;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s ease;
    }
    .nav-link.active {
        color: #F0F6FC;
    }
    .nav-right {
        font-size: 18px;
        color: #8B949E;
        cursor: pointer;
    }

    /* ── Dashboard Header ────────────────────────── */
    .dash-header {
        margin-bottom: 24px;
    }
    .dash-title {
        font-size: 28px;
        font-weight: 700;
        color: #F0F6FC;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .dash-subtitle {
        font-size: 14px;
        color: #8B949E;
        margin-top: 4px;
        font-weight: 400;
    }

    /* ── Metric Cards ────────────────────────────── */
    .metric-card {
        background-color: #161B22;
        border: 1px solid #21262D;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
        min-height: 140px;
    }
    .metric-card-label {
        font-size: 13px;
        font-weight: 500;
        color: #8B949E;
        margin-bottom: 12px;
        letter-spacing: 0.2px;
    }
    .metric-card-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 32px;
        font-weight: 700;
        color: #F0F6FC;
        letter-spacing: -1px;
    }
    .metric-card-value.gain { color: #2DD4A0; }
    .metric-card-value.loss { color: #F85149; }
    .metric-card-delta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        margin-top: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .metric-card-delta.gain { color: #2DD4A0; }
    .metric-card-delta.loss { color: #F85149; }
    .metric-card-sub {
        font-size: 12px;
        color: #484F58;
        margin-top: 16px;
        display: flex;
        gap: 24px;
    }
    .metric-card-sub span {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .metric-card-sub .sub-label {
        color: #8B949E;
        font-size: 11px;
    }
    .metric-card-sub .sub-value {
        font-family: 'IBM Plex Mono', monospace;
        color: #C9D1D9;
        font-size: 13px;
        font-weight: 500;
    }

    /* ── Section Headers ─────────────────────────── */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #F0F6FC;
        margin-bottom: 4px;
        letter-spacing: -0.3px;
    }
    .section-sub {
        font-size: 13px;
        color: #8B949E;
        margin-bottom: 16px;
    }

    /* ── Trade Recommendation Cards ──────────────── */
    .trade-card {
        background-color: #161B22;
        border: 1px solid #21262D;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .trade-card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }
    .trade-ticker {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 20px;
        font-weight: 700;
        color: #F0F6FC;
    }
    .badge {
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .badge-buy {
        background-color: rgba(45, 212, 160, 0.15);
        color: #2DD4A0;
        border: 1px solid rgba(45, 212, 160, 0.3);
    }
    .badge-sell {
        background-color: rgba(248, 81, 73, 0.15);
        color: #F85149;
        border: 1px solid rgba(248, 81, 73, 0.3);
    }
    .trade-rationale {
        font-size: 14px;
        color: #8B949E;
        line-height: 1.5;
        margin-top: 8px;
    }
    .trade-meta {
        display: flex;
        gap: 32px;
        margin-top: 16px;
        padding-top: 14px;
        border-top: 1px solid #21262D;
    }
    .trade-meta-item {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .trade-meta-label {
        font-size: 11px;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .trade-meta-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 14px;
        font-weight: 600;
        color: #F0F6FC;
    }
    .trade-meta-value.gain { color: #2DD4A0; }
    .trade-meta-value.loss { color: #F85149; }

    /* ── Activity Feed ───────────────────────────── */
    .activity-card {
        background-color: #161B22;
        border: 1px solid #21262D;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .activity-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }
    .activity-left {
        display: flex;
        gap: 14px;
        align-items: flex-start;
    }
    .activity-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background-color: #21262D;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .activity-info .activity-title {
        font-size: 15px;
        font-weight: 600;
        color: #F0F6FC;
    }
    .activity-info .activity-title .side-label {
        color: #8B949E;
        font-weight: 400;
    }
    .activity-info .activity-detail {
        font-size: 13px;
        color: #8B949E;
        font-family: 'IBM Plex Mono', monospace;
    }
    .activity-info .activity-reason {
        font-size: 13px;
        color: #6E7681;
        margin-top: 8px;
        line-height: 1.4;
    }
    .activity-info .activity-time {
        font-size: 12px;
        color: #484F58;
        margin-top: 6px;
    }
    .activity-pnl {
        text-align: right;
        flex-shrink: 0;
    }
    .activity-pnl .pnl-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 16px;
        font-weight: 600;
    }
    .activity-pnl .pnl-value.gain { color: #2DD4A0; }
    .activity-pnl .pnl-value.loss { color: #F85149; }

    /* ── Info Banner ──────────────────────────────── */
    .info-banner {
        background-color: #161B22;
        border: 1px solid #21262D;
        border-radius: 12px;
        padding: 16px 24px;
        font-size: 13px;
        color: #8B949E;
        text-align: center;
        margin-bottom: 16px;
    }

    /* ── Disclaimer ───────────────────────────────── */
    .disclaimer {
        background-color: #161B22;
        border: 1px solid #21262D;
        border-radius: 12px;
        padding: 16px 24px;
        font-size: 12px;
        color: #484F58;
        line-height: 1.5;
        margin-top: 24px;
    }

    /* ── Plotly overrides ─────────────────────────── */
    .stPlotlyChart { margin-bottom: 0; }

    /* ── Streamlit overrides ─────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #161B22;
        border-radius: 8px;
        padding: 4px;
        border: 1px solid #21262D;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8B949E;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #21262D;
        color: #F0F6FC;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ───────────────────────────────────────────

def load_json(path: Path, default=None):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return default

def load_state() -> dict:
    return load_json(_STATE_FILE, default={
        "equity_history": [],
        "latest_snapshot": {},
        "engine_started_at": None,
    })

def load_trades() -> list[dict]:
    return load_json(_TRADES_FILE, default=[])


# ── Load Data ──────────────────────────────────────────────

state = load_state()
trades = load_trades()
snapshot = state.get("latest_snapshot", {})
equity_history = state.get("equity_history", [])
engine_started = state.get("engine_started_at")


# ── Top Navigation Bar ────────────────────────────────────

st.markdown("""
<div class="top-nav">
    <div style="display:flex; align-items:center;">
        <div class="nav-brand">
            <div class="nav-logo">📈</div>
            <div class="nav-title">Paper Trade AI</div>
        </div>
        <div class="nav-links">
            <span class="nav-link active">Dashboard</span>
            <span class="nav-link">Portfolio</span>
            <span class="nav-link">Analytics</span>
            <span class="nav-link">History</span>
        </div>
    </div>
    <div class="nav-right">⚙</div>
</div>
""", unsafe_allow_html=True)


# ── Dashboard Header ──────────────────────────────────────

st.markdown("""
<div class="dash-header">
    <div class="dash-title">Portfolio Dashboard</div>
    <div class="dash-subtitle">Real-time market analysis and AI-powered trade recommendations</div>
</div>
""", unsafe_allow_html=True)


# ── Top Metric Cards ──────────────────────────────────────

equity = snapshot.get("equity", 0)
cash = snapshot.get("cash", 0)
pnl = snapshot.get("unrealized_pnl", 0)
drawdown_pct = snapshot.get("drawdown_pct", 0)
positions = snapshot.get("positions", 0)
exposure = snapshot.get("exposure_pct", 0)

# Compute day performance (approximation from equity history)
day_change = 0.0
day_change_pct = 0.0
if len(equity_history) >= 2:
    prev_eq = equity_history[-2].get("equity", equity)
    if prev_eq > 0:
        day_change = equity - prev_eq
        day_change_pct = (day_change / prev_eq) * 100

pnl_css = "gain" if pnl >= 0 else "loss"
pnl_sign = "+" if pnl >= 0 else ""
day_css = "gain" if day_change >= 0 else "loss"
day_sign = "+" if day_change >= 0 else ""

c1, c2, c3 = st.columns([2, 1, 1])

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">Total Portfolio Value</div>
        <div class="metric-card-value">${equity:,.2f}</div>
        <div class="metric-card-delta {day_css}">
            ↗ {day_sign}${abs(day_change):,.2f} ({day_sign}{day_change_pct:.2f}%) Today
        </div>
        <div class="metric-card-sub">
            <span>
                <span class="sub-label">Positions</span>
                <span class="sub-value">{positions}</span>
            </span>
            <span>
                <span class="sub-label">Exposure</span>
                <span class="sub-value">{exposure:.1f}%</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">Day Performance</div>
        <div class="metric-card-value {day_css}">{day_sign}{day_change_pct:.1f}%</div>
        <div class="metric-card-delta {day_css}" style="margin-top:4px;">
            ${abs(day_change):,.2f} {'gain' if day_change >= 0 else 'loss'}
        </div>
        <div class="metric-card-sub" style="margin-top:12px;">
            <span>
                <span class="sub-label">Drawdown</span>
                <span class="sub-value">{drawdown_pct:.2f}%</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    buying_power = cash * 2 if cash > 0 else 0
    cash_pct = (cash / equity * 100) if equity > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">Available Cash</div>
        <div class="metric-card-value">${cash:,.0f}</div>
        <div class="metric-card-delta" style="color:#8B949E;">
            {cash_pct:.1f}% of portfolio
        </div>
        <div class="metric-card-sub" style="margin-top:12px;">
            <span>
                <span class="sub-label">Buying Power</span>
                <span class="sub-value">${buying_power:,.0f}</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Equity Performance Chart ──────────────────────────────

st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)

if equity_history:
    df_eq = pd.DataFrame(equity_history)
    df_eq["datetime"] = pd.to_datetime(df_eq["timestamp"], unit="s", utc=True)

    tab_eq, tab_dd = st.tabs(["Equity Performance", "Drawdown Analysis"])

    with tab_eq:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_eq["datetime"], y=df_eq["equity"],
            mode="lines", name="Equity",
            line=dict(color="#2DD4A0", width=2),
            fill="tozeroy",
            fillcolor="rgba(45, 212, 160, 0.06)",
        ))
        fig.add_trace(go.Scatter(
            x=df_eq["datetime"], y=df_eq["cash"],
            mode="lines", name="Cash",
            line=dict(color="#484F58", width=1.5, dash="dot"),
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0D1117",
            margin=dict(l=50, r=20, t=10, b=40),
            height=350,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(family="Inter", size=12, color="#8B949E"),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis=dict(gridcolor="#21262D", zerolinecolor="#21262D",
                       tickfont=dict(family="IBM Plex Mono", size=11, color="#484F58")),
            yaxis=dict(gridcolor="#21262D", zerolinecolor="#21262D", title="USD",
                       title_font=dict(family="Inter", size=12, color="#484F58"),
                       tickfont=dict(family="IBM Plex Mono", size=11, color="#484F58"),
                       tickprefix="$"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab_dd:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=df_eq["datetime"], y=df_eq["drawdown_pct"].fillna(0),
            mode="lines", name="Drawdown",
            line=dict(color="#F85149", width=2),
            fill="tozeroy",
            fillcolor="rgba(248, 81, 73, 0.08)",
        ))
        fig_dd.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0D1117",
            margin=dict(l=50, r=20, t=10, b=40),
            height=350,
            xaxis=dict(gridcolor="#21262D", zerolinecolor="#21262D",
                       tickfont=dict(family="IBM Plex Mono", size=11, color="#484F58")),
            yaxis=dict(gridcolor="#21262D", zerolinecolor="#21262D", title="Drawdown %",
                       title_font=dict(family="Inter", size=12, color="#484F58"),
                       tickfont=dict(family="IBM Plex Mono", size=11, color="#484F58"),
                       ticksuffix="%"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_dd, use_container_width=True, config={"displayModeBar": False})
else:
    st.markdown('<div class="info-banner">Awaiting equity data. Charts will render once the trading engine begins execution.</div>', unsafe_allow_html=True)


# ── AI Trade Signals + Recent Activity ────────────────────

st.markdown('<div style="margin-top:24px;"></div>', unsafe_allow_html=True)

left_col, right_col = st.columns([3, 2])

# ── Left: Recent Trade Activity (styled as recommendation cards) ──
with left_col:
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
        <div class="section-header">Recent Trade Activity</div>
    </div>
    <div class="section-sub">Algorithmic decisions and execution records</div>
    """, unsafe_allow_html=True)

    if trades:
        for trade in reversed(trades[-8:]):
            side = trade.get("side", "buy")
            symbol = trade.get("symbol", "???")
            qty = trade.get("qty", 0)
            price = trade.get("price", 0)
            reason = trade.get("reason", "")
            ts = trade.get("timestamp", 0)

            badge_css = "badge-buy" if side == "buy" else "badge-sell"
            badge_label = side.upper()
            icon = "📈" if side == "buy" else "📉"
            time_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%b %d, %H:%M") if ts else ""

            st.markdown(f"""
            <div class="activity-card">
                <div class="activity-row">
                    <div class="activity-left">
                        <div class="activity-icon">{icon}</div>
                        <div class="activity-info">
                            <div class="activity-title">
                                <span style="font-family:'IBM Plex Mono',monospace; font-weight:700;">{symbol}</span>
                                <span class="side-label"> • {badge_label}</span>
                            </div>
                            <div class="activity-detail">{int(qty)} shares @ ${price:,.2f}</div>
                            <div class="activity-reason">{reason}</div>
                            <div class="activity-time">{time_str}</div>
                        </div>
                    </div>
                    <div class="activity-pnl">
                        <span class="badge {badge_css}">{badge_label}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-banner">No trades recorded. The engine has not yet executed any orders.</div>', unsafe_allow_html=True)


# ── Right: Strategy Overview + System Status ──
with right_col:
    st.markdown("""
    <div class="section-header">Active Strategy</div>
    <div class="section-sub">Algorithm configuration and risk parameters</div>
    """, unsafe_allow_html=True)

    strategies = [
        ("MA Crossover", "Trend detection via moving average disparities", "#2DD4A0"),
        ("RSI", "Momentum evaluation at overbought/oversold levels", "#58A6FF"),
        ("Bollinger Bands", "Mean reversion using volatility bands", "#D2A8FF"),
        ("XGBoost", "Gradient-boosted classification on 25+ features", "#FFA657"),
        ("LSTM", "Deep sequence modeling (PyTorch)", "#FF7B72"),
        ("LLM Agent", "Market reasoning via Groq (Llama 3)", "#79C0FF"),
        ("Ensemble", "Multi-model consensus voting protocol", "#2DD4A0"),
    ]

    for name, desc, color in strategies:
        st.markdown(f"""
        <div style="background-color:#161B22; border:1px solid #21262D; border-radius:8px;
                    padding:12px 16px; margin-bottom:8px; display:flex; align-items:center; gap:12px;">
            <div style="width:4px; height:32px; background-color:{color}; border-radius:2px; flex-shrink:0;"></div>
            <div>
                <div style="font-size:14px; font-weight:600; color:#F0F6FC;">{name}</div>
                <div style="font-size:12px; color:#8B949E;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # System Status
    st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">System Status</div>
    <div class="section-sub">Engine diagnostics and observability</div>
    """, unsafe_allow_html=True)

    if engine_started:
        started_dt = datetime.fromtimestamp(engine_started, tz=timezone.utc)
        status_text = f"Online since {started_dt.strftime('%Y-%m-%d %H:%M UTC')}"
        status_color = "#2DD4A0"
        status_dot = "🟢"
    else:
        status_text = "Offline — awaiting engine initialization"
        status_color = "#F85149"
        status_dot = "🔴"

    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    st.markdown(f"""
    <div style="background-color:#161B22; border:1px solid #21262D; border-radius:12px; padding:20px 24px;">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
            <span>{status_dot}</span>
            <span style="font-size:14px; color:{status_color}; font-weight:600;">
                {status_text}
            </span>
        </div>
        <div style="display:flex; flex-direction:column; gap:8px;">
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:12px; color:#8B949E;">Equity Snapshots</span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#C9D1D9;">{len(equity_history)}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:12px; color:#8B949E;">Total Trades</span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#C9D1D9;">{len(trades)}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:12px; color:#8B949E;">Last Refresh</span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#C9D1D9;">{now_str}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:12px; color:#8B949E;">Metrics Endpoint</span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#58A6FF;">:8000/metrics</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Disclaimer Footer ─────────────────────────────────────

st.markdown("""
<div class="disclaimer">
    <strong>Paper Trading Simulation</strong> — No real capital is at risk.
    Actual trading involves execution risk, slippage, and fees not reflected in this analysis.
    Historical performance does not guarantee future returns. Risk management protocols take precedence over maximum returns.
</div>
""", unsafe_allow_html=True)


# ── Auto-refresh ──────────────────────────────────────────

time.sleep(30)
st.rerun()
