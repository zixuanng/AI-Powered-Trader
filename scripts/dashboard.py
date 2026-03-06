"""
Streamlit Dashboard for the AI Paper Trading Bot.

Reads from data/state.json and data/trades.json to display:
- Account overview cards
- Equity curve (Plotly)
- Positions table
- Trade log
- Strategy info

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

# Project root
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

_DATA_DIR = _ROOT / "data"
_STATE_FILE = _DATA_DIR / "state.json"
_TRADES_FILE = _DATA_DIR / "trades.json"


# ── Page Config ────────────────────────────────────────────

st.set_page_config(
    page_title="AI Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ─────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-label {
        color: #8892b0;
        font-size: 14px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        color: #ccd6f6;
        font-size: 28px;
        font-weight: 700;
        margin-top: 4px;
    }
    .positive { color: #64ffda !important; }
    .negative { color: #ff6b6b !important; }
    .stApp {
        background-color: #0a0a1a;
    }
    div[data-testid="stSidebar"] {
        background-color: #0f0f23;
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


# ── Sidebar ────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 AI Trading Bot")
    st.markdown("---")

    refresh_rate = st.selectbox(
        "Auto-refresh interval",
        options=[10, 30, 60, 120],
        index=1,
        format_func=lambda x: f"{x} seconds",
    )

    if st.button("🔄 Refresh Now"):
        st.rerun()

    st.markdown("---")

    # Strategy info
    st.markdown("### Available Strategies")
    strategies = [
        ("📊 MA Crossover", "Moving average crossover"),
        ("📉 RSI", "Relative Strength Index"),
        ("📈 Bollinger Bands", "Mean reversion at bands"),
        ("🌲 XGBoost", "ML classifier"),
        ("🧠 LSTM", "Deep learning sequence model"),
        ("💬 LLM Agent", "Groq-powered reasoning"),
        ("🗳️ Ensemble", "Multi-strategy voting"),
    ]
    for icon_name, desc in strategies:
        st.markdown(f"**{icon_name}** — {desc}")

    st.markdown("---")
    st.markdown(
        "**Prometheus Metrics**  \n"
        "`http://localhost:8000/metrics`"
    )


# ── Load Data ──────────────────────────────────────────────

state = load_state()
trades = load_trades()
snapshot = state.get("latest_snapshot", {})
equity_history = state.get("equity_history", [])
engine_started = state.get("engine_started_at")


# ── Header ─────────────────────────────────────────────────

st.markdown("# 📈 AI Paper Trading Dashboard")

if engine_started:
    started_dt = datetime.fromtimestamp(engine_started, tz=timezone.utc)
    st.caption(f"Engine started: {started_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
else:
    st.caption("Engine has not started yet. Run `python scripts/live_paper.py` to begin.")

st.markdown("---")


# ── Account Overview Cards ─────────────────────────────────

def metric_card(label: str, value: str, delta: str = "", color: str = ""):
    delta_html = ""
    if delta:
        css = "positive" if not delta.startswith("-") else "negative"
        delta_html = f'<div class="{css}" style="font-size:14px;margin-top:2px">{delta}</div>'

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color}">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


if snapshot:
    col1, col2, col3, col4 = st.columns(4)

    equity = snapshot.get("equity", 0)
    cash = snapshot.get("cash", 0)
    positions = snapshot.get("positions", 0)
    drawdown = snapshot.get("drawdown_pct", 0)
    exposure = snapshot.get("exposure_pct", 0)
    pnl = snapshot.get("unrealized_pnl", 0)

    with col1:
        metric_card("Portfolio Equity", f"${equity:,.2f}")
    with col2:
        metric_card("Cash", f"${cash:,.2f}")
    with col3:
        pnl_sign = "+" if pnl >= 0 else ""
        pnl_color = "positive" if pnl >= 0 else "negative"
        metric_card("Unrealized P&L", f"{pnl_sign}${pnl:,.2f}", color=pnl_color)
    with col4:
        metric_card("Drawdown", f"{drawdown:.2f}%",
                     color="negative" if drawdown > 5 else "")

    col5, col6 = st.columns(2)
    with col5:
        metric_card("Open Positions", str(positions))
    with col6:
        metric_card("Exposure", f"{exposure:.1f}%")
else:
    st.info("No portfolio data yet. The dashboard will populate once the engine starts trading.")


# ── Equity Curve ───────────────────────────────────────────

st.markdown("## 📊 Equity Curve")

if equity_history:
    df_equity = pd.DataFrame(equity_history)
    df_equity["datetime"] = pd.to_datetime(df_equity["timestamp"], unit="s", utc=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_equity["datetime"],
        y=df_equity["equity"],
        mode="lines",
        name="Equity",
        line=dict(color="#64ffda", width=2),
        fill="tozeroy",
        fillcolor="rgba(100, 255, 218, 0.05)",
    ))

    fig.add_trace(go.Scatter(
        x=df_equity["datetime"],
        y=df_equity["cash"],
        mode="lines",
        name="Cash",
        line=dict(color="#8892b0", width=1, dash="dot"),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,26,0.8)",
        margin=dict(l=40, r=20, t=40, b=40),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="USD"),
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No equity data yet.")


# ── Drawdown Chart ─────────────────────────────────────────

if equity_history and any(e.get("drawdown_pct", 0) > 0 for e in equity_history):
    st.markdown("## 📉 Drawdown")

    df_dd = pd.DataFrame(equity_history)
    df_dd["datetime"] = pd.to_datetime(df_dd["timestamp"], unit="s", utc=True)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=df_dd["datetime"],
        y=df_dd["drawdown_pct"],
        mode="lines",
        name="Drawdown %",
        line=dict(color="#ff6b6b", width=2),
        fill="tozeroy",
        fillcolor="rgba(255, 107, 107, 0.1)",
    ))

    fig_dd.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,26,0.8)",
        margin=dict(l=40, r=20, t=30, b=40),
        height=250,
        yaxis=dict(title="Drawdown %", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )

    st.plotly_chart(fig_dd, use_container_width=True)


# ── Trade Log ──────────────────────────────────────────────

st.markdown("## 📋 Recent Trades")

if trades:
    df_trades = pd.DataFrame(trades[-50:])  # last 50

    if "timestamp" in df_trades.columns:
        df_trades["time"] = pd.to_datetime(df_trades["timestamp"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

    display_cols = [c for c in ["time", "side", "symbol", "qty", "price", "reason"] if c in df_trades.columns]

    if display_cols:
        st.dataframe(
            df_trades[display_cols].sort_index(ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Trade data available but no displayable columns.")
else:
    st.info("No trades recorded yet.")


# ── Footer ─────────────────────────────────────────────────

st.markdown("---")
st.caption(
    f"Last refreshed: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} | "
    f"Equity snapshots: {len(equity_history)} | Trades: {len(trades)}"
)


# ── Auto-refresh ───────────────────────────────────────────

time.sleep(refresh_rate)
st.rerun()
