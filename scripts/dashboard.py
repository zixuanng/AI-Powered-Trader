"""
Streamlit Dashboard for the AI Paper Trading Bot.

Reads from data/state.json and data/trades.json to display:
- Institutional Portfolio Overview
- Equity & Drawdown Analytics
- Historical Trade Records
- Strategy Education & Rationale

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
    page_title="Institutional Trading Protocol | AI Bot",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: Finance-Grade Typography & Palette ─────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #1A2332;
        color: #F8F9FA;
    }
    
    div[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #2C3E50;
    }

    /* Professional Metric Cards */
    .metric-card {
        background-color: #2C3E50;
        border-radius: 6px;
        padding: 24px;
        margin: 8px 0;
        border: 1px solid #34495E;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .metric-label {
        color: #8892b0;
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #FFFFFF;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 28px;
        font-weight: 600;
    }
    .metric-subtext {
        font-size: 13px;
        margin-top: 6px;
        font-family: 'Inter', sans-serif;
    }
    
    /* Institutional Status Colors */
    .positive { color: #4CAF50 !important; }
    .negative { color: #E74C3C !important; }
    .neutral { color: #546E7A !important; }
    .accent { color: #26A69A !important; }

    /* Minimal Headers */
    h1, h2, h3 {
        font-weight: 500;
        letter-spacing: -0.5px;
    }
    
    /* Tables precision */
    .dataframe {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
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

# ── Sidebar & Education ────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏛️ Trading Protocol")
    st.caption("Intelligent guidance grounded in data.")
    st.markdown("---")

    refresh_rate = st.selectbox(
        "Data Synchronization",
        options=[10, 30, 60, 120],
        index=1,
        format_func=lambda x: f"Every {x} seconds",
        help="Adjusts how frequently the dashboard polls local state files for updates."
    )

    if st.button("Manual Resync"):
        st.rerun()

    st.markdown("---")

    st.markdown("#### Operational Strategies")
    st.caption("Algorithm rationales and risk models.")
    
    strategies = [
        ("MA Crossover", "Trend identification via moving average disparities.", "Low volatility, trend-following base."),
        ("RSI", "Momentum evaluation (Relative Strength Index).", "Mean-reversion during over-extended conditions."),
        ("Bollinger Bands", "Volatility assessment using standard deviations.", "Identifies statistical outliers in price action."),
        ("XGBoost", "Gradient-boosted decision trees.", "Non-linear pattern recognition on 25+ technical features."),
        ("LSTM Neural Net", "Sequence modeling (PyTorch).", "Time-series forecasting based on temporal dependencies."),
        ("LLM Agent", "Heuristic market reasoning (Groq).", "Contextual synthesis of price action and technicals."),
        ("Ensemble Voting", "Multi-model consensus protocol.", "Reduces single-model failure risk via majority/weighted voting."),
    ]
    
    for name, desc, risk in strategies:
        with st.expander(name):
            st.markdown(f"**Mechanism**: {desc}")
            st.markdown(f"**Risk Profile**: {risk}")

    st.markdown("---")
    st.info(
        "**Compliance & Risk Disclosure**\n\n"
        "This platform operates strictly via paper trading. "
        "Historical performance parameters suggest past results do not guarantee future returns. "
        "Risk management protocols take precedence over maximum returns."
    )

# ── Load Data ──────────────────────────────────────────────
state = load_state()
trades = load_trades()
snapshot = state.get("latest_snapshot", {})
equity_history = state.get("equity_history", [])
engine_started = state.get("engine_started_at")

# ── Header ─────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("## Portfolio Analytics")
    if engine_started:
        started_dt = datetime.fromtimestamp(engine_started, tz=timezone.utc)
        st.caption(f"Engine Initialized: {started_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} • Protocol Active")
    else:
        st.caption("Engine offline. Awaiting protocol execution via `scripts/live_paper.py`.")

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        f"Data Currency: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}<br>"
        f"Snapshots Logged: {len(equity_history)}",
        unsafe_allow_html=True
    )

st.markdown("---")

# ── Account Analytics Cards ────────────────────────────────
def metric_card(label: str, value: str, subtext: str = "", css_class: str = "", tooltip: str = ""):
    help_icon = f"<span title='{tooltip}' style='cursor:help'>ⓘ</span>" if tooltip else ""
    sub_html = f'<div class="metric-subtext {css_class}">{subtext}</div>' if subtext else ""
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label} {help_icon}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

if snapshot:
    # Row 1
    c1, c2, c3, c4 = st.columns(4)
    
    equity = snapshot.get("equity", 0)
    cash = snapshot.get("cash", 0)
    drawdown = snapshot.get("drawdown_pct", 0) * 100
    pnl = snapshot.get("unrealized_pnl", 0)
    positions = snapshot.get("positions", 0)
    exposure = snapshot.get("exposure_pct", 0) * 100

    # Ensure precision and formatting
    with c1:
        metric_card(
            "Total Equity", 
            f"${equity:,.2f}", 
            "Net liquidation value", 
            "neutral",
            "Total value of cash plus all open positions."
        )
    with c2:
        metric_card(
            "Cash Balance", 
            f"${cash:,.2f}", 
            "Available for allocation", 
            "neutral",
            "Liquid capital not currently deployed in the market."
        )
    with c3:
        pnl_css = "positive" if pnl > 0.0 else ("negative" if pnl < 0.0 else "neutral")
        pnl_sign = "+" if pnl > 0.0 else ""
        metric_card(
            "Unrealized P&L", 
            f"{pnl_sign}${pnl:,.2f}", 
            "Open position variance", 
            pnl_css,
            "Floating profit or loss on current holdings."
        )
    with c4:
        dd_css = "negative" if drawdown > 5.0 else ("neutral" if drawdown > 0 else "positive")
        dd_text = f"Warning: High Drawdown" if drawdown > 10.0 else "Within nominal limits"
        metric_card(
            "Current Drawdown", 
            f"{drawdown:.2f}%", 
            dd_text, 
            dd_css,
            "Peak-to-trough decline measured from highest portfolio value."
        )

    # Row 2 (Risk and Allocation)
    st.markdown("<br>", unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        metric_card(
            "Active Positions", 
            str(positions), 
            "Equities held", 
            "neutral",
            "Number of distinct assets currently held in the portfolio."
        )
    with c6:
        exposure_css = "negative" if exposure > 80.0 else "neutral"
        metric_card(
            "Market Exposure", 
            f"{exposure:.1f}%", 
            f"{100 - exposure:.1f}% capital reserved", 
            exposure_css,
            "Percentage of total equity currently deployed in the market."
        )
else:
    st.info("Awaiting initial telemetry data. Portfolio analytics will populate upon engine initialization.")

# ── Equity & Drawdown Visualizations ───────────────────────
st.markdown("<br><br>### Performance History", unsafe_allow_html=True)

if equity_history:
    df_equity = pd.DataFrame(equity_history)
    df_equity["datetime"] = pd.to_datetime(df_equity["timestamp"], unit="s", utc=True)

    # Shared Plotly configuration to ensure institutional aesthetic
    plotly_layout_kwargs = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1A2332",
        margin=dict(l=40, r=20, t=10, b=30),
        xaxis=dict(
            gridcolor="#2C3E50", 
            zerolinecolor="#2C3E50",
            tickfont=dict(family="IBM Plex Mono", size=11, color="#8892b0")
        ),
        yaxis=dict(
            gridcolor="#2C3E50", 
            zerolinecolor="#2C3E50",
            title_font=dict(family="Inter", size=12, color="#8892b0"),
            tickfont=dict(family="IBM Plex Mono", size=11, color="#8892b0")
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(family="Inter", size=12, color="#8892b0")
        ),
        hovermode="x unified"
    )

    t1, t2 = st.tabs(["Equity Progression", "Drawdown Profile"])

    with t1:
        st.caption("Historical performance tracking absolute equity against cash baseline.")
        fig_equity = go.Figure()
        
        # Muted Teal (#26A69A) for Equity
        fig_equity.add_trace(go.Scatter(
            x=df_equity["datetime"],
            y=df_equity["equity"],
            mode="lines",
            name="Total Equity",
            line=dict(color="#26A69A", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(38, 166, 154, 0.05)",
        ))
        
        # Slate Gray (#546E7A) for Cash
        fig_equity.add_trace(go.Scatter(
            x=df_equity["datetime"],
            y=df_equity["cash"],
            mode="lines",
            name="Cash Reserve",
            line=dict(color="#546E7A", width=1.5, dash="dot"),
        ))
        
        layout_eq = plotly_layout_kwargs.copy()
        layout_eq["yaxis"]["title"] = "Valuation (USD)"
        layout_eq["height"] = 400
        fig_equity.update_layout(**layout_eq)
        st.plotly_chart(fig_equity, use_container_width=True, config={'displayModeBar': False})

    with t2:
        st.caption("Quantitative measurement of capital impairment from peak valuation.")
        # Ensure we are plotting the actual percentage
        # Some data might have drawdown_pct as a float 0.02 = 2% or 2.0 = 2%
        # Our risk metric outputs raw percentage decimals (e.g. 0.02). Let's multiply if necessary, or just plot raw.
        # Looking at original code: drawdown = max(0.0, (last_equity - equity) / last_equity)
        # We will multiply by 100 for display.
        
        df_equity["drawdown_display"] = df_equity["drawdown_pct"].fillna(0) * 100
        
        fig_dd = go.Figure()
        
        # Professional Red (#E74C3C)
        fig_dd.add_trace(go.Scatter(
            x=df_equity["datetime"],
            y=df_equity["drawdown_display"],
            mode="lines",
            name="Drawdown Rate",
            line=dict(color="#E74C3C", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(231, 76, 60, 0.1)",
        ))
        
        layout_dd = plotly_layout_kwargs.copy()
        layout_dd["yaxis"]["title"] = "Impairment (%)"
        layout_dd["height"] = 400
        fig_dd.update_layout(**layout_dd)
        st.plotly_chart(fig_dd, use_container_width=True, config={'displayModeBar': False})
else:
    st.info("Insufficient historical data to render analytical charts.")

# ── Order Execution Ledger ─────────────────────────────────
st.markdown("<br>### Order Execution Ledger", unsafe_allow_html=True)
st.caption("Chronological record of algorithmic decision outputs and enacted orders.")

if trades:
    df_trades = pd.DataFrame(trades[-100:])  # analyze last 100
    
    # Format precision columns
    if "timestamp" in df_trades.columns:
        df_trades["Timestamp (UTC)"] = pd.to_datetime(df_trades["timestamp"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    
    if "price" in df_trades.columns:
        df_trades["Execution Price"] = df_trades["price"].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
        
    if "qty" in df_trades.columns:
        df_trades["Volume"] = df_trades["qty"].apply(lambda x: str(int(x)) if pd.notnull(x) else "")

    # Rename for professional context
    col_mapping = {
        "side": "Direction",
        "symbol": "Ticker",
        "reason": "Algorithmic Rationale"
    }
    df_trades = df_trades.rename(columns=col_mapping)
    
    # Select columns to display
    display_cols = [c for c in ["Timestamp (UTC)", "Direction", "Ticker", "Volume", "Execution Price", "Algorithmic Rationale"] if c in df_trades.columns]

    if display_cols:
        # Sort newest first
        df_display = df_trades[display_cols].sort_index(ascending=False)
        
        # Display as a dataframe using Streamlit's native tools, which will inherit our CSS overrides for fonts
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=300
        )
    else:
        st.info("Ledger data present, but required formatting columns are missing.")
else:
    st.info("No orders enacted. Ledger is empty.")

