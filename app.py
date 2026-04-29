import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
 
# ── Config ──────────────────────────────────────────────────────────────────
TICKERS  = ["AAPL", "AMZN", "APLD", "GOOGL", "MSFT"]
WEIGHTS  = [0.20,   0.25,   0.15,   0.25,   0.15]
BENCH    = "^GSPC"
 
st.set_page_config(page_title="Portfolio vs S&P 500", layout="wide")
 
st.title("📈 Custom Portfolio vs S&P 500")
st.caption(
    "  |  ".join(f"{t} {w:.0%}" for t, w in zip(TICKERS, WEIGHTS))
)
 
# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    period = st.selectbox("Period", ["1y","2y","3y","5y","10y"], index=1)
    start_val = st.number_input("Starting value ($)", value=10_000, step=1_000)
 
# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load(tickers, period):
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    return raw.dropna()
 
all_tickers = TICKERS + [BENCH]
with st.spinner("Fetching prices…"):
    prices = load(all_tickers, period)
 
returns = prices.pct_change().dropna()
 
port_ret  = (returns[TICKERS] * WEIGHTS).sum(axis=1)
bench_ret = returns[BENCH]
 
port_val  = (1 + port_ret).cumprod()  * start_val
bench_val = (1 + bench_ret).cumprod() * start_val
 
# ── Metrics ───────────────────────────────────────────────────────────────────
def ann(r):
    return (1 + r.mean()) ** 252 - 1
 
def vol(r):
    return r.std() * np.sqrt(252)
 
def sharpe(r, rf=0.05):
    excess = ann(r) - rf
    return excess / vol(r) if vol(r) else np.nan
 
def max_dd(cum):
    roll_max = cum.cummax()
    return ((cum - roll_max) / roll_max).min()
 
col1, col2, col3, col4 = st.columns(4)
metrics = {
    "Total Return":    (port_val.iloc[-1]/start_val - 1, bench_val.iloc[-1]/start_val - 1),
    "Ann. Return":     (ann(port_ret), ann(bench_ret)),
    "Ann. Volatility": (vol(port_ret), vol(bench_ret)),
    "Sharpe Ratio":    (sharpe(port_ret), sharpe(bench_ret)),
}
 
for col, (label, (pv, bv)) in zip([col1,col2,col3,col4], metrics.items()):
    fmt = "{:.2f}" if label == "Sharpe Ratio" else "{:.1%}"
    delta_color = "normal" if label != "Ann. Volatility" else "inverse"
    col.metric(
        label,
        fmt.format(pv),
        f"{fmt.format(pv - bv)} vs S&P",
        delta_color=delta_color,
    )
 
st.divider()
 
# ── Cumulative Growth Chart ────────────────────────────────────────────────────
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=port_val.index, y=port_val,
    name="Portfolio", line=dict(color="#00C8FF", width=2.5),
))
fig.add_trace(go.Scatter(
    x=bench_val.index, y=bench_val,
    name="S&P 500", line=dict(color="#FF6B35", width=2, dash="dot"),
))
fig.update_layout(
    title="Cumulative Growth",
    yaxis_title="Portfolio Value ($)",
    yaxis_tickprefix="$",
    yaxis_tickformat=",.0f",
    legend=dict(orientation="h", y=1.08),
    hovermode="x unified",
    height=420,
    margin=dict(t=60, b=40),
    template="plotly_dark",
)
st.plotly_chart(fig, use_container_width=True)
 
# ── Drawdown Chart ─────────────────────────────────────────────────────────────
def drawdown_series(cum):
    roll_max = cum.cummax()
    return (cum - roll_max) / roll_max
 
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=port_val.index, y=drawdown_series(port_val),
    name="Portfolio", fill="tozeroy",
    line=dict(color="#00C8FF", width=1.5),
    fillcolor="rgba(0,200,255,0.15)",
))
fig2.add_trace(go.Scatter(
    x=bench_val.index, y=drawdown_series(bench_val),
    name="S&P 500", fill="tozeroy",
    line=dict(color="#FF6B35", width=1.5, dash="dot"),
    fillcolor="rgba(255,107,53,0.10)",
))
fig2.update_layout(
    title="Drawdown",
    yaxis_title="Drawdown",
    yaxis_tickformat=".0%",
    legend=dict(orientation="h", y=1.08),
    hovermode="x unified",
    height=300,
    margin=dict(t=60, b=40),
    template="plotly_dark",
)
st.plotly_chart(fig2, use_container_width=True)
 
# ── Rolling 30-day Correlation ─────────────────────────────────────────────────
rolling_corr = port_ret.rolling(30).corr(bench_ret)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=rolling_corr.index, y=rolling_corr,
    name="30-day Corr", line=dict(color="#A78BFA", width=2),
))
fig3.update_layout(
    title="Rolling 30-Day Correlation (Portfolio vs S&P 500)",
    yaxis_title="Correlation",
    height=280,
    margin=dict(t=60, b=40),
    template="plotly_dark",
)
st.plotly_chart(fig3, use_container_width=True)
 
# ── Holdings Table ─────────────────────────────────────────────────────────────
st.subheader("Holdings Summary")
rows = []
for t, w in zip(TICKERS, WEIGHTS):
    r = returns[t]
    rows.append({
        "Ticker":        t,
        "Weight":        f"{w:.0%}",
        "Ann. Return":   f"{ann(r):.1%}",
        "Ann. Vol":      f"{vol(r):.1%}",
        "Sharpe":        f"{sharpe(r):.2f}",
        "Max Drawdown":  f"{max_dd(prices[t]):.1%}",
    })
st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)
 
st.caption("Data via Yahoo Finance · prices adjust for splits & dividends")
