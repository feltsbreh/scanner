import pandas as pd
import requests
from io import StringIO

def fetch_russell2000_tickers_from_iwm():
    """
    Fetch tickers of Russell 2000 via IWM ETF holdings CSV (or similar fund).
    Returns list of tickers (strings).
    """
    # Example URL: (you’ll need to find a current URL that provides CSV of holdings for IWM)
    csv_url = "https://www.ishares.com/us/literature/fact-sheet/iwm‑fund‑holdings.csv"
    try:
        resp = requests.get(csv_url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        # Assume column "Ticker" or "Symbol" exists
        tickers = df["Ticker"].dropna().unique().tolist()
        return [t.strip().upper() for t in tickers if isinstance(t, str)]
    except Exception as e:
        print("Failed to fetch IWM holdings:", e)
        return []

# In your Streamlit UI, allow an option “Use Russell 2000 universe”:
use_russell2000 = st.sidebar.checkbox("Use Russell 2000 universe (via IWM holdings)", value=False)

if use_russell2000:
    tickers = fetch_russell2000_tickers_from_iwm()
else:
    # existing tickers input logic
    ...

# Then proceed to run_scan(tickers, …)





"""
Simple Swing Scanner + Streamlit UI

Usage:
  1) pip install -r requirements.txt
  2) streamlit run app.py
  3) open http://localhost:8501 in your browser
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
import math

st.set_page_config(page_title="Swing Scanner", layout="wide")

DEFAULT_TICKERS = ["AAPL", "MSFT", "AMD", "NVDA", "INTC"]

@st.cache_data(ttl=300)
def fetch_history(ticker: str, days: int = 400):
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    try:
        df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False, threads=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df.dropna(how="any")
    return df

def compute_indicators(df: pd.DataFrame):
    df = df.copy()
    df["SMA50"] = df["Close"].rolling(window=50, min_periods=30).mean()
    df["SMA200"] = df["Close"].rolling(window=200, min_periods=120).mean()
    df["RSI14"] = ta.rsi(df["Close"], length=14)
    df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    return df

def analyze_ticker(ticker: str, lookback_days: int, atr_multiplier: float, min_avg_volume: int, account_size: float, risk_pct: float):
    df = fetch_history(ticker, days=lookback_days)
    if df is None or df.shape[0] < 60:
        return None
    df = compute_indicators(df)
    latest = df.iloc[-1]

    avg_vol_30 = df["Volume"].tail(30).mean()
    pass_filter = (min_avg_volume == 0) or (not math.isnan(avg_vol_30) and avg_vol_30 >= min_avg_volume)

    trend_up = not pd.isna(latest["SMA50"]) and not pd.isna(latest["SMA200"]) and latest["SMA50"] > latest["SMA200"]
    near_sma50 = not pd.isna(latest["SMA50"]) and latest["Close"] <= (latest["SMA50"] * 1.03)
    rsi_ok = not pd.isna(latest["RSI14"]) and latest["RSI14"] < 65

    buy_signal = trend_up and near_sma50 and rsi_ok and pass_filter
    sell_signal = (not pd.isna(latest["SMA50"]) and latest["Close"] < latest["SMA50"]) or (not pd.isna(latest["RSI14"]) and latest["RSI14"] > 75)

    atr = latest["ATR14"] if not pd.isna(latest["ATR14"]) else None
    suggested_stop = None
    suggested_shares = 0
    position_risk_amount = account_size * risk_pct / 100.0
    if buy_signal and atr is not None and atr > 0:
        suggested_stop = latest["Close"] - atr_multiplier * atr
        if suggested_stop >= latest["Close"]:
            suggested_stop = latest["Close"] * 0.99
        risk_per_share = latest["Close"] - suggested_stop
        if risk_per_share > 0:
            suggested_shares = int(position_risk_amount / risk_per_share)

    return {
        "ticker": ticker,
        "price": round(latest["Close"], 2),
        "sma50": round(latest["SMA50"], 2) if not pd.isna(latest["SMA50"]) else None,
        "sma200": round(latest["SMA200"], 2) if not pd.isna(latest["SMA200"]) else None,
        "rsi14": round(latest["RSI14"], 1) if not pd.isna(latest["RSI14"]) else None,
        "atr14": round(atr, 3) if atr is not None else None,
        "avg_vol_30": int(avg_vol_30) if not math.isnan(avg_vol_30) else None,
        "buy": buy_signal,
        "sell": sell_signal,
        "suggested_stop": round(suggested_stop, 2) if suggested_stop else None,
        "suggested_shares": suggested_shares,
        "position_risk_amount": round(position_risk_amount, 2)
    }

def run_scan(tickers, lookback_days, atr_multiplier, min_avg_volume, account_size, risk_pct, max_tickers=500):
    rows = []
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t and t.strip()]))[:max_tickers]
    for t in tickers:
        try:
            r = analyze_ticker(t, lookback_days, atr_multiplier, min_avg_volume, account_size, risk_pct)
            if r is not None:
                rows.append(r)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["score"] = df["buy"].apply(lambda x: 2 if x else 1)
    df = df.sort_values(by=["buy", "avg_vol_30"], ascending=[False, False]).reset_index(drop=True)
    return df

# -------------------------
# Streamlit UI
# -------------------------

st.title("🔎 Swing Scanner — SMA / RSI / ATR")
st.sidebar.header("Scanner parameters")
lookback_days = st.sidebar.number_input("Historical lookback (days)", value=400, min_value=120, max_value=2000, step=30)
atr_multiplier = st.sidebar.number_input("ATR multiplier for stop", value=1.5, min_value=0.5, step=0.1, format="%.2f")
min_avg_volume = st.sidebar.number_input("Min avg daily volume (0 to disable)", value=100000, step=1000)
account_size = st.sidebar.number_input("Account size (USD)", value=10000.0, min_value=100.0, step=100.0, format="%.2f")
risk_pct = st.sidebar.number_input("Risk percent per trade (%)", value=1.0, min_value=0.1, max_value=20.0, step=0.1, format="%.2f")
max_tickers = st.sidebar.number_input("Max tickers to scan", value=200, min_value=10, max_value=2000, step=10)
run_button = st.sidebar.button("Run scan")

st.header("Tickers (paste or upload CSV)")
uploaded = st.file_uploader("Upload CSV", type=["csv","txt"])
paste = st.text_area("Or paste tickers (one per line)", height=120, value="\n".join(DEFAULT_TICKERS))

tickers = []
if uploaded:
    try:
        df_in = pd.read_csv(uploaded, header=None)
        tickers = df_in.iloc[:,0].astype(str).tolist()
    except Exception:
        txt = uploaded.getvalue().decode("utf-8")
        tickers = [line.strip() for line in txt.splitlines() if line.strip()]
elif paste:
    tickers = [line.strip().upper() for line in paste.splitlines() if line.strip()]

st.markdown(f"**Tickers to scan:** {len(tickers)}")

if run_button:
    if not tickers:
        st.error("No tickers provided.")
    else:
        with st.spinner("Running scanner..."):
            df_res = run_scan(tickers, lookback_days, atr_multiplier, min_avg_volume, account_size, risk_pct, max_tickers)
        if df_res.empty:
            st.warning("No results. Try fewer tickers or lower min volume.")
        else:
            buys = df_res[df_res["buy"]==True]
            st.markdown(f"### Results — {len(df_res)} scanned | {len(buys)} buy signals")
            st.dataframe(df_res, height=480)
            csv = df_res.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="scanner_results.csv", mime="text/csv")

