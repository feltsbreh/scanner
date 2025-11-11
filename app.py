"""
Swing Scanner + Streamlit UI + Russell2000 Universe Option (auto CSV)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import math
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

st.set_page_config(page_title="Swing Scanner", layout="wide")
DEFAULT_TICKERS = ["AAPL", "MSFT", "AMD", "NVDA", "INTC"]
RUSSSELL_CSV = "russell2000.csv"

# ---------------------------
# Data Fetching
# ---------------------------

@st.cache_data(ttl=300)
def fetch_history(ticker: str, days: int = 400):
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    try:
        df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                         progress=False, threads=False)
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

def analyze_ticker(ticker: str, lookback_days: int, atr_multiplier: float,
                   min_avg_volume: int, account_size: float, risk_pct: float):
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
    sell_signal = (not pd.isna(latest["SMA50"]) and latest["Close"] < latest["SMA50"]) or \
                  (not pd.isna(latest["RSI14"]) and latest["RSI14"] > 75)

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

# ---------------------------
# Russell2000 CSV Handling
# ---------------------------

def build_russell_csv(path=RUSSSELL_CSV):
    """Scrape tickers from SureDividend and save locally."""
    url = "https://www.suredividend.com/russell-2000-stocks/"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if table is None:
            st.warning("Could not find table on SureDividend page.")
            return []
        df = pd.read_html(str(table))[0]
        tickers = df["Ticker"].dropna().astype(str).str.upper().tolist()
        pd.DataFrame({"Ticker": tickers}).to_csv(path, index=False)
        return tickers
    except Exception as e:
        st.warning(f"Failed to fetch Russell2000 tickers: {e}")
        return []

def load_russell_csv(path=RUSSSELL_CSV, max_tickers=2000):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, usecols=["Ticker"])
            tickers = df["Ticker"].dropna().astype(str).str.upper().tolist()
            return tickers[:max_tickers]
        except Exception as e:
            st.warning(f"Failed to load local CSV: {e}")
            return []
    else:
        return build_russell_csv(path)[:max_tickers]

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("🔎 Swing Scanner — SMA / RSI / ATR (Russell2000 CSV)")
st.sidebar.header("Scanner parameters")

lookback_days = st.sidebar.number_input("Historical lookback (days)", value=400, min_value=120, max_value=2000, step=30)
atr_multiplier = st.sidebar.number_input("ATR multiplier for stop", value=1.5, min_value=0.5, step=0.1, format="%.2f")
min_avg_volume = st.sidebar.number_input("Min avg daily volume (0 to disable)", value=100000, step=1000)
account_size = st.sidebar.number_input("Account size (USD)", value=10000.0, min_value=100.0, step=100.0, format="%.2f")
risk_pct = st.sidebar.number_input("Risk percent per trade (%)", value=1.0, min_value=0.1, max_value=20.0, step=0.1, format="%.2f")
max_tickers = st.sidebar.number_input("Max tickers to scan", value=500, min_value=50, max_value=3000, step=50)

use_russell2000 = st.sidebar.checkbox("Use Russell2000 universe (auto CSV)", value=False)
run_button = st.sidebar.button("Run scan")

st.header("Tickers / Universe Selection")
if use_russell2000:
    tickers = load_russell_csv(max_tickers=max_tickers)
    st.write(f"Loaded {len(tickers)} tickers from Russell2000 CSV.")
else:
    uploaded = st.file_uploader("Upload CSV (one ticker per line)", type=["csv","txt"])
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
        with st.spinner("Running scanner…"):
            df_res = run_scan(tickers, lookback_days, atr_multiplier, min_avg_volume, account_size, risk_pct, max_tickers)
        if df_res.empty:
            st.warning("No results—try fewer tickers or relax filters.")
        else:
            buys = df_res[df_res["buy"] == True]
            st.markdown(f"### Results — {len(df_res)} scanned | {len(buys)} buy signals")
            st.dataframe(df_res, height=480)
            csv = df_res.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="scanner_results.csv", mime="text/csv")
            if len(buys) > 0:
                st.markdown("#### Buy candidates")
                st.table(buys[["ticker","price","sma50","sma200","rsi14","atr14","avg_vol_30","suggested_stop","suggested_shares"]].reset_index(drop=True))

st.markdown("---")
st.markdown("### Notes & Next Steps")
st.markdown("""
- If you enable “Use Russell2000 universe”, tickers are fetched from SureDividend and stored locally.
- The CSV will be reused next time to speed up loading.
- Scanning many tickers (>500) may take time or hit data limits.
- Stop-loss is ATR-based: stop = price − ATR * multiplier.
- Position sizing is driven by your account size & risk percent.
- Always back-test before relying on live signals.
""")
