{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 """\
Simple Swing Scanner + Streamlit UI\
\
Usage:\
  1) pip install -r requirements.txt\
  2) streamlit run app.py\
  3) open http://localhost:8501 in your browser\
\
Notes:\
 - Provide your own ticker list by uploading a CSV or pasting tickers (one per line).\
 - By default the app uses a small example list.\
 - The app uses yfinance to fetch historical OHLCV and pandas_ta for indicators.\
"""\
\
import streamlit as st\
import yfinance as yf\
import pandas as pd\
import pandas_ta as ta\
import numpy as np\
from datetime import datetime, timedelta\
import io\
import math\
\
st.set_page_config(page_title="Swing Scanner", layout="wide")\
\
# -------------------------\
# Helper & scanner logic\
# -------------------------\
\
DEFAULT_TICKERS = [\
    "AAPL", "MSFT", "AMD", "NVDA", "INTC"\
]  # small example. Replace with your universe (upload or paste list in UI)\
\
@st.cache_data(ttl=300)\
def fetch_history(ticker: str, days: int = 400):\
    """\
    Fetch historical daily bars using yfinance.\
    Cached for 5 minutes by default to avoid repeated downloads during testing.\
    """\
    end = datetime.utcnow().date()\
    start = end - timedelta(days=days)\
    try:\
        df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False, threads=False)\
    except Exception as e:\
        return None\
    if df is None or df.empty:\
        return None\
    df = df.dropna(how="any")\
    return df\
\
def compute_indicators(df: pd.DataFrame):\
    """\
    Adds SMA50, SMA200, RSI14, ATR14 to dataframe and returns df.\
    """\
    df = df.copy()\
    # Simple moving averages\
    df["SMA50"] = df["Close"].rolling(window=50, min_periods=30).mean()\
    df["SMA200"] = df["Close"].rolling(window=200, min_periods=120).mean()\
    # RSI (pandas_ta)\
    try:\
        df["RSI14"] = ta.rsi(df["Close"], length=14)\
    except Exception:\
        # fallback: compute simple RSI if pandas_ta fails\
        df["RSI14"] = df["Close"].diff().apply(lambda x: np.nan)\
    # ATR (pandas_ta)\
    df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)\
    return df\
\
def analyze_ticker(ticker: str,\
                   lookback_days: int,\
                   atr_multiplier: float,\
                   min_avg_volume: int,\
                   account_size: float,\
                   risk_pct: float):\
    """\
    Fetch data for ticker, compute indicators, produce signal dict.\
    Returns None on failure or a dict with results.\
    """\
    df = fetch_history(ticker, days=lookback_days)\
    if df is None or df.shape[0] < 60:\
        return None\
\
    df = compute_indicators(df)\
\
    # latest row\
    latest = df.iloc[-1]\
    prev = df.iloc[-2]\
\
    # liquidity filter: compute 30-day average volume\
    avg_vol_30 = df["Volume"].tail(30).mean()\
    if min_avg_volume and (math.isnan(avg_vol_30) or avg_vol_30 < min_avg_volume):\
        # fails liquidity filter\
        pass_filter = False\
    else:\
        pass_filter = True\
\
    # trend filter (SMA50 > SMA200)\
    trend_up = (not pd.isna(latest["SMA50"]) and not pd.isna(latest["SMA200"]) and latest["SMA50"] > latest["SMA200"])\
\
    # near SMA50 (within 3%) - adjustable threshold later in UI\
    near_sma50 = False\
    if not pd.isna(latest["SMA50"]) and latest["SMA50"] > 0:\
        near_sma50 = latest["Close"] <= (latest["SMA50"] * 1.03)\
\
    rsi_ok = (not pd.isna(latest["RSI14"]) and latest["RSI14"] < 65)\
\
    buy_signal = trend_up and near_sma50 and rsi_ok and pass_filter\
\
    # Sell / exit signal - simple\
    sell_signal = (not pd.isna(latest["SMA50"]) and latest["Close"] < latest["SMA50"]) or (not pd.isna(latest["RSI14"]) and latest["RSI14"] > 75)\
\
    atr = latest["ATR14"] if not pd.isna(latest["ATR14"]) else None\
\
    suggested_stop = None\
    suggested_shares = 0\
    position_risk_amount = account_size * risk_pct / 100.0  # e.g. 1% -> 0.01\
    if buy_signal and atr is not None and atr > 0:\
        suggested_stop = latest["Close"] - atr_multiplier * atr\
        # ensure stop is below price\
        if suggested_stop >= latest["Close"]:\
            suggested_stop = latest["Close"] * 0.99  # fallback tiny stop\
        risk_per_share = latest["Close"] - suggested_stop\
        if risk_per_share <= 0:\
            suggested_shares = 0\
        else:\
            suggested_shares = int(position_risk_amount / risk_per_share)\
    # Package results\
    res = \{\
        "ticker": ticker,\
        "price": float(latest["Close"]),\
        "sma50": float(latest["SMA50"]) if not pd.isna(latest["SMA50"]) else None,\
        "sma200": float(latest["SMA200"]) if not pd.isna(latest["SMA200"]) else None,\
        "rsi14": float(latest["RSI14"]) if not pd.isna(latest["RSI14"]) else None,\
        "atr14": float(atr) if atr is not None else None,\
        "avg_vol_30": int(avg_vol_30) if not math.isnan(avg_vol_30) else None,\
        "buy": buy_signal,\
        "sell": sell_signal,\
        "suggested_stop": float(suggested_stop) if suggested_stop is not None else None,\
        "suggested_shares": int(suggested_shares),\
        "position_risk_amount": float(position_risk_amount)\
    \}\
    return res\
\
def run_scan(tickers, lookback_days, atr_multiplier, min_avg_volume, account_size, risk_pct, max_tickers=500):\
    """Run scanner over tickers list and return DataFrame of results."""\
    rows = []\
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t and t.strip()]))[:max_tickers]\
    for t in tickers:\
        try:\
            r = analyze_ticker(t, lookback_days, atr_multiplier, min_avg_volume, account_size, risk_pct)\
            if r is not None:\
                rows.append(r)\
        except Exception as e:\
            # skip problematic tickers - continue scan\
            print(f"error scanning \{t\}: \{e\}")\
            continue\
    if not rows:\
        return pd.DataFrame()\
    df = pd.DataFrame(rows)\
    # add kind of score to sort: buy first\
    df["score"] = df["buy"].apply(lambda x: 2 if x else (1 if df["sell"].any() else 0))\
    df = df.sort_values(by=["buy", "avg_vol_30"], ascending=[False, False]).reset_index(drop=True)\
    # nice formatting\
    df["price"] = df["price"].round(2)\
    df["sma50"] = df["sma50"].round(2)\
    df["sma200"] = df["sma200"].round(2)\
    df["rsi14"] = df["rsi14"].round(1)\
    df["atr14"] = df["atr14"].round(3)\
    df["suggested_stop"] = df["suggested_stop"].round(2)\
    return df\
\
# -------------------------\
# Streamlit UI\
# -------------------------\
\
st.title("\uc0\u55357 \u56590  Swing Scanner \'97 SMA / RSI / ATR (Streamlit)")\
st.markdown("A small, free scanner for swing trading (mid/small caps). Use your own tickers or paste/upload a CSV. "\
            "Signals use: SMA50/SMA200 trend filter, RSI14, ATR-based stop, and position sizing by risk %.")\
\
# Sidebar controls\
st.sidebar.header("Scanner parameters")\
lookback_days = st.sidebar.number_input("Historical lookback (days)", value=400, min_value=120, max_value=2000, step=30)\
atr_multiplier = st.sidebar.number_input("ATR multiplier for stop (e.g. 1.5)", value=1.5, min_value=0.5, step=0.1, format="%.2f")\
min_avg_volume = st.sidebar.number_input("Min avg daily volume (30-day) to include (0 to disable)", value=100000, step=1000)\
account_size = st.sidebar.number_input("Account size (USD)", value=10000.0, min_value=100.0, step=100.0, format="%.2f")\
risk_pct = st.sidebar.number_input("Risk percent per trade (%)", value=1.0, min_value=0.1, max_value=20.0, step=0.1, format="%.2f")\
max_tickers = st.sidebar.number_input("Max tickers to scan (safety)", value=200, min_value=10, max_value=2000, step=10)\
run_button = st.sidebar.button("Run scan")\
\
st.sidebar.markdown("---")\
st.sidebar.markdown("How position sizing works:")\
st.sidebar.markdown("`position_risk_amount = account_size * (risk_pct / 100)` \'97 that's how much USD you're risking per trade. "\
                    "Shares = floor(risk_amount / (entry_price - suggested_stop)).")\
\
# Tickers input area\
st.header("Tickers (upload CSV or paste list)")\
col1, col2 = st.columns([2,1])\
\
with col1:\
    uploaded = st.file_uploader("Upload CSV file with a single column of tickers (no header required)", type=["csv", "txt"])\
    paste = st.text_area("Or paste tickers (one per line)", height=120, value="\\n".join(DEFAULT_TICKERS))\
\
with col2:\
    sample_button = st.button("Use sample tickers")\
    if sample_button:\
        paste = "\\n".join(DEFAULT_TICKERS)\
    st.markdown("Tip: use Russell2000 / S&P MidCap 400 lists (paste or upload).")\
\
# Build tickers list\
tickers = []\
if uploaded:\
    try:\
        df_in = pd.read_csv(uploaded, header=None)\
        tickers = df_in.iloc[:,0].astype(str).tolist()\
    except Exception:\
        try:\
            txt = uploaded.getvalue().decode("utf-8")\
            tickers = [line.strip() for line in txt.splitlines() if line.strip()]\
        except Exception:\
            st.error("Unable to parse uploaded file. Please upload a simple CSV or paste tickers.")\
elif paste and paste.strip():\
    tickers = [line.strip().upper() for line in paste.splitlines() if line.strip()]\
\
st.markdown(f"**Tickers to scan:** \{len(tickers)\} tickers")\
\
# Run scan on button press\
if run_button:\
    if not tickers:\
        st.error("No tickers provided. Paste tickers or upload a CSV file.")\
    else:\
        with st.spinner("Running scanner. This can take a little while depending on how many tickers you provided..."):\
            df_res = run_scan(tickers=tickers,\
                              lookback_days=lookback_days,\
                              atr_multiplier=atr_multiplier,\
                              min_avg_volume=min_avg_volume,\
                              account_size=account_size,\
                              risk_pct=risk_pct,\
                              max_tickers=max_tickers)\
        if df_res.empty:\
            st.warning("No results (all tickers failed or none passed filters). Try lowering min avg volume or scanning fewer tickers.")\
        else:\
            # Show summary counts\
            buys = df_res[df_res["buy"] == True]\
            sells = df_res[df_res["sell"] == True]\
            st.markdown(f"### Results \'97 \{len(df_res)\} scanned | \{len(buys)\} buy signals | \{len(sells)\} sell signals")\
            st.dataframe(df_res.drop(columns=["score"]), height=480)\
\
            # Provide CSV download\
            csv = df_res.to_csv(index=False).encode("utf-8")\
            st.download_button("Download CSV", data=csv, file_name="scanner_results.csv", mime="text/csv")\
\
            # Quick export: show only buys\
            if len(buys):\
                st.markdown("#### Buy candidates")\
                st.table(buys[["ticker","price","sma50","sma200","rsi14","atr14","avg_vol_30","suggested_stop","suggested_shares"]].reset_index(drop=True))\
\
# Footer / instructions\
st.markdown("---")\
st.markdown("### Next steps / tips")\
st.markdown("""\
- Replace the small example tickers with a full list (Russell 2000 or S&P MidCap 400 constituents). You can find them online and paste/upload as CSV.\
- If you plan to trade real money: backtest this strategy on historical data first (simulate fills, slippage, commissions, and execution rules).\
- To schedule this to run automatically and host it online: deploy to Streamlit Cloud (free tier available) or run on a small VPS and open port 8501.\
- To improve signals: add volume spikes, MACD filters, fundamental filters (market cap), or use a more robust backtester like backtrader/vectorbt.\
""")\
}