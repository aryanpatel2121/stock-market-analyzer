import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from stock_analyzer import sma, ema, rsi, macd, bollinger, atr, compute_metrics, sma_crossover_strategy

st.set_page_config(page_title="Stock Market Analyzer", layout="wide")

st.title("📈 Stock Market Analyzer")

# Sidebar inputs
ticker = st.sidebar.text_input("Ticker (e.g., AAPL, RELIANCE.NS)", "AAPL")
start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("today"))
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"])
fast_sma = st.sidebar.number_input("Fast SMA", 5, 100, 20)
slow_sma = st.sidebar.number_input("Slow SMA", 10, 200, 50)
risk_free = st.sidebar.number_input("Risk-free rate", 0.0, 0.1, 0.02, step=0.01)

if st.sidebar.button("Run Analysis"):
    # Fetch data
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        st.error("No data returned. Try another ticker or date range.")
    else:
        # Add indicators
        df[f"SMA_{fast_sma}"] = sma(df["Close"], fast_sma)
        df[f"SMA_{slow_sma}"] = sma(df["Close"], slow_sma)
        df["RSI_14"] = rsi(df["Close"])
        macd_line, signal_line = macd(df["Close"])
        df["MACD"] = macd_line
        df["MACD_Signal"] = signal_line
        mid, upper, lower = bollinger(df["Close"])
        df["BB_Mid"], df["BB_Upper"], df["BB_Lower"] = mid, upper, lower

        # Metrics
        metrics = compute_metrics(df, risk_free)
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Annual Return", f"{metrics.ann_return:.2%}")
        col2.metric("Volatility", f"{metrics.ann_volatility:.2%}")
        col3.metric("Sharpe", f"{metrics.sharpe:.2f}")
        col4.metric("Max DD", f"{metrics.max_drawdown:.2%}")
        col5.metric("CAGR", f"{metrics.cagr:.2%}")

        # Charts
        st.subheader(f"{ticker} Price with SMA + Bollinger")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df["Close"], label="Close")
        ax.plot(df.index, df[f"SMA_{fast_sma}"], label=f"SMA {fast_sma}")
        ax.plot(df.index, df[f"SMA_{slow_sma}"], label=f"SMA {slow_sma}")
        ax.plot(df.index, df["BB_Mid"], label="BB Mid", linestyle="--")
        ax.plot(df.index, df["BB_Upper"], label="BB Upper", linestyle="--")
        ax.plot(df.index, df["BB_Lower"], label="BB Lower", linestyle="--")
        ax.legend()
        st.pyplot(fig)

        st.subheader("MACD")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df.index, df["MACD"], label="MACD")
        ax.plot(df.index, df["MACD_Signal"], label="Signal")
        ax.axhline(0, linestyle="--", color="gray")
        ax.legend()
        st.pyplot(fig)

        st.subheader("RSI (14)")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df.index, df["RSI_14"], label="RSI")
        ax.axhline(70, linestyle="--", color="red")
        ax.axhline(30, linestyle="--", color="green")
        ax.legend()
        st.pyplot(fig)
        
        st.sidebar.subheader("Multi-Stock Screener")
tickers_input = st.sidebar.text_area(
    "Enter tickers (comma separated)", 
    "AAPL,MSFT,TSLA,GOOGL,AMZN"
)

if st.sidebar.button("Run Screener"):
    tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    results = []

    st.subheader("📊 Multi-Stock Screener Results")

    for tk in tickers_list:
        df = yf.download(tk, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
        if df.empty:
            st.warning(f"{tk}: No data found.")
            continue
        try:
            metrics = compute_metrics(df, risk_free)
            results.append({
                "Ticker": tk,
                "Annual Return": metrics.ann_return,
                "Volatility": metrics.ann_volatility,
                "Sharpe": metrics.sharpe,
                "Max DD": metrics.max_drawdown,
                "CAGR": metrics.cagr,
            })
        except Exception as e:
            st.error(f"{tk}: {e}")

    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="Sharpe", ascending=False)

        st.dataframe(df_results.style.format({
            "Annual Return": "{:.2%}",
            "Volatility": "{:.2%}",
            "Sharpe": "{:.2f}",
            "Max DD": "{:.2%}",
            "CAGR": "{:.2%}",
        }))

        st.success(f"✅ Best Stock Pick (by Sharpe Ratio): **{df_results.iloc[0]['Ticker']}**")


        # Option to download data
        csv = df.to_csv().encode("utf-8")
        st.download_button("Download CSV ✅ ", data=csv, file_name=f"{ticker}_analysis.csv", mime="text/csv")
