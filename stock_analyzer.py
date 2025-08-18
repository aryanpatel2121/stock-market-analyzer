#!/usr/bin/env python3
"""
Stock Market Data Analyzer
- Downloads price data with yfinance
- Computes indicators (SMA/EMA/RSI/MACD/Bollinger/ATR)
- Computes metrics (returns, volatility, Sharpe, drawdown)
- Optional simple SMA crossover backtest
- Saves indicators CSV and summary PNG per ticker
"""

import argparse
import math
import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


@dataclass
class Metrics:
    ann_return: float
    ann_volatility: float
    sharpe: float
    max_drawdown: float
    cagr: float


def fetch_data(ticker: str, start: str, end: Optional[str], interval: str) -> pd.DataFrame:
    """Download OHLCV data. auto_adjust=True adjusts for splits/dividends."""
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the symbol/date range/interval.")

    # If MultiIndex (e.g., ('Open','AAPL')), flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] else col[1] for col in df.columns.values]

    # Standardize expected columns
    rename_map = {}
    for col in df.columns:
        if col.lower().startswith("adj close"):
            rename_map[col] = "Close"
        elif col.lower().startswith("close"):
            rename_map[col] = "Close"
        elif col.lower().startswith("open"):
            rename_map[col] = "Open"
        elif col.lower().startswith("high"):
            rename_map[col] = "High"
        elif col.lower().startswith("low"):
            rename_map[col] = "Low"
        elif col.lower().startswith("volume"):
            rename_map[col] = "Volume"
    df = df.rename(columns=rename_map)

    # Make sure Close at least exists
    if "Close" not in df.columns:
        raise ValueError(f"{ticker}: No 'Close' column in data returned")

    return df



def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    # Ensure we always have a 1D Series
    if isinstance(series, pd.DataFrame):
        series = series.squeeze("columns")

    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)

    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/window, adjust=False).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi



def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def to_float(val):
    """Safely convert pandas/NumPy scalars to Python float."""
    if hasattr(val, "item"):
        try:
            return val.item()
        except Exception:
            pass
    return float(val)


def compute_metrics(df: pd.DataFrame, risk_free: float = 0.0):
    returns = df["Close"].pct_change().dropna()

    ann_return = to_float((1 + returns.mean()) ** 252 - 1)
    ann_vol = to_float(returns.std() * (252 ** 0.5))

    sharpe = np.nan
    if not np.isnan(ann_vol) and ann_vol > 0:
        sharpe = (ann_return - risk_free) / ann_vol

    max_dd = to_float(((df["Close"] / df["Close"].cummax()) - 1).min())
    cagr = to_float((df["Close"].iloc[-1] / df["Close"].iloc[0]) ** (252 / len(df)) - 1)

    return Metrics(ann_return, ann_vol, sharpe, max_dd, cagr)



def sma_crossover_strategy(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    fast_sma = sma(close, fast)
    slow_sma = sma(close, slow)
    signal = (fast_sma > slow_sma).astype(int)
    # Avoid look-ahead bias
    strat_rets = close.pct_change().fillna(0) * signal.shift(1).fillna(0)
    equity = (1 + strat_rets).cumprod()
    out = pd.DataFrame({
        "Close": close,
        f"SMA_{fast}": fast_sma,
        f"SMA_{slow}": slow_sma,
        "Signal": signal,
        "Strategy_Equity": equity
    })
    return out


def plot_summary(ticker: str, df: pd.DataFrame, outpath: str, fast_sma: int, slow_sma: int, show: bool = False):
    # Compute indicators for plotting
    mid, upper, lower = bollinger(df["Close"])
    rsi_vals = rsi(df["Close"])
    macd_line, signal_line = macd(df["Close"])
    vol = df["Close"].pct_change().rolling(20).std() * np.sqrt(252)

    fig = plt.figure(figsize=(12, 10))

    # Price + Bollinger + SMAs
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(df.index, df["Close"], label="Close")
    ax1.plot(df.index, sma(df["Close"], fast_sma), label=f"SMA {fast_sma}")
    ax1.plot(df.index, sma(df["Close"], slow_sma), label=f"SMA {slow_sma}")
    ax1.plot(df.index, mid, label="BB Mid")
    ax1.plot(df.index, upper, label="BB Upper")
    ax1.plot(df.index, lower, label="BB Lower")
    ax1.set_title(f"{ticker} — Price & Bands")
    ax1.legend(loc="best")

    # MACD
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(df.index, macd_line, label="MACD")
    ax2.plot(df.index, signal_line, label="Signal")
    ax2.axhline(0, linestyle="--", linewidth=1)
    ax2.set_title("MACD")
    ax2.legend(loc="best")

    # RSI & Vol
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(df.index, rsi_vals, label="RSI(14)")
    ax3.axhline(30, linestyle="--", linewidth=1)
    ax3.axhline(70, linestyle="--", linewidth=1)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df.index, vol, label="Ann. Vol (20d)", alpha=0.6)
    ax3.set_title("RSI & Rolling Volatility")
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def analyze_ticker(ticker: str, start: str, end: Optional[str], interval: str,
                   risk_free: float, fast_sma: int, slow_sma: int, outdir: str, show_plots: bool):
    df = fetch_data(ticker, start, end, interval)
    df = df.dropna()

    # Indicators
    df_out = pd.DataFrame(index=df.index)
    df_out["Open"] = df["Open"]
    df_out["High"] = df["High"]
    df_out["Low"] = df["Low"]
    df_out["Close"] = df["Close"]
    df_out["Volume"] = df["Volume"]
    df_out["Return"] = df["Close"].pct_change()
    df_out[f"SMA_{fast_sma}"] = sma(df["Close"], fast_sma)
    df_out[f"SMA_{slow_sma}"] = sma(df["Close"], slow_sma)
    df_out["EMA_12"] = ema(df["Close"], 12)
    df_out["EMA_26"] = ema(df["Close"], 26)
    macd_line, signal_line = macd(df["Close"])
    df_out["MACD"] = macd_line
    df_out["MACD_Signal"] = signal_line
    mid, upper, lower = bollinger(df["Close"])
    df_out["BB_Mid"] = mid
    df_out["BB_Upper"] = upper
    df_out["BB_Lower"] = lower
    df_out["RSI_14"] = rsi(df["Close"])
    df_out["ATR_14"] = atr(df["High"], df["Low"], df["Close"])
    df_out["RollVol_20"] = df["Close"].pct_change().rolling(20).std() * np.sqrt(252)

    # Strategy
    strat = sma_crossover_strategy(df["Close"], fast_sma, slow_sma)
    df_out["Strategy_Equity"] = strat["Strategy_Equity"]

    # Metrics
    metrics = compute_metrics(df, risk_free=risk_free)
    metrics_dict = {
        "ticker": ticker,
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "annual_return": metrics.ann_return,
        "annual_volatility": metrics.ann_volatility,
        "sharpe": metrics.sharpe,
        "max_drawdown": metrics.max_drawdown,
        "cagr": metrics.cagr,
    }

    # Save outputs
        # Save outputs
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"{ticker}_indicators.csv")

    # Reset index so Date is a normal column
    df_out_reset = df_out.reset_index()
    df_out_reset.rename(columns={"index": "Date"}, inplace=True)
    df_out_reset.to_csv(csv_path, index=False)

    json_path = os.path.join(outdir, f"{ticker}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    png_path = os.path.join(outdir, f"{ticker}_summary.png")
    plot_summary(ticker, df, png_path, fast_sma, slow_sma, show=show_plots)

    # Console summary
    print(f"\n=== {ticker} Results ===")
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            print(f"{k:17s}: {v:,.4f}")
        else:
            print(f"{k:17s}: {v}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {png_path}")


def main():
    p = argparse.ArgumentParser(description="Stock Market Data Analyzer")
    p.add_argument("--tickers", "-t", type=str, required=True,
                   help="Comma-separated tickers, e.g. AAPL,MSFT,TSLA")
    p.add_argument("--start", "-s", type=str, default="2018-01-01",
                   help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", "-e", type=str, default=None,
                   help="End date (YYYY-MM-DD). Default: latest")
    p.add_argument("--interval", "-i", type=str, default="1d",
                   choices=["1d", "1wk", "1mo", "1h", "30m", "15m", "5m", "1m"],
                   help="Data interval (note: intraday needs recent start date)")
    p.add_argument("--risk_free", type=float, default=0.0, help="Annual risk-free rate for Sharpe (e.g., 0.02)")
    p.add_argument("--fast_sma", type=int, default=20, help="Fast SMA window for plots/strategy")
    p.add_argument("--slow_sma", type=int, default=50, help="Slow SMA window for plots/strategy")
    p.add_argument("--outdir", type=str, default="output", help="Output directory")
    p.add_argument("--show_plots", action="store_true", help="Display plots interactively")

    args = p.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    for tk in tickers:
        try:
            analyze_ticker(
                ticker=tk,
                start=args.start,
                end=args.end,
                interval=args.interval,
                risk_free=args.risk_free,
                fast_sma=args.fast_sma,
                slow_sma=args.slow_sma,
                outdir=args.outdir,
                show_plots=args.show_plots,
            )
        except Exception as ex:
            print(f"[ERROR] {tk}: {ex}")


if __name__ == "__main__":
    main()
