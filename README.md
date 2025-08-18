# 📈 Stock Market Analyzer  

A Python-based **Stock Market Analysis Tool** with:  
- Command Line (CLI) for fast analysis  
- Streamlit Web App for visualization  
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)  
- Portfolio metrics (Annual Return, Volatility, Sharpe Ratio, Max Drawdown, CAGR)  
- **Multi-stock screener** to rank and find the best stocks  

---

## 🚀 Features
- Download historical stock data from **Yahoo Finance**  
- Calculate common **technical indicators**  
- Generate metrics like **Sharpe ratio, CAGR, Max Drawdown**  
- Visualize stock performance and indicators  
- **Rank multiple stocks** to find top performers  
- Export results to CSV  

---

## 📦 Installation  

```bash
git clone https://github.com/yourusername/stock-market-analyzer.git
cd stock-market-analyzer
python -m venv my_env
source my_env/bin/activate   # (Mac/Linux)
my_env\Scripts\activate      # (Windows)
pip install -r requirements.txt
```

---

## 🖥️ Usage  

### CLI (Command Line)
Run analysis for a stock:  
```bash
python stock_analyzer.py --tickers AAPL --start 2020-01-01 --end 2024-01-01
```

Run for multiple tickers:  
```bash
python stock_analyzer.py --tickers AAPL,MSFT,NVDA,TSLA --start 2018-01-01 --end 2024-01-01 --rank
```

This will generate:  
- Individual CSV files with stock data + indicators  
- `stock_ranking.csv` with best stocks ranked by Sharpe Ratio  

---

### Streamlit App  

Launch the app:  
```bash
streamlit run app.py
```

Features:  
- Enter **one ticker** → Get price chart + SMA, RSI, MACD, Bollinger  
- Metrics dashboard (Annual Return, Volatility, Sharpe, Max DD, CAGR)  
- Download analysis CSV  
- Enter **multiple tickers** → Screener ranks stocks by performance  

---

## 📊 Example  

**Streamlit Dashboard**  
- Price chart with SMA + Bollinger  
- MACD & RSI charts  
- Metrics cards  
- Multi-stock screener (table + best pick suggestion)  

**CLI Ranking Output**  
```
Ticker   | Annual Return | Volatility | Sharpe | Max DD | CAGR
--------------------------------------------------------------
AAPL     | 0.25          | 0.18       | 1.38   | -0.20  | 0.22
MSFT     | 0.23          | 0.16       | 1.44   | -0.18  | 0.20
TSLA     | 0.35          | 0.40       | 0.87   | -0.45  | 0.32
```
