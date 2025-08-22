# app/services/fetcher.py

import yfinance as yf
import pandas as pd

def get_stock_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df
    