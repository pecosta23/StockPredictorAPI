# app/tests.py

from app.services.fetcher import get_stock_data
from app.services.features import calculate_features
from app.services.predictor import train_model
import pandas as pd

df_tickers = pd.read_excel("app/data/TickersB3.xlsx")
#df_tickers = pd.read_excel("app/data/classificacao_tickers.xlsx")

tickers = df_tickers.iloc[:, 0].dropna().tolist()

# Coloca SA no final (B3 no yf)
tickers = [ticker if ticker.endswith(".SA") else f"{ticker}.SA" for ticker in tickers]

print(f"{len(tickers)} tickers carregados")
print(tickers[:10])

df_list = []

for ticker in tickers:
    df = get_stock_data(ticker, period="10y")
    df_features = calculate_features(df)
    df_features["ticker"] = ticker
    df_list.append(df_features)

df_full = pd.concat(df_list).reset_index(drop=True)

df_full["ticker_code"] = df_full["ticker"].astype('category').cat.codes

train_model(df_full)