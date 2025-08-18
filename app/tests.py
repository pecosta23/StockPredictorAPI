from services.fetcher import get_stock_data
from services.features import calculate_features
from services.predictor import train_model
import pandas as pd

tickers = ["PETR4.SA", "NVDA", "NKE", "DIS"] 

df_list = []

for ticker in tickers:
    df = get_stock_data(ticker)
    df_features = calculate_features(df)
    df_features["ticker"] = ticker
    df_list.append(df_features)
    print(df_features[["return_1d", "ma7", "ma21", "volatility", "rsi", "target"]].tail())

df_full = pd.concat(df_list).reset_index(drop=True)
train_model(df_full)