# app/services/features.py
import pandas as pd
import numpy as np

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Retorno diário
    df["return_1d"] = df["Close"].pct_change()

    # MAs
    df["ma7"] = df["Close"].rolling(window=7).mean()
    df["ma21"] = df["Close"].rolling(window=21).mean()

    # Volatilidade
    df["volatility"] = df["return_1d"].rolling(window=21).std()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).ewm(span=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14).mean()

    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Próximo retorno (target)
    df["return_next"] = df["Close"].shift(-1) / df["Close"] - 1

    threshold = 0.005
    df["target"] = 0
    df.loc[df["return_next"] > threshold, "target"] = 1
    df.loc[df["return_next"] < -threshold, "target"] = -1

    df.dropna(inplace=True)

    # ⚠ NÃO RETORNAR JSON AQUI
    return df
