# app/services/features.py

import pandas as pd
import numpy as np

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Retorno diário
    df["return_1d"] = df["Close"].pct_change()

    # Média móvel, 7 dias, 21 dias
    df["ma7"] = df["Close"].rolling(window=7).mean()
    df["ma21"] = df["Close"].rolling(window=21).mean()

    # Volatividade (desvio padrão retornos)
    df["volatility"] = df["return_1d"].rolling(window=21).std()

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14).mean()

    rs = gain/loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Retorno do próximo dia
    df["return_next"] = df["Close"].shift(-1) / df["Close"] - 1

    # Define target
    threshold = 0.005 #0.5%
    df["target"] = 0
    df.loc[df["return_next"] > threshold, "target"] = 1
    df.loc[df["return_next"] < -threshold, "target"] = -1

    # Remove NaNs
    df.dropna(inplace=True)

    # Revome as timezones só quando for salvar no excel
    if "Date" in df.columns:
       df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)    
    
    df_export = df.copy()
    df_export["Date"] = df_export["Date"].astype(str)
    df_export.to_excel("app/data/Metricas.xlsx", index=False)
    return df


