import pandas as pd
import numpy as np

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Retorno diário
    df["return_1d"] = df["Close"].pct_change()

    # Média móvel, 7 dias, 21 dias
    df["ma7"] = df["Close"].rolling(window=7).mean()
    df["ma21"] = df["Close"].rolling(window=7).mean()

    # Volatividade (desvio padrão retornos)
    df["volatility"] = df["return_1d"].rolling(window=7).std()

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain/loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Target, preço alvo (caso o fechamento do dia seguinte seja maior do que o de hoje)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Remove NaNs
    df.dropna(inplace=True)

    # Remove as timezones do excel
    #if "Date" in df.columns:
    #    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)    
    df["Date"] = df["Date"].astype(str)

    df.to_excel("app/data/Metricas.xlsx", index=False)
    return df


