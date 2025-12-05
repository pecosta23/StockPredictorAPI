# app/main.py

import hashlib
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.services.fetcher import get_stock_data
from app.services.features import calculate_features
from app.services.predictor import load_model_and_predict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder

app = FastAPI(title="Stock Trend Predictor")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # adicione aqui outros domínios que usar
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def ticker_code(ticker: str) -> int:
    return int(hashlib.sha256(ticker.encode()).hexdigest(), 16) % 10000


@app.get("/")
def root():
    return {"message": "API - Stock Trend Predictor is alive!"}


@app.get("/predict/{ticker}")
def predict_stock(ticker: str):
    try:
        # 1. Baixa dados
        df = get_stock_data(ticker)

        # 2. Calcula features
        df_feat = calculate_features(df)

        # 3. Adiciona ticker_code fixo
        df_feat["ticker_code"] = ticker_code(ticker)

        # 4. Predição
        pred = load_model_and_predict(df_feat)

        pred_value = pred["prediction"]
        pred_action = pred["action"]

        # 5. Texto da tendência
        trend_text = {
            2: "subir",
            1: "manter",
            0: "cair"
        }.get(pred_value, "desconhecido")

        # 6. Histórico formatado para o React
        hist = df.tail(30).copy()
        hist["Date"] = hist["Date"].dt.strftime("%Y-%m-%d")
        history = hist[["Date", "Close"]].to_dict(orient="records")

        # 7. Métricas para análise
        metrics = df_feat.tail(10).copy()
        if "Date" in metrics.columns:
            metrics["Date"] = metrics["Date"].dt.strftime("%Y-%m-%d")

        metrics = metrics.to_dict(orient="records")

        return JSONResponse(content={
            "ticker": ticker,
            "prediction": pred_value,
            "action": pred_action,
            "message": f"A ação tende a {trend_text} no próximo dia.",
            "history": history,
            "metrics": metrics
        })

    except Exception as e:
        return {"error": str(e)}
