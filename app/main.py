# app/main.py

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.services.fetcher import get_stock_data
from app.services.features import calculate_features
from app.services.predictor import load_model_and_predict

app = FastAPI(title="Stock Trend Predictor")

@app.get("/")
def root():
    return {"message": "API - Stock Trend Predictor is alive!"}

@app.get("/predict/{ticker}")
def predict_stock(ticker: str):
    try:
        df = get_stock_data(ticker)
        df_feat = calculate_features(df)
        prediction = load_model_and_predict(df_feat)

        trend = "subir" if prediction == 1 else "cair"

        sample_data = df_feat.tail(10).to_dict(orient="records")

        return JSONResponse(content={
            "ticker": ticker, 
            "predicition": int(prediction),
            "message": f"A ação tende a {trend} no próximo dia.",
            "metrics": sample_data
        })
    
    except Exception as e:
        return {"error": str(e)}
    