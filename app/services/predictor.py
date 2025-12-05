# app/services/predictor.py

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def train_model(df: pd.DataFrame, model_path: str = "app/models/ml_model.pkl"):
    features = ["return_1d", "ma7", "ma21", "rsi", "volatility", "ticker_code"]
    df['target_xgb'] = df['target'].map({-1:0, 0:1, 1:2})

    X = df[features]
    y = df['target_xgb']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    use_label_encoder=False
    )
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("Relatório de Classificação:")
    print(report)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Modelo salvo em -> {model_path}")


def load_model_and_predict(df_feat: pd.DataFrame, model_path: str = "app/models/ml_model.pkl") -> int:
    features = ["return_1d", "ma7", "ma21", "rsi", "volatility", "ticker_code"]
    model = joblib.load(model_path)

    last_row = df_feat[features].iloc[-1].values.reshape(1, -1)

    prediction = model.predict(last_row)[0]

    action_map = {
    2: "Comprar",
    1: "Manter",
    0: "Vender"
    }


    result = {
        "prediction": int(prediction),
        "action": action_map[int(prediction)]
    }

    return result


