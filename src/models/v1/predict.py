import pandas as pd
import joblib
from pathlib import Path
from src.config import PROCESSED_DATA_DIR, MODELS_DIR

FEATURES = None  # déduit automatiquement


def predict_top_clients(
    model_name: str,
    top_n: int,
    csv_name: str = "processed_sales_2024.csv",
    include_true_label: bool = True  # 👈 switch PROD / DEBUG
):
    # -----------------------
    # Charger données
    # -----------------------
    data_path = PROCESSED_DATA_DIR / csv_name
    data = pd.read_csv(data_path)

    if "CLV" not in data.columns:
        raise ValueError("La colonne 'CLV' est absente du dataset")

    global FEATURES
    FEATURES = [c for c in data.columns if c != "CLV"]

    X = data[FEATURES]

    # -----------------------
    # Charger modèle
    # -----------------------
    model_path = MODELS_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    model = joblib.load(model_path)

    # -----------------------
    # Prédictions
    # -----------------------
    data["predicted_clv"] = model.predict(X)

    # -----------------------
    # Top clients
    # -----------------------
    cols = ["predicted_clv"]
    if include_true_label:
        cols.append("CLV")

    top_clients = (
        data
        .sort_values("predicted_clv", ascending=False)
        .head(top_n)
        [cols]
        .reset_index()
        .rename(columns={
            "index": "client_id",
            "CLV": "true_clv"
        })
    )

    return top_clients.to_dict(orient="records")
