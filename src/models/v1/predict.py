import pandas as pd
import joblib
from pathlib import Path
from src.config import PROCESSED_DATA_DIR, MODELS_DIR

FEATURES = None  # sera déduit automatiquement


def predict_top_clients(
    model_name: str,
    top_n: int,
    csv_name: str = "processed_sales_2024.csv"
):
    # -----------------------
    # Charger données
    # -----------------------
    data = pd.read_csv(PROCESSED_DATA_DIR / csv_name)

    global FEATURES
    FEATURES = [c for c in data.columns if c != "CLV"]

    X = data[FEATURES]

    # -----------------------
    # Charger modèle
    # -----------------------
    model_path = MODELS_DIR / "v1" / model_name
    model = joblib.load(model_path)

    model_path = MODELS_DIR / "v1" / model_name

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}"
        )

    # -----------------------
    # Prédictions
    # -----------------------
    data["predicted_clv"] = model.predict(X)

    # -----------------------
    # Top clients
    # -----------------------
    top_clients = (
        data
        .sort_values("predicted_clv", ascending=False)
        .head(top_n)
        [["predicted_clv"]]
        .reset_index()
        .rename(columns={"index": "client_id"})
    )

    return top_clients.to_dict(orient="records")
