import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import re
from src.config import MODELS_DIR, PROCESSED_DATA_DIR


# -----------------------
# Utils
# -----------------------
def parse_model_tag(model_path: Path):
    """
    Extrait v, sub, param, size depuis :
    v1.2.param1.size15.joblib
    """
    name = model_path.stem  # sans .joblib
    pattern = r"(v\d+)\.(\d+)\.param(\d+)\.size(\d+)"
    match = re.match(pattern, name)
    if not match:
        raise ValueError(f"Nom de modèle invalide : {name}")

    v_key = match.group(1)
    sub_key = match.group(2)
    param_key = f"param{match.group(3)}"
    size_key = f"size{match.group(4)}"

    return v_key, sub_key, param_key, size_key


# -----------------------
# Main
# -----------------------
def evaluate_model(model_path: Path,
                   csv_name: str = "processed_sales_2024.csv",
                   target_col: str = "CLV") -> None:
    """
    Évalue un modèle et injecte les métriques dans le registre :

    v1 -> sub -> param -> size -> metrics

    Metrics :
    - RMSE
    - MAE
    - MAPE
    - R2
    sur train et test

    + Intervalle de confiance bootstrap de l'écart-type de y
    Validation :
    - les features du CSV doivent correspondre à celles de l'entraînement
    - la taille du test doit correspondre
    """

    # -----------------------
    # Charger modèle
    # -----------------------
    model = joblib.load(model_path)

    # -----------------------
    # Identifier sa clé registre
    # -----------------------
    v_key, sub_key, param_key, size_key = parse_model_tag(model_path)

    model_dir = Path(model_path).parent
    registry_path = model_dir / "registry.json"

    if not registry_path.exists():
        raise FileNotFoundError(f"Registre introuvable : {registry_path}")

    with open(registry_path, "r") as f:
        registry = json.load(f)

    try:
        entry = registry[v_key]["models"][sub_key][param_key][size_key]
    except KeyError:
        raise KeyError(
            f"Entrée modèle introuvable dans registre : "
            f"{v_key} → {sub_key} → {param_key} → {size_key}"
        )

    # -----------------------
    # Charger données
    # -----------------------
    data_path = PROCESSED_DATA_DIR / csv_name
    data = pd.read_csv(data_path)
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # -----------------------
    # Validation features
    # -----------------------
    trained_features = entry["features"]
    if set(X.columns) != set(trained_features):
        raise ValueError(
            f"Mismatch des features !\n"
            f"CSV : {list(X.columns)}\n"
            f"Training : {trained_features}"
        )

    # -----------------------
    # Validation test_size
    # -----------------------
    test_size_train = entry.get("test_size", 0.2)
    if abs(len(X) * 0.2 - len(X) * test_size_train) > 1:
        print(
            f"Warning : la taille du test dans le CSV ({len(X)*0.2:.0f}) "
            f"diffère de celle utilisée à l'entraînement ({len(X)*test_size_train:.0f})"
        )

    # -----------------------
    # Split train/test
    # -----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_train, random_state=42
    )

    # -----------------------
    # Prédictions
    # -----------------------
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # -----------------------
    # Metrics
    # -----------------------
    def calc_metrics(y_true, y_pred):
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        mape = float(
            np.mean(
                np.abs((y_true - y_pred) /
                       np.where(y_true == 0, 1e-8, y_true))
            ) * 100
        )
        r2 = float(r2_score(y_true, y_pred))
        return {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2
        }

    metrics_train = calc_metrics(y_train, y_pred_train)
    metrics_test = calc_metrics(y_test, y_pred_test)

    # -----------------------
    # IC bootstrap sur std(y)
    # -----------------------
    B = 1000
    s_boot = [
        np.std(np.random.choice(y, len(y), replace=True), ddof=1)
        for _ in range(B)
    ]
    ci_lower, ci_upper = np.percentile(s_boot, [2.5, 97.5])
    ci = {
        "y_std": float(np.std(y, ddof=1)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "method": "bootstrap",
        "B": B
    }

    # -----------------------
    # Injection registre
    # -----------------------
    entry["metrics"] = {
        "train": metrics_train,
        "test": metrics_test,
        "y_std_interval": ci
    }
    entry["evaluated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # -----------------------
    # Sauvegarde
    # -----------------------
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)

    # -----------------------
    # Logs
    # -----------------------
    print("Évaluation complétée pour :", model_path.name)
    print("Chemin registre :", registry_path)
    print("Train metrics :", metrics_train)
    print("Test metrics :", metrics_test)
    print("IC std(y) :", ci)


# -----------------------
# CLI simple
# -----------------------
if __name__ == "__main__":
    print("Évaluation du modèle...")

    model_file = MODELS_DIR / "v1" / "v1.2.param1.size0.joblib"

    evaluate_model(
        model_path=model_file,
        csv_name="processed_sales_2024.csv",
        target_col="CLV"
    )

    print("Évaluation terminée.")
