import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from datetime import datetime
import json
from src.config import PROCESSED_DATA_DIR, MODELS_DIR


def _load_registry(registry_path: Path) -> dict:
    if registry_path.exists():
        with open(registry_path, "r") as f:
            return json.load(f)
    return {}


def _save_registry(registry_path: Path, registry: dict):
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)


def _features_equal(f1, f2):
    return set(f1) == set(f2)


def _hyperparams_equal(h1, h2):
    return h1 == h2


def _next_int(keys):
    if not keys:
        return 0
    return max(int(k) for k in keys) + 1


def train_rf_model(csv_name: str = "processed_sales_2024.csv",
                   version: str = "v1",
                   target_col: str = "CLV",
                   algo_name: str = "rf",
                   n_estimators: int = 100,
                   test_size: float = 0.2) -> Path:
    """
    Entraîne un RandomForestRegressor et enregistre le modèle dans un registre versionné.

    Naming du modèle :
        v{version}.{sub}.param{param}.size{k}.joblib

    Logique registre :
        v1 -> sous_version -> param_version -> size_k
        - features changent => new sub_version
        - hyperparams changent => new param_version
        - size change => nouvelle entrée size
        - combinaison exacte => écrase
    """

    # -----------------------
    # Charger les données
    # -----------------------
    data_path = PROCESSED_DATA_DIR / csv_name
    data = pd.read_csv(data_path)

    X = data.drop(columns=[target_col])
    y = data[target_col]

    features = list(X.columns)
    size_k = len(X) // 1000

    print("Features utilisées :", features)
    print(f"Taille dataset : {size_k}k")

    # -----------------------
    # Registre
    # -----------------------
    version_clean = version.replace("v", "")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    registry_path = MODELS_DIR / "registry.json"
    registry = _load_registry(registry_path)

    v_key = f"v{version_clean}"
    registry.setdefault(v_key, {
        "target": target_col,
        "algo": algo_name,
        "models": {}
    })

    models = registry[v_key]["models"]

    # -----------------------
    # Trouver sub_version
    # -----------------------
    sub_version = None

    for sub_k, params_block in models.items():
        for param_block in params_block.values():
            for entry in param_block.values():
                if _features_equal(entry["features"], features):
                    sub_version = int(sub_k)
                    break
            if sub_version is not None:
                break
        if sub_version is not None:
            break

    if sub_version is None:
        sub_version = _next_int(models.keys())

    # -----------------------
    # Trouver param_version
    # -----------------------
    hyperparams = {
        "n_estimators": n_estimators,
        "random_state": 42
    }

    sub_key = str(sub_version)
    models.setdefault(sub_key, {})

    param_version = None
    for param_k, size_block in models[sub_key].items():
        for entry in size_block.values():
            if _hyperparams_equal(entry["hyperparameters"], hyperparams):
                param_version = int(param_k.replace("param", ""))
                break
        if param_version is not None:
            break

    if param_version is None:
        param_keys = [k.replace("param", "") for k in models[sub_key].keys()]
        param_version = _next_int(param_keys)

    # -----------------------
    # Entraînement
    # -----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42
    )
    model.fit(X_train, y_train)

    # -----------------------
    # Naming & sauvegarde
    # -----------------------
    model_tag = f"v{version_clean}.{sub_version}.param{param_version}.size{size_k}"
    model_name = f"{model_tag}.joblib"
    model_path = MODELS_DIR / model_name

    joblib.dump(model, model_path)

    # -----------------------
    # Mise à jour registre
    # -----------------------
    param_key = f"param{param_version}"
    size_key = f"size{size_k}"

    models[sub_key].setdefault(param_key, {})
    models[sub_key][param_key][size_key] = {
        "model_file": model_name,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_rows": len(X),
        "features": features,
        "hyperparameters": hyperparams,
        "test_size": test_size,
        "source_csv": csv_name
    }

    _save_registry(registry_path, registry)

    # -----------------------
    # Logs
    # -----------------------
    print(f"Modèle sauvegardé : {model_path}")
    print(f"Registre mis à jour : {registry_path}")
    print(f"Tag modèle : {model_tag}")

    return model_path


# ----------------------- # CLI simple # ----------------------- 
if __name__ == "__main__": 
    print("Entraînement du modèle...")
    train_rf_model(csv_name= "processed_sales_2024.csv",
                   version="v1",
                   target_col="CLV",
                   algo_name="rf",
                   n_estimators=100,
                   test_size=0.2)