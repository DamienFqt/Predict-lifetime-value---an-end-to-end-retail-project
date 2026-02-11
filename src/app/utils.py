import json
from pathlib import Path
from src.config import MODELS_DIR

REGISTRY_PATH = MODELS_DIR / "registry.json"

def load_registry():
    """Charge le registry JSON"""
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def get_model_metrics(model_name: str, registry: dict):
    import re
    pattern = r"(v\d+)\.(\d+)\.(param\d+)\.(size\d+)"
    match = re.match(pattern, model_name.replace(".joblib", ""))

    if not match:
        raise ValueError("Nom de modèle invalide")

    v, sub, param, size = match.groups()
    entry = registry[v]["models"][sub][param][size]

    # Metrics avec fallback
    metrics = entry.get("metrics") or {}
    train_metrics = metrics.get("train") or {}
    test_metrics = metrics.get("test") or {}

    # Business metrics avec fallback complet
    bm = entry.get("business_metrics") or {}
    train_bm = bm.get("train") or {}
    test_bm = bm.get("test") or {}

    return {
        "algo": registry[v].get("algo"),
        "target": registry[v].get("target"),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "business_metrics": {
            "train": train_bm,   # <-- garder toutes les clés train (precision, recall, business)
            "test": test_bm      # <-- idem
        },
        "evaluated_at": entry.get("evaluated_at"),
        "features": entry.get("features")
    }

