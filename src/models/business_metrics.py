import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def compute_business_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    prop_engaged: float = 0.1,
    cost_program: float = 1500,
    gain_top_client: float = 5000
):
    # --- Sécurisation types ---
    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true)

    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred, index=y_true.index)

    n = len(y_true)
    k = int(np.ceil(prop_engaged * n))

    # True top-k
    true_thresh = y_true.nlargest(k).min()
    y_true_bin = (y_true >= true_thresh).astype(int)

    # Predicted top-k
    pred_thresh = y_pred.nlargest(k).min()
    y_pred_bin = (y_pred >= pred_thresh).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()

    # ML metrics
    metrics = {
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

    # Business metrics
    business_gain = tp * (gain_top_client - cost_program)
    business_cost = fp * cost_program

    metrics["business"] = {
        "n_engaged": int(k),
        "business_gain": int(business_gain),
        "business_cost": int(business_cost),
        "profit": int(business_gain - business_cost),
    }

    return metrics
