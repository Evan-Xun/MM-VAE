import csv
import json
import math
import os
import random
from pathlib import Path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def first_float(text):
    import re
    match = re.search(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", text or "")
    return float(match.group(0)) if match else math.nan


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    npv = tn / max(1, tn + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    roc_auc = None
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "specificity": specificity, "npv": npv, "f1": f1, "roc_auc": roc_auc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def append_summary(path, row):
    fieldnames = [
        "experiment_name", "model", "feature_set", "image_input", "pooling",
        "latent_dim", "accuracy", "precision", "recall", "specificity",
        "npv", "f1", "roc_auc", "tn", "fp", "fn", "tp",
    ]
    rows = []
    if os.path.exists(path):
        rows = read_csv(path)
        rows = [r for r in rows if r.get("experiment_name") != row.get("experiment_name")]
    rows.append({k: row.get(k, "") for k in fieldnames})
    write_csv(path, rows, fieldnames)
