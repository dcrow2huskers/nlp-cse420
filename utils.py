# Shared helpers for loading data and scoring models
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = "data"
LABELS = ["negative", "neutral", "positive", "ambiguous"]


def load_split(name):
    df = pd.read_csv(f"{DATA_DIR}/{name}.csv")
    return df["text"].tolist(), df["sentiment"].tolist()


def bootstrap_ci(y_true, y_pred, metric="accuracy", n=1000, seed=0):
    # resample the test set to get a 95% confidence interval for the metric
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if metric == "accuracy":
            scores.append(accuracy_score(y_true[idx], y_pred[idx]))
        else:
            scores.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(np.mean(scores)), float(lo), float(hi)


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc_mean, acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, "accuracy")
    f1_mean, f1_lo, f1_hi = bootstrap_ci(y_true, y_pred, "macro_f1")
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "accuracy_ci": (acc_lo, acc_hi),
        "macro_f1_ci": (f1_lo, f1_hi),
    }
