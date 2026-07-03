# Compare the two models: table, plots, and confusion matrices
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

from utils import LABELS

RESULTS_DIR = "results"

MODELS = [
    ("TF-IDF + LogReg", "baseline"),
    ("DistilBERT", "transformer"),
]


def load(name):
    with open(f"{RESULTS_DIR}/{name}.json") as f:
        metrics = json.load(f)
    preds = pd.read_csv(f"{RESULTS_DIR}/{name}_preds.csv")
    return metrics, preds


def print_table(data):
    print("\n=== Overall performance (95% CI from bootstrap) ===")
    print(f"{'Model':<20}{'Accuracy':<24}{'Macro-F1':<24}")
    for label, m, _ in data:
        acc = f"{m['accuracy']:.3f} [{m['accuracy_ci'][0]:.3f}, {m['accuracy_ci'][1]:.3f}]"
        f1 = f"{m['macro_f1']:.3f} [{m['macro_f1_ci'][0]:.3f}, {m['macro_f1_ci'][1]:.3f}]"
        print(f"{label:<20}{acc:<24}{f1:<24}")


def per_class_table(data):
    print("\n=== Per-class F1 ===")
    header = f"{'Class':<12}" + "".join(f"{lbl:<20}" for lbl, _, _ in data)
    print(header)
    rows = {}
    for lbl, _, preds in data:
        f1s = f1_score(preds["true"], preds["pred"], labels=LABELS, average=None)
        rows[lbl] = dict(zip(LABELS, f1s))
    for cls in LABELS:
        line = f"{cls:<12}" + "".join(f"{rows[lbl][cls]:<20.3f}" for lbl, _, _ in data)
        print(line)
    return rows


def plot_accuracy(data):
    # accuracy with 95% CI error bars
    fig, ax = plt.subplots(figsize=(6, 4))
    names = [lbl for lbl, _, _ in data]
    accs = [m["accuracy"] for _, m, _ in data]
    lo = [m["accuracy"] - m["accuracy_ci"][0] for _, m, _ in data]
    hi = [m["accuracy_ci"][1] - m["accuracy"] for _, m, _ in data]
    ax.bar(names, accs, yerr=[lo, hi], capsize=8, color=["#7fa8d4", "#d98e73"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Test accuracy with 95% confidence intervals")
    for i, a in enumerate(accs):
        ax.text(i, a + 0.03, f"{a:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/accuracy_ci.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR}/accuracy_ci.png")


def plot_per_class_f1(rows):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(LABELS))
    width = 0.35
    for i, (lbl, vals) in enumerate(rows.items()):
        ax.bar(x + (i - 0.5) * width, [vals[c] for c in LABELS], width, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/per_class_f1.png", dpi=150)
    print(f"Saved {RESULTS_DIR}/per_class_f1.png")


def plot_confusion(data):
    fig, axes = plt.subplots(1, len(data), figsize=(11, 4.5))
    for ax, (lbl, _, preds) in zip(axes, data):
        cm = confusion_matrix(preds["true"], preds["pred"], labels=LABELS, normalize="true")
        disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f")
        ax.set_title(lbl)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/confusion_matrices.png", dpi=150)
    print(f"Saved {RESULTS_DIR}/confusion_matrices.png")


def significance(data):
    # CIs don't overlap -> significant difference
    (_, a, _), (_, b, _) = data
    a_hi, b_lo = a["accuracy_ci"][1], b["accuracy_ci"][0]
    print("\n=== Significance ===")
    if b_lo > a_hi:
        print(f"DistilBERT's accuracy CI lower bound ({b_lo:.3f}) is above the "
              f"baseline's upper bound ({a_hi:.3f}); the 95% CIs do not overlap, "
              f"so the improvement is statistically significant.")
    else:
        print("The 95% confidence intervals overlap; the difference is not clearly significant.")


def main():
    data = [(lbl, *load(name)) for lbl, name in MODELS]
    print_table(data)
    rows = per_class_table(data)
    plot_accuracy(data)
    plot_per_class_f1(rows)
    plot_confusion(data)
    significance(data)


if __name__ == "__main__":
    main()
