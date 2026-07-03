# Model 1: TF-IDF features + Logistic Regression
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from utils import load_split, evaluate, LABELS

RESULTS_DIR = "results"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    # word + char n-grams tend to help on short, noisy Reddit text
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # dataset is imbalanced
            C=1.0,
        )),
    ])

    print("Training TF-IDF + Logistic Regression ...")
    model.fit(X_train, y_train)

    print("Evaluating on test set ...")
    y_pred = model.predict(X_test)

    print("\n" + classification_report(y_test, y_pred, labels=LABELS, digits=3))

    metrics = evaluate(y_test, y_pred)
    print(f"Accuracy: {metrics['accuracy']:.3f} "
          f"(95% CI {metrics['accuracy_ci'][0]:.3f}-{metrics['accuracy_ci'][1]:.3f})")
    print(f"Macro-F1: {metrics['macro_f1']:.3f} "
          f"(95% CI {metrics['macro_f1_ci'][0]:.3f}-{metrics['macro_f1_ci'][1]:.3f})")

    metrics["model"] = "tfidf_logreg"
    with open(f"{RESULTS_DIR}/baseline.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {RESULTS_DIR}/baseline.json")


if __name__ == "__main__":
    main()
