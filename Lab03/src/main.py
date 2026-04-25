"""
Main entry-point — runs both assignments and prints the comparison table.

Usage:
    python main.py
"""

import numpy as np
import pandas as pd

from data_loader import load_data
from svm_numpy import standardize, train_svm, predict, evaluate as np_evaluate
from svm_sklearn import train_sklearn_svm, evaluate as sk_evaluate

np.random.seed(42)


def main():
    # ── Data loading (val merged into train) ─────────────────────────────
    X_train, y_train, X_test, y_test = load_data()

    # ── Standardize ──────────────────────────────────────────────────────
    X_train_std, X_test_std = standardize(X_train, X_test)

    # ── Assignment 1: NumPy Soft-margin SVM ──────────────────────────────
    print("\n=== Assignment 1 — NumPy Soft-margin SVM ===")
    w, b = train_svm(X_train_std, y_train)
    y_pred_np = predict(X_test_std, w, b)
    metrics_np = np_evaluate(y_test, y_pred_np)
    metrics_np["Model"] = "NumPy Soft-margin SVM (TEST)"

    print(f"Precision: {metrics_np['Precision']:.4f}")
    print(f"Recall   : {metrics_np['Recall']:.4f}")
    print(f"F1-score : {metrics_np['F1']:.4f}")

    # ── Assignment 2: Sklearn LinearSVC ──────────────────────────────────
    print("\n=== Assignment 2 — Sklearn LinearSVC ===")
    model = train_sklearn_svm(X_train_std, y_train)
    y_pred_sk = model.predict(X_test_std)
    metrics_sk = sk_evaluate(y_test, y_pred_sk)
    metrics_sk["Model"] = "Sklearn LinearSVC (TEST)"

    print(f"Precision: {metrics_sk['Precision']:.4f}")
    print(f"Recall   : {metrics_sk['Recall']:.4f}")
    print(f"F1-score : {metrics_sk['F1']:.4f}")

    # ── Comparison table ─────────────────────────────────────────────────
    print("\n=== Comparison ===")
    comparison_df = pd.DataFrame(
        [
            {"Metric": "Precision", "NumPy": metrics_np["Precision"], "Sklearn": metrics_sk["Precision"]},
            {"Metric": "Recall",    "NumPy": metrics_np["Recall"],    "Sklearn": metrics_sk["Recall"]},
            {"Metric": "F1",        "NumPy": metrics_np["F1"],        "Sklearn": metrics_sk["F1"]},
        ]
    )
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
