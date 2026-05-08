"""Run all assignments and print comparison."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier as SklearnDT

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_loader import load_data
from task1 import DecisionTreeClassifier
from task2 import RandomForestClassifier

np.random.seed(42)


def main():
    X_train, X_test, y_train, y_test = load_data()

    print("\n=== Assignment 1 - NumPy Decision Tree ===")
    dt_np = DecisionTreeClassifier(max_depth=12, min_samples_split=5, seed=42)
    dt_np.fit(X_train, y_train)
    f1_dt_np = f1_score(y_test, dt_np.predict(X_test), average="weighted", zero_division=0)
    print(f"F1-score (weighted): {f1_dt_np:.4f}")

    print("\n=== Assignment 2 - NumPy Random Forest ===")
    rf_np = RandomForestClassifier(
        n_estimators=20, max_depth=12, min_samples_split=5,
        max_features=int(np.sqrt(X_train.shape[1])), seed=42,
    )
    rf_np.fit(X_train, y_train)
    f1_rf_np = f1_score(y_test, rf_np.predict(X_test), average="weighted", zero_division=0)
    print(f"F1-score (weighted): {f1_rf_np:.4f}")

    print("\n=== Assignment 3 - Sklearn Library ===")
    dt_sk = SklearnDT(max_depth=12, min_samples_split=5, random_state=42)
    dt_sk.fit(X_train, y_train)
    f1_dt_sk = f1_score(y_test, dt_sk.predict(X_test), average="weighted", zero_division=0)
    print(f"Decision Tree F1-score (weighted): {f1_dt_sk:.4f}")

    rf_sk = SklearnRF(
        n_estimators=20, max_depth=12, min_samples_split=5,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    rf_sk.fit(X_train, y_train)
    f1_rf_sk = f1_score(y_test, rf_sk.predict(X_test), average="weighted", zero_division=0)
    print(f"Random Forest F1-score (weighted): {f1_rf_sk:.4f}")

    print("\n=== Comparison ===")
    print(
        pd.DataFrame(
            [
                {"Model": "NumPy Decision Tree", "F1-score": f1_dt_np},
                {"Model": "NumPy Random Forest", "F1-score": f1_rf_np},
                {"Model": "Sklearn Decision Tree", "F1-score": f1_dt_sk},
                {"Model": "Sklearn Random Forest", "F1-score": f1_rf_sk},
            ]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
