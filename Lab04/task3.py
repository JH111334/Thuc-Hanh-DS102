"""Assignment 3 - Decision Tree & Random Forest with sklearn."""

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_loader import load_data


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"F1": f1_score(y_true, y_pred, average="weighted", zero_division=0)}


def main():
    print("\n=== Assignment 3 - Sklearn Decision Tree & Random Forest ===")
    X_train, X_test, y_train, y_test = load_data()

    print("\n--- Decision Tree ---")
    dt = DecisionTreeClassifier(max_depth=12, min_samples_split=5, random_state=42)
    dt.fit(X_train, y_train)
    print(f"F1-score (weighted): {evaluate(y_test, dt.predict(X_test))['F1']:.4f}")

    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=20,
        max_depth=12,
        min_samples_split=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print(f"F1-score (weighted): {evaluate(y_test, rf.predict(X_test))['F1']:.4f}")


if __name__ == "__main__":
    main()
