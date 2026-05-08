"""Assignment 2 - Random Forest with NumPy."""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_loader import load_data
from task1 import DecisionTreeClassifier


class RandomForestClassifier:
    """Random forest using bootstrap aggregation."""

    def __init__(
        self,
        n_estimators: int = 20,
        max_depth: int = 12,
        min_samples_split: int = 5,
        max_features: int | None = None,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.trees: list[DecisionTreeClassifier] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        rng = np.random.default_rng(self.seed)
        n_samples = X.shape[0]
        self.trees = []

        for i in range(self.n_estimators):
            idx = rng.integers(0, n_samples, size=n_samples)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                seed=self.seed + i,
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise ValueError("Forest not fitted.")

        predictions = np.array([tree.predict(X) for tree in self.trees])
        result = np.empty(X.shape[0], dtype=np.int32)
        for i in range(X.shape[0]):
            result[i] = int(np.bincount(predictions[:, i]).argmax())
        return result


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"F1": f1_score(y_true, y_pred, average="weighted", zero_division=0)}


def main():
    print("\n=== Assignment 2 - NumPy Random Forest ===")
    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=12,
        min_samples_split=5,
        max_features=int(np.sqrt(X_train.shape[1])),
        seed=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"F1-score (weighted): {evaluate(y_test, y_pred)['F1']:.4f}")


if __name__ == "__main__":
    main()
