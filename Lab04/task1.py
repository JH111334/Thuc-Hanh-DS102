"""Assignment 1 - Decision Tree with NumPy."""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_loader import load_data


class _Node:
    """Internal tree node."""

    def __init__(self, gini: float, num_samples: int, num_classes: int, prediction: int):
        self.gini = gini
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.prediction = prediction
        self.feature_index: int | None = None
        self.threshold: float | None = None
        self.left: "_Node | None" = None
        self.right: "_Node | None" = None

    def __repr__(self) -> str:
        return f"_Node(pred={self.prediction}, n={self.num_samples}, gini={self.gini:.4f})"


class DecisionTreeClassifier:
    """Decision tree classifier using Gini impurity."""

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features: int | None = None,
        seed: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.root: _Node | None = None
        self.rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError("Tree not fitted.")
        return np.array([self._traverse(x, self.root) for x in X])

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        n_samples, n_classes = y.size, len(np.unique(y))
        prediction = int(np.bincount(y, minlength=np.max(y) + 1).argmax())
        node = _Node(gini=self._gini(y), num_samples=n_samples, num_classes=n_classes, prediction=prediction)

        if depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            return node

        feature_indices = np.arange(X.shape[1])
        if self.max_features is not None:
            feature_indices = self.rng.choice(feature_indices, size=min(self.max_features, X.shape[1]), replace=False)

        best_idx, best_thr, best_gain = self._best_split(X, y, feature_indices)
        if best_idx is None:
            return node

        left_mask = X[:, best_idx] <= best_thr
        if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
            return node

        node.feature_index = best_idx
        node.threshold = best_thr
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray
    ) -> tuple[int | None, float | None, float]:
        best_gain = -1.0
        best_idx, best_thr = None, None
        parent_gini = self._gini(y)

        for idx in feature_indices:
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                left = y[X[:, idx] <= thr]
                right = y[X[:, idx] > thr]
                if left.size == 0 or right.size == 0:
                    continue
                gain = parent_gini - (len(left) * self._gini(left) + len(right) * self._gini(right)) / len(y)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = int(idx)
                    best_thr = float(thr)

        return best_idx, best_thr, best_gain

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        proportions = np.bincount(y) / y.size
        return 1.0 - np.sum(proportions ** 2)

    @staticmethod
    def _traverse(x: np.ndarray, node: _Node) -> int:
        if node.left is None or node.right is None:
            return node.prediction
        if x[node.feature_index] <= node.threshold:
            return DecisionTreeClassifier._traverse(x, node.left)
        return DecisionTreeClassifier._traverse(x, node.right)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"F1": f1_score(y_true, y_pred, average="weighted", zero_division=0)}


def main():
    print("=== Assignment 1 - NumPy Decision Tree ===")
    X_train, X_test, y_train, y_test = load_data()

    model = DecisionTreeClassifier(max_depth=12, min_samples_split=5, seed=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"F1-score (weighted): {evaluate(y_test, y_pred)['F1']:.4f}")


if __name__ == "__main__":
    main()
