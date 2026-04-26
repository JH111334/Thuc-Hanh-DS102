"""
Bài tập 2 — SVM sử dụng LinearSVC của thư viện sklearn.
"""

from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


def train_sklearn_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 2.0,
    max_iter: int = 8000,
    seed: int = 42,
) -> LinearSVC:
    """Huấn luyện (fit) một LinearSVC trên dữ liệu train và trả về mô hình."""
    model = LinearSVC(
        C=C,
        loss="hinge",
        max_iter=max_iter,
        random_state=seed,
        dual=True,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(y_true, y_pred):
    """Trả về một dictionary chứa Precision, Recall, và F1."""
    return {
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
