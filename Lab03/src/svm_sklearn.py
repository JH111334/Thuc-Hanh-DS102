"""
Assignment 2 — SVM using sklearn's LinearSVC.
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
    """Fit a LinearSVC on the training data and return the model."""
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
    """Return a dict with Precision, Recall, and F1."""
    return {
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
