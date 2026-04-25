"""
Assignment 1 — NumPy Soft-margin SVM trained with Mini-batch SGD.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def standardize(X_train, X_test):
    """Zero-mean, unit-variance standardization fitted on train."""
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lr: float = 5e-4,
    C: float = 2.0,
    epochs: int = 8,
    seed: int = 42,
):
    """Train a soft-margin SVM with full-batch gradient descent.

    Returns
    -------
    w : np.ndarray   – weight vector
    b : float        – bias
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features, dtype=np.float32)
    b = 0.0

    for epoch in range(epochs):
        perm = rng.permutation(n_samples)
        X_epoch = X_train[perm]
        y_epoch = y_train[perm]

        scores = X_epoch @ w + b
        active = (y_epoch * scores) < 1.0

        grad_w = w.copy()
        grad_b = 0.0
        if np.any(active):
            grad_w -= C * (X_epoch[active].T @ y_epoch[active]) / n_samples
            grad_b = -C * np.sum(y_epoch[active]) / n_samples

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def predict(X, w, b):
    """Return predictions in {-1, +1}."""
    return np.where((X @ w + b) >= 0.0, 1, -1).astype(np.int32)


def evaluate(y_true, y_pred):
    """Return a dict with Precision, Recall, and F1."""
    return {
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
