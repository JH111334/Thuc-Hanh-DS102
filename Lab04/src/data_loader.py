"""Load Wine Quality dataset (red + white)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_data(
    data_root: Path | str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and split Wine Quality data."""
    if data_root is None:
        data_root = _PROJECT_ROOT / "Data"
    data_root = Path(data_root)

    red_df = pd.read_csv(data_root / "winequality-red.csv", sep=";")
    white_df = pd.read_csv(data_root / "winequality-white.csv", sep=";")

    red_df["wine_type"] = 0
    white_df["wine_type"] = 1

    df = pd.concat([red_df, white_df], ignore_index=True)
    y = df["quality"].values.astype(np.int32)
    X = df.drop(columns=["quality"]).values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[data_loader] train: {X_train.shape[0]} | test: {X_test.shape[0]}")
    print(f"[data_loader] features: {X_train.shape[1]} | classes: {len(np.unique(y))}")
    return X_train, X_test, y_train, y_test
