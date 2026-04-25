"""
Data loader for the Chest X-Ray Pneumonia dataset.

Loads images from the dataset directory, resizes them to a target size,
converts to grayscale, and returns flattened feature vectors with labels.

Expected data layout
--------------------
    data/
    ├── train/raw/train/{NORMAL, PNEUMONIA}/
    └── test/raw/test/{NORMAL, PNEUMONIA}/
"""

import numpy as np
from pathlib import Path
from PIL import Image

IMG_SIZE = (128, 128)
CLASS_TO_LABEL = {"NORMAL": -1, "PNEUMONIA": 1}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Resolve the project root (Lab03/) relative to this file's location (src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_split(data_root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load all images from a single split directory.

    The images are expected at:
        <data_root>/<split>/raw/<split>/<class_name>/*.{jpg,png,...}

    Parameters
    ----------
    data_root : Path
        Root ``data/`` directory (e.g. ``Lab03/data``).
    split : str
        Name of the split folder (``"train"`` or ``"test"``).

    Returns
    -------
    X : np.ndarray of shape (n_samples, IMG_SIZE[0]*IMG_SIZE[1])
    y : np.ndarray of shape (n_samples,)
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = data_root / split / "raw" / split / class_name
        if not class_dir.is_dir():
            print(f"[data_loader] WARNING: directory not found, skipping — {class_dir}")
            continue
        for p in class_dir.iterdir():
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS:
                try:
                    with Image.open(p) as img:
                        arr = (
                            np.asarray(img.convert("L").resize(IMG_SIZE), dtype=np.float32)
                            / 255.0
                        )
                    X_list.append(arr.reshape(-1))
                    y_list.append(label)
                except Exception as exc:
                    print(f"[data_loader] WARNING: failed to load {p.name} — {exc}")

    if not X_list:
        raise FileNotFoundError(
            f"No images found for split '{split}' under {data_root / split / 'raw' / split}. "
            f"Expected sub-folders: {list(CLASS_TO_LABEL.keys())}"
        )

    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.int32),
    )


def load_data(data_root: Path | str | None = None):
    """Load train and test data.

    Parameters
    ----------
    data_root : Path or str, optional
        Path to the ``data/`` directory.
        Defaults to ``<project_root>/data`` (i.e. ``Lab03/data``).

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
    """
    if data_root is None:
        data_root = _PROJECT_ROOT / "data"
    data_root = Path(data_root)

    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root directory not found: {data_root}")

    X_train, y_train = load_split(data_root, "train")
    X_test, y_test = load_split(data_root, "test")

    print(f"[data_loader] train samples: {X_train.shape[0]}  |  test samples: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test
