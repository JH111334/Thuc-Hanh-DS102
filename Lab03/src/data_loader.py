"""
Data loader for the Chest X-Ray Pneumonia dataset.

Loads images from the dataset directory, resizes them to a target size,
converts to grayscale, and returns flattened feature vectors with labels.
The val split is merged into train so that every sample is used for training.
"""

import numpy as np
from pathlib import Path
from PIL import Image

IMG_SIZE = (128, 128)
CLASS_TO_LABEL = {"NORMAL": -1, "PNEUMONIA": 1}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_split(dataset_root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load all images from a single split directory.

    Parameters
    ----------
    dataset_root : Path
        Root directory of the chest_xray dataset.
    split : str
        Name of the split folder (e.g. "train", "val", "test").

    Returns
    -------
    X : np.ndarray of shape (n_samples, IMG_SIZE[0]*IMG_SIZE[1])
    y : np.ndarray of shape (n_samples,)
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = dataset_root / split / class_name
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
                except Exception:
                    pass
    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.int32),
    )


def load_data(dataset_root: Path | str | None = None):
    """Load train and test data, merging the val split into train.

    Parameters
    ----------
    dataset_root : Path or str, optional
        Path to the ``chest_xray`` directory.  Defaults to ``./data/chest_xray``.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
    """
    if dataset_root is None:
        dataset_root = Path.cwd() / "data" / "chest_xray"
    dataset_root = Path(dataset_root)

    # Load every available split and merge val into train
    X_train, y_train = load_split(dataset_root, "train")
    X_val, y_val = load_split(dataset_root, "val")
    X_test, y_test = load_split(dataset_root, "test")

    # Merge val samples into the training set
    X_train = np.concatenate([X_train, X_val], axis=0)
    y_train = np.concatenate([y_train, y_val], axis=0)

    print(f"[data_loader] train samples: {X_train.shape[0]}  |  test samples: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test
