"""
Module tải dữ liệu cho tập dữ liệu Chest X-Ray Pneumonia.

Tải hình ảnh từ thư mục tập dữ liệu, thay đổi kích thước theo kích thước mục tiêu,
chuyển đổi sang ảnh xám, và trả về các vector đặc trưng 1 chiều kèm theo nhãn.

Cấu trúc dữ liệu dự kiến
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

# Xác định thư mục gốc của dự án (Lab03/) dựa trên vị trí của file này (src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_split(data_root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Tải tất cả hình ảnh từ một thư mục phân chia (split) duy nhất.

    Các hình ảnh được dự kiến nằm tại:
        <data_root>/<split>/raw/<split>/<class_name>/*.{jpg,png,...}

    Tham số
    ----------
    data_root : Path
        Thư mục gốc ``data/`` (ví dụ: ``Lab03/data``).
    split : str
        Tên của thư mục phân chia (``"train"`` hoặc ``"test"``).

    Trả về
    -------
    X : np.ndarray có kích thước (n_samples, IMG_SIZE[0]*IMG_SIZE[1])
    y : np.ndarray có kích thước (n_samples,)
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = data_root / split / "raw" / split / class_name
        if not class_dir.is_dir():
            print(f"[data_loader] CẢNH BÁO: không tìm thấy thư mục, đang bỏ qua — {class_dir}")
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
                    print(f"[data_loader] CẢNH BÁO: không thể tải {p.name} — {exc}")

    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.int32),
    )


def load_data(data_root: Path | str | None = None):
    """Tải dữ liệu huấn luyện (train) và kiểm tra (test).

    Tham số
    ----------
    data_root : Path hoặc str, tùy chọn
        Đường dẫn đến thư mục ``data/``.
        Mặc định là ``<project_root>/data`` (tức là ``Lab03/data``).

    Trả về
    -------
    X_train, y_train, X_test, y_test : np.ndarray
    """
    if data_root is None:
        data_root = _PROJECT_ROOT / "data"
    data_root = Path(data_root)

    X_train, y_train = load_split(data_root, "train")
    X_test, y_test = load_split(data_root, "test")

    print(f"[data_loader] số mẫu train: {X_train.shape[0]}  |  số mẫu test: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test
