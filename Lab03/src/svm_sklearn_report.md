# Báo cáo Module: SVM sử dụng LinearSVC (sklearn)

> **File:** `svm_sklearn.py`
> **Mục đích:** Sử dụng lớp `LinearSVC` từ thư viện scikit-learn để huấn luyện mô hình SVM phân loại ảnh X-quang phổi (NORMAL vs. PNEUMONIA), nhằm so sánh hiệu quả với phiên bản tự triển khai bằng NumPy.

---

## 1. Tổng quan

Module `svm_sklearn.py` đóng gói việc huấn luyện và đánh giá mô hình **Linear Support Vector Classifier** bằng thư viện `sklearn.svm.LinearSVC`. So với module `svm_numpy.py` (Bài tập 1), module này tận dụng thuật toán tối ưu hóa đã được tối ưu cao của sklearn, giúp hội tụ nhanh hơn và đạt hiệu suất ổn định hơn.

---

## 2. Phương pháp sử dụng

### 2.1. Huấn luyện — `train_sklearn_svm(X_train, y_train, C, max_iter, seed)`

| Tham số     | Mặc định | Ý nghĩa |
|-------------|----------|----------|
| `C`         | `2.0`    | Hệ số phạt — cân bằng giữa lề rộng và lỗi phân loại |
| `max_iter`  | `8000`   | Số vòng lặp tối đa cho bộ tối ưu hóa |
| `seed`      | `42`     | Seed cho tính tái lập |

**Cấu hình `LinearSVC`:**

| Thuộc tính      | Giá trị | Giải thích |
|-----------------|---------|------------|
| `loss`          | `"hinge"` | Sử dụng hinge loss — hàm mất mát chuẩn của SVM |
| `dual`          | `True`  | Giải bài toán đối ngẫu (dual) — hiệu quả khi số đặc trưng > số mẫu |
| `random_state`  | `42`    | Đảm bảo kết quả tái lập được |
| `max_iter`      | `8000`  | Đủ lớn để đảm bảo hội tụ |

- Nội bộ, `LinearSVC` sử dụng thư viện **LIBLINEAR** — một bộ giải tối ưu hóa chuyên biệt cho SVM tuyến tính, áp dụng phương pháp **Coordinate Descent** trên bài toán đối ngẫu.

### 2.2. Dự đoán

- Sử dụng trực tiếp phương thức `model.predict(X_test)` của đối tượng `LinearSVC` đã được huấn luyện.

### 2.3. Đánh giá — `evaluate(y_true, y_pred)`

Tương tự module NumPy, sử dụng ba chỉ số:

| Chỉ số    | Ý nghĩa |
|-----------|----------|
| **Precision** | Tỷ lệ dự đoán đúng trong các mẫu được dự đoán là PNEUMONIA. |
| **Recall**    | Tỷ lệ mẫu PNEUMONIA thực tế được phát hiện đúng. |
| **F1-score**  | Trung bình điều hòa của Precision và Recall. |

---

## 3. Lý do phương pháp này hiệu quả

### 3.1. LIBLINEAR — bộ giải tối ưu chuyên biệt

- `LinearSVC` sử dụng thuật toán **Coordinate Descent** trên bài toán đối ngẫu, được triển khai trong thư viện LIBLINEAR viết bằng C/C++.
- So với gradient descent thông thường (bài NumPy), coordinate descent:
  - **Hội tụ nhanh hơn** nhờ cập nhật từng biến đối ngẫu (dual variable) một cách tối ưu.
  - **Ổn định hơn** vì không phụ thuộc vào việc chọn learning rate.

### 3.2. Hinge Loss đảm bảo tính chất SVM

- `loss="hinge"` giữ nguyên bản chất SVM: chỉ các support vector (mẫu nằm gần hoặc vi phạm lề) mới ảnh hưởng đến mô hình.
- Điều này giúp mô hình **chống nhiễu tốt** và tổng quát hóa tốt hơn so với các hàm mất mát khác (ví dụ: squared hinge).

### 3.3. Giải bài toán đối ngẫu (`dual=True`)

- Với dữ liệu ảnh X-quang đã làm phẳng (`16384` đặc trưng), bài toán đối ngẫu thường hiệu quả khi $p \gg n$ (số đặc trưng lớn hơn số mẫu), nhưng LIBLINEAR cũng xử lý tốt trường hợp ngược lại.
- Giải đối ngẫu cho phép mô hình biểu diễn nghiệm thông qua các hệ số Lagrange, tiết kiệm bộ nhớ.

### 3.4. Tái sử dụng thư viện đã kiểm chứng

- sklearn đã được kiểm thử rộng rãi bởi cộng đồng, đảm bảo tính chính xác của thuật toán.
- Giúp tập trung vào bài toán ứng dụng thay vì lo lắng về lỗi triển khai.

### 3.5. So sánh với NumPy SVM

| Tiêu chí              | NumPy SVM         | sklearn LinearSVC |
|------------------------|--------------------|--------------------|
| Thuật toán tối ưu      | Full-batch GD      | Coordinate Descent |
| Tốc độ hội tụ          | Chậm hơn           | Nhanh hơn          |
| Phụ thuộc learning rate | Có                | Không              |
| Mục đích               | Học thuật toán     | Ứng dụng thực tế   |

---

## 4. Quy trình thực thi

```
  Tải dữ liệu ảnh X-quang
         │
         ▼
  Chuẩn hóa Z-score (từ svm_numpy.standardize)
         │
         ▼
  Khởi tạo LinearSVC(C=2.0, loss="hinge", dual=True)
         │
         ▼
  Gọi model.fit(X_train, y_train)
  ┌──────────────────────────────────────┐
  │  LIBLINEAR nội bộ:                   │
  │  • Chuyển sang bài toán đối ngẫu    │
  │  • Coordinate Descent trên các biến │
  │    đối ngẫu α₁, α₂, ..., αₙ        │
  │  • Lặp đến khi hội tụ hoặc hết     │
  │    max_iter = 8000                   │
  └──────────────────────────────────────┘
         │
         ▼
  Dự đoán: model.predict(X_test)
         │
         ▼
  Đánh giá: Precision, Recall, F1
```

### Các bước chi tiết:

1. **Tải dữ liệu:** Module `data_loader.py` đọc ảnh, resize `128x128`, chuyển ảnh xám, làm phẳng thành vector `16384` chiều.
2. **Chuẩn hóa:** Sử dụng hàm `standardize()` từ module `svm_numpy.py` (chia sẻ giữa hai bài tập).
3. **Huấn luyện:** Hàm `train_sklearn_svm()` gọi `LinearSVC.fit()` — LIBLINEAR tự động tối ưu bên trong.
4. **Dự đoán:** Gọi `model.predict()` trên tập test đã chuẩn hóa.
5. **Đánh giá:** Hàm `evaluate()` tính Precision, Recall, F1 với `pos_label=1` (PNEUMONIA).

---

## 5. Cấu trúc hàm

```
svm_sklearn.py
├── train_sklearn_svm(X_train, y_train, ...) → LinearSVC model
└── evaluate(y_true, y_pred)                 → {"Precision", "Recall", "F1"}
```

---

## 6. Ghi chú

- Module này phụ thuộc vào hàm `standardize()` của `svm_numpy.py` để chuẩn hóa dữ liệu trước khi huấn luyện.
- Cả hai module sử dụng cùng giá trị `C=2.0` để đảm bảo việc so sánh công bằng.
- Kết quả được in ra và so sánh trong bảng tổng hợp bởi `main.py`.
