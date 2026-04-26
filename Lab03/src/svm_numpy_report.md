# Báo cáo Module: Soft-margin SVM bằng NumPy

> **File:** `svm_numpy.py`
> **Mục đích:** Xây dựng mô hình SVM (Support Vector Machine) từ đầu bằng NumPy để phân loại ảnh X-quang phổi thành hai lớp NORMAL (-1) và PNEUMONIA (+1).

---

## 1. Tổng quan

Module `svm_numpy.py` triển khai toàn bộ quy trình huấn luyện và đánh giá một mô hình **Soft-margin SVM** sử dụng **Full-batch Gradient Descent**, chỉ dựa trên thư viện NumPy cho phần tối ưu hóa. Đây là bài tập nhằm hiểu rõ bản chất toán học đằng sau SVM, thay vì phụ thuộc hoàn toàn vào thư viện có sẵn.

---

## 2. Phương pháp sử dụng

### 2.1. Chuẩn hóa dữ liệu — `standardize(X_train, X_test)`

- **Z-score Normalization:** Chuẩn hóa đặc trưng về trung bình bằng 0 và phương sai bằng 1.
- Tham số `mean` và `std` được tính **chỉ trên tập train** rồi áp dụng lên cả train và test, tránh rò rỉ dữ liệu (data leakage).
- Thêm hằng số nhỏ `1e-8` vào `std` để tránh chia cho 0.

### 2.2. Huấn luyện — `train_svm(X_train, y_train, lr, C, epochs, seed)`

| Tham số   | Mặc định | Ý nghĩa |
|-----------|----------|----------|
| `lr`      | `5e-4`   | Tốc độ học (learning rate) |
| `C`       | `2.0`    | Hệ số phạt cho các mẫu vi phạm lề (margin) |
| `epochs`  | `8`      | Số vòng lặp huấn luyện |
| `seed`    | `42`     | Seed đảm bảo tái lập kết quả |

- **Hàm mất mát:** Sử dụng **Hinge Loss** kết hợp với **L2 Regularization**:

  $$L(w, b) = \frac{1}{2} \|w\|^2 + C \cdot \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i (w \cdot x_i + b))$$

- **Gradient Descent:** Tại mỗi epoch, mô hình tính gradient của hàm mất mát theo `w` và `b`, sau đó cập nhật:
  - Với các mẫu **vi phạm lề** (active: $y_i \cdot f(x_i) < 1$):
    - $\nabla_w = w - C \cdot \frac{1}{n} \sum_{i \in \text{active}} y_i \cdot x_i$
    - $\nabla_b = -C \cdot \frac{1}{n} \sum_{i \in \text{active}} y_i$
  - Với các mẫu **không vi phạm**: gradient chỉ chứa thành phần regularization $w$.

### 2.3. Dự đoán — `predict(X, w, b)`

- Tính giá trị quyết định $f(x) = w \cdot x + b$.
- Trả về `+1` nếu $f(x) \geq 0$, ngược lại trả về `-1`.

### 2.4. Đánh giá — `evaluate(y_true, y_pred)`

Sử dụng ba chỉ số từ `sklearn.metrics`:

| Chỉ số    | Ý nghĩa |
|-----------|----------|
| **Precision** | Tỷ lệ dự đoán đúng trong số các mẫu được dự đoán là dương (PNEUMONIA). |
| **Recall**    | Tỷ lệ mẫu dương (PNEUMONIA) thực tế được phát hiện đúng. |
| **F1-score**  | Trung bình điều hòa (harmonic mean) của Precision và Recall. |

---

## 3. Lý do phương pháp này hiệu quả

### 3.1. SVM phù hợp cho bài toán phân loại nhị phân

- SVM tìm **siêu phẳng phân tách có lề lớn nhất** (maximum margin hyperplane), giúp mô hình có khả năng tổng quát hóa (generalization) tốt trên dữ liệu chưa thấy.
- Trong bài toán phân loại ảnh X-quang (NORMAL vs. PNEUMONIA), dữ liệu sau khi chuẩn hóa và chuyển sang vector 1 chiều thường có ranh giới tuyến tính hợp lý.

### 3.2. Soft-margin cho phép xử lý nhiễu

- Hệ số `C` kiểm soát sự cân bằng giữa **tối đa hóa lề** và **giảm thiểu lỗi phân loại**:
  - `C` lớn: mô hình cố gắng phân loại đúng mọi mẫu, dễ overfitting.
  - `C` nhỏ: mô hình chấp nhận một số lỗi, lề rộng hơn, tổng quát hóa tốt hơn.

### 3.3. Chuẩn hóa Z-score cải thiện tốc độ hội tụ

- Khi các đặc trưng có cùng thang đo, gradient descent hội tụ nhanh hơn vì bề mặt hàm mất mát đồng đều hơn (tránh hiện tượng "zigzag").

### 3.4. Tự triển khai giúp hiểu sâu thuật toán

- Khác với việc gọi hàm thư viện, tự viết SVM bằng NumPy buộc người học phải hiểu rõ cách tính gradient, ý nghĩa của từng siêu tham số, và cơ chế hoạt động bên trong.

---

## 4. Quy trình thực thi

```
  Tải dữ liệu ảnh X-quang
         │
         ▼
  Chuẩn hóa Z-score (fit trên train)
         │
         ▼
  Khởi tạo w = 0, b = 0
         │
         ▼
  ┌─────────────────────────────┐
  │  Lặp qua từng epoch:       │
  │  1. Hoán vị ngẫu nhiên     │
  │  2. Tính score = Xw + b    │
  │  3. Xác định mẫu vi phạm  │
  │  4. Tính gradient           │
  │  5. Cập nhật w, b           │
  └─────────────────────────────┘
         │
         ▼
  Dự đoán trên tập test
         │
         ▼
  Đánh giá: Precision, Recall, F1
```

### Các bước chi tiết:

1. **Tải dữ liệu:** Module `data_loader.py` đọc ảnh X-quang, resize về `128x128`, chuyển ảnh xám, và làm phẳng thành vector `16384` chiều.
2. **Chuẩn hóa:** Hàm `standardize()` áp dụng Z-score normalization.
3. **Huấn luyện:** Hàm `train_svm()` chạy full-batch gradient descent trong `8` epoch.
4. **Dự đoán:** Hàm `predict()` phân loại các mẫu test dựa trên dấu của $f(x)$.
5. **Đánh giá:** Hàm `evaluate()` tính Precision, Recall, F1 với `pos_label=1` (PNEUMONIA).

---

## 5. Cấu trúc hàm

```
svm_numpy.py
├── standardize(X_train, X_test)      → (X_train_std, X_test_std)
├── train_svm(X_train, y_train, ...)  → (w, b)
├── predict(X, w, b)                  → y_pred
└── evaluate(y_true, y_pred)          → {"Precision", "Recall", "F1"}
```
