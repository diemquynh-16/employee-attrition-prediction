# Báo cáo: IBM Employee Attrition - Classification

## 1. Giới thiệu bài toán

Bài toán được chọn là **IBM Employee Attrition Dataset**, thuộc dạng bài toán **Classification**.

**Mục tiêu:** Dự đoán một nhân viên có nghỉ việc hay không dựa trên các đặc trưng cá nhân và công việc như tuổi tác, thu nhập, phòng ban, sự hài lòng trong công việc, v.v.

**Biến mục tiêu:** `Attrition`
- `Yes` (nghỉ việc) => mã hóa thành `1`
- `No` (ở lại) => mã hóa thành `0`

**Ứng dụng thực tế:** Bài toán này có ý nghĩa quan trọng trong quản lý nhân sự. Dự đoán sớm nhân viên có nguy cơ nghỉ việc giúp doanh nghiệp chủ động giữ chân nhân tài, tiết kiệm chi phí tuyển dụng và đào tạo.

---

## 2. Mô tả dataset

| Thông tin | Giá trị |
|---|---|
| Nguồn | IBM HR Analytics Employee Attrition Dataset |
| Số dòng | 1.470 |
| Số cột | 13 |
| Biến mục tiêu | `Attrition` (Yes/No) |

### Mô tả các cột

| Cột | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| `Age` | Số nguyên | Tuổi của nhân viên |
| `Attrition` | Phân loại | Nhân viên có nghỉ việc không (Yes/No) |
| `Department` | Phân loại | Phòng ban (Sales, R&D, Human Resources) |
| `DistanceFromHome` | Số nguyên | Khoảng cách từ nhà đến nơi làm việc (km) |
| `Education` | Số nguyên | Trình độ học vấn (1–5) |
| `EducationField` | Phân loại | Lĩnh vực học vấn |
| `EnvironmentSatisfaction` | Số nguyên | Mức độ hài lòng với môi trường làm việc (1–4) |
| `JobSatisfaction` | Số nguyên | Mức độ hài lòng với công việc (1–4) |
| `MaritalStatus` | Phân loại | Tình trạng hôn nhân (Single/Married/Divorced) |
| `MonthlyIncome` | Số nguyên | Thu nhập hàng tháng (USD) |
| `NumCompaniesWorked` | Số nguyên | Số công ty đã làm việc |
| `WorkLifeBalance` | Số nguyên | Cân bằng công việc - cuộc sống (1–4) |
| `YearsAtCompany` | Số nguyên | Số năm làm việc tại công ty |

---

## 3. Khám phá dữ liệu (EDA)

### 3.1. Tổng quan

- **Kích thước:** 1.470 dòng × 13 cột
- **Giá trị thiếu:** Không có
- **Dữ liệu trùng lặp:** Không có
- **Biến số:** 9 cột (`Age`, `DistanceFromHome`, `Education`, `EnvironmentSatisfaction`, `JobSatisfaction`, `MonthlyIncome`, `NumCompaniesWorked`, `WorkLifeBalance`, `YearsAtCompany`)
- **Biến phân loại:** 3 cột (`Department`, `EducationField`, `MaritalStatus`)

### 3.2. Phân bố biến mục tiêu

| Nhãn | Số lượng | Tỷ lệ |
|---|---:|---:|
| No (Ở lại) | 1.233 | 83.9% |
| Yes (Nghỉ việc) | 237 | 16.1% |

**Nhận xét:** Dataset bị **mất cân bằng lớp nghiêm trọng** - tỷ lệ nhân viên nghỉ việc chỉ chiếm ~16%. Đây là đặc điểm điển hình của bài toán dự đoán nghỉ việc trong thực tế. Sự mất cân bằng này ảnh hưởng đáng kể đến kết quả mô hình và buộc ta phải ưu tiên F1-score thay vì Accuracy.

### 3.3. Thống kê mô tả

| Cột | Min | Mean | Max | Std |
|---|---:|---:|---:|---:|
| Age | 18 | 36.9 | 60 | 9.1 |
| MonthlyIncome | 1.009 | 6.503 | 19.999 | 4.708 |
| YearsAtCompany | 0 | 7.0 | 40 | 6.1 |
| DistanceFromHome | 1 | 9.2 | 29 | 8.1 |
| NumCompaniesWorked | 0 | 2.7 | 9 | 2.5 |

### 3.4. Nhận xét từ trực quan hóa

Từ biểu đồ boxplot theo Attrition có thể thấy:
- Nhân viên **nghỉ việc có xu hướng trẻ hơn** và có `YearsAtCompany` ngắn hơn.
- Nhân viên nghỉ việc có **MonthlyIncome thấp hơn** đáng kể so với nhân viên ở lại.
- **DistanceFromHome** cao hơn ở nhóm nghỉ việc.
- Nhân viên độc thân (`Single`) có tỷ lệ nghỉ việc cao hơn so với nhóm đã kết hôn.
- Phòng `Sales` có tỷ lệ nghỉ việc cao nhất so với các phòng khác.

---

## 4. Tiền xử lý dữ liệu

### 4.1. Làm sạch dữ liệu

- **Giá trị thiếu:** Không có => không cần xử lý.
- **Dữ liệu trùng lặp:** Không có => không cần xử lý.
- **Chuẩn hóa chuỗi:** Strip khoảng trắng đầu/cuối cho các cột kiểu object.

### 4.2. Mã hóa biến mục tiêu

```python
y = df["Attrition"].map({"Yes": 1, "No": 0})
```

### 4.3. Tách X và y

- `X`: 12 cột đặc trưng (loại bỏ cột `Attrition`)
- `y`: cột `Attrition` (0/1)

### 4.4. Xử lý đặc trưng trong Pipeline

Sử dụng `ColumnTransformer` trong `Pipeline` để tránh data leakage:
- **Biến số (9 cột):** `StandardScaler` - chuẩn hóa về mean=0, std=1
- **Biến phân loại (3 cột):** `OneHotEncoder` - mã hóa one-hot

### 4.5. Chia train/test

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Train: 1.176 mẫu | Test: 294 mẫu
```

Sử dụng `stratify=y` để đảm bảo tỷ lệ lớp tương đồng giữa train và test.

### 4.6. Xử lý mất cân bằng lớp

Do dữ liệu mất cân bằng (16% Yes), tất cả mô hình (trừ KNN) được cấu hình với `class_weight="balanced"` để mô hình chú trọng hơn vào lớp thiểu số.

---

## 5. Mô hình sử dụng

Huấn luyện 4 mô hình:

| Mô hình | Cấu hình chính |
|---|---|
| Logistic Regression | `max_iter=1000`, `class_weight="balanced"` |
| KNN | `n_neighbors=7` |
| Decision Tree | `max_depth=5`, `class_weight="balanced"` |
| Random Forest | `n_estimators=200`, `max_depth=10`, `class_weight="balanced"` |

---

## 6. Kết quả đánh giá

### 6.1. Bảng so sánh

| Mô hình | Train Acc | Test Acc | Train F1 | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6956 | 0.6735 | 0.4281 | **0.3514** | **0.6789** |
| Decision Tree | 0.7968 | 0.6735 | 0.5267 | 0.2381 | 0.4851 |
| Random Forest | 0.9932 | 0.8367 | 0.9787 | 0.2000 | 0.6541 |
| KNN | 0.8554 | 0.8401 | 0.2797 | 0.1132 | 0.6148 |

### 6.2. Phân tích chi tiết từng mô hình

#### Logistic Regression (tốt nhất)

- `Test F1 = 0.3514` - cao nhất trong các mô hình.
- `Test ROC-AUC = 0.6789` - cao nhất, cho thấy khả năng phân biệt lớp tốt nhất.
- `Test Recall = 0.5532` - mô hình tìm được hơn 55% nhân viên có khả năng nghỉ việc.
- Khoảng cách Train F1 (0.428) và Test F1 (0.351) không quá lớn => ít overfitting.
- Mô hình tuyến tính phù hợp làm baseline cho bài toán nhị phân mất cân bằng.

#### KNN

- Accuracy test cao (0.8401) nhưng chủ yếu do dự đoán tốt lớp đa số (ở lại).
- `Test F1 = 0.1132` - rất thấp, cho thấy mô hình hầu như không phát hiện được nhân viên nghỉ việc.
- `Test Recall = 0.064` - chỉ tìm được ~6% nhân viên thực sự nghỉ.
- KNN nhạy với tỷ lệ lớp mất cân bằng và không hỗ trợ `class_weight`.

#### Decision Tree (max_depth=5)

- `Test F1 = 0.2381`, `Test Recall = 0.3191`.
- Train F1 (0.527) cao hơn Test F1 (0.238) đáng kể => có dấu hiệu overfitting dù đã giới hạn độ sâu.
- ROC-AUC thấp (0.485) - gần với mô hình đoán ngẫu nhiên.

#### Random Forest

- Accuracy test cao (0.8367) nhưng Test F1 chỉ đạt 0.2000.
- `Test Recall = 0.1277` - rất thấp, mô hình bỏ sót nhiều nhân viên nghỉ việc.
- Train F1 (0.979) vs Test F1 (0.200) - chênh lệch rất lớn => **overfitting nghiêm trọng** mặc dù đã dùng `class_weight="balanced"`.
- Paradox của Random Forest trên dữ liệu mất cân bằng: accuracy cao nhưng recall lớp thiểu số rất thấp.

### 6.3. Confusion Matrix (Logistic Regression - mô hình tốt nhất)

|  | Dự đoán: No | Dự đoán: Yes |
|---|---:|---:|
| **Thực tế: No** | 145 | 102 |
| **Thực tế: Yes** | 21 | 26 |

- Mô hình phát hiện được 26/47 nhân viên nghỉ việc (Recall = 55%).
- 102 trường hợp False Positive (dự đoán nghỉ nhưng thực ra ở lại) - chi phí thấp (có thể can thiệp giữ chân không cần thiết).
- 21 trường hợp False Negative (bỏ sót nhân viên nghỉ) - chi phí cao hơn về mặt nghiệp vụ.

---

## 7. So sánh và phân tích

### 7.1. Mô hình tốt nhất

**Logistic Regression** là mô hình tốt nhất trong thí nghiệm này.

Lý do chọn:
- Test F1 cao nhất (0.3514), cân bằng giữa Precision và Recall.
- Test ROC-AUC cao nhất (0.6789), cho thấy khả năng phân biệt lớp tốt nhất.
- Ít overfitting nhất trong các mô hình (gap train-test nhỏ).
- Recall 0.553 - tốt nhất, phù hợp với mục tiêu nghiệp vụ là phát hiện nhân viên có nguy cơ nghỉ.

### 7.2. Dấu hiệu overfitting/underfitting

| Mô hình | Nhận định |
|---|---|
| Logistic Regression | Ổn định, gap train-test nhỏ - ít overfitting |
| KNN | Underfitting đối với lớp thiểu số - không học được pattern nghỉ việc |
| Decision Tree | Overfitting nhẹ - Train F1 cao hơn Test F1 đáng kể |
| Random Forest | **Overfitting nặng** - Train F1=0.98 nhưng Test F1=0.20 |

### 7.3. Validation Curve của Decision Tree

| max_depth | Train F1 | Test F1 | Test ROC-AUC |
|---:|---:|---:|---:|
| 1 | 0.377 | 0.260 | 0.549 |
| 2 | 0.438 | 0.317 | 0.576 |
| 3 | 0.459 | 0.267 | 0.532 |
| 5 | 0.527 | 0.238 | 0.485 |
| 10 | 0.801 | 0.140 | 0.464 |
| None | 1.000 | 0.174 | 0.510 |

**Nhận xét:** Test F1 tốt nhất ở `max_depth=2` (0.317). Khi cây sâu hơn, Train F1 tăng nhưng Test F1 giảm - dấu hiệu overfitting điển hình. Với `max_depth=None`, Train F1 đạt 1.0 nhưng Test F1 chỉ còn 0.174.

### 7.4. Khó khăn gặp phải

1. **Mất cân bằng lớp nghiêm trọng (16% vs 84%):** Đây là thách thức lớn nhất. Các mô hình có xu hướng dự đoán tất cả là "No" để đạt Accuracy cao, nhưng lại bỏ sót hầu hết trường hợp nghỉ việc thực sự. Đã xử lý bằng `class_weight="balanced"` nhưng vẫn chưa đủ.

2. **Feature hạn chế:** Dataset chỉ có 12 đặc trưng. Dataset IBM đầy đủ gốc có tới 35 cột - việc dùng bản rút gọn 13 cột làm mất nhiều thông tin quan trọng như `OverTime`, `JobLevel`, `StockOptionLevel`.

3. **F1-score tổng thể thấp:** Ngay cả mô hình tốt nhất chỉ đạt Test F1 = 0.35, cho thấy bài toán thực sự khó với bộ đặc trưng hiện tại.

---

## 8. Kết luận và hướng cải thiện

### Kết luận

- Dataset được chọn: IBM Employee Attrition (1.470 mẫu, 13 đặc trưng).
- Biến mục tiêu: `Attrition` (Yes/No) - bài toán Classification nhị phân.
- Mô hình tốt nhất: **Logistic Regression** với `Test F1 = 0.3514`, `Test ROC-AUC = 0.6789`.
- Kết quả chưa cao chủ yếu do mất cân bằng lớp nặng và bộ đặc trưng hạn chế.
- Không nên chỉ dùng Accuracy để đánh giá - cần ưu tiên F1-score và Recall trong bài toán này.

### Hướng cải thiện

1. **Kỹ thuật xử lý mất cân bằng lớp:** Áp dụng SMOTE (Synthetic Minority Oversampling Technique) để tạo thêm mẫu lớp thiểu số, hoặc undersampling lớp đa số.

2. **Tuning hyperparameter:** Dùng GridSearchCV hoặc RandomizedSearchCV để tìm bộ tham số tối ưu cho từng mô hình.

3. **Thử các mô hình mạnh hơn:** GradientBoostingClassifier hoặc XGBoost thường cho kết quả tốt hơn trên dữ liệu mất cân bằng.

4. **Điều chỉnh threshold phân loại:** Thay vì dùng ngưỡng 0.5, hạ threshold xuống (ví dụ 0.3) để tăng Recall - phù hợp với mục tiêu nghiệp vụ là phát hiện sớm nhân viên có nguy cơ nghỉ.

5. **Feature engineering:** Tạo thêm đặc trưng mới như `IncomePerYear = MonthlyIncome / YearsAtCompany`, hoặc nhóm các mức satisfaction thành High/Low.
