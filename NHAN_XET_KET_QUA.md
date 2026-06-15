# **Nhận xét kết quả**

## **1. Tổng quan kết quả**

4 mô hình được đánh giá trên tập test bằng Accuracy, Precision, Recall, F1-score và ROC-AUC.

Bảng kết quả chính:

| Mô hình | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.673 | 0.257 | 0.553 | **0.351** | **0.679** |
| Decision Tree | 0.653 | 0.222 | 0.468 | 0.301 | 0.57 |
| Random Forest | 0.837 | 0.462 | 0.128 | 0.200 | 0.654 |
| KNN | 0.840 | 0.500 | 0.064 | 0.113 | 0.615 |

**Nhận xét:** Dataset bị mất cân bằng lớp nghiêm trọng (16.1% Yes / 83.9% No). Accuracy không phản ánh đúng hiệu quả thực tế. KNN và Random Forest có Accuracy cao nhưng Recall lớp Yes cực thấp. Chỉ số đánh giá chính là F1-score và ROC-AUC.

---

## **2. Mô hình tốt nhất**

Mô hình tốt nhất là **Logistic Regression**.

Các chỉ số chính:
- `Test Accuracy = 0.673`
- `Test Precision = 0.257`
- `Test Recall = 0.553`
- `Test F1 = 0.351`
- `Test ROC-AUC = 0.679`

Giải thích:
- **Recall 0.553:** Mô hình phát hiện được hơn 55% nhân viên thực sự có khả năng nghỉ việc - đây là chỉ số quan trọng nhất về mặt nghiệp vụ nhân sự.

- **F1 = 0.351:** Cao nhất trong các mô hình - cân bằng tốt nhất giữa Precision và Recall.

- **ROC-AUC = 0.679:** Cao nhất - khả năng phân biệt 2 lớp tốt hơn các mô hình còn lại.

- **Accuracy = 0.673** thấp hơn Random Forest và KNN, nhưng đây là hệ quả tất yếu khi mô hình chủ động dự đoán nhiều trường hợp là "Yes" hơn thay vì an toàn bằng cách dự đoán tất cả là "No".

---

## **3. So sánh từng mô hình**

### Logistic Regression

- Train F1 = 0.428 | Test F1 = 0.351 => sai lệch nhỏ, ít overfitting.

- Recall cao nhất (0.553): tìm được hơn nửa số nhân viên thực sự nghỉ.

- Mô hình tuyến tính đơn giản nhưng phù hợp làm baseline và cho kết quả tốt nhất trong bộ dữ liệu này.

- Learning curve: Train và Test F1 hội tụ dần khi thêm dữ liệu - dấu hiệu mô hình ổn định, không overfitting.

### Decision Tree (max_depth=5, max_leaf_nodes=10)

- Train F1 = 0.48 | Test F1 = 0.301 => có dấu hiệu overfitting rõ rệt và khả năng tổng quát hóa trên dữ liệu mới còn kém

- ROC-AUC = 0.57 - gần với mô hình đoán ngẫu nhiên (0.5), cho thấy mô hình khó phân biệt 2 lớp trong bộ dữ liệu này.

- Confusion matrix: đúng 25/47 ca Yes (Recall = 46.8%).

- Learning curve: gap Train–Test F1 lớn và không thu hẹp khi tăng dữ liệu => dấu hiệu overfitting.

### Random Forest

- Train F1 = 0.979 | Test F1 = 0.200 => **overfitting nặng nhất**.

- Accuracy = 0.837 nhưng Recall chỉ 0.128: chỉ phát hiện 6/47 nhân viên nghỉ (bỏ sót 41 người).

- Confusion matrix: 240 TN, 7 FP, 41 FN, 6 TP - mô hình gần như chỉ dự đoán "No".

- Learning curve: Train F1 luôn xấp xỉ 1.0 bất kể số mẫu, Test F1 dao động thấp ~ 0.20 => mô hình học thuộc train hoàn toàn nhưng không tổng quát hóa được.

### KNN

- Test Accuracy = 0.840 nhưng Recall chỉ 0.064 - tệ nhất trong tất cả mô hình.

- Confusion matrix: chỉ đúng 3/47 ca Yes, bỏ sót 44 người nghỉ.

- KNN không hỗ trợ `class_weight` nên bị chi phối bởi lớp đa số (No).

- Learning curve: Train và Test Accuracy gần nhau nhưng cả hai đều không phản ánh khả năng phát hiện lớp Yes.

---

## **4. Nhận xét Confusion Matrix**

Tập test có: 247 nhân viên ở lại (No), 47 nhân viên nghỉ (Yes).

| Mô hình | TP | FP | FN | TN | Recall (Yes) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 26 | 75 | 21 | 172 | 55.3% |
| Decision Tree | 22 | 77 | 25 | 170 | 46.8% |
| Random Forest | 6 | 7 | 41 | 240 | 12.8% |
| KNN | 3 | 3 | 44 | 244 | 6.4% |

Trong bối cảnh nhân sự:
- **False Negative (FN):** Bỏ sót nhân viên thực sự nghỉ => rủi ro cao, công ty mất nhân tài mà không kịp giữ chân.

- **False Positive (FP):** Dự đoán nhầm nghỉ khi thực ra ở lại => HR gặp gỡ nhân viên không cần thiết, chi phí thấp hơn.

Vì vậy, trong bài toán này nên **ưu tiên Recall** cao hơn Precision, vì bỏ sót nhân viên nghỉ nguy hiểm hơn việc can thiệp thừa.

---

## **5. Nhận xét Learning Curve**

### Logistic Regression
- Train F1 (~ 0.43) và Test F1 (~ 0.35) gần nhau và ổn định khi tăng dữ liệu.

- Hai đường hội tụ dần => mô hình ổn định, không overfitting.

- F1 không tăng nhiều khi thêm dữ liệu.

### Decision Tree
- Train F1 (~ 0.47) luôn cao hơn Test F1 (~ 0.3) và gap không thu hẹp.

- Có dấu hiệu overfitting, có thể do mô hình quá phức tạp tương đối với pattern của lớp thiểu số.

### Random Forest
- Train F1 ~ 1.0 từ rất sớm (ngay cả khi chỉ dùng 100 mẫu train).

- Test F1 chỉ ~ 0.20 và không cải thiện dù tăng dữ liệu.

=> Overfitting nghiêm trọng.

### KNN
- Train F1 tăng dần từ 0 lên ~ 0.28 khi thêm dữ liệu, nhưng Test F1 chỉ ~ 0.11.

- Accuracy train và test gần nhau nhưng cả hai không phản ánh khả năng phát hiện lớp Yes.

---

## **6. Nhận xét Validation Curve - Decision Tree**

| max_depth | Train F1 | Test F1 | Nhận xét |
|---|---:|---:|---|
| 1 | 0.38 | 0.26 | Cây quá đơn giản - underfitting nhẹ |
| 2 | 0.43 | **0.32** | Test F1 cao nhất |
| 3 | 0.46 | 0.27 | Bắt đầu giảm |
| 5 | 0.53 | 0.24 | Overfitting tăng |
| 7 | 0.64 | 0.25 | |
| 10 | 0.80 | 0.15 | Overfitting rõ |
| 15 | 0.99 | 0.18 | Train gần hoàn hảo, Test thấp |
| None | 1.00 | 0.18 | Overfitting cực lớn |

- Test F1 tốt nhất tại `max_depth=2` - cây càng sâu thì Train F1 tăng nhưng Test F1 giảm.

- `max_depth=None` (không giới hạn): Train F1 = 1.0, Test F1 = 0.18 => mô hình bị overfitting.

---

## **7. Kết luận**

Trong bài toán IBM Employee Attrition, **Logistic Regression** là mô hình phù hợp nhất trong 4 mô hình đã thử.

Lý do:
- Test F1 cao nhất (0.351) - cân bằng tốt Precision/Recall.

- Recall cao nhất (0.553) - quan trọng nhất về mặt nghiệp vụ nhân sự.

- ROC-AUC cao nhất (0.679) - phân biệt 2 lớp tốt nhất.

- Ít overfitting nhất - gap Train/Test F1 nhỏ nhất.

Nhận xét:
- **Feature quan trọng nhất** theo Random Forest: MonthlyIncome, Age, YearsAtCompany - giống với phân tích EDA.

Hướng cải thiện:
- Áp dụng **SMOTE** để tạo thêm mẫu tổng hợp cho lớp Yes trước khi train.

- **Hạ threshold** phân loại từ 0.5 xuống ~ 0.3 để tăng Recall.

- Thử **GradientBoosting hoặc XGBoost** với `scale_pos_weight` để xử lý mất cân bằng tốt hơn.