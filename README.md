# **IBM Employee Attrition - Classification**

**Môn học:** Nhập môn Khoa học Dữ liệu  
**Bài toán:** Dự đoán nhân viên có nghỉ việc hay không (`Attrition`: Yes / No)  
**Loại bài toán:** Binary Classification  

---

## ***Cấu trúc thư mục***

```
HW2_24280042_QuangDiemQuynh/
├── data/
│   ├── IBM.csv                         # Dataset gốc (1470 mẫu, 13 cột)
│   └── README.md                       # Mô tả dataset
│
├── notebooks/
│   ├── 00_eda.ipynb                    # Phân tích dữ liệu khám phá (EDA)
│   └── 01_data_preprocessing.ipynb     # Tiền xử lý dữ liệu
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                # Hàm load, clean, prepare_features, build_preprocessor, build_pipeline
│   ├── classification_models.py        # Định nghĩa mô hình + train_all_models
│   └── evaluation.py                   # Tính chỉ số, vẽ biểu đồ, lưu kết quả
│
├── outputs/
│   ├── figures_eda/                    # Biểu đồ EDA (target distribution, continuous, ordinal, categorical, outlier, correlation)
│   ├── figures_evaluate/               # Biểu đồ đánh giá (confusion matrix, ROC, model comparison, learning/validation curve, feature importance)
│   ├── models/                         # Mô hình đã train (.pkl)
│   └── metrics/                        # Kết quả đánh giá (results.csv, classification_reports.csv, learning/validation curve, JSON từng mô hình)
│
├── REPORT.md                           # Báo cáo
├── NHAN_XET_KET_QUA.md                 # Nhận xét chi tiết kết quả từng mô hình
├── main.py                             # Pipeline end-to-end
├── README.md
└── requirements.txt
```

---

## ***Cài đặt môi trường***

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# hoặc: .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## ***Chạy chương trình***

```bash
python main.py
```

Chương trình sẽ tự động chạy toàn bộ quy trình và lưu kết quả vào `outputs/`.

---

## ***Kết quả chính***

| Mô hình | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.673 | 0.257 | 0.553 | **0.351** | **0.679** |
| Decision Tree | 0.653 | 0.222 | 0.468 | 0.301 | 0.57 |
| Random Forest | 0.837 | 0.462 | 0.128 | 0.200 | 0.654 |
| KNN | 0.840 | 0.500 | 0.064 | 0.113 | 0.615 |

**Mô hình tốt nhất: Logistic Regression** với F1 (0.351) và ROC-AUC (0.679) cao nhất, Recall cao nhất (0.553).

> **Nhận xét:** Accuracy không phải chỉ số đáng tin cậy ở bài toán này vì dataset bị mất cân bằng lớp nặng (16.1% Yes / 83.9% No). KNN và Random Forest có Accuracy cao hơn nhưng Recall lớp Yes cực thấp - tức là gần như không phát hiện được nhân viên nghỉ việc.

---

## ***Outputs sau khi chạy***

```
outputs/
├── figures_eda/
│   ├── eda_01_target_distribution.png
│   ├── eda_02_continuous_features.png
│   ├── eda_03_ordinal_features.png
│   ├── eda_04_categorical.png
│   ├── eda_05_outliers_boxplot.png
│   ├── eda_06_correlation_heatmap.png
│   └── eda_07_correlation_with_target.png
│
├── figures_evaluate/
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_decision_tree.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_knn.png
│   ├── roc_curves.png
│   ├── model_comparison_all_metrics.png
│   ├── feature_importance_random_forest.png
│   ├── learning_curve_logistic_regression.png
│   ├── learning_curve_decision_tree.png
│   ├── learning_curve_random_forest.png
│   ├── learning_curve_knn.png
│   └── validation_curve_decision_tree.png
│
├── models/
│   ├── logistic_regression.pkl
│   ├── knn.pkl
│   ├── decision_tree.pkl
│   └── random_forest.pkl
│
└── metrics/
    ├── results.csv
    ├── classification_reports.csv
    ├── learning_curves.csv
    ├── validation_curve_decision_tree.csv
    ├── logistic_regression_metrics.json
    ├── decision_tree_metrics.json
    ├── random_forest_metrics.json
    └── knn_metrics.json
```