# **IBM Employee Attrition Prediction - Classification**

**Subject:** Introduction to Data Science  
**Task:** Predicting employee turnover (`Attrition`: Yes / No)  
**Problem Type:** Binary Classification (Imbalanced Dataset)  
**Tech Stack:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  

---
## **Project Overview & Business Context**

Employee turnover imposes significant financial and operational costs on organizations due to recruitment, onboarding, and productivity loss. This project builds an end-to-end machine learning pipeline on the **IBM HR Analytics Dataset** to:
1. Uncover key demographic, compensation, and workplace satisfaction drivers behind employee attrition through statistical exploratory data analysis (EDA).
2. Train and benchmark multiple binary classification models under severe class imbalance.
3. Prioritize **Recall** and **ROC-AUC** over raw accuracy so HR teams can proactively identify high-risk employees before they leave.
---

## **Project Structure**

```
├── data/
│   ├── IBM.csv             # Raw dataset (1,470 records, 13 features)
│   └── README.md           # Dataset description
│
├── notebooks/
│   ├── 00_eda.ipynb        # Exploratory Data Analysis (EDA)
│   └── 01_data_preprocessing.ipynb     # Data preprocessing
│
├── src/
│   ├── init.py
│   ├── preprocessing.py      # Data loading, cleaning, feature prep
│   ├── classification_models.py        # Model training workflow
│   └── evaluation.py         # Metrics computation, visualization & result logging
│
├── outputs/
│   ├── figures_eda/          # EDA plots
│   ├── figures_evaluate/     # Evaluation plots
│   ├── models/               # Saved trained models (.pkl)
│   └── metrics/              # Evaluation results
│
├── REPORT.md                 # Detailed project report (Vietnamese)
├── NHAN_XET_KET_QUA.md       # Model-by-model performance analysis (Vietnamese)
├── main.py                   # End-to-end pipeline
├── README.md
└── requirements.txt
```

---

## **Environment Setup & Usage**

### ***Installation***
```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# or: .venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### ***Run End-to-End Pipeline***
```bash
python main.py
```
Executing `main.py` automatically runs preprocessing, trains all 4 classification models, computes evaluation metrics, and exports all plots and artifacts to `outputs/`.

---

## **Key Results**

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.673 | 0.257 | **0.553** | **0.351** | **0.679** |
| Decision Tree | 0.653 | 0.222 | 0.468 | 0.301 | 0.570 |
| Random Forest | 0.837 | 0.462 | 0.128 | 0.200 | 0.654 |
| KNN | 0.840 | 0.500 | 0.064 | 0.113 | 0.615 |

**Best Model: Logistic Regression**, achieving the highest F1-Score (`0.351`), ROC-AUC (`0.679`), and Recall (`0.553`) for the minority class.

> **Key Takeaway:** Accuracy is misleading for this problem due to severe class imbalance (`16.1% Yes` vs. `83.9% No`). While KNN and Random Forest achieve higher overall accuracy, their Recall on the `Yes` class is extremely low—meaning they fail to identify most at-risk employees.

---

## **Generated Outputs**

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

## **Future Improvements**

* **Resampling & Threshold Tuning:** Experiment with SMOTE/ADASYN inside cross-validation folds and tune probability classification thresholds using Precision-Recall curves to improve the Precision-Recall trade-off.
* **Gradient Boosting & Feature Engineering:** Evaluate regularized gradient boosting models (`XGBoost`, `LightGBM`) and engineer domain ratios (e.g., `IncomePerYearAtCompany`, `TenureToAgeRatio`).