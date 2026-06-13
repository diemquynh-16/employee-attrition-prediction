"""
main.py
Pipeline end-to-end cho bài toán IBM Employee Attrition Classification.

Chạy:
    python main.py
"""

import os
from pathlib import Path

from src.preprocessing import load_and_clean, prepare_features
from src.classification_models import train_all_models
from src.evaluation import (
    evaluate_all_models,
    save_metrics,
    save_confusion_matrices,
    save_roc_curves,
    save_model_comparison_chart,
    save_feature_importance,
    save_learning_curves,
    save_validation_curve_dt,
)

# Đường dẫn ───────────────────────────────────────────
DATA_PATH    = Path("data/IBM.csv")
FIGURES_EVALUATE_DIR  = Path("outputs/figures_evaluate")
MODELS_DIR   = Path("outputs/models")
METRICS_DIR  = Path("outputs/metrics")


def main():
    print("== IBM Employee Attrition - Classification Pipeline == ")

    # 1. Load & clean
    df = load_and_clean(DATA_PATH)

    # 2. Prepare features & split
    X_train, X_test, y_train, y_test, num_feats, cat_feats = prepare_features(df)

    # 3. Train
    print("\n[train] Bắt đầu huấn luyện mô hình ...")
    trained = train_all_models(X_train, y_train, num_feats, cat_feats, models_dir=MODELS_DIR)

    # 4. Evaluate
    print("\n[evaluate] Đánh giá mô hình ...")
    results_df = evaluate_all_models(trained, X_train, y_train, X_test, y_test)
    print("\nBảng so sánh kết quả:")
    print(results_df.to_string(index=False))

    # 5. Save metrics
    save_metrics(results_df, trained, X_test, y_test, METRICS_DIR)

    # 6. Save figures
    save_confusion_matrices(trained, X_test, y_test, FIGURES_EVALUATE_DIR)
    save_roc_curves(trained, X_test, y_test, FIGURES_EVALUATE_DIR)
    save_model_comparison_chart(results_df, FIGURES_EVALUATE_DIR)
    save_feature_importance(trained, num_feats, cat_feats, FIGURES_EVALUATE_DIR)
    save_learning_curves(X_train, y_train, X_test, y_test,
                         num_feats, cat_feats, FIGURES_EVALUATE_DIR, METRICS_DIR)
    save_validation_curve_dt(X_train, y_train, X_test, y_test,
                              num_feats, cat_feats, FIGURES_EVALUATE_DIR, METRICS_DIR)
    print("== Hoàn tất! Kết quả lưu tại outputs/ ==")

if __name__ == "__main__":
    main()