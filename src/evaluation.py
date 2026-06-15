""" 
evaluation.py chứa các hàm đánh giá mô hình, vẽ biểu đồ và lưu kết quả đánh giá vào .../outputs/evaluation
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import build_pipeline

POSITIVE_CLASS = 1
CLASS_NAME = ["No (ở lại)", "Yes (nghỉ)"]

# Helpers ─────────────────────────────────────────────────────────
def _slug(text: str) -> str:
    return text.lower().replace(" ", "_")

def _predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, POSITIVE_CLASS]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)

# Single-model evaluation ─────────────────────────────────────────────────────────
def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    """
    Tính các chỉ số đánh giá cho một mô hình.

    Returns
    -------
    dict chứa Accuracy, Precision, Recall, F1, ROC-AUC (train + test).
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_score = _predict_scores(model, X_test)

    return {
        "Model": name,
        "Train Accuracy": accuracy_score(y_train, y_train_pred),
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Train Precision": precision_score(y_train, y_train_pred, zero_division=0),
        "Test Precision": precision_score(y_test, y_test_pred, zero_division=0),
        "Train Recall": recall_score(y_train, y_train_pred, zero_division=0),
        "Test Recall": recall_score(y_test,  y_test_pred, zero_division=0),
        "Train F1": f1_score(y_train, y_train_pred, zero_division=0),
        "Test F1": f1_score(y_test,  y_test_pred, zero_division=0),
        "Test ROC-AUC": roc_auc_score(y_test, y_test_score),
    }

def evaluate_all_models(trained_models: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """
    Đánh giá tất cả mô hình và trả về DataFrame so sánh,
    sắp xếp theo Test F1 giảm dần.
    """
    rows = []

    for name, model in trained_models.items():
        metrics_dict = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        rows.append(metrics_dict)

    return pd.DataFrame(rows).sort_values("Test F1", ascending=False).reset_index(drop=True)

# Save metrics ─────────────────────────────────────────────────────────
def save_metrics(results_df: pd.DataFrame, trained_models: dict, X_test, y_test, metrics_dir: Path):
    """Lưu results.csv, classification_reports.csv và metrics JSON."""
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bảng so sánh chính
    results_df.to_csv(metrics_dir/"results.csv", index=False)

    # 2. Classification reports chi tiết
    rows = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=CLASS_NAME,
            output_dict=True,
            zero_division=0
        )

        for label in CLASS_NAME + ["macro avg", "weighted avg"]:
            rows.append({
                "Model": name,
                "Label": label,
                "Precision": report[label]["precision"],
                "Recall": report[label]["recall"],
                "F1": report[label]["f1-score"],
                "Support": report[label]["support"]
            })
    pd.DataFrame(rows).to_csv(metrics_dir/"classification_reports.csv", index=False)

    # 3. JSON từng mô hình
    for _, row in results_df.iterrows():
        slug = _slug(row["Model"])
        with open(metrics_dir / f"{slug}_metrics.json", "w", encoding="utf-8") as f:
            json.dump({k: round(v, 6) if isinstance(v, float) else v
                       for k, v in row.items()},
                       f, indent=2, ensure_ascii=False)
            
    print(f"[Evaluation] Đã lưu vào metrics: {metrics_dir}/")

# Figures ─────────────────────────────────────────────────────────
def save_confusion_matrices(trained_models: dict, X_test, y_test, figures_evaluate_dir: Path):
    figures_evaluate_dir = Path(figures_evaluate_dir)
    figures_evaluate_dir.mkdir(parents=True, exist_ok=True)
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAME)
        display.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix: {name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(figures_evaluate_dir / f"confusion_matrix_{_slug(name)}.png", dpi = 150)
        plt.close()
    
def save_roc_curves(trained_models: dict, X_test, y_test, figures_evaluate_dir: Path):
    figures_evaluate_dir = Path(figures_evaluate_dir)
    figures_evaluate_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for (name, model), color in zip(trained_models.items(), colors):
        scores = _predict_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve tất cả mô hình", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_evaluate_dir / "roc_curves.png", dpi=150)
    plt.close()

def save_model_comparison_chart(results_df: pd.DataFrame, figures_evaluate_dir: Path):
    figures_evaluate_dir = Path(figures_evaluate_dir)
    figures_evaluate_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["Test Accuracy", "Test Precision", "Test Recall", "Test F1", "Test ROC-AUC"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 5))
    palette = sns.color_palette("Set2", len(results_df))
    for ax, metric in zip(axes, metrics):
        bars = ax.bar(results_df["Model"], results_df[metric], color=palette, edgecolor="white")
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel("Score")
    plt.suptitle("So sánh các mô hình", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_evaluate_dir / "model_comparison_all_metrics.png", dpi=150)
    plt.close()


def save_feature_importance(trained_models: dict, numeric_features: list,
                             categorical_features: list, figures_evaluate_dir: Path):
    figures_evaluate_dir = Path(figures_evaluate_dir)
    figures_evaluate_dir.mkdir(parents=True, exist_ok=True)
    rf = trained_models.get("Random Forest")
    if rf is None:
        return
    preprocessor = rf.named_steps["preprocessor"]
    classifier   = rf.named_steps["model"]
    cat_names = preprocessor.named_transformers_["cat"] \
                             .get_feature_names_out(categorical_features).tolist()
    all_names = numeric_features + cat_names
    importances = classifier.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(10, 7))
    plt.barh([all_names[i] for i in reversed(top_idx)],
             importances[list(reversed(top_idx))], color="#5C6BC0")
    plt.xlabel("Feature Importance")
    plt.title("Top 20 Feature Importance - Random Forest", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_evaluate_dir / "feature_importance_random_forest.png", dpi=150)
    plt.close()

# Learning curves ─────────────────────────────────────────────────────────
def save_learning_curves(
    X_train, y_train, X_test, y_test,
    numeric_features: list, categorical_features: list,
    figures_evaluate_dir: Path, metrics_dir: Path,
):
    print("[evaluate] Đang tính learning curves ...")
    from .classification_models import get_estimators

    figures_evaluate_dir = Path(figures_evaluate_dir)
    figures_evaluate_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(metrics_dir)
    ratios = np.linspace(0.1, 1.0, 10)
    rows = []

    for name, estimator in get_estimators().items():
        for ratio in ratios:
            n = max(10, int(len(X_train) * ratio))
            Xs, ys = X_train.iloc[:n], y_train.iloc[:n]
            pipe = build_pipeline(clone(estimator), numeric_features, categorical_features)
            pipe.fit(Xs, ys)
            r = evaluate_model(name, pipe, Xs, ys, X_test, y_test)
            rows.append({
                "Model": name, "Ratio": round(ratio, 2), "Train Size": n,
                "Train F1": r["Train F1"], "Test F1": r["Test F1"],
                "Train Accuracy": r["Train Accuracy"], "Test Accuracy": r["Test Accuracy"],
                "Test ROC-AUC": r["Test ROC-AUC"],
            })

    lc_df = pd.DataFrame(rows)
    lc_df.to_csv(metrics_dir / "learning_curves.csv", index=False)

    for name in lc_df["Model"].unique():
        mdf = lc_df[lc_df["Model"] == name]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, metric in zip(axes, ["F1", "Accuracy"]):
            ax.plot(mdf["Train Size"], mdf[f"Train {metric}"],
                    "o-", label=f"Train {metric}", color="#2196F3")
            ax.plot(mdf["Train Size"], mdf[f"Test {metric}"],
                    "o-", label=f"Test {metric}",  color="#F44336")
            ax.set_xlabel("Số mẫu train"); ax.set_ylabel(metric)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Learning Curve {metric} — {name}", fontweight="bold")
            ax.legend()
        plt.tight_layout()
        plt.savefig(figures_evaluate_dir / f"learning_curve_{_slug(name)}.png", dpi=150)
        plt.close()

    print(f"[Evaluation] Learning curves to {figures_evaluate_dir}/")

# Validation curve (Decision Tree max_depth) ─────────────────────────────────────────────────────────
def save_validation_curve_dt(
    X_train, y_train, X_test, y_test,
    numeric_features: list, categorical_features: list,
    figures_evaluate_dir: Path, metrics_dir: Path,
):
    figures_evaluate_dir = Path(figures_evaluate_dir)
    figures_evaluate_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    depths = [1, 2, 3, 4, 5, 7, 10, 15, None]
    rows = []

    for i, d in enumerate(depths, 1):
        pipe = build_pipeline(
            DecisionTreeClassifier(max_depth=d, random_state=42, class_weight="balanced"),
            numeric_features, categorical_features,
        )
        pipe.fit(X_train, y_train)
        r = evaluate_model(f"Decision Tree d={d}", pipe, X_train, y_train, X_test, y_test)
        rows.append({
            "max_depth": "None" if d is None else d,
            "idx": i,
            "Train F1": r["Train F1"], "Test F1": r["Test F1"],
            "Train Accuracy": r["Train Accuracy"], "Test Accuracy": r["Test Accuracy"],
            "Test ROC-AUC": r["Test ROC-AUC"],
        })

    vc_df = pd.DataFrame(rows)
    vc_df.to_csv(metrics_dir / "validation_curve_decision_tree.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ["F1", "Accuracy"]):
        ax.plot(vc_df["idx"], vc_df[f"Train {metric}"],
                "o-", label=f"Train {metric}", color="#2196F3")
        ax.plot(vc_df["idx"], vc_df[f"Test {metric}"],
                "o-", label=f"Test {metric}",  color="#F44336")
        ax.set_xticks(vc_df["idx"]); ax.set_xticklabels(vc_df["max_depth"])
        ax.set_xlabel("max_depth"); ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Validation Curve Decision Tree — {metric}", fontweight="bold")
        ax.legend()
    plt.tight_layout()
    plt.savefig(figures_evaluate_dir / "validation_curve_decision_tree.png", dpi=150)
    plt.close()

    print(f"[Evaluate] Validation curve to {figures_evaluate_dir}/")
