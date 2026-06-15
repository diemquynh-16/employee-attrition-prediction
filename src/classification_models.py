import pickle
from pathlib import Path

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import build_pipeline

# Model definitions ─────────────────────────────────────────────────────────

def get_estimators() -> dict:
    """
    Trả về dict {tên_mô_hình: estimator}.

    Tất cả mô hình hỗ trợ class_weight được đặt "balanced"
    để xử lý mất cân bằng lớp (16% Yes / 84% No).
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            # KNN không hỗ trợ class_weight
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_leaf_nodes=10,
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }

# Training ─────────────────────────────────────────────────────────
def train_all_models(X_train, y_train, numeric_features: list, categorical_features: list, models_dir: Path | None = None) -> dict:
    """
    Huấn luyện tất cả mô hình, lưu .pkl.

    Parameters
    ----------
    X_train, y_train : train split
    numeric_features, categorical_features : danh sách cột
    models_dir : Path hoặc None — nếu không None thì lưu .pkl vào đây

    Returns
    -------
    dict {tên_mô_hình: fitted Pipeline}
    """
    trained = {}

    for name, estimator in get_estimators().items():
        print(f"  [Training] Huấn luyện: {name} ...")
        pipeline = build_pipeline(clone(estimator), numeric_features, categorical_features)
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline

        if models_dir is not None:
            Path(models_dir).mkdir(parents=True, exist_ok=True)
            slug = name.lower().replace(" ", "_")
            pkl_path = Path(models_dir) / f"{slug}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(pipeline, f)
            print(f"Đã lưu vào {pkl_path}")

    return trained

def load_model(path: Path):
    """Load mô hình đã lưu từ file .pkl."""
    with open(path, "rb") as f:
        return pickle.load(f)