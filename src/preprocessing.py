""""
preprocessing.py chứa các hàm đọc, làm sạch, mã hóa và chuẩn bị dữ liệu
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Helpers ─────────────────────────────────────────────────────────
def _make_ohe() -> OneHotEncoder:
    """ Tạo OneHotEncoder tương thích với nhiều phiên bản scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
    
# Load and clean ─────────────────────────────────────────────────────────
def load_and_clean(path: Path) -> pd.DataFrame:
    """
    Đọc CSV, strip whitespace, loại trùng lặp và giá trị thiếu.

    Parameters
    ----------
    path : Path
        Đường dẫn tới file IBM.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame đã được làm sạch.
    """

    if not Path(path).exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    before = len(df)
    df = df.drop_duplicates()
    n_dup = before - len(df)

    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        # Điền median cho số, mode cho phân loại
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ["int64", "float64"]:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
    
    print(f"[Preprocessing] Load: {before} dòng | Trùng lặp xóa: {n_dup} | Xử lý missing values: {n_missing}")
    print(f"[preprocessing] Kích thước sau làm sạch: {df.shape}")
    return df

# Feature preparation ─────────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Tách X/y, chia train/test với stratify.

    Parameters
    ----------
    df : pd.DataFrame
    test_size : float
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test, numeric_features, categorical_features
    """
    TARGET = "Attrition"
    X = df.drop(columns=[TARGET])
    y = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    print(f"[Preprocessing] Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"[Preprocessing] Numeric ({len(numeric_features)}): {numeric_features}")
    print(f"[Preprocessing] Categorical ({len(categorical_features)}): {categorical_features}")
    print(f"[Preprocessing] Phân bố nhãn (train): {dict(y_train.value_counts().rename(index={0:'No', 1:'Yes'}))}")

    return X_train, X_test, y_train, y_test, numeric_features, categorical_features

# Processor (dùng trong pipeline) ─────────────────────────────────────────────────────────
def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Tạo ColumnTransformer:
      - StandardScaler  cho biến số
      - OneHotEncoder   cho biến phân loại

    Parameters
    ----------
    numeric_features : list
    categorical_features : list

    Returns
    -------
    ColumnTransformer
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", _make_ohe(), categorical_features),
        ],
        remainder="drop",
    )

# Pipeline ─────────────────────────────────────────────────────────
def build_pipeline(estimator, numeric_features: list, categorical_features: list) -> Pipeline:
    """
    Tạo Pipeline gồm preprocessor + estimator.

    Parameters
    ----------
    estimator : sklearn estimator
    numeric_features : list
    categorical_features : list

    Returns
    -------
    Pipeline
    """
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
        ("model", estimator),
    ])


