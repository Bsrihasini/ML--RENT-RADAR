import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score, davies_bouldin_score
)

MODEL_DIR = os.path.dirname(__file__)

REGRESSION_FEATURES = [
    "bedrooms", "bathrooms", "square_feet",
    "cityname", "state", "pets_allowed", "fee",
    "has_photo", "price_type"
]

UNSUPERVISED_FEATURES = [
    "price", "square_feet", "bedrooms", "bathrooms", "latitude", "longitude"
]


def _prepare_frame(df):
    data = df.copy()
    for col in ["price", "square_feet", "bedrooms", "bathrooms", "latitude", "longitude"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.drop_duplicates().copy()


def _preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])


def _regression_data(df):
    data = _prepare_frame(df).dropna(subset=["price"])
    features = [c for c in REGRESSION_FEATURES if c in data.columns]
    if not features:
        raise ValueError("No suitable regression features were found.")
    X = data[features]
    y = data["price"].astype(float)
    return train_test_split(X, y, test_size=0.20, random_state=42), features


def _classification_data(df):
    data = _prepare_frame(df).dropna(subset=["price"])
    features = [c for c in REGRESSION_FEATURES if c in data.columns]
    if not features:
        raise ValueError("No suitable classification features were found.")

    def rent_band(value):
        if value < 500:
            return "Low"
        if value <= 1500:
            return "Medium"
        return "High"

    y_text = data["price"].astype(float).map(rent_band)
    X = data[features]
    return train_test_split(X, y_text, test_size=0.20, random_state=42, stratify=y_text), features


def _regression_model(key):
    models = {
        "linear-regression": LinearRegression(),
        "multiple-linear-regression": LinearRegression(),
        "ridge-regression": Ridge(alpha=1.0),
        "lasso-regression": Lasso(alpha=0.1, max_iter=5000),
        "decision-tree-regression": DecisionTreeRegressor(criterion="squared_error", max_depth=12, random_state=42),
        "bagging": BaggingRegressor(n_estimators=10, max_samples=0.6, max_features=1.0, bootstrap=True, random_state=42, n_jobs=-1),
        "adaboost": AdaBoostRegressor(n_estimators=15, learning_rate=0.05, random_state=42),
        "gradient-boosting": GradientBoostingRegressor(n_estimators=30, learning_rate=0.05, max_depth=3, random_state=42),
        "random-forest": RandomForestRegressor(n_estimators=10, max_depth=12, random_state=42, n_jobs=-1),
    }
    if key in models:
        return models[key]
    if key == "xgboost":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=30, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=2)
        except ImportError as exc:
            raise ValueError("XGBoost is not installed. Run: python -m pip install xgboost") from exc
    if key == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=30, learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.8, random_state=42, verbosity=-1)
        except ImportError as exc:
            raise ValueError("LightGBM is not installed. Run: python -m pip install lightgbm") from exc
    raise ValueError("Unknown regression model.")


def _classification_model(key):
    models = {
        "logistic-regression": LogisticRegression(max_iter=300, solver="lbfgs"),
        "decision-tree-id3": DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=12, random_state=42),
        "decision-tree-c45": DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=12, random_state=42),
        "decision-tree-cart": DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=12, random_state=42),
    }
    if key not in models:
        raise ValueError("Unknown classification model.")
    return models[key]


def _save_result(key, result):
    path = os.path.join(MODEL_DIR, f"{key}.pkl")
    try:
        joblib.dump(result, path)
    except Exception:
        pass
    return result


def _supervised_result(df, key):
    regression_keys = {
        "linear-regression", "multiple-linear-regression", "ridge-regression", "lasso-regression",
        "decision-tree-regression", "bagging", "adaboost", "gradient-boosting", "xgboost", "lightgbm", "random-forest"
    }
    if key in regression_keys:
        work_df = df
        if key in {"bagging", "random-forest", "adaboost", "gradient-boosting", "xgboost", "lightgbm"} and len(work_df) > 10000:
            work_df = work_df.sample(10000, random_state=42)
        (X_train, X_test, y_train, y_test), features = _regression_data(work_df)
        preprocessor = _preprocessor(X_train)
        X_train_p = preprocessor.fit_transform(X_train)
        X_test_p = preprocessor.transform(X_test)
        model = _regression_model(key)
        model.fit(X_train_p, y_train)
        predictions = model.predict(X_test_p)
        mse = mean_squared_error(y_test, predictions)
        result = {
            "kind": "Regression", "target": "price", "target_label": "Monthly Rent (price)",
            "features": features, "train_rows": len(X_train), "test_rows": len(X_test),
            "mae": float(mean_absolute_error(y_test, predictions)), "mse": float(mse),
            "rmse": float(np.sqrt(mse)), "r2": float(r2_score(y_test, predictions)),
            "actual": y_test.to_numpy(), "predictions": predictions,
            "model": model, "preprocessor": preprocessor,
        }
        return _save_result(key, result)

    cls_df = df.sample(20000, random_state=42) if len(df) > 20000 else df
    (X_train, X_test, y_train, y_test), features = _classification_data(cls_df)
    preprocessor = _preprocessor(X_train)
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    model = _classification_model(key)
    model.fit(X_train_p, y_train)
    predictions = model.predict(X_test_p)
    labels = ["Low", "Medium", "High"]
    cm = confusion_matrix(y_test, predictions, labels=labels)
    result = {
        "kind": "Classification", "target": "rent_category", "target_label": "Rent Category (Low / Medium / High)",
        "features": features, "train_rows": len(X_train), "test_rows": len(X_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(), "labels": labels,
        "actual": y_test.to_numpy(), "predictions": predictions,
        "model": model, "preprocessor": preprocessor,
    }
    return _save_result(key, result)


def _unsupervised_data(df, max_rows=5000):
    data = _prepare_frame(df)
    features = [c for c in UNSUPERVISED_FEATURES if c in data.columns]
    data = data.dropna(subset=features).copy()
    # Keep the page responsive while still calculating directly from the uploaded dataset.
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=42)
    X = data[features].astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return data, features, Xs, scaler


def _unsupervised_result(df, key):
    max_rows = 5000 if key == "kmeans" else (1500 if key in {"hierarchical", "agglomerative"} else 3000)
    data, features, Xs, scaler = _unsupervised_data(df, max_rows=max_rows)
    if key == "kmeans":
        model = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = model.fit_predict(Xs)
        params = {"n_clusters": 3, "n_init": 10, "random_state": 42}
        extra = {"inertia": float(model.inertia_)}
    elif key == "hierarchical":
        model = AgglomerativeClustering(n_clusters=3, linkage="ward")
        labels = model.fit_predict(Xs)
        params = {"n_clusters": 3, "linkage": "ward"}
        extra = {}
    elif key == "agglomerative":
        model = AgglomerativeClustering(n_clusters=3, linkage="average")
        labels = model.fit_predict(Xs)
        params = {"n_clusters": 3, "linkage": "average"}
        extra = {}
    elif key == "dbscan":
        model = DBSCAN(eps=0.8, min_samples=8)
        labels = model.fit_predict(Xs)
        params = {"eps": 0.8, "min_samples": 8}
        extra = {"noise_points": int((labels == -1).sum())}
    elif key == "pca":
        n_components = min(2, Xs.shape[1])
        model = PCA(n_components=n_components, random_state=42)
        transformed = model.fit_transform(Xs)
        labels = np.zeros(len(transformed), dtype=int)
        total_variance = float(model.explained_variance_ratio_.sum())
        result = {
            "kind": "PCA", "target_label": "No target — structure of numeric rent data",
            "features": features, "rows_used": len(data), "params": {"n_components": n_components},
            "explained_variance": [float(v) for v in model.explained_variance_ratio_],
            "explained_variance_total": total_variance,
            "transformed": transformed, "model": model,
        }
        return _save_result(key, result)
    else:
        raise ValueError("Unknown unsupervised model.")

    unique = set(labels)
    valid = labels != -1
    if len(unique - {-1}) > 1 and valid.sum() > len(unique - {-1}):
        silhouette = float(silhouette_score(Xs[valid], labels[valid]))
        db = float(davies_bouldin_score(Xs[valid], labels[valid]))
    else:
        silhouette = None
        db = None
    result = {
        "kind": "Clustering", "target_label": "No target — structure of numeric rent data",
        "features": features, "rows_used": len(data), "params": params,
        "cluster_count": int(len(set(labels)) - (1 if -1 in labels else 0)),
        "silhouette": silhouette, "davies_bouldin": db,
        "labels": labels, "X_scaled": Xs, "model": model, **extra,
    }
    return _save_result(key, result)


def train_model(df, key):
    if key in {"kmeans", "hierarchical", "agglomerative", "dbscan", "pca"}:
        return _unsupervised_result(df, key)
    return _supervised_result(df, key)
