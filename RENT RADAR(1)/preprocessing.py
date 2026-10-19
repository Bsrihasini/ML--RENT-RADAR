import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

TARGET = "price"

MODEL_FEATURES = [
    "bedrooms", "bathrooms", "square_feet",
    "cityname", "state", "pets_allowed", "fee",
    "has_photo", "price_type"
]


def preprocess_data(data):
    df = data.copy()

    before = len(df)
    df = df.drop_duplicates().copy()
    duplicates_removed = before - len(df)

    for col in ["price", "square_feet", "bedrooms", "bathrooms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET]).copy()

    features = [c for c in MODEL_FEATURES if c in df.columns]
    if not features:
        raise ValueError("No valid model features were found in the dataset.")

    X = df[features].copy()
    y = df[TARGET].astype(float)

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_count": int(X_train_processed.shape[1]),
        "duplicates_removed": int(duplicates_removed),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "features": features,
        "cleaned_data": df,
    }


def transform_input(payload, preprocessor):
    row = {}
    for col in MODEL_FEATURES:
        value = payload.get(col, "")
        if col in ["bedrooms", "bathrooms", "square_feet"]:
            row[col] = pd.to_numeric(value, errors="coerce")
        else:
            row[col] = value if value != "" else None

    return preprocessor.transform(pd.DataFrame([row], columns=MODEL_FEATURES))
