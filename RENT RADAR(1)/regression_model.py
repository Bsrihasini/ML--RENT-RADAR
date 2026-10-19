import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_FILE = os.path.join(MODEL_DIR, "rent_linear_regression.pkl")


def train_linear_regression(preprocessed):
    os.makedirs(MODEL_DIR, exist_ok=True)

    model = LinearRegression()
    model.fit(preprocessed["X_train"], preprocessed["y_train"])

    predictions = model.predict(preprocessed["X_test"])
    actual = preprocessed["y_test"]

    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(actual, predictions)

    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessed["preprocessor"],
            "features": preprocessed["features"],
        },
        MODEL_FILE,
    )

    return {
        "model": model,
        "predictions": predictions,
        "actual": actual,
        "mae": float(mae),
        "mse": float(mse),
        "rmse": rmse,
        "r2": float(r2),
    }


def load_model():
    return joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else None
