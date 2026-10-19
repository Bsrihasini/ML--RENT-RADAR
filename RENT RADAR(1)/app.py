import os
from flask import Flask, render_template, request, redirect, url_for, flash

from load_data import load_data, get_data_summary, DEFAULT_DATA_FILE
from rent_eda import run_eda, create_rent_distribution_chart
from preprocessing import preprocess_data, transform_input, MODEL_FEATURES
from regression_model import train_linear_regression, load_model
from models.model_suite import train_model
from visualization import (
    create_model_visualizations, create_classification_visualization,
    create_clustering_visualization, create_pca_visualization
)

app = Flask(__name__)
app.secret_key = "rent-radar-local-key"

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def current_data():
    uploaded = os.path.join(UPLOAD_DIR, "active_dataset.csv")
    return load_data(uploaded if os.path.exists(uploaded) else DEFAULT_DATA_FILE)


def dashboard_stats(df):
    price = df["price"].dropna()
    total = len(df)
    avg_rent = float(price.mean()) if len(price) else 0
    total_rent = float(price.sum()) if len(price) else 0
    avg_sqft = float(df["square_feet"].mean()) if df["square_feet"].notna().any() else 0

    low = int((price < 500).sum())
    mid = int(((price >= 500) & (price <= 1500)).sum())
    high = int((price > 1500).sum())

    def pct(n):
        return round((n / len(price) * 100), 1) if len(price) else 0

    return {
        "total_listings": total,
        "average_rent": avg_rent,
        "total_listed_rent": total_rent,
        "average_sqft": avg_sqft,
        "locations": int(df["cityname"].nunique()) if "cityname" in df.columns else 0,
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "low_count": low,
        "mid_count": mid,
        "high_count": high,
        "low_pct": pct(low),
        "mid_pct": pct(mid),
        "high_pct": pct(high),
    }


@app.route("/")
def index():
    try:
        df = current_data()
        summary = get_data_summary(df)
        stats = dashboard_stats(df)
        create_rent_distribution_chart(df)
        return render_template(
            "index.html",
            active="dashboard",
            summary=summary,
            stats=stats,
            error=None,
        )
    except Exception as e:
        return render_template(
            "index.html",
            active="dashboard",
            summary=None,
            stats=None,
            error=str(e),
        )


@app.route("/data-loading", methods=["GET", "POST"])
def data_loading():
    error = None
    summary = None

    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or not file.filename:
            error = "Please select a CSV dataset."
        else:
            path = os.path.join(UPLOAD_DIR, "active_dataset.csv")
            file.save(path)
            try:
                summary = get_data_summary(load_data(path))
                flash("Rent dataset uploaded and activated successfully.", "success")
            except Exception as e:
                if os.path.exists(path):
                    os.remove(path)
                error = f"Could not load the uploaded dataset: {e}"
    else:
        try:
            summary = get_data_summary(current_data())
        except Exception as e:
            error = str(e)

    return render_template(
        "data_loading.html",
        active="data-loading",
        summary=summary,
        error=error,
    )


@app.route("/eda")
def eda_page():
    try:
        results = run_eda(current_data())
        return render_template("eda.html", active="eda", results=results, error=None)
    except Exception as e:
        return render_template("eda.html", active="eda", results=None, error=str(e))


@app.route("/preprocessing")
def preprocessing_page():
    try:
        processed = preprocess_data(current_data())
        cleaned = processed["cleaned_data"].head(12).fillna("").to_dict(orient="records")
        return render_template(
            "preprocessing.html",
            active="preprocessing",
            results=processed,
            cleaned=cleaned,
            error=None,
        )
    except Exception as e:
        return render_template(
            "preprocessing.html",
            active="preprocessing",
            results=None,
            cleaned=[],
            error=str(e),
        )


@app.route("/visualization")
def visualization_page():
    try:
        df = current_data()
        results = run_eda(df)
        return render_template(
            "visualization.html",
            active="visualization",
            charts=results["charts"],
            error=None,
        )
    except Exception as e:
        return render_template(
            "visualization.html",
            active="visualization",
            charts=[],
            error=str(e),
        )


MODEL_GROUPS = {
    "Regression": [
        ("linear-regression", "Linear Regression"),
        ("multiple-linear-regression", "Multiple Linear Regression"),
        ("ridge-regression", "Ridge Regression"),
        ("lasso-regression", "Lasso Regression"),
    ],
    "Decision Trees": [
        ("decision-tree-regression", "Decision Tree Regression"),
        ("decision-tree-id3", "Decision Tree — ID3"),
        ("decision-tree-c45", "Decision Tree — C4.5"),
        ("decision-tree-cart", "Decision Tree — CART"),
    ],
    "Bagging": [
        ("bagging", "Bagging"),
        ("random-forest", "Random Forest"),
    ],
    "Boosting": [
        ("adaboost", "AdaBoost"),
        ("gradient-boosting", "Gradient Boosting"),
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
    ],
    "Classification": [
        ("logistic-regression", "Logistic Regression"),
    ],
}

UNSUPERVISED_GROUPS = {
    "Clustering": [
        ("kmeans", "K-Means"),
        ("hierarchical", "Hierarchical Clustering"),
        ("agglomerative", "Agglomerative Clustering"),
        ("dbscan", "DBSCAN"),
    ],
    "Dimensionality Reduction": [
        ("pca", "PCA — Principal Component Analysis"),
    ],
}


MODEL_INFO = {
    "linear-regression": {"type":"Regression","description":"Fits a linear relationship between the rental features and monthly rent.","attributes":["fit_intercept = True","multiple predictors","copy_X = True"]},
    "multiple-linear-regression": {"type":"Regression","description":"Uses multiple rental variables together to predict monthly rent.","attributes":["fit_intercept = True","multiple predictors","copy_X = True"]},
    "ridge-regression": {"type":"Regression","description":"Linear regression with L2 regularization to reduce coefficient magnitude.","attributes":["alpha = 1.0","solver = auto","fit_intercept = True","max_iter = None"]},
    "lasso-regression": {"type":"Regression","description":"Linear regression with L1 regularization and sparse coefficients.","attributes":["alpha = 0.1","max_iter = 5000","fit_intercept = True"]},
    "decision-tree-regression": {"type":"Regression","description":"Predicts rent using recursive decision rules on the rental data.","attributes":["criterion = squared_error","max_depth = 12","random_state = 42"]},
    "decision-tree-id3": {"type":"Classification","description":"Decision tree using entropy/information gain for rent-band classification.","attributes":["criterion = entropy","max_depth = 12","splitter = best"]},
    "decision-tree-c45": {"type":"Classification","description":"Entropy-based C4.5-style decision tree exposed for comparison.","attributes":["criterion = entropy","max_depth = 12","splitter = best"]},
    "decision-tree-cart": {"type":"Classification","description":"CART-style classification tree using Gini impurity.","attributes":["criterion = gini","max_depth = 12","splitter = best"]},
    "logistic-regression": {"type":"Classification","description":"Classifies listings into Low, Medium, or High rent bands.","attributes":["max_iter = 1000","solver = lbfgs","3 rent classes"]},
    "bagging": {"type":"Regression","description":"Combines predictions from bootstrap-sampled base estimators.","attributes":["n_estimators = 10","max_samples = 0.6","bootstrap = True","random_state = 42"]},
    "random-forest": {"type":"Regression","description":"A bagged ensemble of randomized decision trees for rent prediction.","attributes":["n_estimators = 10","max_depth = 12","random_state = 42"]},
    "adaboost": {"type":"Regression","description":"Sequentially improves weak learners by emphasizing difficult observations.","attributes":["n_estimators = 15","learning_rate = 0.05","random_state = 42"]},
    "gradient-boosting": {"type":"Regression","description":"Builds additive trees sequentially to reduce prediction error.","attributes":["n_estimators = 30","learning_rate = 0.05","max_depth = 3"]},
    "xgboost": {"type":"Regression","description":"Gradient-boosted decision trees with regularized boosting.","attributes":["n_estimators = 30","learning_rate = 0.05","max_depth = 6","subsample = 0.8"]},
    "lightgbm": {"type":"Regression","description":"Gradient boosting using a leaf-wise tree growth strategy.","attributes":["n_estimators = 30","learning_rate = 0.05","num_leaves = 31","subsample = 0.8"]},
    "kmeans": {"type":"Clustering","description":"Groups listings into three clusters using distance from cluster centroids.","attributes":["n_clusters = 3","n_init = 10","random_state = 42"]},
    "hierarchical": {"type":"Clustering","description":"Builds a hierarchy of listings and forms three clusters using Ward linkage.","attributes":["n_clusters = 3","linkage = ward"]},
    "agglomerative": {"type":"Clustering","description":"Bottom-up clustering using average linkage between groups.","attributes":["n_clusters = 3","linkage = average"]},
    "dbscan": {"type":"Clustering","description":"Density-based clustering that also identifies noise points.","attributes":["eps = 0.8","min_samples = 8"]},
    "pca": {"type":"PCA","description":"Reduces the numeric rental dataset to principal components while retaining variance.","attributes":["n_components = 2","standardized input","random_state = 42"]},
}


def _all_model_names():
    names = {}
    for group in list(MODEL_GROUPS.values()) + list(UNSUPERVISED_GROUPS.values()):
        names.update(dict(group))
    return names


@app.route("/models")
def models_page():
    return render_template(
        "models.html", active="models", model_groups=MODEL_GROUPS,
        unsupervised_groups=UNSUPERVISED_GROUPS, model_info=MODEL_INFO, selected_model=None,
        results=None, error=None, charts=[]
    )


@app.route("/models/<model_key>")
def model_detail(model_key):
    names = _all_model_names()
    if model_key not in names:
        return redirect(url_for("models_page"))

    try:
        results = train_model(current_data(), model_key)
        charts = []
        if results["kind"] == "Regression":
            charts = create_model_visualizations(results["actual"], results["predictions"], model_key)
        elif results["kind"] == "Classification":
            charts = create_classification_visualization(results["confusion_matrix"], results["labels"], model_key)
        elif results["kind"] == "Clustering":
            charts = create_clustering_visualization(results["X_scaled"], results["labels"], model_key)
        elif results["kind"] == "PCA":
            charts = create_pca_visualization(results["transformed"], results["explained_variance"])
        results["charts"] = charts
        return render_template(
            "models.html", active="models", model_groups=MODEL_GROUPS,
            unsupervised_groups=UNSUPERVISED_GROUPS, model_info=MODEL_INFO,
            selected_model={"key": model_key, "name": names[model_key]},
            results=results, error=None, charts=charts
        )
    except Exception as e:
        return render_template(
            "models.html", active="models", model_groups=MODEL_GROUPS,
            unsupervised_groups=UNSUPERVISED_GROUPS, model_info=MODEL_INFO,
            selected_model={"key": model_key, "name": names[model_key]},
            results=None, charts=[], error=f"Model error: {e}"
        )


@app.route("/regression")
def regression_page():
    return redirect(url_for("model_detail", model_key="linear-regression"))


@app.route("/prediction", methods=["GET", "POST"])
def prediction_page():
    prediction = None
    error = None
    values = {c: "" for c in MODEL_FEATURES}

    if request.method == "POST":
        values.update({c: request.form.get(c, "") for c in MODEL_FEATURES})
        try:
            bundle = load_model()
            if bundle is None:
                processed = preprocess_data(current_data())
                train_linear_regression(processed)
                bundle = load_model()

            X = transform_input(values, bundle["preprocessor"])
            prediction = float(bundle["model"].predict(X)[0])
        except Exception as e:
            error = str(e)

    return render_template(
        "prediction.html",
        active="prediction",
        prediction=prediction,
        error=error,
        values=values,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
