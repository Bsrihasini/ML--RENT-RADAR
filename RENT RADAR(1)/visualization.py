import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CHART_DIR = os.path.join(os.path.dirname(__file__), "static", "charts", "models")
os.makedirs(CHART_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f"charts/models/{name}"


def create_model_visualizations(actual, predictions, key="model"):
    actual = np.asarray(actual, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    slug = key.replace("-", "_")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(x=actual, y=predictions, alpha=0.45, s=28, ax=ax)
    low = min(actual.min(), predictions.min())
    high = max(actual.max(), predictions.max())
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Actual Rent")
    ax.set_ylabel("Predicted Rent")
    ax.set_title("Actual vs Predicted Rent")
    actual_path = _save(fig, f"{slug}_actual_vs_predicted.png")

    residuals = actual - predictions
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, bins=40, kde=True, ax=ax)
    ax.axvline(0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Distribution")
    residual_path = _save(fig, f"{slug}_residual_distribution.png")
    return [actual_path, residual_path]


def create_classification_visualization(cm, labels, key):
    slug = key.replace("-", "_")
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Rent Category Confusion Matrix")
    return [_save(fig, f"{slug}_confusion_matrix.png")]


def create_clustering_visualization(X_scaled, labels, key):
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    slug = key.replace("-", "_")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=labels, s=16, alpha=0.65, cmap="Greens")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"{key.replace('-', ' ').title()} — Cluster Visualization")
    if len(set(labels)) <= 12:
        fig.colorbar(scatter, ax=ax, label="Cluster")
    return [_save(fig, f"{slug}_clusters.png")]


def create_pca_visualization(transformed, explained_variance):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(transformed[:, 0], transformed[:, 1], s=16, alpha=0.6)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA — First Two Principal Components")
    p1 = _save(fig, "pca_components.png")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(range(1, len(explained_variance) + 1), np.asarray(explained_variance) * 100)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("PCA Explained Variance")
    p2 = _save(fig, "pca_explained_variance.png")
    return [p1, p2]
