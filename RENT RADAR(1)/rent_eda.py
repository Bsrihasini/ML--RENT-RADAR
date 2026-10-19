import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data

CHART_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="#0b0f0c")
    plt.close(fig)
    return name


def create_rent_distribution_chart(df):
    price = df["price"].dropna()
    counts = [
        int((price < 500).sum()),
        int(((price >= 500) & (price <= 1500)).sum()),
        int((price > 1500).sum()),
    ]
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0b0f0c")
    wedges, _ = ax.pie(
        counts,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.32, "edgecolor": "#0b0f0c"},
        colors=["#2e7d45", "#45c86f", "#72e69a"],
    )
    ax.text(0, 0.08, "Total Listings", ha="center", va="center",
            color="#aab5ae", fontsize=12)
    ax.text(0, -0.12, f"{len(price):,}", ha="center", va="center",
            color="white", fontsize=18, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_facecolor("#0b0f0c")
    return _save(fig, "rent_distribution.png")


def run_eda(df=None):
    df = df if df is not None else load_data()
    charts = []

    numeric = df.select_dtypes(include="number")
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        missing.sort_values(ascending=False).plot(kind="bar", ax=ax)
        ax.set_title("Missing Values by Column")
        ax.set_ylabel("Count")
        charts.append(_save(fig, "missing_values.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["price"].dropna(), bins=40, kde=True, ax=ax)
    ax.set_title("Monthly Rent Distribution")
    ax.set_xlabel("Rent (USD)")
    charts.append(_save(fig, "rent_distribution_hist.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df["price"], ax=ax)
    ax.set_title("Rent Boxplot")
    ax.set_xlabel("Rent (USD)")
    charts.append(_save(fig, "rent_boxplot.png"))

    if "bedrooms" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        order = sorted(df["bedrooms"].dropna().unique())[:12]
        sns.boxplot(data=df[df["bedrooms"].isin(order)], x="bedrooms", y="price", ax=ax)
        ax.set_title("Rent by Bedrooms")
        charts.append(_save(fig, "rent_by_bedrooms.png"))

    if "square_feet" in df.columns:
        valid = df[["square_feet", "price"]].dropna()
        sample = valid.sample(min(5000, len(valid)), random_state=42) if len(valid) else valid
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=sample, x="square_feet", y="price", alpha=0.45, ax=ax)
        ax.set_title("Square Feet vs Rent")
        charts.append(_save(fig, "sqft_vs_rent.png"))

    if "cityname" in df.columns:
        top_cities = df["cityname"].value_counts().head(12).index
        city_avg = df[df["cityname"].isin(top_cities)].groupby("cityname")["price"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(9, 6))
        city_avg.plot(kind="barh", ax=ax)
        ax.set_title("Average Rent — Top Cities by Listing Count")
        ax.set_xlabel("Average Rent (USD)")
        charts.append(_save(fig, "city_average_rent.png"))

    if not numeric.empty:
        corr = numeric.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Greens", ax=ax)
        ax.set_title("Numeric Correlation Heatmap")
        charts.append(_save(fig, "correlation_heatmap.png"))

    stats = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "missing": missing.to_dict(),
        "average_rent": float(df["price"].mean()),
        "median_rent": float(df["price"].median()),
        "min_rent": float(df["price"].min()),
        "max_rent": float(df["price"].max()),
        "charts": charts,
    }
    return stats
