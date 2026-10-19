import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_DATA_FILE = os.path.join(DATA_DIR, "apartments_for_rent_classified_100K.csv")

EXPECTED_COLUMNS = [
    "id", "category", "title", "body", "amenities", "bathrooms", "bedrooms",
    "currency", "fee", "has_photo", "pets_allowed", "price", "price_display",
    "price_type", "square_feet", "address", "cityname", "state",
    "latitude", "longitude", "source", "time"
]


def _read_rent_csv(path):
    """Read the supplied dataset safely.

    The source is semicolon-delimited and contains commas inside free-text fields.
    Splitting each physical row on semicolons is therefore more reliable than
    pandas' default comma parser for this particular file.
    """
    with open(path, "r", encoding="latin1", errors="replace") as f:
        lines = [line.rstrip("\r\n") for line in f if line.strip()]

    if not lines:
        raise ValueError("Dataset is empty.")

    header = lines[0].split(";")
    if header != EXPECTED_COLUMNS:
        # Fall back to pandas for uploaded standard CSVs.
        return pd.read_csv(path, low_memory=False)

    rows = []
    for line_no, line in enumerate(lines[1:], start=2):
        parts = line.split(";")
        if len(parts) != len(header):
            raise ValueError(
                f"Invalid dataset row {line_no}: expected {len(header)} fields, "
                f"found {len(parts)}."
            )
        rows.append(parts)

    return pd.DataFrame(rows, columns=header)


def load_data(path=None):
    path = path or DEFAULT_DATA_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = _read_rent_csv(path)

    # Normalize numeric fields.
    for col in ["price", "square_feet", "bedrooms", "bathrooms", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_data_summary(df):
    missing = df.isna().sum()
    return {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_counts": missing.to_dict(),
        "total_missing": int(missing.sum()),
        "duplicates": int(df.duplicated().sum()),
        "preview": df.head(10).fillna("").to_dict(orient="records"),
    }


def clean_for_display(df, rows=10):
    return df.head(rows).fillna("").to_dict(orient="records")
