# src/data/preprocess_cars24.py
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


CURRENT_YEAR = 2025

# --- simple normalizers -------------------------------------------------------
def norm_str(s: str) -> str:
    """Trim, collapse spaces, and title-case common names."""
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_fuel(s: str) -> str:
    """Map fuel names to a closed set."""
    s = norm_str(s).lower()
    if not s:
        return "unknown"
    if "petrol" in s or "gas" in s:
        return "petrol"
    if "diesel" in s:
        return "diesel"
    if "cng" in s:
        return "cng"
    if "lpg" in s:
        return "lpg"
    if "electric" in s or s == "ev":
        return "electric"
    return "other"

def norm_transmission(s: str) -> str:
    """Reduce transmission/drive variants to {manual, automatic, other}."""
    s = norm_str(s).lower()
    if not s:
        return "unknown"
    if "auto" in s:
        return "automatic"
    if "man" in s:
        return "manual"
    return "other"

def norm_body_type(s: str) -> str:
    """Map body/Type to a small vocabulary."""
    s = norm_str(s).lower()
    if not s:
        return "unknown"
    if "hatch" in s:
        return "hatchback"
    if "sedan" in s:
        return "sedan"
    if "suv" in s:
        return "suv"
    if "mpv" in s:
        return "mpv"
    if "van" in s:
        return "van"
    if "truck" in s or "pickup" in s:
        return "truck"
    if "coupe" in s:
        return "coupe"
    return "other"

def parse_brand_model(car_name: str) -> tuple[str, str]:
    """Split 'Car Name' into brand (first token) and model (rest)."""
    s = norm_str(car_name)
    if not s:
        return "", ""
    parts = s.split(" ", 1)
    if len(parts) == 1:
        return parts[0], ""
    brand, model = parts[0], parts[1]
    return brand, model

def to_int_safe(x) -> int | None:
    """Coerce strings like '103,781' or '103781 km' to int; return None if invalid."""
    if pd.isna(x):
        return None
    s = str(x)
    s = s.replace(",", "").strip()
    s = re.sub(r"[^\d\-]", "", s)  # keep digits and minus
    if s == "" or s == "-":
        return None
    try:
        return int(s)
    except Exception:
        return None


# --- core preprocessing --------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, normalize, and feature-engineer Cars24 dataset."""
    # Standardize column names to snake_case
    df = df.rename(columns={
        "Car Name": "car_name",
        "Year": "year",
        "Distance": "distance",
        "Owner": "owner",
        "Fuel": "fuel",
        "Location": "location",
        "Drive": "drive",
        "Type": "body_type",
        "Price": "price",
        # tolerate alternative column names if present
        "Transmission": "drive",
    })

    # Keep only expected columns
    expected = [
        "car_name","year","distance","owner","fuel",
        "location","drive","body_type","price"
    ]
    df = df[[c for c in expected if c in df.columns]].copy()

    # Strip whitespace on strings
    for col in ["car_name","fuel","location","drive","body_type"]:
        if col in df.columns:
            df[col] = df[col].apply(norm_str)

    # Numeric coercion
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["distance"] = df["distance"].apply(to_int_safe)
    df["owner"] = pd.to_numeric(df["owner"], errors="coerce")
    df["price"] = df["price"].apply(to_int_safe)

    # Drop obvious invalids
    df = df.dropna(subset=["year","distance","owner","price"]).copy()
    df = df[
        (df["year"].between(1995, CURRENT_YEAR)) &
        (df["distance"].between(0, 500_000)) &
        (df["owner"].between(1, 5)) &
        (df["price"] > 0)
    ].copy()

    # Normalize categoricals
    df["fuel"] = df["fuel"].apply(norm_fuel)
    df["drive"] = df["drive"].apply(norm_transmission)
    df["body_type"] = df["body_type"].apply(norm_body_type)
    df["location"] = df["location"].replace("", np.nan).fillna("UNKNOWN")

    # Brand/Model features
    bm = df["car_name"].apply(parse_brand_model)
    df["brand"] = bm.apply(lambda t: t[0])
    df["model"] = bm.apply(lambda t: t[1])

    # Feature engineering
    df["age"] = CURRENT_YEAR - df["year"].astype(int)
    df["age"] = df["age"].clip(lower=0, upper=40)
    df["km_per_year"] = (df["distance"] / df["age"].replace(0, 1)).round(2)

    # Log target for modeling stability
    df["log_price"] = np.log1p(df["price"])

    # Cast to categories for downstream CatBoost or for memory efficiency
    cat_cols = ["brand","model","fuel","location","drive","body_type"]
    for c in cat_cols:
        df[c] = df[c].astype("category")

    return df


def main(in_path: str, out_path: str, schema_path: str):
    in_fp = Path(in_path)
    out_fp = Path(out_path)
    schema_fp = Path(schema_path)

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(in_fp)

    cleaned = preprocess(df)
    cleaned.to_csv(out_fp, index=False)

    # Save a minimal schema to align GPT/ingest later
    schema = {
        "required_inputs": [
            "year","distance","owner","fuel","location","drive","body_type","car_name"
        ],
        "derived_inputs": ["brand","model","age","km_per_year"],
        "target": "price",
        "target_transformed": "log_price",
        "categorical_vocab": {
            "fuel": sorted(cleaned["fuel"].astype(str).unique().tolist()),
            "drive": sorted(cleaned["drive"].astype(str).unique().tolist()),
            "body_type": sorted(cleaned["body_type"].astype(str).unique().tolist()),
        }
    }
    schema_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(schema_fp, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    # Quick printout
    print(f"[OK] rows in: {len(df):,}  -> rows out: {len(cleaned):,}")
    print(f"[OK] saved cleaned dataset: {out_fp}")
    print(f"[OK] saved schema: {schema_fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Cars24 dataset.")
    parser.add_argument("--in", dest="in_path", default="data/raw/cars24.csv")
    parser.add_argument("--out", dest="out_path", default="data/processed/cars24_clean.csv")
    parser.add_argument("--schema", dest="schema_path", default="data/processed/cars24_schema.json")
    args = parser.parse_args()
    main(args.in_path, args.out_path, args.schema_path)
