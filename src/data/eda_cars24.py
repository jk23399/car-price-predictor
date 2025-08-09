# src/data/eda_cars24.py
import pandas as pd
from pathlib import Path

def main(in_path: str):
    fp = Path(in_path)
    df = pd.read_csv(fp)

    print("\n=== Dataset Shape ===")
    print(df.shape)

    print("\n=== First 5 Rows ===")
    print(df.head())

    print("\n=== Missing Values (count) ===")
    print(df.isna().sum())

    print("\n=== Numeric Columns Summary ===")
    print(df.describe().T)

    # Detect categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print("\n=== Categorical Columns Unique Values ===")
    for col in cat_cols:
        uniq_vals = df[col].nunique()
        sample_vals = df[col].dropna().unique()[:10]
        print(f"{col}: {uniq_vals} unique | sample: {sample_vals}")

    print("\n=== Price Distribution ===")
    print(df["price"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))

    print("\n=== Age Distribution ===")
    print(df["age"].value_counts().sort_index())

if __name__ == "__main__":
    main("data/processed/cars24_clean.csv")
