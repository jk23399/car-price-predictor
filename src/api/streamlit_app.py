# streamlit_app.py
import os
import math
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =============================
# Config
# =============================
# Project root (…/car-price-predictor/)
BASE_DIR = Path(__file__).resolve().parents[2]

CURRENT_YEAR = 2025
THRESH_EXCELLENT, THRESH_GOOD, THRESH_FAIR = -0.15, -0.05, 0.05

# Default artifact under models/
DEFAULT_ARTIFACT = str(BASE_DIR / "src" / "models" / "lgbm_cargurus_pricer_derived.joblib")

# =============================
# Helpers
# =============================
def _resolve_path(p: str | os.PathLike) -> Path:
    """Resolve to absolute path. If relative, treat as relative to project root."""
    p = Path(p)
    return p if p.is_absolute() else (BASE_DIR / p)

@st.cache_resource(show_spinner=False)
def load_artifacts(path_str: str):
    """Load joblib artifact (uses user-provided path)."""
    p = _resolve_path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found: {p}")
    artifacts = joblib.load(p)
    model = artifacts["model"]
    return artifacts, model, p

def classify_deal(actual_price: float, predicted_price: float) -> str:
    if predicted_price is None or predicted_price <= 0 or pd.isna(predicted_price):
        return "Unknown"
    if actual_price is None or pd.isna(actual_price):
        return "Unknown"
    r = (actual_price - predicted_price) / predicted_price
    if r <= THRESH_EXCELLENT: return "Excellent Deal"
    if r <= THRESH_GOOD:      return "Good Deal"
    if r <= THRESH_FAIR:      return "Fair Deal"
    return "Bad Deal"

def preprocess_for_inference(df_raw: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Align raw rows to training feature order, recreate derived features, and apply TE."""
    enc_map = artifacts["encoding_map"]
    global_mean = artifacts["global_target_mean"]
    model_cols = artifacts["model_columns"]
    num_cols = artifacts["num_cols"]
    bin_cols = artifacts["bin_cols"]
    med = artifacts["numeric_medians"]

    X_new = df_raw.copy()

    # numerics
    for c in ["year", "mileage", "engine_displacement"]:
        X_new[c] = pd.to_numeric(X_new.get(c, np.nan), errors="coerce").fillna(med[c])

    # derived features (must mirror training)
    X_new["car_age"] = (CURRENT_YEAR - pd.to_numeric(X_new["year"], errors="coerce")).clip(lower=0)
    X_new["miles_per_year"] = pd.to_numeric(X_new["mileage"], errors="coerce") / X_new["car_age"].replace(0, np.nan)
    X_new["miles_per_year"] = X_new["miles_per_year"].fillna(0)
    X_new["log_mileage"] = np.log1p(pd.to_numeric(X_new["mileage"], errors="coerce"))
    X_new["price_per_mile"] = (
        pd.to_numeric(X_new.get("price", np.nan), errors="coerce")
        / (1.0 + pd.to_numeric(X_new["mileage"], errors="coerce"))
    )
    X_new["price_per_mile"] = X_new["price_per_mile"].fillna(med.get("price_per_mile", 0.0))

    # optional binary flags
    for b in ["has_accidents", "salvage", "theft_title"]:
        if b in bin_cols:
            X_new[b] = pd.to_numeric(X_new.get(b, 0), errors="coerce").fillna(0).astype(int)

    # strings for target encoding
    X_new["model_name"] = X_new.get("model_name", "Unknown").fillna("Unknown").astype(str)
    X_new["trim_name"]  = X_new.get("trim_name",  "Unknown").fillna("Unknown").astype(str)
    X_new["model_trim"] = X_new["model_name"] + "_" + X_new["trim_name"]
    X_new["model_trim_encoded"] = X_new["model_trim"].map(enc_map).fillna(global_mean)

    # drop raw strings; align columns to training order
    X_new = X_new.drop(columns=["model_name", "trim_name", "model_trim"], errors="ignore")
    for col in model_cols:
        if col not in X_new.columns:
            X_new[col] = 0
    return X_new[model_cols]

# =============================
# UI
# =============================
st.set_page_config(page_title="Car Deal Rater", page_icon="🚗", layout="centered")
st.title("🚗 Car Deal Rater")
st.caption("Listing price only (taxes/fees excluded). Uses your LightGBM model artifact.")

# Sidebar: artifact path input
artifact_path_input = st.sidebar.text_input("Artifact path", value=DEFAULT_ARTIFACT)

try:
    artifacts, model, resolved = load_artifacts(artifact_path_input)
    st.sidebar.success("Artifact loaded")
    st.sidebar.caption(f"Resolved: {resolved}")
except Exception as e:
    st.sidebar.error(f"Failed to load artifact: {e}")
    st.stop()

tab_manual, tab_batch, tab_url = st.tabs(["Manual Input", "Batch CSV", "URL (TODO)"])

# -------- Manual Input --------
with tab_manual:
    st.subheader("Manual Listing")
    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Year", min_value=1980, max_value=2025, value=2018)
        mileage = st.number_input("Mileage (miles)", min_value=0, max_value=500_000, value=45_000)
        engine_disp = st.number_input(
            "Engine displacement (cc or L)", min_value=0.0, value=2000.0,
            help="Enter liters if you know; cc is fine too (we'll convert if value looks like cc)."
        )
    with c2:
        model_name = st.text_input("Model name", value="Camry")
        trim_name = st.text_input("Trim name", value="SE")
        price = st.number_input("Listing price (USD)", min_value=0, max_value=500_000, value=20_000)

    adv = st.expander("Optional flags")
    with adv:
        has_accidents = st.checkbox("Has accidents", value=False)
        salvage = st.checkbox("Salvage", value=False)
        theft_title = st.checkbox("Theft title", value=False)

    if st.button("Evaluate deal", type="primary"):
        row = {
            "year": year,
            "mileage": mileage,
            "engine_displacement": engine_disp,
            "model_name": model_name,
            "trim_name": trim_name,
            "price": price,
            "has_accidents": int(has_accidents),
            "salvage": int(salvage),
            "theft_title": int(theft_title),
        }
        df_in = pd.DataFrame([row])

        # Heuristic: if displacement > 10, assume cc and convert to liters
        if df_in.loc[0, "engine_displacement"] > 10:
            df_in.loc[0, "engine_displacement"] = df_in.loc[0, "engine_displacement"] / 1000.0

        Xn = preprocess_for_inference(df_in, artifacts)
        pred_price = float(np.expm1(model.predict(Xn))[0])
        rating = classify_deal(price, pred_price)
        delta_pct = (price - pred_price) / max(pred_price, 1e-9) * 100.0

        st.metric("Predicted price", f"${pred_price:,.0f}", f"{delta_pct:+.1f}% vs listing")
        st.success(f"Deal rating: **{rating}**")

# -------- Batch CSV --------
with tab_batch:
    st.subheader("Batch CSV")
    st.caption("Required: year, mileage, price, model_name. Optional: engine_displacement, trim_name, has_accidents, salvage, theft_title.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df_csv = pd.read_csv(up)

            # best-effort typing
            for c in ["year","mileage","price","engine_displacement","has_accidents","salvage","theft_title"]:
                if c in df_csv.columns:
                    df_csv[c] = pd.to_numeric(df_csv[c], errors="coerce")
            for s in ["model_name","trim_name"]:
                if s in df_csv.columns:
                    df_csv[s] = df_csv[s].astype(str)

            Xn = preprocess_for_inference(df_csv, artifacts)
            preds = np.expm1(model.predict(Xn))

            out = df_csv.copy()
            out["pred_price"] = preds
            if "price" in out.columns:
                out["delta_pct"] = (out["price"] - out["pred_price"]) / np.clip(out["pred_price"], 1e-9, None) * 100.0
                out["deal_rating"] = out.apply(lambda r: classify_deal(r.get("price", np.nan), r["pred_price"]), axis=1)

            st.dataframe(out.head(50))
            st.download_button(
                "Download results CSV",
                out.to_csv(index=False).encode("utf-8"),
                file_name="deal_ratings.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")

# -------- URL (placeholder) --------
with tab_url:
    st.subheader("Listing URL (TODO: GPT extraction)")
    url = st.text_input("Craigslist URL")
    st.caption("In production, fetch the page and use GPT to extract year/mileage/price/model/trim, then call the same pipeline.")
    if st.button("Parse & Evaluate (demo)"):
        st.warning("URL extraction not implemented in this demo. Use Manual or Batch tabs.")
