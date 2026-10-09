"""
pipeline.py — Core scoring logic for the churn prediction app.

Full chain: raw transactions → clean → features → score → rank → target list.
All feature engineering is identical to notebooks/07_data_preparation.ipynb
and notebooks/09_evaluation.ipynb to ensure no train/serve skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ── Model artefacts ──────────────────────────────────────────────────────────
_ARTEFACT_DIR = Path(__file__).parent / "model_artefacts"

FULL_FEATURES = [
    "recency_days", "frequency_so_far", "monetary_so_far",
    "avg_order_value_so_far", "tenure_days", "mean_gap_so_far",
    "recency_to_typical_gap", "activity_last_90d",
    "distinct_products_so_far", "cancel_count_so_far",
    "cancel_value_share_so_far",
]

LOG_FEATURES = {
    "monetary_so_far", "avg_order_value_so_far", "tenure_days",
    "mean_gap_so_far", "recency_to_typical_gap", "activity_last_90d",
    "distinct_products_so_far", "cancel_count_so_far", "frequency_so_far",
}

R_MAX_DAYS = 136   # P90 of inter-purchase interval (from Chapter 6)
H_DAYS     = 90    # Prediction horizon


def load_artefacts():
    """Load the trained model, imputer, and scaler from disk."""
    model   = joblib.load(_ARTEFACT_DIR / "model.pkl")
    imputer = joblib.load(_ARTEFACT_DIR / "imputer.pkl")
    scaler  = joblib.load(_ARTEFACT_DIR / "scaler.pkl")
    return model, imputer, scaler


# ── Step 1: Data loading ─────────────────────────────────────────────────────

def load_transactions(path: str) -> pd.DataFrame:
    """Load raw Online Retail II data from Excel or Parquet."""
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in (".xlsx", ".xls"):
        sheets = []
        for sheet in ["Year 2009-2010", "Year 2010-2011"]:
            try:
                sheets.append(pd.read_excel(p, sheet_name=sheet))
            except Exception:
                pass
        if not sheets:
            df = pd.read_excel(p)  # fallback: single sheet
        else:
            df = pd.concat(sheets, ignore_index=True)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    df["Invoice"]     = df["Invoice"].astype(str)
    df["StockCode"]   = df["StockCode"].astype(str)
    df["Description"] = df["Description"].fillna("").astype(str)
    return df


# ── Step 2: Transaction cleaning ─────────────────────────────────────────────

def _find_non_product_codes(df: pd.DataFrame) -> set:
    codes = df["StockCode"].unique()
    return {c for c in codes if not c[:1].isdigit()}


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Section 7.2 cleaning rules. Returns valid purchase rows."""
    non_product = _find_non_product_codes(df)

    has_customer   = df["Customer ID"].notna()
    not_cancelled  = ~df["Invoice"].str.startswith("C")
    valid_amounts  = (df["Quantity"] > 0) & (df["Price"] > 0)
    is_product     = ~df["StockCode"].isin(non_product)
    not_duplicate  = ~df.duplicated()

    mask = has_customer & not_cancelled & valid_amounts & is_product & not_duplicate
    clean = df[mask].copy()

    clean = clean.rename(columns={
        "Invoice":      "invoice_id",
        "StockCode":    "product_id",
        "Description":  "description",
        "Quantity":     "quantity",
        "InvoiceDate":  "invoice_time",
        "Price":        "unit_price",
        "Customer ID":  "customer_id",
        "Country":      "country",
    })
    clean["customer_id"]  = clean["customer_id"].astype(int)
    clean["invoice_time"] = pd.to_datetime(clean["invoice_time"])
    clean["total_price"]  = clean["quantity"] * clean["unit_price"]
    return clean


# ── Step 3: Collapse to purchase occasions ────────────────────────────────────

def compute_occasions(clean: pd.DataFrame) -> pd.DataFrame:
    """One row per (customer_id, invoice_id) — revenue, products, etc."""
    occ = (
        clean.groupby(["customer_id", "invoice_id"], as_index=False)
        .agg(
            invoice_time=("invoice_time", "min"),
            revenue=("total_price", "sum"),
            quantity=("quantity", "sum"),
            products=("product_id", lambda s: frozenset(s)),
        )
        .sort_values(["customer_id", "invoice_time"])
        .reset_index(drop=True)
    )
    return occ


# ── Step 4: Feature engineering ───────────────────────────────────────────────

def _compute_cumulative_distinct_products(occ: pd.DataFrame) -> pd.DataFrame:
    ordered = occ.sort_values(["customer_id", "invoice_time"]).copy()
    counts = []
    for _, rows in ordered.groupby("customer_id"):
        seen = set()
        for products in rows["products"]:
            seen.update(products)
            counts.append(len(seen))
    ordered["distinct_products_so_far"] = counts
    return ordered


def build_features(clean: pd.DataFrame, occ: pd.DataFrame,
                   snapshot_date: pd.Timestamp, r_max: int = R_MAX_DAYS
                   ) -> pd.DataFrame:
    """
    Compute the full feature set for every active customer at snapshot_date.
    Identical logic to notebook 07 — no leakage (only history up to snapshot_date).
    """
    snap = snapshot_date

    # History up to snapshot_date only
    hist = occ[occ["invoice_time"] <= snap].copy()
    if hist.empty:
        return pd.DataFrame()

    # Cumulative distinct products
    hist = _compute_cumulative_distinct_products(hist)

    # Cumulative stats per customer (up to snapshot_date)
    cum = hist.groupby("customer_id").agg(
        first_purchase=("invoice_time", "min"),
        last_purchase=("invoice_time", "max"),
        frequency_so_far=("invoice_id", "nunique"),
        monetary_so_far=("revenue", "sum"),
        avg_order_value_so_far=("revenue", "mean"),
        distinct_products_so_far=("distinct_products_so_far", "last"),
    ).reset_index()

    cum["recency_days"]  = (snap - cum["last_purchase"]).dt.total_seconds() / 86400
    cum["tenure_days"]   = (snap - cum["first_purchase"]).dt.total_seconds() / 86400

    # Activity condition
    active = cum[cum["recency_days"] <= r_max].copy()
    if active.empty:
        return pd.DataFrame()

    # Inter-purchase gap (mean) — needs ≥ 2 occasions
    gaps = []
    for cid, rows in hist.groupby("customer_id"):
        times = rows["invoice_time"].sort_values()
        if len(times) >= 2:
            diffs = times.diff().dropna().dt.total_seconds() / 86400
            gaps.append({"customer_id": cid, "mean_gap_so_far": diffs.mean()})
        else:
            gaps.append({"customer_id": cid, "mean_gap_so_far": np.nan})
    gap_df = pd.DataFrame(gaps)
    active = active.merge(gap_df, on="customer_id", how="left")

    # Recency-to-typical-gap ratio
    active["recency_to_typical_gap"] = (
        active["recency_days"] / active["mean_gap_so_far"].replace(0, np.nan)
    )

    # Activity in last 90 days
    cutoff_90 = snap - pd.Timedelta(days=90)
    act_90 = (
        hist[hist["invoice_time"] > cutoff_90]
        .groupby("customer_id")["invoice_id"]
        .nunique()
        .reset_index()
        .rename(columns={"invoice_id": "activity_last_90d"})
    )
    active = active.merge(act_90, on="customer_id", how="left")
    active["activity_last_90d"] = active["activity_last_90d"].fillna(0)

    # Cancellation features (from raw clean data, up to snapshot_date)
    cancel_hist = clean[
        (clean["invoice_id"].str.startswith("C")) &
        (clean["invoice_time"] <= snap)
    ].copy()
    cancel_hist["cancel_value"] = cancel_hist["quantity"].abs() * cancel_hist["unit_price"]

    cancel_agg = cancel_hist.groupby("customer_id").agg(
        cancel_count_so_far=("invoice_id", "nunique"),
        cancel_value_so_far=("cancel_value", "sum"),
    ).reset_index()
    active = active.merge(cancel_agg, on="customer_id", how="left")
    active["cancel_count_so_far"]  = active["cancel_count_so_far"].fillna(0)
    active["cancel_value_so_far"]  = active["cancel_value_so_far"].fillna(0)
    active["cancel_value_share_so_far"] = (
        active["cancel_value_so_far"]
        / (active["monetary_so_far"] + active["cancel_value_so_far"])
    ).fillna(0)

    # Segment
    active["segment"] = np.where(active["frequency_so_far"] == 1, "one-time", "repeat")

    # Revenue at risk proxy
    active["revenue_at_risk"] = active["avg_order_value_so_far"]

    # Country (latest known)
    country_map = (
        clean[clean["invoice_time"] <= snap]
        .sort_values("invoice_time")
        .groupby("customer_id")["country"]
        .last()
    )
    active["country"] = active["customer_id"].map(country_map).fillna("Unknown")

    active["snapshot_date"] = snap
    return active.reset_index(drop=True)


# ── Step 5 + 6: Score + rank ─────────────────────────────────────────────────

def score_and_rank(features_df: pd.DataFrame,
                   model, imputer, scaler) -> pd.DataFrame:
    """Apply the trained pipeline and rank customers by churn probability."""
    if features_df.empty:
        return pd.DataFrame()

    X = features_df[FULL_FEATURES].copy()
    for col in LOG_FEATURES:
        if col in X.columns:
            X[col] = np.log1p(X[col])

    X_imp    = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)

    scored = features_df.copy()
    scored["churn_probability"] = model.predict_proba(X_scaled)[:, 1]
    scored = scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    scored["rank"] = range(1, len(scored) + 1)

    n = len(scored)
    thresholds = [0, int(n * 0.30), int(n * 0.60), n]
    labels_map = {}
    for i, row in scored.iterrows():
        r = row["rank"]
        if r <= thresholds[1]:
            labels_map[i] = "🔴 High"
        elif r <= thresholds[2]:
            labels_map[i] = "🟡 Medium"
        else:
            labels_map[i] = "🟢 Low"
    scored["risk_band"] = pd.Series(labels_map)

    return scored


# ── Step 7: Target list selection ────────────────────────────────────────────

def select_targets(scored: pd.DataFrame, k_frac: float) -> pd.DataFrame:
    """Return the top K% of ranked customers as the target list."""
    k = max(int(round(len(scored) * k_frac)), 1)
    return scored.head(k).copy()


# ── Bootstrap confidence interval for churn probability ──────────────────────

def bootstrap_churn_ci(features_df: pd.DataFrame,
                        model, imputer, scaler,
                        n_bootstrap: int = 200,
                        ci: float = 0.90) -> pd.DataFrame:
    """
    Estimate per-customer churn probability with a confidence interval
    using parametric bootstrap on the logistic regression coefficients.

    This simulates the uncertainty around the point estimate by
    sampling from the asymptotic normal distribution of the LR weights.
    Returns the original scored df with two extra columns:
        churn_prob_lo, churn_prob_hi
    """
    if features_df.empty:
        return features_df

    X = features_df[FULL_FEATURES].copy()
    for col in LOG_FEATURES:
        if col in X.columns:
            X[col] = np.log1p(X[col])
    X_imp    = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)

    # Point estimate
    p_point = model.predict_proba(X_scaled)[:, 1]

    # Bootstrap: sample LR coefficients from N(coef, se)
    coef_mean = model.coef_[0]
    # Approximate SE from the Fisher information matrix
    # For LR: Var(β) ≈ (X'WX)^-1  where W = p*(1-p)
    W = p_point * (1 - p_point)
    XtWX = X_scaled.T @ (X_scaled * W[:, np.newaxis])
    try:
        coef_cov  = np.linalg.inv(XtWX + np.eye(XtWX.shape[0]) * 1e-6)
        coef_se   = np.sqrt(np.diag(coef_cov))
    except np.linalg.LinAlgError:
        # Fallback: uniform small SE
        coef_se = np.ones_like(coef_mean) * 0.05

    rng       = np.random.default_rng(42)
    boot_probs = np.zeros((n_bootstrap, len(X_scaled)))
    intercept  = model.intercept_[0]

    for b in range(n_bootstrap):
        coef_sample = rng.normal(coef_mean, coef_se)
        logit       = X_scaled @ coef_sample + intercept
        boot_probs[b] = 1 / (1 + np.exp(-logit))

    alpha_lo = (1 - ci) / 2
    alpha_hi = 1 - alpha_lo

    out = features_df.copy()
    out["churn_probability"] = p_point
    out["churn_prob_lo"]     = np.quantile(boot_probs, alpha_lo, axis=0)
    out["churn_prob_hi"]     = np.quantile(boot_probs, alpha_hi, axis=0)
    return out


# ── Top-level pipeline entry point ───────────────────────────────────────────

def run_pipeline(path: str,
                 snapshot_date: pd.Timestamp | None = None,
                 k_frac: float = 0.30,
                 r_max: int = R_MAX_DAYS,
                 with_ci: bool = True):
    """
    Full pipeline: raw file → cleaned data → features → scores → target list.

    Args:
        path:          Path to the transaction Excel/Parquet file.
        snapshot_date: Scoring date. Defaults to the last transaction date.
        k_frac:        Target list fraction (e.g. 0.30 → top 30%).
        r_max:         Activity threshold in days.
        with_ci:       Whether to compute bootstrap confidence intervals.

    Returns:
        dict with keys:
            raw_df, clean_df, occ_df, features_df,
            scored_df, target_df,
            snapshot_date, stats
    """
    # Load
    raw_df  = load_transactions(path)

    # Clean
    clean_df = clean_transactions(raw_df)

    # Occasions
    occ_df = compute_occasions(clean_df)

    # Snapshot date
    if snapshot_date is None:
        snapshot_date = clean_df["invoice_time"].max().normalize()

    # Features
    features_df = build_features(clean_df, occ_df, snapshot_date, r_max)
    if features_df.empty:
        return None

    # Load artefacts
    model, imputer, scaler = load_artefacts()

    # Score + rank
    scored_df = score_and_rank(features_df, model, imputer, scaler)

    # Confidence intervals
    if with_ci:
        scored_df = bootstrap_churn_ci(scored_df, model, imputer, scaler)

    # Target list
    target_df = select_targets(scored_df, k_frac)

    # Pipeline stats
    stats = {
        "total_transactions":    len(raw_df),
        "valid_transactions":    len(clean_df),
        "unique_customers_raw":  raw_df["Customer ID"].nunique(),
        "active_customers":      len(scored_df),
        "churn_rate_estimated":  float(scored_df["churn_probability"].mean()),
        "target_list_size":      len(target_df),
        "total_revenue_at_risk": float(
            target_df["revenue_at_risk"].sum()
        ),
        "snapshot_date": snapshot_date,
    }

    return {
        "raw_df":       raw_df,
        "clean_df":     clean_df,
        "occ_df":       occ_df,
        "features_df":  features_df,
        "scored_df":    scored_df,
        "target_df":    target_df,
        "snapshot_date": snapshot_date,
        "stats":        stats,
    }
