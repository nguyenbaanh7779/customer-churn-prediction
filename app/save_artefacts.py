"""
save_artefacts.py — Run once after training to save model artefacts for the app.

Usage (from project root):
    python app/save_artefacts.py

Requires: data/processed/modeling_dataset_h90.parquet  (produced by notebook 07)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Same constants as notebooks/09_evaluation.ipynb ──────────────────────────
H            = 90
COOLDOWN     = H
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

ARTEFACT_DIR = Path(__file__).parent / "model_artefacts"
ARTEFACT_DIR.mkdir(exist_ok=True)
DATA_PATH    = Path(__file__).parent.parent / "data" / "processed" / "modeling_dataset_h90.parquet"


def apply_episode_cooldown(active_grid, cooldown_days):
    cooldown = pd.Timedelta(days=cooldown_days)
    ordered  = active_grid.sort_values(["customer_id", "snapshot_date"])
    selected = []
    for _, rows in ordered.groupby("customer_id"):
        next_allowed = None
        for row in rows.itertuples():
            if next_allowed is None or row.snapshot_date >= next_allowed:
                selected.append(row.Index)
                next_allowed = row.snapshot_date + cooldown
    return active_grid.loc[selected]


def prepare_X(df, features):
    X = df[features].copy()
    for col in features:
        if col in LOG_FEATURES:
            X[col] = np.log1p(X[col])
    return X


def main():
    print(f"Loading dataset from {DATA_PATH} ...")
    ds = pd.read_parquet(DATA_PATH)

    # ── Refit on train + validation (same as notebook 09) ────────────────────
    train_df = ds[(ds["split"] == "train") & ds["in_training_sample"]]
    val_df   = ds[ds["split"] == "validation"]

    train_start = train_df["snapshot_date"].min()
    val_end     = val_df["snapshot_date"].max()

    refit_pool = ds[
        (ds["snapshot_date"] >= train_start) &
        (ds["snapshot_date"] <= val_end) &
        ds["labelable"]
    ]
    refit_rows    = apply_episode_cooldown(
        refit_pool[["customer_id", "snapshot_date"]], cooldown_days=COOLDOWN
    )
    refit_train   = refit_pool.loc[refit_rows.index]

    print(f"Refit training rows: {len(refit_train):,}")

    X_raw = prepare_X(refit_train, FULL_FEATURES)

    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_raw)

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_imp)

    model   = LogisticRegression(max_iter=1000, solver="liblinear", C=0.01)
    model.fit(X_sc, refit_train["churn"])

    # ── Save ─────────────────────────────────────────────────────────────────
    joblib.dump(model,   ARTEFACT_DIR / "model.pkl")
    joblib.dump(imputer, ARTEFACT_DIR / "imputer.pkl")
    joblib.dump(scaler,  ARTEFACT_DIR / "scaler.pkl")
    print(f"Saved artefacts to {ARTEFACT_DIR}/")
    print("  model.pkl   imputer.pkl   scaler.pkl")


if __name__ == "__main__":
    main()
