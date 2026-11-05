"""Shared helpers for the policy analysis (notebook 10).

The model recipe is copied unchanged from notebooks/09_evaluation.ipynb
(Logistic Regression, Extended features, C=0.01, rows weighted 1/n per
customer), but fitted on the TRAIN period only so that validation
predictions are out-of-sample and a calibrator can be fitted on them.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RFM_FEATURES = ["recency_days", "frequency_so_far", "monetary_so_far"]
EXTENDED_FEATURES = RFM_FEATURES + [
    "avg_order_value_so_far", "tenure_days", "mean_gap_so_far",
    "recency_to_typical_gap", "activity_last_90d",
]
LOG_FEATURES = [
    "monetary_so_far", "avg_order_value_so_far", "tenure_days",
    "mean_gap_so_far", "recency_to_typical_gap", "activity_last_90d",
    "distinct_products_so_far", "cancel_count_so_far", "frequency_so_far",
]
FINAL_C = 0.01


def prepare_features(df, features):
    X = df[features].copy()
    for col in features:
        if col in LOG_FEATURES:
            X[col] = np.log1p(X[col])
    return X


def fit_train_only(train_df, features=EXTENDED_FEATURES, c=FINAL_C):
    """Fit the final recipe on `train_df` only; return a scoring callable."""
    w = 1.0 / train_df.groupby("customer_id")["customer_id"].transform("size")
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X = sc.fit_transform(imp.fit_transform(prepare_features(train_df, features)))
    lr = LogisticRegression(max_iter=1000, solver="liblinear", C=c)
    lr.fit(X, train_df["churn"], sample_weight=w)

    def predict(df):
        Z = sc.transform(imp.transform(prepare_features(df, features)))
        return lr.predict_proba(Z)[:, 1]

    return predict, lr


def expected_margin(p, v, d, e, m, r, c_contact):
    """E[dM] per customer (Strategy 7.7)."""
    return p * e * v * (m - d) - (1 - p) * r * v * d - c_contact


def realized_margin(df, mask, d, e, m, r, c_contact, vcol="value_v"):
    """Realized scenario margin of targeted rows using actual labels (7.8)."""
    t = df[mask]
    churn = t["churn"] == 1
    v = t[vcol]
    return (
        (e * v[churn] * (m - d)).sum()
        - (r * v[~churn] * d).sum()
        - len(t) * c_contact
    )


def top_k_mask(df, score_col, k, by="snapshot_date"):
    """Boolean mask of the top-k fraction of each week by `score_col`."""
    rank = df.groupby(by)[score_col].rank(ascending=False, method="first")
    size = df.groupby(by)[score_col].transform("size")
    return rank <= np.maximum(np.round(size * k), 1)


def row_gain(df, d, e, m, r, c, vcol="value_v"):
    """Per-row realized scenario margin of contacting that row (7.8)."""
    v = df[vcol]
    return np.where(df["churn"] == 1, e * v * (m - d), -r * v * d) - c


def weekly_margin(df, mask, scen, vcol="value_v"):
    """Realized margin of the targeted rows, summed per scoring week."""
    d, e, m, r, c = scen
    g = pd.Series(row_gain(df, d, e, m, r, c, vcol), index=df.index)
    weeks = df["snapshot_date"].unique()
    return (
        g[mask].groupby(df["snapshot_date"][mask]).sum()
        .reindex(weeks, fill_value=0)
    )


def margin_breakdown(df, mask, scen, vcol="value_v"):
    """Weekly decomposition of a policy's scenario margin (7.8).

    Returns per-week averages: contacts, revenue_preserved (e * V of
    targeted churners), gross_margin (m * preserved revenue), discount on
    responders, discount given away to would-be buyers (cannibalization),
    contact cost and net margin. Also the break-even gross margin m*, the
    margin rate at which net margin is zero (net is linear in m).
    """
    d, e, m, r, c = scen
    t = df[mask]
    weeks = df["snapshot_date"].nunique()
    churn = t["churn"] == 1
    vc, vn = t.loc[churn, vcol].sum(), t.loc[~churn, vcol].sum()
    n = len(t)
    out = {
        "contacts_week": n / weeks,
        "revenue_preserved_week": e * vc / weeks,
        "gross_margin_week": e * vc * m / weeks,
        "discount_on_responders_week": -e * vc * d / weeks,
        "discount_wasted_on_buyers_week": -r * vn * d / weeks,
        "contact_cost_week": -n * c / weeks,
    }
    out["net_margin_week"] = (
        out["gross_margin_week"] + out["discount_on_responders_week"]
        + out["discount_wasted_on_buyers_week"] + out["contact_cost_week"]
    )
    out["break_even_margin_rate"] = (
        (e * d * vc + r * d * vn + c * n) / (e * vc) if vc > 0 else np.nan
    )
    return out
