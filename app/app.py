"""
app.py — Streamlit churn scoring demo.

One command:  streamlit run app/app.py
Full chain:   raw transactions → clean → features → score → rank → target list
              with churn probability + 90% confidence interval per customer.

Tabs:
    1. Pipeline Summary   — run stats, pipeline trace
    2. Risk Dashboard     — distribution, revenue at risk, top customers
    3. Target List        — filterable table + CSV download
    4. Customer Lookup    — individual customer deep-dive
"""

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Churn Scoring Pipeline",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import pipeline (same directory) ─────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import run_pipeline, R_MAX_DAYS, FULL_FEATURES

# ── Colour palette (matches project notebooks) ───────────────────────────────
C_BLUE    = "#2a78d6"
C_ORANGE  = "#eb6834"
C_GREEN   = "#1baf7a"
C_NEUTRAL = "#8a8a85"
C_RED     = "#d62a2a"

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — controls
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🎯 Churn Scoring Pipeline")
st.sidebar.markdown("*Customer retention targeting — weekly batch scoring*")
st.sidebar.markdown("---")

# Data source
data_source = st.sidebar.radio(
    "Data source",
    ["Use bundled sample data", "Upload transaction file"],
    index=0,
)

uploaded_file = None
data_path     = None

DEFAULT_DATA = Path(__file__).parent.parent / "data" / "raw" / "online_retail_II.parquet"
FALLBACK_XLS = Path(__file__).parent.parent / "data" / "raw" / "online_retail_II.xlsx"

if data_source == "Upload transaction file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload file (.xlsx or .parquet)",
        type=["xlsx", "xls", "parquet"],
    )
    if uploaded_file:
        suffix = Path(uploaded_file.name).suffix
        tmp    = Path("/tmp") / f"upload{suffix}"
        tmp.write_bytes(uploaded_file.getvalue())
        data_path = str(tmp)
else:
    if DEFAULT_DATA.exists():
        data_path = str(DEFAULT_DATA)
    elif FALLBACK_XLS.exists():
        data_path = str(FALLBACK_XLS)

st.sidebar.markdown("---")

# Snapshot date
st.sidebar.subheader("⚙️ Parameters")
use_latest = st.sidebar.checkbox("Use latest date in data", value=True)
custom_date = None
if not use_latest:
    custom_date = st.sidebar.date_input("Snapshot date", value=None)

# K%
k_pct = st.sidebar.slider(
    "Target list size K%",
    min_value=10, max_value=50, value=30, step=5,
    help="Top K% of ranked customers to include in the retention campaign.",
)

# R_max
r_max = st.sidebar.slider(
    "Activity threshold R_max (days)",
    min_value=60, max_value=200, value=R_MAX_DAYS, step=1,
    help="Customers with recency > R_max are considered inactive and excluded.",
)

run_btn = st.sidebar.button("▶  Run Pipeline", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: Logistic Regression, Full feature set, C=0.01 · "
    "Trained on UCI Online Retail II · H=90 days"
)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None

# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if data_path is None:
        st.error("⚠️ No data file available. Upload a file or ensure sample data exists.")
    else:
        with st.spinner("Running pipeline…"):
            snap = None
            if not use_latest and custom_date:
                snap = pd.Timestamp(custom_date)
            try:
                result = run_pipeline(
                    path=data_path,
                    snapshot_date=snap,
                    k_frac=k_pct / 100,
                    r_max=r_max,
                    with_ci=True,
                )
                if result is None:
                    st.error("No active customers found at this snapshot date.")
                else:
                    st.session_state.result = result
                    st.success("Pipeline complete ✅")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
# Landing page (before first run)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.result is None:
    st.title("🎯 Customer Churn Scoring Pipeline")
    st.markdown("""
    This app runs the **full churn prediction pipeline** end-to-end:

    ```
    Raw transactions  →  Cleaning  →  Feature engineering
        →  Model scoring  →  Risk ranking  →  Retention target list
    ```

    **How to use:**
    1. Select a data source in the sidebar (bundled sample or upload your own).
    2. Adjust K% (how many customers to target) and other parameters.
    3. Click **▶ Run Pipeline**.
    4. Explore the results across the four tabs.

    ---
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "Logistic Regression")
    col2.metric("Prediction horizon", "90 days")
    col3.metric("Default target share", f"{k_pct}%")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Results tabs
# ─────────────────────────────────────────────────────────────────────────────
result      = st.session_state.result
stats       = result["stats"]
scored_df   = result["scored_df"]
target_df   = result["target_df"]
snap_date   = result["snapshot_date"]

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Pipeline Summary",
    "📊 Risk Dashboard",
    "🎯 Target List",
    "🔍 Customer Lookup",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Pipeline Summary
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header(f"Scoring Run — {snap_date.date()}")

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Raw transactions",    f"{stats['total_transactions']:,}")
    c2.metric("Valid transactions",  f"{stats['valid_transactions']:,}")
    c3.metric("Active customers",    f"{stats['active_customers']:,}")
    c4.metric(f"Target list (K={k_pct}%)",
              f"{stats['target_list_size']:,}")
    c5.metric("Revenue at risk (target)",
              f"£{stats['total_revenue_at_risk']:,.0f}")

    st.markdown("---")

    # Pipeline trace
    st.subheader("Pipeline trace")
    trace = [
        ("1. Data load",           f"{stats['total_transactions']:,} raw rows"),
        ("2. Cleaning",            f"{stats['valid_transactions']:,} valid purchases "
                                   f"({stats['total_transactions'] - stats['valid_transactions']:,} removed)"),
        ("3. Active customers",    f"{stats['active_customers']:,} customers with recency ≤ {r_max} days"),
        ("4. Feature engineering", f"{len(FULL_FEATURES)} features per customer, no leakage"),
        ("5. Model scoring",       "Logistic Regression · Full feature set · C=0.01"),
        ("6. Risk ranking",        "Sorted by churn probability (descending) · 90% CI via bootstrap"),
        (f"7. Target list (K={k_pct}%)",
                                   f"{stats['target_list_size']:,} customers selected for campaign"),
    ]
    for step, detail in trace:
        col_s, col_d = st.columns([1, 3])
        col_s.markdown(f"**{step}**")
        col_d.write(detail)

    st.markdown("---")

    # Risk band summary
    st.subheader("Risk band breakdown")
    bands = scored_df["risk_band"].value_counts().reset_index()
    bands.columns = ["risk_band", "count"]
    band_order = ["🔴 High", "🟡 Medium", "🟢 Low"]
    bands = bands.set_index("risk_band").reindex(band_order).dropna().reset_index()

    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.dataframe(bands, use_container_width=True, hide_index=True)
    with col_b:
        fig, ax = plt.subplots(figsize=(5, 2.5))
        colors  = [C_RED, C_ORANGE, C_GREEN]
        bars    = ax.barh(bands["risk_band"], bands["count"], color=colors)
        ax.bar_label(bars, fmt="%d", padding=4, fontsize=9)
        ax.set_xlabel("Customers")
        ax.set_title("Customers by risk band")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Risk Dashboard
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Risk Dashboard")

    col1, col2 = st.columns(2)

    # ── Chart 1: Churn probability distribution ───────────────────────────
    with col1:
        st.subheader("Churn probability distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(scored_df["churn_probability"], bins=30, color=C_BLUE,
                edgecolor="white", linewidth=0.4)
        cutoff = scored_df["churn_probability"].quantile(1 - k_pct / 100)
        ax.axvline(cutoff, color=C_ORANGE, linewidth=2,
                   linestyle="--", label=f"Top {k_pct}% cutoff ({cutoff:.2f})")
        ax.set_xlabel("Predicted churn probability")
        ax.set_ylabel("Customers")
        ax.set_title("All active customers")
        ax.legend(fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Chart 2: Revenue at risk by risk band ─────────────────────────────
    with col2:
        st.subheader("Revenue at risk by risk band")
        rev_by_band = (
            scored_df.groupby("risk_band")["revenue_at_risk"]
            .sum()
            .reindex(["🔴 High", "🟡 Medium", "🟢 Low"])
            .dropna()
        )
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = [C_RED, C_ORANGE, C_GREEN]
        bars   = ax.bar(rev_by_band.index, rev_by_band.values, color=colors)
        ax.bar_label(bars, fmt="£%.0f", padding=4, fontsize=8)
        ax.set_ylabel("Revenue at risk (£)")
        ax.set_title("Total estimated revenue at risk")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"£{x:,.0f}"
        ))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")

    # ── Chart 3: Top 20 customers — churn prob + CI ───────────────────────
    st.subheader("Top 20 highest-risk customers — churn probability with 90% CI")

    top20 = scored_df.head(20).copy()
    fig, ax = plt.subplots(figsize=(10, 4))

    y_pos = np.arange(len(top20))[::-1]   # highest risk at top

    # CI bars
    if "churn_prob_lo" in top20.columns:
        xerr_lo = top20["churn_probability"] - top20["churn_prob_lo"]
        xerr_hi = top20["churn_prob_hi"]     - top20["churn_probability"]
        ax.barh(y_pos, top20["churn_probability"], color=C_BLUE, alpha=0.85,
                xerr=[xerr_lo.clip(0), xerr_hi.clip(0)],
                error_kw={"ecolor": C_ORANGE, "capsize": 3, "linewidth": 1.2})
    else:
        ax.barh(y_pos, top20["churn_probability"], color=C_BLUE, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"ID {int(cid)}" for cid in top20["customer_id"]],
        fontsize=8,
    )
    ax.set_xlabel("Churn probability")
    ax.set_title("Orange bars = 90% bootstrap confidence interval")
    ax.axvline(0.5, color=C_NEUTRAL, linewidth=1, linestyle=":")
    ax.set_xlim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "Confidence intervals estimated via parametric bootstrap on "
        "Logistic Regression coefficients (200 samples, 90% CI). "
        "Wide bars indicate customers where the model has less certainty."
    )

    st.markdown("---")

    # ── Chart 4: Segment breakdown ────────────────────────────────────────
    st.subheader("Churn probability by customer segment")
    col3, col4 = st.columns(2)

    with col3:
        seg_stats = (
            scored_df.groupby("segment")["churn_probability"]
            .agg(["mean", "median", "count"])
            .rename(columns={"mean": "Avg prob", "median": "Median prob", "count": "Customers"})
            .round(3)
        )
        st.dataframe(seg_stats, use_container_width=True)

    with col4:
        fig, ax = plt.subplots(figsize=(4, 3))
        for seg, color in [("one-time", C_ORANGE), ("repeat", C_BLUE)]:
            data = scored_df[scored_df["segment"] == seg]["churn_probability"]
            if not data.empty:
                ax.hist(data, bins=20, alpha=0.6, color=color, label=seg, edgecolor="white")
        ax.set_xlabel("Churn probability")
        ax.set_ylabel("Customers")
        ax.legend(fontsize=8, frameon=False)
        ax.set_title("Distribution by segment")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Target List
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header(f"Target List — {len(target_df):,} customers (K={k_pct}%)")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        seg_filter = st.multiselect(
            "Segment",
            options=["one-time", "repeat"],
            default=["one-time", "repeat"],
        )
    with col_f2:
        band_opts = [b for b in ["🔴 High", "🟡 Medium", "🟢 Low"]
                     if b in target_df["risk_band"].values]
        band_filter = st.multiselect("Risk band", options=band_opts, default=band_opts)
    with col_f3:
        min_rev = st.slider(
            "Min revenue at risk (£)",
            min_value=0,
            max_value=int(target_df["revenue_at_risk"].max()),
            value=0,
            step=50,
        )

    filtered = target_df[
        target_df["segment"].isin(seg_filter) &
        target_df["risk_band"].isin(band_filter) &
        (target_df["revenue_at_risk"] >= min_rev)
    ]

    st.markdown(f"**{len(filtered):,}** customers after filters")

    # Display columns
    display_cols = ["customer_id", "churn_probability", "risk_band",
                    "rank", "segment", "revenue_at_risk", "country"]
    if "churn_prob_lo" in filtered.columns:
        display_cols += ["churn_prob_lo", "churn_prob_hi"]

    display_df = filtered[display_cols].copy()
    display_df["churn_probability"] = display_df["churn_probability"].round(3)
    display_df["revenue_at_risk"]   = display_df["revenue_at_risk"].round(2)
    if "churn_prob_lo" in display_df.columns:
        display_df["churn_prob_lo"] = display_df["churn_prob_lo"].round(3)
        display_df["churn_prob_hi"] = display_df["churn_prob_hi"].round(3)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "customer_id":       st.column_config.NumberColumn("Customer ID"),
            "churn_probability": st.column_config.ProgressColumn(
                "Churn Prob", min_value=0, max_value=1, format="%.3f"
            ),
            "risk_band":         st.column_config.TextColumn("Risk"),
            "rank":              st.column_config.NumberColumn("Rank"),
            "revenue_at_risk":   st.column_config.NumberColumn("Rev at Risk (£)", format="£%.2f"),
        }
    )

    # CSV download
    csv_buf = io.StringIO()
    display_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️  Download target list as CSV",
        data=csv_buf.getvalue(),
        file_name=f"target_list_{snap_date.date()}_K{k_pct}.csv",
        mime="text/csv",
        type="primary",
    )

    st.caption(
        "Upload this CSV to your campaign platform. "
        "Fields: customer_id, churn_probability, 90% CI bounds, risk_band, rank, segment, revenue_at_risk."
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Customer Lookup
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Individual Customer Lookup")
    st.markdown("Enter a Customer ID to see their churn risk profile.")

    cid_input = st.number_input(
        "Customer ID",
        min_value=int(scored_df["customer_id"].min()),
        max_value=int(scored_df["customer_id"].max()),
        value=int(scored_df["customer_id"].iloc[0]),
        step=1,
    )

    row = scored_df[scored_df["customer_id"] == cid_input]

    if row.empty:
        st.warning(f"Customer {cid_input} is not in this week's Scoring Population. "
                   "They may be inactive (recency > R_max) or not in the dataset.")
    else:
        r = row.iloc[0]

        # Header cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Churn probability",  f"{r['churn_probability']:.1%}")
        c2.metric("Risk band",          r["risk_band"])
        c3.metric("Rank",               f"#{int(r['rank'])} of {len(scored_df):,}")
        c4.metric("Revenue at risk",    f"£{r['revenue_at_risk']:.2f}")

        # CI display
        if "churn_prob_lo" in r.index:
            st.markdown(
                f"**90% Confidence interval:** "
                f"{r['churn_prob_lo']:.1%} – {r['churn_prob_hi']:.1%}"
            )

        in_target = int(r["rank"]) <= len(target_df)
        if in_target:
            st.success(f"✅ This customer **IS** in this week's target list (top {k_pct}%).")
        else:
            st.info(f"ℹ️ This customer is **NOT** in the target list at K={k_pct}%. "
                    f"They are ranked #{int(r['rank'])}.")

        st.markdown("---")

        # Key features
        st.subheader("Key features driving the prediction")
        feature_labels = {
            "recency_days":             "Days since last purchase",
            "frequency_so_far":         "Number of purchase occasions",
            "monetary_so_far":          "Total spend to date (£)",
            "avg_order_value_so_far":   "Average order value (£)",
            "tenure_days":              "Customer tenure (days)",
            "mean_gap_so_far":          "Mean days between purchases",
            "recency_to_typical_gap":   "Recency ÷ typical gap (>1 = overdue)",
            "activity_last_90d":        "Purchase occasions in last 90 days",
            "distinct_products_so_far": "Distinct products bought",
            "cancel_count_so_far":      "Cancellation occasions",
            "cancel_value_share_so_far":"Cancel value share (%)",
        }
        feat_data = []
        for col, label in feature_labels.items():
            if col in r.index:
                val = r[col]
                if "share" in col or "ratio" in col:
                    formatted = f"{val:.1%}"
                elif isinstance(val, float):
                    formatted = f"{val:,.1f}"
                else:
                    formatted = str(val)
                feat_data.append({"Feature": label, "Value": formatted})

        feat_df = pd.DataFrame(feat_data)
        col_feat, col_bar = st.columns([2, 3])
        with col_feat:
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

        with col_bar:
            # Mini probability gauge
            fig, ax = plt.subplots(figsize=(4, 1.5))
            p = r["churn_probability"]
            ax.barh([0], [p], color=C_RED if p > 0.6 else (C_ORANGE if p > 0.35 else C_GREEN),
                    height=0.5)
            ax.barh([0], [1 - p], left=[p], color="#e5e5e0", height=0.5)
            if "churn_prob_lo" in r.index:
                ax.barh([0], [r["churn_prob_hi"] - r["churn_prob_lo"]],
                        left=[r["churn_prob_lo"]],
                        color=C_ORANGE, height=0.15, alpha=0.6,
                        label="90% CI")
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("Churn probability")
            ax.axvline(0.5, color=C_NEUTRAL, linewidth=1, linestyle=":")
            ax.set_title(f"P(churn) = {p:.1%}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("---")
        st.subheader("Customer info")
        info_cols = ["customer_id", "segment", "country", "snapshot_date"]
        st.json({
            col: str(r[col]) if col in r.index else "n/a"
            for col in info_cols
        })
