"""
app.py — Customer Churn Prediction & Monitoring Dashboard

Run:  streamlit run app/app.py

Tabs:
    1. Overview        — run summary, pipeline trace, risk band breakdown
    2. Churn Prediction — score distribution, top-risk chart, segment view
    3. Target List     — filterable table, CSV export for campaign platform
    4. Customer Profile — individual lookup, gauge, feature explain, recommendation
    5. Monitoring      — score stability, feature drift (PSI), risk band trend
"""

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import run_pipeline, R_MAX_DAYS, FULL_FEATURES

# ── Palette ───────────────────────────────────────────────────────────────────
C_BLUE    = "#2a78d6"
C_ORANGE  = "#eb6834"
C_GREEN   = "#1baf7a"
C_RED     = "#d62a2a"
C_NEUTRAL = "#8a8a85"
C_BG      = "#f7f7f5"

BAND_COLORS = {"🔴 High": C_RED, "🟡 Medium": C_ORANGE, "🟢 Low": C_GREEN}

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_pct(v):  return f"{v:.1%}"
def fmt_gbp(v):  return f"£{v:,.0f}"
def fmt_num(v):  return f"{v:,.0f}"

def risk_color(p):
    if p >= 0.6: return C_RED
    if p >= 0.35: return C_ORANGE
    return C_GREEN

def spark_gauge(p, lo=None, hi=None, width=4, height=1.2):
    """Horizontal probability gauge bar."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.barh(0, p,     color=risk_color(p), height=0.45)
    ax.barh(0, 1 - p, left=p, color="#e8e8e4", height=0.45)
    if lo is not None and hi is not None:
        ax.barh(0, hi - lo, left=lo, color=C_ORANGE, height=0.15, alpha=0.55)
    ax.set_xlim(0, 1); ax.set_yticks([])
    ax.axvline(0.5, color=C_NEUTRAL, lw=1, ls=":")
    ax.set_xlabel("Churn probability", fontsize=8)
    ax.set_title(f"P(churn) = {fmt_pct(p)}", fontsize=9, fontweight="bold")
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig

def psi(expected, actual, bins=10):
    """Population Stability Index between two 1-D arrays."""
    mn, mx = min(expected.min(), actual.min()), max(expected.max(), actual.max())
    if mx == mn: return 0.0
    edges = np.linspace(mn, mx, bins + 1)
    e_cnt, _ = np.histogram(expected, bins=edges)
    a_cnt, _ = np.histogram(actual,   bins=edges)
    e_pct = np.where(e_cnt == 0, 1e-6, e_cnt / len(expected))
    a_pct = np.where(a_cnt == 0, 1e-6, a_cnt / len(actual))
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

def psi_status(v):
    if v < 0.10: return "✅ Stable"
    if v < 0.20: return "🟡 Minor drift"
    return "🔴 Major drift"

# ── Feature labels (plain language) ──────────────────────────────────────────
FEATURE_LABELS = {
    "recency_days":              "Days since last purchase",
    "frequency_so_far":          "Total purchase occasions",
    "monetary_so_far":           "Total spend (£)",
    "avg_order_value_so_far":    "Avg order value (£)",
    "tenure_days":               "Customer tenure (days)",
    "mean_gap_so_far":           "Avg days between orders",
    "recency_to_typical_gap":    "Overdue ratio (recency ÷ avg gap)",
    "activity_last_90d":         "Orders in last 90 days",
    "distinct_products_so_far":  "Distinct products bought",
    "cancel_count_so_far":       "Cancellation count",
    "cancel_value_share_so_far": "Cancellation value share",
}

# ── Data paths ────────────────────────────────────────────────────────────────
DEFAULT_DATA = Path(__file__).parent.parent / "data" / "raw" / "online_retail_II.parquet"
FALLBACK_XLS = Path(__file__).parent.parent / "data" / "raw" / "online_retail_II.xlsx"

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Churn Intelligence")
    st.caption("Weekly customer retention scoring")
    st.markdown("---")

    # ── Data source ───────────────────────────────────────────────────────
    st.markdown("**Data source**")
    data_source = st.radio(
        "data_source", ["Sample data", "Upload file"],
        label_visibility="collapsed", index=0,
    )

    data_path = None
    if data_source == "Upload file":
        uploaded = st.file_uploader("Upload .xlsx or .parquet", type=["xlsx", "xls", "parquet"])
        if uploaded:
            suffix = Path(uploaded.name).suffix
            tmp = Path(__file__).parent / f"_upload{suffix}"
            tmp.write_bytes(uploaded.getvalue())
            data_path = str(tmp)
    else:
        if DEFAULT_DATA.exists():   data_path = str(DEFAULT_DATA)
        elif FALLBACK_XLS.exists(): data_path = str(FALLBACK_XLS)

    st.markdown("---")

    # ── Scoring parameters ────────────────────────────────────────────────
    st.markdown("**Scoring parameters**")

    use_latest = st.checkbox("Use latest date in data", value=True)
    custom_date = None
    if not use_latest:
        custom_date = st.date_input("Snapshot date")

    k_pct = st.slider(
        "Campaign target size (K%)",
        min_value=10, max_value=50, value=30, step=5,
        help="Top K% of ranked customers to contact this week."
    )
    r_max = st.slider(
        "Max inactivity window (days)",
        min_value=60, max_value=200, value=R_MAX_DAYS, step=1,
        help="Customers not seen within this window are considered inactive."
    )

    st.markdown("---")
    run_btn = st.button("▶  Run Scoring", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<small>Model: Logistic Regression · C=0.01<br>"
        "Horizon: 90 days · CI: 90% bootstrap</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "result"      not in st.session_state: st.session_state.result      = None
if "prev_scored" not in st.session_state: st.session_state.prev_scored = None  # for monitoring

# ─────────────────────────────────────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if data_path is None:
        st.error("⚠️ No data file found. Upload a file or ensure sample data exists in data/raw/.")
    else:
        with st.spinner("Scoring customers…"):
            snap = pd.Timestamp(custom_date) if (not use_latest and custom_date) else None
            try:
                # Keep previous run for monitoring comparison
                if st.session_state.result is not None:
                    st.session_state.prev_scored = (
                        st.session_state.result["scored_df"].copy()
                    )
                result = run_pipeline(
                    path=data_path, snapshot_date=snap,
                    k_frac=k_pct / 100, r_max=r_max, with_ci=True,
                )
                if result is None:
                    st.error("No active customers found for this date.")
                else:
                    st.session_state.result = result
                    st.success("✅ Scoring complete")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.result is None:
    st.title("📡 Customer Churn Intelligence")
    st.markdown(
        "Predict which customers are likely to leave in the next **90 days** "
        "and build a targeted retention campaign list — in one click."
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.info("**Step 1**\nSelect data source in the sidebar")
    c2.info("**Step 2**\nSet campaign size K% and inactivity window")
    c3.info("**Step 3**\nClick **▶ Run Scoring**")
    c4.info("**Step 4**\nExplore tabs: Prediction · Target List · Monitoring")

    st.markdown("---")
    st.markdown(
        "**What this app does:**  \n"
        "Raw transactions → cleaning → feature engineering → "
        "model scoring → risk ranking → retention target list with 90% confidence intervals."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# UNPACK RESULTS
# ─────────────────────────────────────────────────────────────────────────────
result    = st.session_state.result
stats     = result["stats"]
scored    = result["scored_df"]
target    = result["target_df"]
snap_date = result["snapshot_date"]
prev      = st.session_state.prev_scored   # may be None

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview",
    "📊 Churn Prediction",
    "🎯 Target List",
    "🔍 Customer Profile",
    "📡 Monitoring",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"### Scoring run — {snap_date.date()}")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions loaded",  fmt_num(stats["total_transactions"]))
    c2.metric("Valid transactions",   fmt_num(stats["valid_transactions"]))
    c3.metric("Active customers",     fmt_num(stats["active_customers"]))
    c4.metric(f"Campaign list (K={k_pct}%)", fmt_num(stats["target_list_size"]))
    c5.metric("Revenue at risk",      fmt_gbp(stats["total_revenue_at_risk"]))

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("##### Pipeline trace")
        trace = [
            ("Load",             f"{fmt_num(stats['total_transactions'])} raw rows ingested"),
            ("Clean",            f"{fmt_num(stats['valid_transactions'])} valid · "
                                 f"{fmt_num(stats['total_transactions'] - stats['valid_transactions'])} removed"),
            ("Active filter",    f"{fmt_num(stats['active_customers'])} customers "
                                 f"with recency ≤ {r_max} days"),
            ("Feature engineer", f"{len(FULL_FEATURES)} features · no future leakage"),
            ("Score",            "Logistic Regression · C=0.01 · Full feature set"),
            ("Rank + CI",        "Sorted by churn probability · 90% bootstrap CI"),
            (f"Campaign list",   f"{fmt_num(stats['target_list_size'])} customers selected (K={k_pct}%)"),
        ]
        for step, detail in trace:
            a, b = st.columns([1, 3])
            a.markdown(f"**{step}**")
            b.write(detail)

    with col_r:
        st.markdown("##### Risk band breakdown")
        band_order = ["🔴 High", "🟡 Medium", "🟢 Low"]
        bands = (
            scored["risk_band"].value_counts()
            .reindex(band_order).dropna().reset_index()
        )
        bands.columns = ["Band", "Customers"]
        bands["Churn rate"] = bands["Band"].map(
            scored.groupby("risk_band")["churn_probability"]
            .mean().apply(lambda v: fmt_pct(float(v)))
        )

        fig, ax = plt.subplots(figsize=(4.5, 2.2))
        colors = [BAND_COLORS[b] for b in bands["Band"]]
        bars = ax.barh(bands["Band"], bands["Customers"], color=colors)
        ax.bar_label(bars, fmt="%d", padding=4, fontsize=9)
        ax.set_xlabel("Customers")
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.dataframe(
            bands[["Band", "Customers", "Churn rate"]],
            use_container_width=True, hide_index=True,
        )

    # Avg churn probability gauge
    st.markdown("---")
    avg_p = float(scored["churn_probability"].mean())
    st.markdown(f"##### Population churn risk — avg score: **{fmt_pct(avg_p)}**")
    fig = spark_gauge(avg_p, width=8, height=0.9)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHURN PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Churn Prediction Results")

    col1, col2 = st.columns(2)

    # Score distribution
    with col1:
        st.markdown("##### Score distribution")
        cutoff = scored["churn_probability"].quantile(1 - k_pct / 100)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.hist(scored["churn_probability"], bins=30,
                color=C_BLUE, edgecolor="white", linewidth=0.4)
        ax.axvline(cutoff, color=C_ORANGE, lw=2, ls="--",
                   label=f"Top {k_pct}% cutoff ({cutoff:.2f})")
        ax.set_xlabel("Predicted churn probability")
        ax.set_ylabel("Customers")
        ax.legend(fontsize=8, frameon=False)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            f"Mean: {fmt_pct(scored['churn_probability'].mean())} · "
            f"Median: {fmt_pct(scored['churn_probability'].median())} · "
            f"High risk (≥60%): {fmt_num((scored['churn_probability'] >= 0.6).sum())} customers"
        )

    # Revenue at risk by band
    with col2:
        st.markdown("##### Revenue at risk by risk band")
        rev = (
            scored.groupby("risk_band")["revenue_at_risk"]
            .sum().reindex(["🔴 High", "🟡 Medium", "🟢 Low"]).dropna()
        )
        fig, ax = plt.subplots(figsize=(5, 3.2))
        colors = [BAND_COLORS[b] for b in rev.index]
        bars = ax.bar(rev.index, rev.values, color=colors)
        ax.bar_label(bars, labels=[fmt_gbp(v) for v in rev.values],
                     padding=4, fontsize=8)
        ax.set_ylabel("Revenue at risk (£)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_gbp(x)))
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        total_rar = scored["revenue_at_risk"].sum()
        target_rar = target["revenue_at_risk"].sum()
        st.caption(
            f"Total revenue at risk: {fmt_gbp(total_rar)} · "
            f"In campaign list: {fmt_gbp(target_rar)} ({target_rar/total_rar:.0%})"
        )

    st.markdown("---")

    # Top 20 customers
    st.markdown(f"##### Top 20 highest-risk customers")
    top_n = scored.head(20).copy()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    y = np.arange(len(top_n))[::-1]

    if "churn_prob_lo" in top_n.columns:
        lo_err = (top_n["churn_probability"] - top_n["churn_prob_lo"]).clip(0)
        hi_err = (top_n["churn_prob_hi"] - top_n["churn_probability"]).clip(0)
        ax.barh(y, top_n["churn_probability"], color=C_BLUE, alpha=0.85,
                xerr=[lo_err, hi_err],
                error_kw={"ecolor": C_ORANGE, "capsize": 3, "lw": 1.2})
    else:
        ax.barh(y, top_n["churn_probability"], color=C_BLUE, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels([f"ID {int(c)}" for c in top_n["customer_id"]], fontsize=8)
    ax.set_xlabel("Churn probability")
    ax.set_title("Orange whiskers = 90% confidence interval", fontsize=8)
    ax.axvline(0.5, color=C_NEUTRAL, lw=1, ls=":")
    ax.set_xlim(0, 1)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # Segment comparison
    st.markdown("##### Churn risk by customer segment")
    c3, c4 = st.columns(2)

    with c3:
        seg = (
            scored.groupby("segment")["churn_probability"]
            .agg(
                Customers="count",
                Mean=lambda x: fmt_pct(float(x.mean())),
                Median=lambda x: fmt_pct(float(x.median())),
            )
            .reset_index().rename(columns={"segment": "Segment"})
        )
        st.dataframe(seg, use_container_width=True, hide_index=True)
        st.caption(
            "One-time buyers churn more — they have less purchase history "
            "and fewer reasons to return."
        )

    with c4:
        fig, ax = plt.subplots(figsize=(4.5, 3))
        for seg_name, color in [("one-time", C_ORANGE), ("repeat", C_BLUE)]:
            d = scored[scored["segment"] == seg_name]["churn_probability"]
            if not d.empty:
                ax.hist(d, bins=20, alpha=0.6, color=color,
                        label=f"{seg_name} ({len(d):,})", edgecolor="white")
        ax.set_xlabel("Churn probability")
        ax.set_ylabel("Customers")
        ax.legend(fontsize=8, frameon=False)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — TARGET LIST
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"### Campaign Target List — {len(target):,} customers (K={k_pct}%)")
    st.caption("These customers are most at risk of churning in the next 90 days.")

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        seg_sel = st.multiselect(
            "Customer type",
            ["one-time", "repeat"], default=["one-time", "repeat"],
        )
    with f2:
        band_opts = [b for b in ["🔴 High", "🟡 Medium", "🟢 Low"]
                     if b in target["risk_band"].values]
        band_sel = st.multiselect("Risk level", band_opts, default=band_opts)
    with f3:
        max_rev = int(target["revenue_at_risk"].max())
        min_rev = st.slider("Min revenue at risk (£)", 0, max(max_rev, 1), 0, step=50)

    filtered = target[
        target["segment"].isin(seg_sel) &
        target["risk_band"].isin(band_sel) &
        (target["revenue_at_risk"] >= min_rev)
    ].copy()

    st.markdown(f"**{len(filtered):,}** customers match your filters")

    # Build display table
    disp_cols = ["customer_id", "churn_probability", "risk_band", "rank",
                 "segment", "revenue_at_risk", "country"]
    if "churn_prob_lo" in filtered.columns:
        disp_cols += ["churn_prob_lo", "churn_prob_hi"]

    disp = filtered[disp_cols].copy()
    disp["churn_probability"] = disp["churn_probability"].round(3)
    disp["revenue_at_risk"]   = disp["revenue_at_risk"].round(2)
    if "churn_prob_lo" in disp.columns:
        disp["churn_prob_lo"] = disp["churn_prob_lo"].round(3)
        disp["churn_prob_hi"] = disp["churn_prob_hi"].round(3)

    st.dataframe(
        disp, use_container_width=True, hide_index=True, height=400,
        column_config={
            "customer_id":       st.column_config.NumberColumn("Customer ID"),
            "churn_probability": st.column_config.ProgressColumn(
                "Churn Risk", min_value=0, max_value=1, format="%.3f"),
            "risk_band":         st.column_config.TextColumn("Risk Level"),
            "rank":              st.column_config.NumberColumn("Rank"),
            "revenue_at_risk":   st.column_config.NumberColumn("Revenue at Risk (£)", format="£%.2f"),
            "country":           st.column_config.TextColumn("Country"),
        }
    )

    # Summary metrics below table
    m1, m2, m3 = st.columns(3)
    m1.metric("Customers in list",      fmt_num(len(filtered)))
    m2.metric("Total revenue at risk",  fmt_gbp(filtered["revenue_at_risk"].sum()))
    m3.metric("Avg churn probability",  fmt_pct(filtered["churn_probability"].mean()))

    # CSV download
    buf = io.StringIO()
    disp.to_csv(buf, index=False)
    st.download_button(
        "⬇️  Download as CSV (for campaign platform)",
        data=buf.getvalue(),
        file_name=f"campaign_list_{snap_date.date()}_K{k_pct}.csv",
        mime="text/csv",
        type="primary",
    )
    st.caption("Upload this file to your email/SMS campaign platform.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOMER PROFILE
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Customer Profile")
    st.caption("Look up any customer to see their churn risk, key signals, and recommended action.")

    cid = st.number_input(
        "Customer ID",
        min_value=int(scored["customer_id"].min()),
        max_value=int(scored["customer_id"].max()),
        value=int(scored["customer_id"].iloc[0]),
        step=1,
    )

    row = scored[scored["customer_id"] == cid]

    if row.empty:
        st.warning(
            f"Customer **{cid}** is not in this week's scoring population. "
            "They may be inactive (recency > R_max) or not in the dataset."
        )
        st.stop()

    r = row.iloc[0]
    p = float(r["churn_probability"])
    in_target = int(r["rank"]) <= len(target)

    # Header
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("Churn probability",  fmt_pct(p))
    hc2.metric("Risk level",         r["risk_band"])
    hc3.metric("Rank",               f"#{int(r['rank'])} of {len(scored):,}")
    hc4.metric("Revenue at risk",    fmt_gbp(r["revenue_at_risk"]))

    if "churn_prob_lo" in r.index:
        st.caption(
            f"90% confidence interval: {fmt_pct(r['churn_prob_lo'])} – {fmt_pct(r['churn_prob_hi'])}"
        )

    # Action recommendation
    if in_target:
        if p >= 0.6:
            st.error(
                f"🔴 **High priority — include in this week's campaign.**  \n"
                f"Probability of churning: {fmt_pct(p)}. "
                f"Estimated revenue at stake: {fmt_gbp(r['revenue_at_risk'])}."
            )
        else:
            st.warning(
                f"🟡 **In campaign list — medium risk.**  \n"
                f"Probability of churning: {fmt_pct(p)}. "
                f"Consider a standard re-engagement offer."
            )
    else:
        st.info(
            f"🟢 **Not in this week's campaign list** (ranked #{int(r['rank'])}, K={k_pct}%).  \n"
            "Risk is below the current campaign threshold. Monitor next week."
        )

    st.markdown("---")

    left, right = st.columns([2, 3])

    with left:
        st.markdown("##### Customer signals")
        rows_feat = []
        for col, label in FEATURE_LABELS.items():
            if col not in r.index: continue
            val = r[col]
            if "share" in col:  fmt = fmt_pct(val)
            elif isinstance(val, float): fmt = f"{val:,.1f}"
            else: fmt = str(val)
            rows_feat.append({"Signal": label, "Value": fmt})
        st.dataframe(pd.DataFrame(rows_feat), use_container_width=True, hide_index=True)

        st.markdown("##### Customer info")
        st.json({
            "Customer ID": int(r["customer_id"]),
            "Segment":     r["segment"],
            "Country":     r.get("country", "Unknown"),
            "Snapshot":    str(r["snapshot_date"].date()),
        })

    with right:
        st.markdown("##### Churn probability gauge")
        lo = r.get("churn_prob_lo") if "churn_prob_lo" in r.index else None
        hi = r.get("churn_prob_hi") if "churn_prob_hi" in r.index else None
        fig = spark_gauge(p, lo, hi, width=5.5, height=1.4)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("##### Where this customer sits in the population")
        fig2, ax2 = plt.subplots(figsize=(5.5, 2.5))
        ax2.hist(scored["churn_probability"], bins=30, color="#d0dff5",
                 edgecolor="white", lw=0.4, label="All customers")
        ax2.axvline(p, color=risk_color(p), lw=2.5, label=f"This customer ({fmt_pct(p)})")
        ax2.set_xlabel("Churn probability")
        ax2.set_ylabel("Customers")
        ax2.legend(fontsize=8, frameon=False)
        for s in ["top", "right"]: ax2.spines[s].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — MONITORING
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Model & Data Monitoring")
    st.caption(
        "Tracks whether the model and data remain healthy week over week. "
        "Run the pipeline again to compare this week against the previous run."
    )

    if prev is None:
        st.info(
            "ℹ️ Run the pipeline a **second time** (or with different parameters) "
            "to enable week-over-week comparison charts."
        )

    # ── Section 1: Score distribution ─────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 1 · Risk Score Distribution")

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Mean churn probability",
               fmt_pct(scored["churn_probability"].mean()))
    mc2.metric("High-risk customers (≥60%)",
               fmt_num((scored["churn_probability"] >= 0.6).sum()),
               help="Customers with churn probability ≥ 0.60")
    mc3.metric("Customers scored",
               fmt_num(len(scored)))

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.hist(scored["churn_probability"], bins=35, color=C_BLUE,
            edgecolor="white", lw=0.4, alpha=0.85, label="Current run")
    if prev is not None:
        ax.hist(prev["churn_probability"], bins=35, color=C_NEUTRAL,
                edgecolor="white", lw=0.4, alpha=0.45, label="Previous run")
    ax.axvline(0.35, color=C_ORANGE, lw=1.2, ls=":", label="Medium threshold (0.35)")
    ax.axvline(0.60, color=C_RED,    lw=1.2, ls=":", label="High threshold (0.60)")
    ax.set_xlabel("Churn probability")
    ax.set_ylabel("Customers")
    ax.legend(fontsize=8, frameon=False)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if prev is not None:
        ks_stat, ks_p = ks_2samp(
            scored["churn_probability"], prev["churn_probability"]
        )
        ks_status = "✅ No material shift" if ks_stat < 0.05 else \
                    "🟡 Minor shift" if ks_stat < 0.10 else "🔴 Distribution shifted"
        st.caption(f"KS statistic vs previous run: **{ks_stat:.3f}** — {ks_status}")

    # ── Section 2: Risk band trend ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 2 · Risk Band Breakdown")

    band_order = ["🔴 High", "🟡 Medium", "🟢 Low"]
    curr_bands = scored["risk_band"].value_counts().reindex(band_order).fillna(0)

    if prev is not None:
        prev_bands = prev["risk_band"].value_counts().reindex(band_order).fillna(0)
        band_df = pd.DataFrame({
            "Band":     band_order,
            "Current":  curr_bands.values.astype(int),
            "Previous": prev_bands.values.astype(int),
            "Change":   (curr_bands.values - prev_bands.values).astype(int),
        })
        st.dataframe(band_df, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(7, 2.5))
        x = np.arange(len(band_order))
        w = 0.35
        bars_c = ax.bar(x - w/2, band_df["Current"],  w, color=[BAND_COLORS[b] for b in band_order], alpha=0.9, label="Current")
        bars_p = ax.bar(x + w/2, band_df["Previous"], w, color=[BAND_COLORS[b] for b in band_order], alpha=0.40, label="Previous")
        ax.set_xticks(x); ax.set_xticklabels(band_order)
        ax.set_ylabel("Customers")
        ax.legend(fontsize=8, frameon=False)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        band_df = pd.DataFrame({
            "Band":      band_order,
            "Customers": curr_bands.values.astype(int),
            "Avg churn": [
                fmt_pct(scored[scored["risk_band"] == b]["churn_probability"].mean())
                if b in scored["risk_band"].values else "—"
                for b in band_order
            ],
        })
        st.dataframe(band_df, use_container_width=True, hide_index=True)

    # ── Section 3: Feature drift (PSI) ────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 3 · Feature Drift (PSI)")
    st.caption(
        "Population Stability Index measures how much each feature's "
        "distribution has shifted. PSI < 0.10 = stable · 0.10–0.20 = minor drift · ≥ 0.20 = investigate."
    )

    numeric_features = [f for f in FULL_FEATURES if f in scored.columns]

    if prev is not None:
        psi_rows = []
        for feat in numeric_features:
            curr_vals = scored[feat].dropna()
            prev_vals = prev[feat].dropna()
            if len(curr_vals) < 10 or len(prev_vals) < 10:
                continue
            psi_val = psi(prev_vals.values, curr_vals.values)
            psi_rows.append({
                "Feature":      FEATURE_LABELS.get(feat, feat),
                "PSI":          round(psi_val, 4),
                "Status":       psi_status(psi_val),
                "Current mean": f"{curr_vals.mean():.2f}",
                "Prev mean":    f"{prev_vals.mean():.2f}",
            })
        psi_df = pd.DataFrame(psi_rows).sort_values("PSI", ascending=False)
        st.dataframe(psi_df, use_container_width=True, hide_index=True)

        # Highlight drifted features
        drifted = psi_df[psi_df["PSI"] >= 0.10]
        if len(drifted) > 0:
            st.warning(
                f"⚠️ {len(drifted)} feature(s) showing drift: "
                + ", ".join(drifted["Feature"].tolist())
            )
        else:
            st.success("✅ All features stable vs previous run.")
    else:
        # Show current feature summary without comparison
        feat_summary = []
        for feat in numeric_features:
            vals = scored[feat].dropna()
            feat_summary.append({
                "Feature": FEATURE_LABELS.get(feat, feat),
                "Mean":    f"{vals.mean():.2f}",
                "Median":  f"{vals.median():.2f}",
                "Std":     f"{vals.std():.2f}",
                "Null %":  fmt_pct(scored[feat].isna().mean()),
            })
        st.dataframe(pd.DataFrame(feat_summary), use_container_width=True, hide_index=True)
        st.caption("Run the pipeline a second time to see PSI drift comparison.")

    # ── Section 4: Data quality ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 4 · Data Quality")

    raw_df   = result["raw_df"]
    clean_df = result["clean_df"]

    removed_pct = (stats["total_transactions"] - stats["valid_transactions"]) / max(stats["total_transactions"], 1)
    null_cid_pct = raw_df["Customer ID"].isna().mean()

    dq_rows = [
        {"Check": "Rows removed by cleaning",
         "Value": fmt_pct(removed_pct),
         "Status": "✅ OK" if removed_pct < 0.30 else "🟡 High"},
        {"Check": "Missing Customer ID rate",
         "Value": fmt_pct(null_cid_pct),
         "Status": "✅ OK" if null_cid_pct < 0.05 else "🔴 Alert"},
        {"Check": "Active customers found",
         "Value": fmt_num(stats["active_customers"]),
         "Status": "✅ OK" if stats["active_customers"] > 0 else "🔴 Alert"},
    ]
    # Feature null rates
    for feat in ["recency_days", "frequency_so_far", "monetary_so_far"]:
        if feat in scored.columns:
            null_r = scored[feat].isna().mean()
            dq_rows.append({
                "Check":  f"Null rate — {FEATURE_LABELS.get(feat, feat)}",
                "Value":  fmt_pct(null_r),
                "Status": "✅ OK" if null_r < 0.10 else "🟡 High",
            })

    st.dataframe(pd.DataFrame(dq_rows), use_container_width=True, hide_index=True)
