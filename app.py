"""
Customer Churn Prediction – Analytics Dashboard
================================================
4 tabs:
  1. Classification Performance
  2. Risk Band Analysis & Revenue at Risk
  3. Business Impact Scenarios
  4. Scoring Output (deployment period: 2011-09-12 → 2011-12-05)

Run:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Brand palette (matches notebook) ──────────────────────────────────────────
C_BLUE    = "#2a78d6"
C_ORANGE  = "#eb6834"
C_GREEN   = "#1baf7a"
C_YELLOW  = "#eda100"
C_GRAY    = "#8a8a85"
C_LIGHT   = "#e8f0fb"
BAND_SEQ  = px.colors.sequential.Blues_r  # risk-band heat

CANDIDATE_COLORS = {
    "Final Model":   C_BLUE,
    "Recency Rule":  C_ORANGE,
    "RFM Rule":      C_GREEN,
    "Random":        C_GRAY,
}

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- global ---- */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { color: #1a1a2e; letter-spacing: -0.5px; }
    h2, h3 { color: #2a2a4a; }

    /* ---- metric tiles ---- */
    div[data-testid="metric-container"] {
        background: #f7f9ff;
        border: 1px solid #d9e4f5;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    div[data-testid="metric-container"] label { font-size: 0.78rem; color: #6b7280; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.6rem; font-weight: 700; color: #1a1a2e;
    }

    /* ---- info boxes ---- */
    .info-box {
        background: #f0f6ff;
        border-left: 4px solid #2a78d6;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #374151;
        margin-bottom: 8px;
    }

    /* ---- tab styling ---- */
    button[data-baseweb="tab"] { font-size: 0.93rem; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

RFM_FEATURES = ["recency_days", "frequency_so_far", "monetary_so_far"]
K_FRACS      = [0.10, 0.20, 0.30, 0.40, 0.50]
EFF_LEVELS   = [0.30, 0.40, 0.50]


@st.cache_data(show_spinner="Loading data…")
def load_raw():
    scores   = pd.read_parquet("model/churn_scores_h90.parquet")
    modeling = pd.read_parquet("model/modeling_dataset_h90.parquet")
    return scores, modeling


@st.cache_data(show_spinner="Preparing evaluation data…")
def build_test_df(_scores, _modeling):
    """Merge test scores with features; add baseline scores."""
    test_s = _scores[_scores["split"] == "test"].copy()
    test_m = _modeling[_modeling["split"] == "test"].copy()
    train  = _modeling[_modeling["split"] == "train"]

    # baseline: RFM rule (using train stats for normalisation)
    rfm_mean = train[RFM_FEATURES].mean()
    rfm_std  = train[RFM_FEATURES].std()
    z = (test_m[RFM_FEATURES] - rfm_mean) / rfm_std
    test_m["score_rfm_rule"] = (
        z["recency_days"] - z["frequency_so_far"] - z["monetary_so_far"]
    )
    # baseline: recency rule
    test_m["score_recency_rule"] = test_m["recency_days"]
    # baseline: random
    rng = np.random.default_rng(0)
    test_m["score_random"] = rng.uniform(size=len(test_m))

    keep = [
        "customer_id", "snapshot_date",
        "monetary_so_far", "recency_days",
        "score_rfm_rule", "score_recency_rule", "score_random",
    ]
    merged = test_s.merge(test_m[keep], on=["customer_id", "snapshot_date"], how="left")
    merged.rename(columns={"churn_probability": "score_final_model"}, inplace=True)
    return merged


@st.cache_data(show_spinner="Preparing deployment data…")
def build_deploy_df(_scores, _modeling):
    """Merge deployment scores with features (no churn labels)."""
    deploy_s = _scores[_scores["split"] == "deployment"].copy()
    deploy_m = _modeling[_modeling["snapshot_date"] >= pd.Timestamp("2011-09-12")].copy()

    keep = ["customer_id", "snapshot_date",
            "monetary_so_far", "recency_days", "frequency_so_far",
            "avg_order_value_so_far", "tenure_days"]
    merged = deploy_s.merge(deploy_m[keep], on=["customer_id", "snapshot_date"], how="left")
    merged.rename(columns={"churn_probability": "churn_probability"}, inplace=True)
    return merged


# ── Helper: weekly business counts ────────────────────────────────────────────

def weekly_business_counts(df, score_col, revenue_col, k_frac):
    rows = {}
    for week, wdf in df.groupby("snapshot_date"):
        n   = max(int(round(len(wdf) * k_frac)), 1)
        top = wdf.nlargest(n, score_col)
        is_churn = top["churn"] == 1
        rows[week] = {
            "n":               n,
            "tp":              int(is_churn.sum()),
            "fp":              int((~is_churn).sum()),
            "revenue_at_risk": top.loc[is_churn, revenue_col].sum(),
            "positives":       int(wdf["churn"].sum()),
        }
    return pd.DataFrame(rows).T.astype(float)


def business_impact(counts_sum, e, m, c_contact, c_inc, r):
    rev_preserved   = counts_sum["revenue_at_risk"] * e
    gp_preserved    = rev_preserved * m
    cost_contact    = counts_sum["n"] * c_contact
    cost_incentive  = (counts_sum["tp"] * e + counts_sum["fp"] * r) * c_inc
    net             = gp_preserved - cost_contact - cost_incentive
    return {
        "Customers Targeted":    int(counts_sum["n"]),
        "Churners Caught (TP)":  int(counts_sum["tp"]),
        "Non-Churners (FP)":     int(counts_sum["fp"]),
        "Recall @ K":            counts_sum["tp"] / counts_sum["positives"],
        "Revenue at Risk (£)":   counts_sum["revenue_at_risk"],
        "Revenue Preserved (£)": rev_preserved,
        "Gross Profit (£)":      gp_preserved,
        "Contact Cost (£)":      cost_contact,
        "Incentive Cost (£)":    cost_incentive,
        "Net Impact (£)":        net,
    }


# ── Plotly theme helper ────────────────────────────────────────────────────────

def clean_layout(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1a1a2e")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#374151", size=12),
        height=height,
        margin=dict(l=40, r=20, t=50 if title else 20, b=40),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#d9e4f5",
            borderwidth=1,
        ),
        xaxis=dict(showgrid=False, linecolor="#c0c0c0", linewidth=1),
        yaxis=dict(gridcolor="#f0f0f0", linecolor="#c0c0c0"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<h1>📊 Customer Churn Prediction Dashboard</h1>"
        "<p style='color:#6b7280;margin-top:-8px;'>Online Retail II · Logistic Regression · H=90 days · Model <code>lr_full_c0.01_h90</code></p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Load data ─────────────────────────────────────────────────────────────
    scores_raw, modeling_raw = load_raw()
    test_df   = build_test_df(scores_raw, modeling_raw)
    deploy_df = build_deploy_df(scores_raw, modeling_raw)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Classification Performance",
        "📊 Risk Band Analysis",
        "💰 Business Impact Scenarios",
        "🗓️ Scoring Output",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – CLASSIFICATION PERFORMANCE
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("#### Model vs Baselines — Test Period (Jul 4 – Sep 5, 2011 · 10 weeks)")

        y      = test_df["churn"].to_numpy()
        s_mod  = test_df["score_final_model"].to_numpy()
        preds  = (s_mod >= 0.5).astype(int)

        # ── KPI row ───────────────────────────────────────────────────────────
        c0, c1, c2, c3, c4, c5 = st.columns(6)
        c0.metric("Churn Rate",  f"{y.mean():.1%}")
        c1.metric("ROC-AUC",     f"{roc_auc_score(y, s_mod):.3f}")
        c2.metric("PR-AUC",      f"{average_precision_score(y, s_mod):.3f}")
        c3.metric("Precision",   f"{precision_score(y, preds):.3f}")
        c4.metric("Recall",      f"{recall_score(y, preds):.3f}")
        c5.metric("F1",          f"{f1_score(y, preds):.3f}")

        st.markdown(
            "<div class='info-box'>Metrics at default 0.5 probability threshold. "
            "PR-AUC random reference = churn rate (40.1%).</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        # ── ROC + PR curves ───────────────────────────────────────────────────
        col_roc, col_pr = st.columns(2)

        candidates = {
            "Final Model":  test_df["score_final_model"].to_numpy(),
            "Recency Rule": test_df["score_recency_rule"].to_numpy(),
            "RFM Rule":     test_df["score_rfm_rule"].to_numpy(),
            "Random":       test_df["score_random"].to_numpy(),
        }

        # ROC
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                          line=dict(dash="dot", color=C_GRAY, width=1))
        for name, scores_c in candidates.items():
            fpr, tpr, _ = roc_curve(y, scores_c)
            auc = roc_auc_score(y, scores_c)
            lw  = 2.5 if name == "Final Model" else 1.5
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{name} ({auc:.3f})",
                line=dict(color=CANDIDATE_COLORS[name], width=lw),
            ))
        fig_roc.update_xaxes(title_text="False Positive Rate", range=[0, 1])
        fig_roc.update_yaxes(title_text="True Positive Rate",  range=[0, 1])
        clean_layout(fig_roc, "ROC Curve (AUC)")
        col_roc.plotly_chart(fig_roc, use_container_width=True)

        # PR
        fig_pr = go.Figure()
        base_rate = y.mean()
        fig_pr.add_shape(type="line", x0=0, x1=1, y0=base_rate, y1=base_rate,
                         line=dict(dash="dot", color=C_GRAY, width=1))
        for name, scores_c in candidates.items():
            prec, rec, _ = precision_recall_curve(y, scores_c)
            ap = average_precision_score(y, scores_c)
            lw = 2.5 if name == "Final Model" else 1.5
            fig_pr.add_trace(go.Scatter(
                x=rec, y=prec, mode="lines", name=f"{name} (AP={ap:.3f})",
                line=dict(color=CANDIDATE_COLORS[name], width=lw),
            ))
        fig_pr.update_xaxes(title_text="Recall",    range=[0, 1])
        fig_pr.update_yaxes(title_text="Precision", range=[0, 1])
        clean_layout(fig_pr, "Precision-Recall Curve")
        col_pr.plotly_chart(fig_pr, use_container_width=True)

        # ── Calibration + weekly lift ──────────────────────────────────────────
        col_cal, col_lift = st.columns(2)

        # Calibration
        frac_pos, mean_pred = calibration_curve(y, s_mod, n_bins=10, strategy="uniform")
        fig_cal = go.Figure()
        fig_cal.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                          line=dict(dash="dot", color=C_GRAY, width=1))
        fig_cal.add_trace(go.Scatter(
            x=mean_pred, y=frac_pos, mode="lines+markers",
            name="Final Model",
            line=dict(color=C_BLUE, width=2),
            marker=dict(size=7),
        ))
        fig_cal.update_xaxes(title_text="Mean Predicted Probability", range=[0, 1])
        fig_cal.update_yaxes(title_text="Fraction of Positives",      range=[0, 1])
        clean_layout(fig_cal, "Calibration Curve")
        col_cal.plotly_chart(fig_cal, use_container_width=True)

        # Weekly lift @ 10%
        def _lift_at_k(y_true, sc, k=0.10):
            k_n = max(int(round(len(sc) * k)), 1)
            top = np.argsort(-sc)[:k_n]
            return y_true[top].mean() / y_true.mean()

        weekly_lift = {}
        for week, wdf in test_df.groupby("snapshot_date"):
            yw = wdf["churn"].to_numpy()
            for nm, col in [
                ("Final Model",  "score_final_model"),
                ("Recency Rule", "score_recency_rule"),
                ("RFM Rule",     "score_rfm_rule"),
                ("Random",       "score_random"),
            ]:
                weekly_lift.setdefault(nm, []).append(
                    _lift_at_k(yw, wdf[col].to_numpy())
                )

        weeks = sorted(test_df["snapshot_date"].unique())
        week_labels = [pd.Timestamp(w).strftime("%b %d") for w in weeks]
        fig_lift = go.Figure()
        for nm, vals in weekly_lift.items():
            lw = 2.5 if nm == "Final Model" else 1.5
            fig_lift.add_trace(go.Scatter(
                x=week_labels, y=vals, mode="lines+markers",
                name=nm, line=dict(color=CANDIDATE_COLORS[nm], width=lw),
                marker=dict(size=6),
            ))
        fig_lift.update_yaxes(title_text="Lift @ Top 10%")
        clean_layout(fig_lift, "Weekly Lift @ Top 10%")
        col_lift.plotly_chart(fig_lift, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – RISK BAND ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### Risk Band Analysis — Final Model · Test Period")
        st.markdown(
            "<div class='info-box'>Bands 1–10 (10 = highest churn risk). "
            "Revenue at Risk = cumulative £ revenue of churners in each band.</div>",
            unsafe_allow_html=True,
        )

        band_grp = (
            test_df.groupby("risk_band")
            .agg(
                customers=("customer_id", "count"),
                churners=("churn", "sum"),
                revenue_at_risk=("monetary_so_far", lambda x: x[test_df.loc[x.index, "churn"] == 1].sum()),
            )
            .reset_index()
        )
        band_grp["churn_rate"] = band_grp["churners"] / band_grp["customers"]
        band_grp["avg_revenue"] = band_grp["revenue_at_risk"] / band_grp["churners"].clip(lower=1)
        band_grp["band_label"] = band_grp["risk_band"].astype(str)

        c1, c2, c3 = st.columns(3)

        # Customer count
        fig_cnt = px.bar(
            band_grp, x="band_label", y="customers",
            color="risk_band", color_continuous_scale="Blues",
            labels={"band_label": "Risk Band", "customers": "Customers"},
        )
        fig_cnt.update_coloraxes(showscale=False)
        clean_layout(fig_cnt, "Customers per Band")
        c1.plotly_chart(fig_cnt, use_container_width=True)

        # Churn rate
        fig_cr = px.bar(
            band_grp, x="band_label", y="churn_rate",
            color="risk_band", color_continuous_scale="Reds",
            labels={"band_label": "Risk Band", "churn_rate": "Churn Rate"},
        )
        fig_cr.update_yaxes(tickformat=".0%")
        fig_cr.update_coloraxes(showscale=False)
        clean_layout(fig_cr, "Churn Rate per Band")
        c2.plotly_chart(fig_cr, use_container_width=True)

        # Revenue at risk
        fig_rev = px.bar(
            band_grp, x="band_label", y="revenue_at_risk",
            color="risk_band", color_continuous_scale="Oranges",
            labels={"band_label": "Risk Band", "revenue_at_risk": "Revenue at Risk (£)"},
        )
        fig_rev.update_yaxes(tickprefix="£", tickformat=",.0f")
        fig_rev.update_coloraxes(showscale=False)
        clean_layout(fig_rev, "Revenue at Risk per Band (Churners)")
        c3.plotly_chart(fig_rev, use_container_width=True)

        # ── Cumulative revenue chart ───────────────────────────────────────────
        band_grp_desc = band_grp.sort_values("risk_band", ascending=False)
        band_grp_desc["cumul_revenue"]   = band_grp_desc["revenue_at_risk"].cumsum()
        band_grp_desc["cumul_customers"] = band_grp_desc["customers"].cumsum()
        total_rev = band_grp["revenue_at_risk"].sum()
        band_grp_desc["pct_revenue"] = band_grp_desc["cumul_revenue"] / total_rev * 100

        col_cumul, col_seg = st.columns(2)

        fig_cumul = go.Figure()
        fig_cumul.add_trace(go.Scatter(
            x=band_grp_desc["risk_band"].astype(str),
            y=band_grp_desc["pct_revenue"],
            fill="tozeroy", mode="lines+markers",
            line=dict(color=C_BLUE, width=2),
            fillcolor="rgba(42,120,214,0.12)",
            name="Cumulative Revenue at Risk",
        ))
        fig_cumul.update_xaxes(
            title_text="Risk Band (top-down)",
            categoryorder="array",
            categoryarray=band_grp_desc["risk_band"].astype(str).tolist(),
        )
        fig_cumul.update_yaxes(title_text="% of Total Revenue at Risk", ticksuffix="%")
        clean_layout(fig_cumul, "Cumulative Revenue at Risk (High → Low Band)")
        col_cumul.plotly_chart(fig_cumul, use_container_width=True)

        # Segment breakdown by band
        seg_grp = (
            test_df.groupby(["risk_band", "segment"])["customer_id"]
            .count()
            .reset_index()
            .rename(columns={"customer_id": "count"})
        )
        fig_seg = px.bar(
            seg_grp, x="risk_band", y="count", color="segment",
            barmode="stack",
            color_discrete_map={"repeat": C_BLUE, "one-time": C_ORANGE},
            labels={"risk_band": "Risk Band", "count": "Customers", "segment": "Segment"},
        )
        clean_layout(fig_seg, "Segment Mix per Risk Band")
        col_seg.plotly_chart(fig_seg, use_container_width=True)

        # ── Summary table ─────────────────────────────────────────────────────
        st.markdown("##### Band Summary Table")
        display_tbl = band_grp[
            ["risk_band", "customers", "churners", "churn_rate",
             "revenue_at_risk", "avg_revenue"]
        ].copy()
        display_tbl.columns = [
            "Band", "Customers", "Churners", "Churn Rate",
            "Revenue at Risk (£)", "Avg Rev / Churner (£)",
        ]
        display_tbl["Churn Rate"] = display_tbl["Churn Rate"].map("{:.1%}".format)
        display_tbl["Revenue at Risk (£)"] = display_tbl["Revenue at Risk (£)"].map("£{:,.0f}".format)
        display_tbl["Avg Rev / Churner (£)"] = display_tbl["Avg Rev / Churner (£)"].map("£{:,.0f}".format)
        st.dataframe(display_tbl, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – BUSINESS IMPACT SCENARIOS
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("#### Business Impact Scenario Builder")
        st.markdown(
            "<div class='info-box'>"
            "All figures are <strong>scenario estimates</strong> under stated assumptions, "
            "not measured causal impact (no treatment/control in Online Retail II). "
            "Adjust the parameters below to explore different retention campaigns."
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Controls ──────────────────────────────────────────────────────────
        st.markdown("##### Campaign Parameters")
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        k_pct   = p1.slider("Top-K %",          10, 50, 20, 10, format="%d%%")
        eff_pct = p2.slider("Retention Eff. %",  30, 60, 40, 5,  format="%d%%")
        margin  = p3.slider("Gross Margin %",    20, 60, 35, 5,  format="%d%%")
        c_cont  = p4.number_input("Contact Cost (£)", 0.10, 5.0, 0.50, 0.10, format="%.2f")
        c_inc   = p5.number_input("Incentive Cost (£)", 1.0, 20.0, 5.0, 0.5, format="%.1f")
        cannib  = p6.slider("Cannibalization %", 0, 50, 20, 5, format="%d%%")

        k_frac  = k_pct  / 100
        e       = eff_pct / 100
        m       = margin  / 100
        r       = cannib  / 100

        counts_wk = weekly_business_counts(
            test_df, "score_final_model", "monetary_so_far", k_frac
        )
        counts_sum = counts_wk.sum()
        impact = business_impact(counts_sum, e, m, c_cont, c_inc, r)

        # ── KPIs ──────────────────────────────────────────────────────────────
        st.markdown("##### Estimated Campaign Results (10-Week Test Period)")
        km1, km2, km3, km4, km5, km6 = st.columns(6)
        km1.metric("Targeted",        f"{impact['Customers Targeted']:,.0f}")
        km2.metric("Churners Caught",  f"{impact['Churners Caught (TP)']:,.0f}")
        km3.metric("Recall @ K",       f"{impact['Recall @ K']:.1%}")
        km4.metric("Revenue at Risk",  f"£{impact['Revenue at Risk (£)']:,.0f}")
        km5.metric("Gross Profit",     f"£{impact['Gross Profit (£)']:,.0f}")
        km6.metric(
            "Net Business Impact",
            f"£{impact['Net Impact (£)']:,.0f}",
            delta=f"{'▲' if impact['Net Impact (£)'] >= 0 else '▼'} {'Positive' if impact['Net Impact (£)'] >= 0 else 'Negative'}",
        )

        # ── Value of the Model ────────────────────────────────────────────────────
        st.markdown("##### Value of the Model — Final Model vs Baselines")
        st.markdown(
            "<div class='info-box'>"
            "Net business impact and Recall@K for each candidate at the same Top-K% and cost assumptions. "
            "Model uplift = Final Model net impact minus Random baseline net impact."
            "</div>",
            unsafe_allow_html=True,
        )
        model_candidates = {
            "Final Model":  "score_final_model",
            "Recency Rule": "score_recency_rule",
            "RFM Rule":     "score_rfm_rule",
            "Random":       "score_random",
        }
        vm_rows = []
        for cand_name, sc_col in model_candidates.items():
            cnt_c = weekly_business_counts(test_df, sc_col, "monetary_so_far", k_frac).sum()
            imp_c = business_impact(cnt_c, e, m, c_cont, c_inc, r)
            vm_rows.append({
                "Candidate":           cand_name,
                "Net Impact (£)":      imp_c["Net Impact (£)"],
                "Revenue at Risk (£)": imp_c["Revenue at Risk (£)"],
                "Recall @ K":          imp_c["Recall @ K"],
                "Churners Caught":     imp_c["Churners Caught (TP)"],
            })
        vm_df = pd.DataFrame(vm_rows)

        col_vm1, col_vm2 = st.columns(2)

        # Net impact comparison
        fig_vm = go.Figure(go.Bar(
            x=vm_df["Candidate"],
            y=vm_df["Net Impact (£)"],
            marker_color=[CANDIDATE_COLORS[c] for c in vm_df["Candidate"]],
            text=vm_df["Net Impact (£)"].map("£{:,.0f}".format),
            textposition="outside",
        ))
        fig_vm.update_yaxes(title_text="Net Business Impact (£)", tickprefix="£", tickformat=",.0f")
        clean_layout(fig_vm, f"Net Business Impact by Candidate — Top-{k_pct}%")
        col_vm1.plotly_chart(fig_vm, use_container_width=True)

        # Recall @ K comparison
        fig_recall_vm = go.Figure(go.Bar(
            x=vm_df["Candidate"],
            y=vm_df["Recall @ K"],
            marker_color=[CANDIDATE_COLORS[c] for c in vm_df["Candidate"]],
            text=vm_df["Recall @ K"].map("{:.1%}".format),
            textposition="outside",
        ))
        fig_recall_vm.update_yaxes(title_text="Recall @ K", tickformat=".0%", range=[0, 1])
        clean_layout(fig_recall_vm, f"Recall @ Top-{k_pct}% by Candidate")
        col_vm2.plotly_chart(fig_recall_vm, use_container_width=True)

        # Model vs random uplift callout
        base_net   = vm_df.loc[vm_df["Candidate"] == "Random",     "Net Impact (£)"].iloc[0]
        model_net  = vm_df.loc[vm_df["Candidate"] == "Final Model", "Net Impact (£)"].iloc[0]
        uplift_val = model_net - base_net
        uplift_sign = "▲" if uplift_val >= 0 else "▼"
        col_vm1.markdown(
            f"<div class='info-box'>"
            f"Final Model: <strong>£{model_net:,.0f}</strong> &nbsp;·&nbsp; "
            f"Random baseline: <strong>£{base_net:,.0f}</strong> &nbsp;·&nbsp; "
            f"Model uplift: <strong>{uplift_sign} £{abs(uplift_val):,.0f}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("##### Revenue & Cost Breakdown (Final Model)")

        # ── Waterfall breakdown ────────────────────────────────────────────────
        col_wf, col_sc = st.columns(2)

        wf_labels = [
            "Gross Profit", "Contact Cost", "Incentive Cost", "Net Impact",
        ]
        wf_vals = [
            impact["Gross Profit (£)"],
            -impact["Contact Cost (£)"],
            -impact["Incentive Cost (£)"],
            impact["Net Impact (£)"],
        ]
        wf_colors = [C_GREEN, C_ORANGE, C_YELLOW,
                     C_BLUE if impact["Net Impact (£)"] >= 0 else C_ORANGE]
        fig_wf = go.Figure(go.Bar(
            x=wf_labels, y=wf_vals,
            marker_color=wf_colors,
            text=[f"£{abs(v):,.0f}" for v in wf_vals],
            textposition="outside",
        ))
        fig_wf.update_yaxes(title_text="£", tickprefix="£", tickformat=",.0f")
        clean_layout(fig_wf, "Revenue Breakdown")
        col_wf.plotly_chart(fig_wf, use_container_width=True)

        # ── Scenario matrix: effectiveness × K ────────────────────────────────
        sc_rows = []
        for k_f in K_FRACS:
            cnt = weekly_business_counts(
                test_df, "score_final_model", "monetary_so_far", k_f
            ).sum()
            for eff in EFF_LEVELS:
                imp = business_impact(cnt, eff, m, c_cont, c_inc, r)
                sc_rows.append({
                    "Top-K %":   f"{int(k_f*100)}%",
                    "Eff.":      f"{int(eff*100)}%",
                    "Net (£)":   imp["Net Impact (£)"],
                })
        sc_df = pd.DataFrame(sc_rows)

        fig_sc = px.bar(
            sc_df, x="Top-K %", y="Net (£)", color="Eff.",
            barmode="group",
            color_discrete_sequence=[C_GREEN, C_BLUE, C_ORANGE],
            labels={"Net (£)": "Net Business Impact (£)"},
        )
        fig_sc.update_yaxes(tickprefix="£", tickformat=",.0f")
        clean_layout(fig_sc, "Scenario Matrix — Net Impact (£) by K & Effectiveness")
        col_sc.plotly_chart(fig_sc, use_container_width=True)

        # ── Weekly targeting trend ─────────────────────────────────────────────
        st.markdown("##### Weekly Targeting Trends")
        wk_labels = [pd.Timestamp(w).strftime("%b %d") for w in counts_wk.index]
        fig_wk = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Weekly Revenue at Risk (Churners Targeted)",
                            "Weekly Recall @ K"),
        )
        fig_wk.add_trace(go.Bar(
            x=wk_labels, y=counts_wk["revenue_at_risk"],
            marker_color=C_BLUE, name="Revenue at Risk",
            text=counts_wk["revenue_at_risk"].map("£{:,.0f}".format),
            textposition="outside",
        ), row=1, col=1)
        recall_wk = counts_wk["tp"] / counts_wk["positives"]
        fig_wk.add_trace(go.Scatter(
            x=wk_labels, y=recall_wk,
            mode="lines+markers",
            line=dict(color=C_GREEN, width=2), marker=dict(size=7),
            name="Recall @ K",
        ), row=1, col=2)
        fig_wk.update_yaxes(tickprefix="£", tickformat=",.0f", row=1, col=1)
        fig_wk.update_yaxes(tickformat=".0%", row=1, col=2)
        fig_wk.update_layout(
            height=360,
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="#374151"),
            showlegend=False,
            margin=dict(l=40, r=20, t=60, b=40),
        )
        fig_wk.update_xaxes(showgrid=False)
        fig_wk.update_yaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig_wk, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 – SCORING OUTPUT (DEPLOYMENT)
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("#### Deployment Scoring Output — Sep 12 to Dec 5, 2011 (13 Weeks)")
        st.markdown(
            "<div class='info-box'>"
            "Live scoring snapshots produced by the final model. "
            "No churn labels available (future-looking). "
            "Use the filters to explore a specific week or customer segment."
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Filters ───────────────────────────────────────────────────────────
        all_weeks = sorted(deploy_df["snapshot_date"].unique())
        week_opts = {pd.Timestamp(w).strftime("%Y-%m-%d"): w for w in all_weeks}

        f1, f2, f3, f4 = st.columns(4)
        sel_week   = f1.selectbox("Week", list(week_opts.keys()), index=len(week_opts)-1)
        sel_seg    = f2.multiselect("Segment", ["repeat", "one-time"],
                                    default=["repeat", "one-time"])
        sel_bands  = f3.multiselect("Risk Bands", list(range(1, 11)),
                                    default=list(range(1, 11)),
                                    format_func=lambda x: f"Band {x}")
        top_k_pct4 = f4.slider("Top-K % to Highlight", 0, 50, 20, 5, format="%d%%")

        sel_ts = week_opts[sel_week]
        week_df = deploy_df[
            (deploy_df["snapshot_date"] == sel_ts)
            & deploy_df["segment"].isin(sel_seg)
            & deploy_df["risk_band"].isin(sel_bands)
        ].copy()

        top_k_n = max(int(round(len(week_df) * top_k_pct4 / 100)), 0)
        week_df["in_top_k"] = False
        if top_k_n > 0:
            top_idx = week_df.nlargest(top_k_n, "churn_probability").index
            week_df.loc[top_idx, "in_top_k"] = True

        # ── KPIs for selected week ─────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Customers Scored",    f"{len(week_df):,}")
        k2.metric("Avg Churn Probability", f"{week_df['churn_probability'].mean():.1%}")
        k3.metric("In Top-K (Targeted)", f"{week_df['in_top_k'].sum():,}")
        k4.metric("High Risk (Band 8-10)",
                  f"{(week_df['risk_band'] >= 8).sum():,}")

        # ── Charts ────────────────────────────────────────────────────────────
        col_dist, col_trend = st.columns(2)

        # Risk band distribution for selected week
        bd_cnt = week_df.groupby("risk_band")["customer_id"].count().reset_index()
        bd_cnt.columns = ["risk_band", "count"]
        fig_bd = px.bar(
            bd_cnt, x="risk_band", y="count",
            color="risk_band", color_continuous_scale="Blues",
            labels={"risk_band": "Risk Band", "count": "Customers"},
        )
        fig_bd.update_coloraxes(showscale=False)
        clean_layout(fig_bd, f"Risk Band Distribution — {sel_week}")
        col_dist.plotly_chart(fig_bd, use_container_width=True)

        # Weekly high-risk trend (pandas-2.x safe: no groupby key access inside lambda)
        _seg_df = deploy_df[deploy_df["segment"].isin(sel_seg)]
        trend = (
            _seg_df.groupby("snapshot_date")
            .agg(
                avg_p=("churn_probability", "mean"),
                high_r=("risk_band", lambda x: (x >= 8).sum()),
                n=("customer_id", "count"),
            )
            .reset_index()
            .rename(columns={"snapshot_date": "week"})
        )
        trend["week_str"] = trend["week"].apply(lambda w: pd.Timestamp(w).strftime("%b %d"))

        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(
            go.Bar(x=trend["week_str"], y=trend["high_r"],
                   name="High Risk (Band 8–10)", marker_color=C_ORANGE,
                   opacity=0.7),
            secondary_y=False,
        )
        fig_trend.add_trace(
            go.Scatter(x=trend["week_str"], y=trend["avg_p"],
                       mode="lines+markers", name="Avg Churn Prob.",
                       line=dict(color=C_BLUE, width=2), marker=dict(size=6)),
            secondary_y=True,
        )
        fig_trend.update_yaxes(title_text="High Risk Customers",      secondary_y=False)
        fig_trend.update_yaxes(title_text="Avg Churn Probability",
                               tickformat=".1%", secondary_y=True)
        fig_trend.update_layout(
            title="Weekly High-Risk Count & Avg Probability",
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="#374151"), height=380,
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#d9e4f5", borderwidth=1),
        )
        fig_trend.update_xaxes(showgrid=False)
        col_trend.plotly_chart(fig_trend, use_container_width=True)

        # ── Score distribution histogram ──────────────────────────────────────
        col_hist, col_pie = st.columns(2)

        fig_hist = go.Figure()
        for seg, clr in [("repeat", C_BLUE), ("one-time", C_ORANGE)]:
            s = week_df[week_df["segment"] == seg]["churn_probability"]
            if len(s):
                fig_hist.add_trace(go.Histogram(
                    x=s, name=seg.capitalize(),
                    marker_color=clr, opacity=0.7,
                    xbins=dict(start=0, end=1, size=0.05),
                ))
        fig_hist.update_xaxes(title_text="Churn Probability", tickformat=".0%")
        fig_hist.update_yaxes(title_text="Customers")
        fig_hist.update_layout(barmode="overlay")
        clean_layout(fig_hist, f"Churn Probability Distribution — {sel_week}")
        col_hist.plotly_chart(fig_hist, use_container_width=True)

        # Risk band pie
        pie_grp = week_df.groupby("risk_band")["customer_id"].count().reset_index()
        fig_pie = px.pie(
            pie_grp, names="risk_band", values="customer_id",
            color_discrete_sequence=px.colors.sequential.Blues_r,
            hole=0.4,
        )
        fig_pie.update_traces(textinfo="percent+label")
        clean_layout(fig_pie, f"Risk Band Share — {sel_week}", height=380)
        col_pie.plotly_chart(fig_pie, use_container_width=True)

        # ── Batch scoring summary (all 13 weeks) ──────────────────────────────
        st.markdown("##### Batch Scoring Summary — All 13 Deployment Weeks")
        batch_rows = []
        for w in sorted(deploy_df["snapshot_date"].unique()):
            wk = deploy_df[deploy_df["snapshot_date"] == w]
            batch_rows.append({
                "Week":              pd.Timestamp(w).strftime("%Y-%m-%d"),
                "Customers Scored":  len(wk),
                "Avg Churn Prob.":   wk["churn_probability"].mean(),
                "High Risk (8–10)":  int((wk["risk_band"] >= 8).sum()),
                "% High Risk":       (wk["risk_band"] >= 8).mean(),
                "Median Risk Score": int(wk["risk_score"].median()),
                "Repeat":            int((wk["segment"] == "repeat").sum()),
                "One-Time":          int((wk["segment"] == "one-time").sum()),
            })
        batch_df = pd.DataFrame(batch_rows)
        batch_display = batch_df.copy()
        batch_display["Avg Churn Prob."] = batch_display["Avg Churn Prob."].map("{:.1%}".format)
        batch_display["% High Risk"]     = batch_display["% High Risk"].map("{:.1%}".format)
        st.dataframe(batch_display, use_container_width=True, hide_index=True)
        st.markdown("")

        # ── Customer table ─────────────────────────────────────────────────────
        st.markdown(f"##### Customer Scores — {sel_week} ({len(week_df):,} customers)")

        show_cols = ["rank", "customer_id", "segment", "risk_band",
                     "churn_probability", "risk_score",
                     "monetary_so_far", "recency_days",
                     "frequency_so_far", "in_top_k"]
        display = week_df[show_cols].copy().sort_values("rank")
        display.columns = [
            "Rank", "Customer ID", "Segment", "Risk Band",
            "Churn Prob.", "Risk Score",
            "Revenue (£)", "Recency Days",
            "Frequency", "In Top-K",
        ]
        display["Churn Prob."]  = display["Churn Prob."].map("{:.1%}".format)
        display["Revenue (£)"]  = display["Revenue (£)"].map("£{:,.0f}".format)
        display["Recency Days"] = display["Recency Days"].map("{:.0f}".format)
        display["Frequency"]    = display["Frequency"].map("{:.0f}".format)
        display["In Top-K"]     = display["In Top-K"].map({True: "✅", False: ""})

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=400,
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<p style='text-align:center;color:#9ca3af;font-size:0.8rem;'>"
        "Customer Churn Prediction Dashboard · Online Retail II · "
        "Model: Logistic Regression LR Full C=0.01 H=90 · "
        "Data: 2010-12 to 2011-12"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
