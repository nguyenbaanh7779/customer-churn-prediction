"""Retention targeting dashboard (Strategy 8.5).

Run from the repo root:  streamlit run dashboard/app.py
Reads data/results/customer_scores_actions_h90.parquet, recency_days from the modeling
dataset, and the 8 RFM segments (src/make_rfm8.py, rules from output/02_rfm_segmentation.ipynb). No model is retrained here.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from policy_lib import expected_margin, row_gain

st.set_page_config(page_title="Retention targeting", layout="wide", initial_sidebar_state="expanded")

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a2"
INK, MUTED, LINE = "#0b0b0b", "#6b6a66", "#e5e5e0"

st.markdown(f"""
<style>
#MainMenu, footer, header [data-testid="stToolbar"] {{visibility: hidden;}}
.block-container {{padding: 3.5rem 2.5rem 3rem; max-width: 1280px;}}
h1 {{font-size: 1.75rem !important; font-weight: 700; letter-spacing: -0.01em; margin-bottom: 0.1rem; padding: 0;}}
h3 {{font-size: 1rem !important; font-weight: 600; margin: 0 0 0.15rem 0; padding: 0;}}
.sub {{color: {MUTED}; font-size: 0.86rem; margin-bottom: 1.1rem;}}
.kpis {{display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 0.6rem 0 1.6rem;}}
.kpi {{background: #fff; border: 1px solid {LINE}; border-radius: 10px; padding: 14px 16px;}}
.kpi .l {{font-size: 0.72rem; text-transform: uppercase; letter-spacing: .05em; color: {MUTED}; font-weight: 600;}}
.kpi .v {{font-size: 1.65rem; font-weight: 700; line-height: 1.25; margin-top: 4px; font-variant-numeric: tabular-nums;}}
.kpi .n {{font-size: 0.76rem; color: {MUTED}; margin-top: 2px;}}
.chips {{display: flex; gap: 6px; margin-top: 8px;}}
.chip {{font-size: 0.8rem; font-weight: 600; padding: 3px 9px; border-radius: 99px; font-variant-numeric: tabular-nums;}}
.c-r {{background: #dcefe7; color: #0e6d4a;}} .c-n {{background: #fdeccd; color: #8a5a00;}} .c-x {{background: #ecebe7; color: #55544f;}}
.story {{background: #f1f6fd; border-left: 4px solid {BLUE}; border-radius: 6px; padding: 12px 16px; font-size: 0.98rem; line-height: 1.5; margin: 0.2rem 0 0.8rem;}}
.step {{display: flex; align-items: center; gap: 10px; margin: 1.9rem 0 0.1rem;}}
.step .num {{background: {BLUE}; color: #fff; width: 26px; height: 26px; border-radius: 50%; text-align: center; line-height: 26px; font-weight: 700; font-size: 0.85rem;}}
.step .t {{font-size: 1.15rem; font-weight: 700;}}
.take {{font-size: 0.92rem; background: #f6f6f3; border-radius: 6px; padding: 8px 12px; margin: 0.2rem 0 0.4rem;}}
.note {{font-size: 0.78rem; color: {MUTED}; margin-top: 0.4rem;}}
section[data-testid="stSidebar"] {{border-right: 1px solid {LINE};}}
section[data-testid="stSidebar"] h2 {{font-size: 0.78rem; text-transform: uppercase; letter-spacing: .06em; color: {MUTED}; margin-top: 0.8rem;}}
@media (max-width: 900px) {{.kpis {{grid-template-columns: repeat(2, 1fr);}}}}
</style>
""", unsafe_allow_html=True)

PRESETS = {"Light (5%, 20%)": (0.05, 0.20), "Base (10%, 30%)": (0.10, 0.30), "Deep (20%, 40%)": (0.20, 0.40)}
SEG_ORDER = ["Champions", "VIP at risk", "Loyal", "Developing", "One-time recent", "Cooling", "Lapsed", "Dormant"]
SEG_HINT = {"Champions": "top 10% spenders, bought < 90 days ago", "VIP at risk": "top 10% spenders, silent > 90 days",
            "Loyal": "4+ orders, bought < 90 days ago", "Developing": "2-3 orders, bought < 90 days ago",
            "One-time recent": "1 order, bought < 90 days ago", "Cooling": "last order 90-180 days ago",
            "Lapsed": "last order 180-365 days ago", "Dormant": "last order > 1 year ago"}


@st.cache_data
def load():
    s = pd.read_parquet(ROOT / "data/results/customer_scores_actions_h90.parquet")
    rec = pd.read_parquet(ROOT / "data/processed/modeling_dataset_h90.parquet",
                          columns=["customer_id", "snapshot_date", "recency_days"])
    seg = pd.read_parquet(ROOT / "data/results/rfm8_segments.parquet")
    s = s.merge(rec, on=["customer_id", "snapshot_date"], how="left").merge(
        seg[["customer_id", "snapshot_date", "segment8"]], on=["customer_id", "snapshot_date"], how="left")
    s = s.rename(columns={"churn_probability_calibrated": "p", "churn_actual": "churn"})
    all_seg = seg[seg.snapshot_date == seg.snapshot_date.max()].segment8.value_counts()
    return s, all_seg


def exp_outlay(df, d, e, r):
    """Expected discount cost of contacting a customer: d * V * (p*e + (1-p)*r)."""
    return d * df.value_v * (df.p * e + (1 - df.p) * r)


def select(df, rule, mode, k, budget, d, e, m, r, c, score_override=None):
    """Boolean mask of customers who get an offer, chosen separately in each week.
    rule 'risk' = highest churn probability; 'margin' = highest expected margin (only if > 0)."""
    em = expected_margin(df.p, df.value_v, d, e, m, r, c)
    score = df.p if rule == "risk" else em
    if score_override is not None:
        score = score_override
    ok = (em > 0) if (rule == "margin" and score_override is None) else pd.Series(True, index=df.index)
    work = pd.DataFrame({"w": df.snapshot_date, "s": score.where(ok, -np.inf), "o": exp_outlay(df, d, e, r)})
    work = work.sort_values(["w", "s"], ascending=[True, False])
    g = work.groupby("w")
    if mode == "share":
        take = g.cumcount() < np.maximum(np.round(g["s"].transform("size") * k), 1)
    else:
        take = g["o"].cumsum() <= budget
    take &= np.isfinite(work.s)
    return take.reindex(df.index).fillna(False), em


df, all_seg = load()
test = df[df.split == "test"].copy()
last_date = df[df.split == "deployment"].snapshot_date.max()
cur = df[df.snapshot_date == last_date].copy()

# ---- sidebar: two controls only --------------------------------------------
sb = st.sidebar
sb.markdown("## Weekly discount budget")
budget = float(sb.slider("Expected discount cost per week (£)", 1000, 20000, 7740, 20, label_visibility="collapsed"))
sb.markdown("## Offer scenario")
preset = sb.selectbox("Preset (discount, response)", list(PRESETS), index=1, label_visibility="collapsed")
d0, e0 = PRESETS[preset]
with sb.expander("Fine-tune assumptions"):
    d = st.slider("Discount", 0.05, 0.30, d0, 0.01, format="%.2f")
    e = st.slider("Response of customers who would not buy", 0.05, 0.60, e0, 0.05)
    r = st.slider("Redemption by customers who would buy anyway", 0.10, 0.80, 0.40, 0.05)
    m = st.slider("Gross margin", 0.20, 0.60, 0.40, 0.05)
    c = st.number_input("Contact cost (£)", 0.0, 5.0, 0.50, 0.10)

# ---- this Monday's list -----------------------------------------------------
sel, cur["em"] = select(cur, "risk", "budget", None, budget, d, e, m, r, c)
cur["action"] = np.where(sel, "Voucher", "No voucher")
off = cur[sel]
spend = exp_outlay(off, d, e, r).sum()

# ---- replay on the 10 test weeks -------------------------------------------
weeks = test.snapshot_date.nunique()
gain = pd.Series(row_gain(test, d, e, m, r, c), index=test.index)
sim_out = d * test.value_v * np.where(test.churn == 1, e, r)
risk_sel, _ = select(test, "risk", "budget", None, budget, d, e, m, r, c)
marg_sel, _ = select(test, "margin", "share", 0.20, None, d, e, m, r, c)
rec_sel, _ = select(test, "risk", "budget", None, budget, d, e, m, r, c, score_override=test.recency_days.fillna(0))
policies = {"Highest churn risk": risk_sel, "Highest expected margin (top 20%)": marg_sel,
            "Longest since last purchase": rec_sel, "Offer to everyone": pd.Series(True, index=test.index)}
res = pd.DataFrame([dict(policy=n, margin=gain[mk].sum() / weeks, contacts=mk.sum() / weeks, outlay=sim_out[mk].sum() / weeks)
                    for n, mk in policies.items()])
me, rc, ev = res.iloc[0], res.iloc[2], res.iloc[3]

# same rule under the three scenarios (for the closing line)
scen = {}
for name, (dd, ee) in PRESETS.items():
    ts, _ = select(test, "risk", "budget", None, budget, dd, ee, m, r, c)
    scen[name.split(" (")[0]] = pd.Series(row_gain(test, dd, ee, m, r, c), index=test.index)[ts].sum() / weeks

st.title("Who should get this week's discount voucher?")
st.markdown(f'<div class="sub">Online retailer · list for Monday {last_date:%d %b %Y} · {len(cur):,} active customers scored · '
            f'assumptions: {d:.0%} voucher, {e:.0%} of leavers use it to stay, {r:.0%} of others use it anyway, {m:.0%} margin · '
            'simulated, not measured sales</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="kpis">
 <div class="kpi"><div class="l">Active customers</div><div class="v">{len(cur):,}</div><div class="n">scored this Monday</div></div>
 <div class="kpi"><div class="l">Get a voucher</div><div class="v">{len(off):,}</div><div class="n">{len(off)/len(cur):.0%} of active customers</div></div>
 <div class="kpi"><div class="l">Discount cost</div><div class="v">£{spend:,.0f}</div><div class="n">expected, this week's list</div></div>
 <div class="kpi"><div class="l">Revenue at risk</div><div class="v">£{(off.value_v * off.p).sum():,.0f}</div><div class="n">order value × chance of leaving</div></div>
 <div class="kpi"><div class="l">Expected extra profit</div><div class="v">£{off.em.sum():,.0f}</div><div class="n">next 90 days, simulated</div></div>
</div>
""", unsafe_allow_html=True)


def step(n, title, sub):
    st.markdown(f'<div class="step"><span class="num">{n}</span><span class="t">{title}</span></div><div class="note" style="margin:0 0 .6rem 38px">{sub}</div>', unsafe_allow_html=True)


AX = dict(labelColor=MUTED, domain=False, ticks=False, grid=True, gridColor=LINE)
YAX = alt.Axis(labelFontSize=12, labelColor=INK, ticks=False, domain=False, labelLimit=220, labelPadding=8)
scored = cur.groupby("segment8").agg(risk=("p", "mean"), offered=("action", lambda a: (a == "Voucher").sum()))
seg = pd.DataFrame({"segment": SEG_ORDER})
seg["customers"] = seg.segment.map(all_seg).fillna(0).astype(int)
seg["risk"] = seg.segment.map(scored.risk)
seg["offered"] = seg.segment.map(scored.offered).fillna(0).astype(int)
seg["hint"] = seg.segment.map(SEG_HINT)
seg["scored"] = seg.risk.notna()

# ---- 1: who is leaving ------------------------------------------------------
step(1, "Who is likely to stop buying?", f"Customers by group on {last_date:%d %b %Y}, with the model's average chance of no purchase in the next 90 days.")
seg["label"] = seg.apply(lambda x: f"{x.risk:.0%}  ·  {x.customers:,} customers" if x.scored else f"not scored  ·  {x.customers:,} customers, silent > 6 months", axis=1)
seg["x"] = seg.risk.fillna(0)
b1 = alt.Chart(seg).encode(y=alt.Y("segment:N", sort=SEG_ORDER, title=None, axis=YAX))
c1 = b1.mark_bar(cornerRadiusEnd=4, size=22).encode(
    x=alt.X("x:Q", title=None, scale=alt.Scale(domain=[0, 1.25]), axis=alt.Axis(format="%", values=[0, .25, .5, .75, 1], **AX)),
    color=alt.condition(alt.datum.x >= 0.4, alt.value(ORANGE), alt.value("#9ec5f3")),
    tooltip=[alt.Tooltip("segment:N", title="Group"), alt.Tooltip("hint:N", title="Definition"), alt.Tooltip("customers:Q", title="Customers")])
t1 = b1.mark_text(align="left", dx=6, fontSize=12, color=INK).encode(x="x:Q", text="label:N")
st.altair_chart((c1 + t1).properties(height=32 * len(seg) + 10).configure_view(stroke=None), use_container_width=True)
hot = seg[seg.risk >= 0.4]
st.markdown(f'<div class="take">Risk is highest for customers who bought only once or have started to go quiet: '
            f'{", ".join(f"<b>{n}</b> {v:.0%}" for n, v in zip(hot.segment, hot.risk))}. <b>Champions</b> are only {seg.set_index("segment").risk["Champions"]:.0%}. '
            f'Lapsed and Dormant customers have already gone and are not scored.</div>', unsafe_allow_html=True)

# ---- 2: who gets a voucher --------------------------------------------------
step(2, "Who gets a voucher?", "The highest-risk customers, until the weekly discount budget is used up.")
seg["not_offered"] = seg.customers - seg.offered
long = seg[seg.scored].melt("segment", ["offered", "not_offered"], var_name="k", value_name="n")
long["k"] = long.k.map({"offered": "Gets a voucher", "not_offered": "No voucher"})
c2 = alt.Chart(long).mark_bar(size=22).encode(
    y=alt.Y("segment:N", sort=SEG_ORDER, title=None, axis=YAX),
    x=alt.X("n:Q", stack="zero", title=None, axis=alt.Axis(format="~s", **AX)),
    color=alt.Color("k:N", scale=alt.Scale(domain=["Gets a voucher", "No voucher"], range=[BLUE, "#e3e2dc"]), legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
    order=alt.Order("k:N", sort="descending"),
    tooltip=[alt.Tooltip("segment:N", title="Group"), alt.Tooltip("k:N", title=""), alt.Tooltip("n:Q", title="Customers", format=",")])
st.altair_chart(c2.properties(height=32 * int(seg.scored.sum()) + 40).configure_view(stroke=None), use_container_width=True)
big = seg.sort_values("offered", ascending=False)
top3 = big.head(3)
st.markdown(f'<div class="take"><b>{len(off):,}</b> customers get a voucher (about £{spend:,.0f} of discount). '
            f'{top3.offered.sum() / max(len(off), 1):.0%} of them are <b>{top3.iloc[0].segment}</b>, <b>{top3.iloc[1].segment}</b> or <b>{top3.iloc[2].segment}</b>. '
            f'Champions and Loyal get {int(seg.set_index("segment").offered[["Champions", "Loyal"]].sum())}: they mostly buy anyway, so a voucher would only cost margin.</div>', unsafe_allow_html=True)

# ---- 3: does it pay ---------------------------------------------------------
step(3, "Does it pay off?", "Replay on the 10 test weeks (Jul–Sep 2011), which the model never saw: extra profit after discounts, per weekly group.")
res["label"] = res.margin.map(lambda v: f"£{v:,.0f}")
res["sub"] = res.apply(lambda x: f"{x.contacts:,.0f} customers · £{x.outlay:,.0f} discount", axis=1)
order = list(res.policy)
b3 = alt.Chart(res).encode(y=alt.Y("policy:N", sort=order, title=None, axis=YAX))
c3 = b3.mark_bar(cornerRadiusEnd=4, size=26).encode(
    x=alt.X("margin:Q", title=None, scale=alt.Scale(domain=[0, float(res.margin.max()) * 1.7], nice=False), axis=alt.Axis(format="~s", labelExpr="'£' + datum.label", **AX)),
    color=alt.condition(alt.datum.policy == "Highest churn risk", alt.value(BLUE), alt.value(GREY)),
    tooltip=[alt.Tooltip("policy:N", title="Policy"), alt.Tooltip("margin:Q", title="Extra profit (£)", format=",.0f"), alt.Tooltip("contacts:Q", title="Customers", format=",.0f"), alt.Tooltip("outlay:Q", title="Discount (£)", format=",.0f")])
t3 = b3.mark_text(align="left", dx=6, fontSize=12, color=INK).encode(x="margin:Q", text="label:N")
t3b = b3.mark_text(align="left", dx=62, fontSize=11, color=MUTED).encode(x="margin:Q", text="sub:N")
st.altair_chart((c3 + t3 + t3b).properties(height=210).configure_view(stroke=None), use_container_width=True)
st.markdown(f'<div class="take">With the same discount budget, ranking by model risk earns <b>{(me.margin / rc.margin - 1):+.0%}</b> more than picking the customers silent the longest. '
            f'Sending the voucher to everyone costs £{ev.outlay:,.0f} for £{ev.margin:,.0f}. Weekly groups overlap, so do not add them up.</div>', unsafe_allow_html=True)

# ---- 4: what to do ----------------------------------------------------------
step(4, "What do we do next?", "Hand this list to the campaign team, then test the assumptions in a pilot.")
view = off.sort_values("em", ascending=False)[["customer_id", "segment8", "p", "value_v", "em"]].rename(columns={
    "customer_id": "Customer", "segment8": "Group", "p": "Churn risk", "value_v": "Usual order (£)", "em": "Expected profit (£)"})
st.dataframe(view, hide_index=True, height=300, use_container_width=True, column_config={
    "Customer": st.column_config.NumberColumn(format="%d", width="small"),
    "Churn risk": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
    "Usual order (£)": st.column_config.NumberColumn(format="%.0f", width="small"),
    "Expected profit (£)": st.column_config.NumberColumn(format="%.1f", width="small")})
st.download_button("Download voucher list (CSV)", view.to_csv(index=False).encode(), "voucher_list.csv", "text/csv")
st.markdown(f'<div class="take">The response rates and margin are assumptions: the data has no past vouchers. Extra profit stays positive in all three scenarios '
            f'(Light £{scen["Light"]:,.0f}, Base £{scen["Base"]:,.0f}, Deep £{scen["Deep"]:,.0f} per weekly group). '
            'Next: a 6–8 week pilot with a 20% no-voucher control group.</div>', unsafe_allow_html=True)
