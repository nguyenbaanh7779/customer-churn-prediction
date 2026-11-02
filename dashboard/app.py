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


@st.cache_data
def load():
    s = pd.read_parquet(ROOT / "data/results/customer_scores_actions_h90.parquet")
    rec = pd.read_parquet(ROOT / "data/processed/modeling_dataset_h90.parquet",
                          columns=["customer_id", "snapshot_date", "recency_days"])
    seg = pd.read_parquet(ROOT / "data/results/rfm8_segments.parquet", columns=["customer_id", "snapshot_date", "segment8"])
    s = s.merge(rec, on=["customer_id", "snapshot_date"], how="left").merge(seg, on=["customer_id", "snapshot_date"], how="left")
    return s.rename(columns={"churn_probability_calibrated": "p", "churn_actual": "churn"})


def exp_outlay(df, d, e, r):
    """Expected discount cost of contacting a customer: d * V * (p*e + (1-p)*r)."""
    return d * df.value_v * (df.p * e + (1 - df.p) * r)


def select(df, rule, mode, k, budget, d, e, m, r, c, need_positive, score_override=None):
    """Boolean mask of customers who get an offer, chosen separately in each week."""
    em = expected_margin(df.p, df.value_v, d, e, m, r, c)
    score = df.p if rule == "Highest churn risk" else em
    if score_override is not None:
        score, need_positive = score_override, False
    ok = (em > 0) if ((need_positive or rule != "Highest churn risk") and score_override is None) else pd.Series(True, index=df.index)
    work = pd.DataFrame({"w": df.snapshot_date, "s": score.where(ok, -np.inf), "o": exp_outlay(df, d, e, r)})
    work = work.sort_values(["w", "s"], ascending=[True, False])
    g = work.groupby("w")
    if mode == "Share of active customers":
        take = g.cumcount() < np.maximum(np.round(g["s"].transform("size") * k), 1)
    else:
        take = g["o"].cumsum() <= budget
    take &= np.isfinite(work.s)
    return take.reindex(df.index).fillna(False), em


df = load()
test = df[df.split == "test"].copy()
last_date = df[df.split == "deployment"].snapshot_date.max()
cur = df[df.snapshot_date == last_date].copy()

# ---- sidebar controls -------------------------------------------------------
sb = st.sidebar
sb.markdown("## Who gets an offer")
rule = sb.radio("Selection rule", ["Highest churn risk", "Highest expected margin"], label_visibility="collapsed")
need_pos = sb.checkbox("Only if expected margin > 0", value=False, disabled=(rule != "Highest churn risk"))
mode = sb.radio("Limit by", ["Discount budget", "Share of customers"], horizontal=True)
if mode == "Share of customers":
    k, budget, mode = sb.slider("Share contacted", 5, 50, 20, 5, format="%d%%") / 100, None, "Share of active customers"
else:
    k, budget = None, float(sb.slider("Expected discount cost, £ per weekly cohort", 1000, 20000, 7740, 20))

sb.markdown("## Offer scenario")
preset = sb.selectbox("Preset (discount, response)", list(PRESETS), index=1, label_visibility="collapsed")
d0, e0 = PRESETS[preset]
with sb.expander("Fine-tune assumptions"):
    d = st.slider("Discount", 0.05, 0.30, d0, 0.01, format="%.2f")
    e = st.slider("Response of customers who would not buy", 0.05, 0.60, e0, 0.05)
    r = st.slider("Redemption by customers who would buy anyway", 0.10, 0.80, 0.40, 0.05)
    m = st.slider("Gross margin", 0.20, 0.60, 0.40, 0.05)
    c = st.number_input("Contact cost (£)", 0.0, 5.0, 0.50, 0.10)

# ---- current target list ----------------------------------------------------
SEG_ORDER = ["Champions", "Loyal", "Developing", "One-time recent", "Cooling", "VIP at risk"]
SEG_HINT = {"Champions": "top 10% spenders, bought < 90 days ago", "Loyal": "4+ orders, bought < 90 days ago",
            "Developing": "2-3 orders, bought < 90 days ago", "One-time recent": "1 order, < 90 days ago",
            "Cooling": "last order 90-180 days ago", "VIP at risk": "top 10% spenders, silent > 90 days"}
sel, cur["em"] = select(cur, rule, mode, k, budget, d, e, m, r, c, need_pos)
cur["action"] = np.where(sel, "Retain", np.where(cur.p.rank(pct=True) >= 0.80, "Nurture", "No-offer"))
off = cur[sel]
n_act = cur.action.value_counts()
cur["out"] = exp_outlay(cur, d, e, r)

# test-week simulation for this rule (used in step 3)
t_sel, _ = select(test, rule, mode, k, budget, d, e, m, r, c, need_pos)
gain = pd.Series(row_gain(test, d, e, m, r, c), index=test.index)
weeks = test.snapshot_date.nunique()
n_w = t_sel.groupby(test.snapshot_date).sum()
rec_sel, _ = select(test, rule, mode, k, budget, d, e, m, r, c, False, score_override=test.recency_days.fillna(0))
policies = {"Your rule": t_sel, "Longest since purchase": rec_sel,
            "Offer to everyone": pd.Series(True, index=test.index), "No offer": pd.Series(False, index=test.index)}
res = pd.DataFrame([dict(policy=n, margin=gain[mk].sum() / weeks, contacts=mk.sum() / weeks,
                         outlay=(d * test.value_v * np.where(test.churn == 1, e, r))[mk].sum() / weeks) for n, mk in policies.items()])
me, rc, ev = res.iloc[0], res.iloc[1], res.iloc[2]

top = cur[sel].segment8.value_counts()
top_names = [f"{n} ({v})" for n, v in top.head(3).items()]

st.title("Who should get this week's discount voucher?")
st.markdown(f'<div class="sub">Online retailer · list for Monday {last_date:%d %b %Y} · {len(cur):,} active customers scored · '
            f'assumptions: {d:.0%} discount, {e:.0%} of leavers respond, {r:.0%} of loyal buyers redeem, {m:.0%} margin · '
            'simulated results, not measured sales</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="story"><b>In short:</b> send the voucher to the <b>{len(off):,}</b> customers most likely to stop buying
({len(off) / len(cur):.0%} of active customers, about <b>£{exp_outlay(off, d, e, r).sum():,.0f}</b> of discount).
In the 10 test weeks this rule earned about <b>£{me.margin:,.0f}</b> extra profit per week-group, against
<b>£{rc.margin:,.0f}</b> for picking the customers silent the longest, and <b>£{ev.margin:,.0f}</b> for sending the voucher to everyone.</div>
<div class="kpis">
 <div class="kpi"><div class="l">Active customers</div><div class="v">{len(cur):,}</div><div class="n">bought in the last 6 months</div></div>
 <div class="kpi"><div class="l">Get a voucher</div><div class="v">{len(off):,}</div><div class="n">{len(off)/len(cur):.0%} of active customers</div></div>
 <div class="kpi"><div class="l">Discount cost</div><div class="v">£{exp_outlay(off, d, e, r).sum():,.0f}</div><div class="n">expected, this week's list</div></div>
 <div class="kpi"><div class="l">Revenue at risk</div><div class="v">£{(off.value_v * off.p).sum():,.0f}</div><div class="n">order value × chance of leaving</div></div>
 <div class="kpi"><div class="l">Expected extra profit</div><div class="v">£{off.em.sum():,.0f}</div><div class="n">next 90 days, simulated</div></div>
</div>
""", unsafe_allow_html=True)


def step(n, title, sub):
    st.markdown(f'<div class="step"><span class="num">{n}</span><span class="t">{title}</span></div><div class="note" style="margin:0 0 .6rem 38px">{sub}</div>', unsafe_allow_html=True)


AX = dict(labelColor=MUTED, domain=False, ticks=False, grid=True, gridColor=LINE)
YAX = alt.Axis(labelFontSize=12, labelColor=INK, ticks=False, domain=False, labelLimit=220, labelPadding=8)

# ---- step 1 -----------------------------------------------------------------
step(1, "Who is likely to stop buying?", "Average chance of no purchase in the next 90 days, by customer group (Monday " + f"{last_date:%d %b}).")
g1 = cur.groupby("segment8").agg(customers=("p", "size"), risk=("p", "mean")).reindex(SEG_ORDER).dropna().reset_index()
g1["label"] = g1.apply(lambda x: f"{x.risk:.0%}  ·  {int(x.customers):,} customers", axis=1)
g1["hint"] = g1.segment8.map(SEG_HINT)
b1 = alt.Chart(g1).encode(y=alt.Y("segment8:N", sort=SEG_ORDER, title=None, axis=YAX))
c1 = b1.mark_bar(cornerRadiusEnd=4, size=24).encode(
    x=alt.X("risk:Q", title=None, scale=alt.Scale(domain=[0, 1.15]), axis=alt.Axis(format="%", values=[0, .25, .5, .75, 1], **AX)),
    color=alt.condition(alt.datum.risk >= 0.4, alt.value(ORANGE), alt.value("#9ec5f3")),
    tooltip=[alt.Tooltip("segment8:N", title="Group"), alt.Tooltip("hint:N", title="Definition"), alt.Tooltip("customers:Q", title="Customers"), alt.Tooltip("risk:Q", title="Avg risk", format=".0%")])
t1 = b1.mark_text(align="left", dx=6, fontSize=12, color=INK).encode(x="risk:Q", text="label:N")
st.altair_chart((c1 + t1).properties(height=36 * len(g1) + 10).configure_view(stroke=None), use_container_width=True)
hi = g1.sort_values("risk", ascending=False).iloc[0]
lo = g1.sort_values("risk").iloc[0]
st.markdown(f'<div class="take">Risk climbs as customers go quiet or have only ordered once: <b>{hi.segment8}</b> customers are at {hi.risk:.0%}, '
            f'<b>{lo.segment8}</b> only {lo.risk:.0%}.</div>', unsafe_allow_html=True)

# ---- step 2 -----------------------------------------------------------------
step(2, "Who gets a voucher?", "Customers with the highest risk, until the weekly discount budget runs out.")
g2 = cur.groupby("segment8").agg(total=("p", "size"), offered=("action", lambda a: (a == "Retain").sum()), margin=("em", lambda x: x[cur.loc[x.index, "action"] == "Retain"].sum())).reindex(SEG_ORDER).dropna().reset_index()
g2["not_offered"] = g2.total - g2.offered
long = g2.melt("segment8", ["offered", "not_offered"], var_name="k", value_name="n")
long["k"] = long.k.map({"offered": "Gets a voucher", "not_offered": "No voucher"})
c2 = alt.Chart(long).mark_bar(size=24).encode(
    y=alt.Y("segment8:N", sort=SEG_ORDER, title=None, axis=YAX),
    x=alt.X("n:Q", stack="zero", title=None, axis=alt.Axis(format="~s", **AX)),
    color=alt.Color("k:N", scale=alt.Scale(domain=["Gets a voucher", "No voucher"], range=[BLUE, "#e3e2dc"]), legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
    order=alt.Order("k:N", sort="descending"),
    tooltip=[alt.Tooltip("segment8:N", title="Group"), alt.Tooltip("k:N", title=""), alt.Tooltip("n:Q", title="Customers", format=",")])
st.altair_chart(c2.properties(height=36 * len(g2) + 40).configure_view(stroke=None), use_container_width=True)
big = g2.sort_values("offered", ascending=False)
share_top2 = big.offered.head(2).sum() / max(big.offered.sum(), 1)
loyal_off = int(g2[g2.segment8.isin(["Champions", "Loyal"])].offered.sum())
st.markdown(f'<div class="take">Offers go mainly to <b>{big.iloc[0].segment8}</b> and <b>{big.iloc[1].segment8}</b> customers ({share_top2:.0%} of the list). '
            f'Champions and Loyal get only {loyal_off}: they mostly buy anyway, so a voucher would just cost margin.</div>', unsafe_allow_html=True)

# ---- step 3 -----------------------------------------------------------------
step(3, "Does it pay off?", "Replay on the 10 test weeks (Jul–Sep 2011) that the model never saw: extra profit after discounts, per weekly group.")
res["label"] = res.margin.map(lambda v: f"£{v:,.0f}")
res["sub"] = res.apply(lambda x: f"{x.contacts:,.0f} customers · £{x.outlay:,.0f} discount", axis=1)
order = list(res.policy)
b3 = alt.Chart(res).encode(y=alt.Y("policy:N", sort=order, title=None, axis=YAX))
c3 = b3.mark_bar(cornerRadiusEnd=4, size=26).encode(
    x=alt.X("margin:Q", title=None, scale=alt.Scale(domain=[0, float(res.margin.max()) * 1.6], nice=False), axis=alt.Axis(format="~s", labelExpr="'£' + datum.label", **AX)),
    color=alt.condition(alt.datum.policy == "Your rule", alt.value(BLUE), alt.value(GREY)),
    tooltip=[alt.Tooltip("policy:N", title="Policy"), alt.Tooltip("margin:Q", title="Extra profit (£)", format=",.0f"), alt.Tooltip("contacts:Q", title="Customers", format=",.0f"), alt.Tooltip("outlay:Q", title="Discount (£)", format=",.0f")])
t3 = b3.mark_text(align="left", dx=6, fontSize=12, color=INK).encode(x="margin:Q", text=alt.Text("label:N"))
t3b = b3.mark_text(align="left", dx=62, fontSize=11, color=MUTED).encode(x="margin:Q", text="sub:N")
st.altair_chart((c3 + t3 + t3b).properties(height=190).configure_view(stroke=None), use_container_width=True)
st.markdown(f'<div class="take">Same discount budget, but picking by model risk earns <b>{(me.margin / rc.margin - 1):+.0%}</b> versus “longest since purchase”. '
            f'Giving the voucher to everyone costs £{ev.outlay:,.0f} in discounts for £{ev.margin:,.0f}. '
            'Weekly groups overlap, so do not add them up.</div>', unsafe_allow_html=True)

# ---- step 4 -----------------------------------------------------------------
step(4, "What if our assumptions are wrong?", "The data has no past vouchers, so response and margin are guesses. Same rule, three scenarios.")
sc = []
for name, (dd, ee) in PRESETS.items():
    ts, _ = select(test, rule, mode, k, budget, dd, ee, m, r, c, need_pos)
    gg = pd.Series(row_gain(test, dd, ee, m, r, c), index=test.index)[ts]
    sc.append({"Scenario": name.split(" (")[0], "What we assume": f"{dd:.0%} off, {ee:.0%} of leavers respond",
               "Customers / week": ts.sum() / weeks, "Discount (£)": (dd * test.value_v * np.where(test.churn == 1, ee, r))[ts].sum() / weeks,
               "Extra profit (£)": gg.sum() / weeks})
sc = pd.DataFrame(sc)
st.dataframe(sc, hide_index=True, use_container_width=True, column_config={
    "Customers / week": st.column_config.NumberColumn(format="%.0f"), "Discount (£)": st.column_config.NumberColumn(format="%.0f"),
    "Extra profit (£)": st.column_config.NumberColumn(format="%.0f")})
st.markdown('<div class="take">Profit stays positive in all three. The real pilot should replace these guesses with measured response.</div>', unsafe_allow_html=True)

# ---- step 5: the list --------------------------------------------------------
step(5, "The list to act on", "Download and hand to the campaign team.")
choice = st.segmented_control("Action", ["Retain", "Nurture", "No-offer"], default="Retain", label_visibility="collapsed") or "Retain"
cap = {"Retain": "Send the voucher.", "Nurture": "Higher risk but outside the budget: light-touch contact, no discount.", "No-offer": "Leave alone."}[choice]
st.markdown(f'<div class="note" style="margin:0 0 .4rem">{choice}: {cap} ({n_act.get(choice, 0):,} customers)</div>', unsafe_allow_html=True)
view = cur[cur.action == choice].sort_values("em", ascending=False)[["customer_id", "segment8", "customer_type", "p", "value_v", "em"]]
view = view.rename(columns={"customer_id": "Customer", "segment8": "Group", "customer_type": "Type", "p": "Churn risk", "value_v": "Usual order (£)", "em": "Expected profit (£)"})
st.dataframe(view, hide_index=True, height=360, use_container_width=True, column_config={
    "Customer": st.column_config.NumberColumn(format="%d", width="small"),
    "Churn risk": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
    "Usual order (£)": st.column_config.NumberColumn(format="%.0f", width="small"),
    "Expected profit (£)": st.column_config.NumberColumn(format="%.1f", width="small")})
st.download_button(f"Download {choice} list (CSV)", view.to_csv(index=False).encode(), f"{choice.lower()}_list.csv", "text/csv")

st.markdown('<div class="note" style="margin-top:2rem;border-top:1px solid #e5e5e0;padding-top:.8rem">'
            'Churn = no purchase in the next 90 days; risk from the team’s calibrated logistic regression. Groups follow the RFM rules in the segmentation notebook '
            '(Lapsed and Dormant customers are not scored). Response, redemption and margin are assumptions: the dataset has no promotions or costs.</div>', unsafe_allow_html=True)
