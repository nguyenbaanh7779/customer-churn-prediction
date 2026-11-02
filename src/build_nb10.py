"""Generate notebooks/10_policy_analysis.ipynb (run: python src/build_nb10.py)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 10. Policy Analysis: Risk × Value Targeting

Builds on notebooks 07-09 (unchanged). Adds what the evaluation chapter was
missing: **validation predictions from a train-only model**, a
**risk × value diagnosis** of why churn-only targeting trailed the recency
rule, **segment-level actions**, and **policy simulation under Strategy v2**
(Sections 7.7-7.9): E[ΔM] = p·e·V·(m−d) − (1−p)·r·V·d − c.

All £ figures are **scenario estimates** (no treatment/control data); policy
*ranking* is more reliable than absolute levels. Amounts are reported **per
week** (mean over scoring weeks), not summed over overlapping 90-day windows.
""")
code("""
import sys, warnings
sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from policy_lib import *

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e5e5e0"})
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"

# Business assumptions (Strategy v2, Section 2.10)
BASE = dict(d=0.10, e=0.30, m=0.40, r=0.40, c=0.50)
SCENARIOS = {"Light": (0.05, 0.20), "Base": (0.10, 0.30), "Deep": (0.20, 0.40)}
K_GRID = [0.10, 0.20, 0.30, 0.40, 0.50]
""")
md("## 10.1 Load and integrity checks")
code("""
data = pd.read_parquet("../data/processed/modeling_dataset_h90.parquet")
scores = pd.read_parquet("../data/results/churn_scores_h90.parquet")  # notebook 09 output
key = ["customer_id", "snapshot_date"]

assert not data.duplicated(key).any(), "duplicate (customer, snapshot) in dataset"
assert not scores.duplicated(key).any(), "duplicate (customer, snapshot) in scores"
test_raw = data[data.split == "test"]
chk = test_raw.merge(scores[scores.split == "test"][key + ["churn_probability", "churn"]],
                     on=key, how="outer", indicator=True, suffixes=("", "_s"))
print("test rows dataset / scores:", len(test_raw), (scores.split == "test").sum())
print("merge status:", chk["_merge"].value_counts().to_dict())
print("label mismatch:", int((chk.churn != chk.churn_s).sum()))
print("model_version in scores:", scores.model_version.unique())
print(data.groupby("split").agg(rows=("churn", "size"), weeks=("snapshot_date", "nunique"),
      customers=("customer_id", "nunique"), churn=("churn", "mean")).round(3))
""")
md("""
## 10.2 Train-only predictions and calibration check

The scores from notebook 09 come from a model refit on train **+ validation**,
so they cannot be used to choose a policy on validation. Here the same recipe
is fitted on **train only**; validation and test predictions are out-of-sample.
""")
code("""
train = data[data.split == "train"]
val = data[data.split == "validation"].copy()
test = data[data.split == "test"].copy()
predict, lr = fit_train_only(train)
val["p"], test["p"] = predict(val), predict(test)

rows = {}
for name, df in [("validation", val), ("test", test)]:
    rows[name] = dict(churn_rate=df.churn.mean(), mean_p=df.p.mean(),
                      roc_auc=roc_auc_score(df.churn, df.p),
                      pr_auc=average_precision_score(df.churn, df.p),
                      brier=brier_score_loss(df.churn, df.p),
                      brier_base_rate=brier_score_loss(df.churn, np.full(len(df), df.churn.mean())))
pd.DataFrame(rows).T.round(3)
""")
code("""
# Calibration: fit on validation, check on test
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
iso = IsotonicRegression(out_of_bounds="clip").fit(val.p, val.churn)
platt = LogisticRegression(C=1e6).fit(logit(val.p).values.reshape(-1, 1), val.churn)
test["p_iso"] = iso.predict(test.p)
test["p_platt"] = platt.predict_proba(logit(test.p).values.reshape(-1, 1))[:, 1]
pd.DataFrame({c: dict(mean_p=test[c].mean(), brier=brier_score_loss(test.churn, test[c]),
                      roc_auc=roc_auc_score(test.churn, test[c]))
              for c in ["p", "p_platt", "p_iso"]}).T.round(4)
""")
code("""
# Why calibration on validation does not transfer: the base churn rate moves with the season
lab = data[data.labelable]
monthly = lab.groupby(lab.snapshot_date.dt.to_period("M")).churn.mean()
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.bar(monthly.index.astype(str), monthly.values, color=BLUE)
ax.set_ylabel("Churn rate (90-day)"); ax.set_title("Churn rate by snapshot month")
plt.xticks(rotation=45); fig.tight_layout()
fig.savefig("../report/figures/ch10_churn_by_month.png", dpi=150, bbox_inches="tight"); plt.show()
""")
md("""
**Reading.** Validation (Feb-Apr 2011) has a 53% churn rate, test (Jul-Sep) 40%.
A calibrator fitted on validation learns that higher base rate and *worsens*
Brier on test. Within a period the train-only model is roughly calibrated;
across seasons the level shifts. The ranking is stable (AUC ≈ 0.75-0.78), so
policies below use the **raw** probability `p`; absolute £ should be read with
this in mind.
""")
md("""
## 10.3 Churn-only vs recency: who do they pick?

Top 10% each week by (a) model probability and (b) recency (days since last
purchase). Groups: picked by both, only by the model, only by recency.
""")
code("""
K = 0.10
def overlap_profile(df):
    mm, mr = top_k_mask(df, "p", K), top_k_mask(df, "recency_days", K)
    groups = {"both": mm & mr, "only_model": mm & ~mr, "only_recency": mr & ~mm, "all active": df.index == df.index}
    out = {}
    for g, m in groups.items():
        x = df[m]
        out[g] = dict(rows=len(x), churn_rate=x.churn.mean(), mean_p=x.p.mean(),
                      value_v_mean=x.value_v.mean(), value_v_median=x.value_v.median(),
                      recency_median=x.recency_days.median(), frequency_median=x.frequency_so_far.median(),
                      one_time_share=(x.segment == "one-time").mean())
    return pd.DataFrame(out).T, (mm & mr).sum() / mm.sum()
for name, df in [("validation", val), ("test", test)]:
    prof, ov = overlap_profile(df)
    print(f"{name}: share of model top-{K:.0%} also in recency top-{K:.0%} = {ov:.1%}")
    display(prof.round(3))
""")
md("""
**Reading.** The two rules share only about a third of their picks. The
model's exclusive picks are low-value, low-frequency customers (many
one-time buyers) with a *higher* churn rate than recency's exclusive picks,
while recency's exclusive picks are repeat customers with roughly 2× the
value. Churn-only targeting finds more churners but with less money behind
each; value-weighted rules correct for it.
""")
code("""
# Same question from the outcome side: where does each rule's 'catch' come from, and how concentrated is value?
mm = top_k_mask(test, "p", K); mr = top_k_mask(test, "recency_days", K)
rev = lambda m: test[m & (test.churn == 1)].value_v.sum()
print(f"Revenue-at-risk among captured churners (sum of value_v, weekly rows): model {rev(mm):,.0f} | recency {rev(mr):,.0f}")
print(f"Churners captured: model {int((mm & (test.churn==1)).sum()):,} | recency {int((mr & (test.churn==1)).sum()):,}")
top1 = test.groupby("customer_id").avg_order_value_so_far.max().nlargest(int(0.01 * test.customer_id.nunique())).index
print("top-1% customers by raw AOV (wholesale-like):", len(top1), "| share of test rows:", round(test.customer_id.isin(top1).mean(), 3),
      "| share of raw value among row-weighted revenue:", round(test[test.customer_id.isin(top1)].avg_order_value_so_far.sum() / test.avg_order_value_so_far.sum(), 3))
""")
md("""
## 10.4 RFM segments → actions

Per-customer realized margin if a segment is offered (base scenario), using
actual labels. The action is chosen on **validation** and then checked on
**test**; segment names are not used to assign actions.
""")
code("""
def row_gain(df, d, e, m, r, c, vcol="value_v"):
    v = df[vcol]
    return np.where(df.churn == 1, e * v * (m - d), -r * v * d) - c

val["gain"] = row_gain(val, **{k: BASE[k] for k in "d e m r c".split()})
test["gain"] = row_gain(test, **{k: BASE[k] for k in "d e m r c".split()})
def seg_table(df):
    return df.groupby("rfm_segment").agg(share=("gain", lambda s: len(s) / len(df)), churn=("churn", "mean"),
            mean_p=("p", "mean"), value_v=("value_v", "mean"), gain_per_customer=("gain", "mean"))
seg_val, seg_test = seg_table(val), seg_table(test)
seg = seg_val.add_suffix("_val").join(seg_test.add_suffix("_test"))
seg["action"] = np.where((seg.gain_per_customer_val > 0) & (seg.gain_per_customer_test > 0), "Offer",
                  np.where((seg.gain_per_customer_val < 0) & (seg.gain_per_customer_test < 0), "No discount", "Unstable: low-cost contact"))
seg.round(3)
""")
md("""
**Reading.** Offers pay off for At Risk, Hibernating and Potential (high churn
probability → recovered margin outweighs discount given to buyers anyway).
Champions and Loyal lose margin per customer (mostly would-have-bought-anyway
→ cannibalization). **Can't Lose** — whose name suggests a rescue offer — has a
churn rate of only 29-39%, so a discount loses or barely breaks even; treat it
with a low-cost contact, not a discount. The segment name alone would have
given the wrong action.
""")
md("""
## 10.5 Policy simulation (P0-P6)

Weekly Top-K from the Scoring Population (every active customer each week).
`E[ΔM]` uses the train-only probability `p`. P6 additionally requires
E[ΔM] > 0. P4 targets At Risk + Can't Lose (strategy definition) and P4b the
segments chosen by validation in 10.4. Reported: **weekly net margin, mean ± std over weeks**.
""")
code("""
def policy_masks(df, k, scen):
    d, e, m, r, c = scen
    df = df.assign(em=expected_margin(df.p, df.value_v, d, e, m, r, c), rand=np.random.default_rng(0).uniform(size=len(df)))
    masks = {
        "P0 No offer": pd.Series(False, index=df.index),
        "P1 Blanket": pd.Series(True, index=df.index),
        "P2 Random": top_k_mask(df, "rand", k),
        "P3 Recency": top_k_mask(df, "recency_days", k),
        "P4 RFM At Risk+Can't Lose": df.rfm_segment.isin(["At Risk", "Can't Lose"]),
        "P4b RFM validated segments": df.rfm_segment.isin(seg.index[seg.action == "Offer"]),
        "P5 Model risk-only": top_k_mask(df, "p", k),
        "P6 Model value-aware": top_k_mask(df, "em", k) & (df.em > 0),
    }
    return df, masks

def weekly_margin(df, mask, scen):
    d, e, m, r, c = scen
    g = pd.Series(row_gain(df, d, e, m, r, c), index=df.index)
    weeks = df.snapshot_date.unique()
    return g[mask].groupby(df.snapshot_date[mask]).sum().reindex(weeks, fill_value=0)

def policy_table(df, k, scen):
    df, masks = policy_masks(df, k, scen)
    rows = {}
    for name, mk in masks.items():
        w = weekly_margin(df, mk, scen)
        n_week = mk.groupby(df.snapshot_date).sum()
        rows[name] = dict(customers_per_week=n_week.mean(), margin_week_mean=w.mean(), margin_week_std=w.std(), worst_week=w.min(),
                          precision=df[mk].churn.mean() if mk.any() else np.nan,
                          would_be_buyers_share=1 - df[mk].churn.mean() if mk.any() else np.nan)
    return pd.DataFrame(rows).T

SC_BASE = (BASE["d"], BASE["e"], BASE["m"], BASE["r"], BASE["c"])
K_OP = 0.20
print("VALIDATION (selection), base scenario, K=20%"); display(policy_table(val, K_OP, SC_BASE).round(2))
print("TEST (final evaluation), base scenario, K=20%"); display(policy_table(test, K_OP, SC_BASE).round(2))
""")
code("""
# Margin vs K for each ranking policy, validation and test
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
names = ["P2 Random", "P3 Recency", "P5 Model risk-only", "P6 Model value-aware"]
colors = [GREY, ORANGE, BLUE, GREEN]
curves = {}
for ax, (split, df) in zip(axes, [("validation", val), ("test", test)]):
    for n, col in zip(names, colors):
        ys = [policy_table(df, k, SC_BASE).loc[n, "margin_week_mean"] for k in K_GRID]
        curves[(split, n)] = ys
        ax.plot([int(k * 100) for k in K_GRID], ys, marker="o", label=n, color=col)
    ax.set_title(f"Weekly margin vs K ({split}, base scenario)"); ax.set_xlabel("K (% of active customers)")
axes[0].set_ylabel("£ per week"); axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("../report/figures/ch10_policy_vs_k.png", dpi=150, bbox_inches="tight"); plt.show()
pd.DataFrame({f"K={int(k*100)}%": {n: curves[("test", n)][i] for n in names} for i, k in enumerate(K_GRID)}).round(0)
""")
md("""
**Reading.** Under the Strategy v2 margin model (cannibalization proportional
to V·d), the value-aware policy P6 has the highest weekly margin at every K on
both validation and test. Risk-only P5 is still below the recency rule at
K=10% and overtakes it at K≈20% (test) or K≈30% (validation).

**Caveat on blanket (P1).** Blanket contacts every active customer (~2,400 per
week), so its total can exceed a K=20% policy when the discount is cheap
(Light scenario) even though its margin *per contact* is far lower
(≈£1.8 vs ≈£16 for P6 at base). Compare policies at equal budget (same K) and
look at margin per contact; blanket becomes strongly negative in the Deep scenario.
""")
md("""
### 10.5b Where does the margin come from? Decomposition and break-even margin

Per policy at K=20%, base scenario, **test**, £ per week. `revenue_preserved`
is the expected revenue kept from targeted churners (e · V), the only revenue
effect in the model. The gross margin rate `m` is an **assumption**
(the dataset has revenue only, no cost or margin); `break_even_margin_rate`
is the gross margin at which the policy's net margin is zero: the policy pays
off if the true margin exceeds it.
""")
code("""
rows = {}
_, masks = policy_masks(test, K_OP, SC_BASE)
for name, mk in masks.items():
    if name == "P0 No offer": continue
    rows[name] = margin_breakdown(test, mk, SC_BASE)
decomp = pd.DataFrame(rows).T
display(decomp.round(2))
d_, e_, m_, r_, c_ = SC_BASE
chk = weekly_margin(test, masks["P6 Model value-aware"], SC_BASE).mean()
assert abs(chk - decomp.loc["P6 Model value-aware", "net_margin_week"]) < 1e-6
""")
code("""
# Break-even margin by scenario (K=20%): lower is safer
rows = []
for sname, (d, e) in SCENARIOS.items():
    scen = (d, e, BASE["m"], BASE["r"], BASE["c"])
    _, mk = policy_masks(test, K_OP, scen)
    rows.append({"scenario": sname, **{n: margin_breakdown(test, m_, scen)["break_even_margin_rate"] for n, m_ in mk.items() if n != "P0 No offer"}})
pd.DataFrame(rows).set_index("scenario").round(3)
""")
md("""
## 10.6 Selection on validation, one evaluation on test

Rule: among P2/P3/P5/P6, choose the policy with the highest weekly validation
margin per scenario; then report it on test (no further tuning).
""")
code("""
rows = []
for sname, (d, e) in SCENARIOS.items():
    scen = (d, e, BASE["m"], BASE["r"], BASE["c"])
    for k in K_GRID:
        tv = policy_table(val, k, scen).loc[names, "margin_week_mean"]
        tt = policy_table(test, k, scen)
        chosen = tv.idxmax()
        rows.append(dict(scenario=sname, K=f"{int(k*100)}%", chosen_on_validation=chosen,
                         test_margin_week=tt.loc[chosen, "margin_week_mean"],
                         recency_test=tt.loc["P3 Recency", "margin_week_mean"],
                         blanket_test=tt.loc["P1 Blanket", "margin_week_mean"],
                         best_in_test=tt.loc[names, "margin_week_mean"].idxmax()))
sel = pd.DataFrame(rows)
sel["vs_recency"] = sel.test_margin_week - sel.recency_test
sel["vs_blanket"] = sel.test_margin_week - sel.blanket_test
sel.round(0)
""")
md("""
## 10.7 Sensitivity and wholesale influence

(a) Scenario grid d/e × m × r at K=20% (test). (b) Value capped at P99 vs
uncapped raw AOV vs dropping the top-1% customers. (c) Suppression run:
customers contacted in the previous 4 weeks are not contacted again.
""")
code("""
rows = []
for sname, (d, e) in SCENARIOS.items():
    for m in [0.3, 0.4, 0.5]:
        for r in [0.2, 0.4, 0.6]:
            t = policy_table(test, K_OP, (d, e, m, r, BASE["c"])).loc[names + ["P1 Blanket"], "margin_week_mean"]
            rows.append(dict(scenario=sname, m=m, r=r, **t.round(0).to_dict(), winner=t[names].idxmax()))
grid = pd.DataFrame(rows)
display(grid)
print("Winner counts over the 27-cell grid:", grid.winner.value_counts().to_dict())
print("Cells where P6 > P3 Recency:", int((grid["P6 Model value-aware"] > grid["P3 Recency"]).sum()), "/ 27;",
      "P6 > blanket:", int((grid["P6 Model value-aware"] > grid["P1 Blanket"]).sum()), "/ 27")
""")
code("""
def with_value(df, col):
    x = df.copy(); x["value_v"] = x[col]; return x
res = {}
variants = {"capped (P99, default)": test, "uncapped raw AOV": with_value(test, "avg_order_value_so_far"),
            "drop top-1% customers": test[~test.customer_id.isin(top1)]}
for lab, df in variants.items():
    t = policy_table(df, K_OP, SC_BASE).loc[names, "margin_week_mean"]
    res[lab] = t
pd.DataFrame(res).T.round(0)
""")
code("""
# Suppression: a customer contacted in the last S weeks is removed from the next week's pool
def suppressed_run(df, policy_col, k, scen, S=4):
    d, e, m, r, c = scen
    df = df.assign(em=expected_margin(df.p, df.value_v, d, e, m, r, c))
    weeks = sorted(df.snapshot_date.unique()); contacted = {}  # customer -> last contact week index
    total, n_contacts, uniq = [], [], set()
    for i, w in enumerate(weeks):
        wk = df[df.snapshot_date == w]
        elig = wk[[(contacted.get(cid, -99) < i - S) for cid in wk.customer_id]]
        n = max(int(round(len(wk) * k)), 1)  # budget fixed on the full pool
        pick = elig.nlargest(n, policy_col)
        if policy_col == "em": pick = pick[pick.em > 0]
        for cid in pick.customer_id: contacted[cid] = i
        uniq.update(pick.customer_id); n_contacts.append(len(pick))
        total.append(pd.Series(row_gain(pick, d, e, m, r, c)).sum())
    return np.mean(total), np.mean(n_contacts), len(uniq)

rows = {}
for lab, col in [("P3 Recency", "recency_days"), ("P5 Model risk-only", "p"), ("P6 Model value-aware", "em")]:
    base_m = policy_table(test, K_OP, SC_BASE).loc[lab, "margin_week_mean"]
    sm, sc_, su = suppressed_run(test, col, K_OP, SC_BASE)
    rows[lab] = dict(no_suppression_week=base_m, suppression_4w_week=sm, contacts_per_week=sc_, unique_customers_contacted=su)
pd.DataFrame(rows).T.round(0)
""")
md("""
**Multi-week note.** Weekly Top-K re-selects the same at-risk customers
week after week. The weekly figures above are per-week campaign margins; the
sum over 10 weeks overstates a real campaign because (i) the 90-day windows
overlap and (ii) the same customer is counted repeatedly. The suppression run
shows the effect of not re-contacting within 4 weeks.
""")
md("## 10.8 Customer-level recommendation table")
code("""
# Action rule (initial, configurable): Retain = P6 selected at K; Nurture = high risk but E[dM] <= 0; No-offer = rest
def build_actions(df, k, scen):
    d, e, m, r, c = scen
    df = df.assign(expected_margin=expected_margin(df.p, df.value_v, d, e, m, r, c))
    sel_k = top_k_mask(df, "expected_margin", k) & (df.expected_margin > 0)
    high_risk = df.groupby("snapshot_date")["p"].rank(pct=True) >= 0.8
    df["action"] = np.where(sel_k, "Retain", np.where(high_risk, "Nurture", "No-offer"))
    df["rank_by_margin"] = df.groupby("snapshot_date").expected_margin.rank(ascending=False, method="first").astype(int)
    return df

actions = build_actions(test, K_OP, SC_BASE)
print(actions.groupby("action").agg(rows=("churn", "size"), churn=("churn", "mean"), value_v=("value_v", "mean"), p=("p", "mean")).round(3))
print("Realized weekly margin of the action list (Retain only):",
      round(weekly_margin(actions, actions.action == "Retain", SC_BASE).mean(), 0))
cols = ["customer_id", "snapshot_date", "split", "p", "value_v", "rfm_segment", "expected_margin", "rank_by_margin", "action", "churn"]
out = pd.concat([build_actions(val, K_OP, SC_BASE).assign(split="validation"), actions.assign(split="test")])[cols]
out = out.rename(columns={"p": "churn_probability"})
out["model_version"] = "lr_extended_c0.01_weighted_h90_trainonly"
out.to_parquet("../data/results/customer_analysis_h90.parquet", index=False)
print("saved customer_analysis_h90.parquet:", out.shape)
""")
md("""
## 10.9 Notes and caveats

* No treatment/control data: all margins are scenario estimates. Policy *ranking*
  across 27 assumption cells is the robust output; absolute £ are not ROI.
* P6 ranks with the same `e`, `r`, `m`, `d` assumptions that the margin
  evaluation uses; the ranking advantage still depends on real labels and V.
* Probabilities are not recalibrated (10.2); the base rate shifts with season.
* The test window (Jul-Sep 2011) has the lowest churn of the year and no Christmas peak.
""")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/10_policy_analysis.ipynb")
