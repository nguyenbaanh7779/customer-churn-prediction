"""Generate notebooks/13_equal_budget.ipynb (run: python src/build_nb13.py)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 13. Equal-Budget Comparison and One-at-a-Time Scenarios

Notebooks 10-12 compare policies with the **same number of contacts**. Contacts are
not the real constraint: a discount costs money in proportion to the customer's
value. This notebook compares policies at the **same expected discount spend** and
varies one scenario parameter at a time. Its tables feed the executive brief
(`data/results/budget_comparison.csv`, `equal_budget_scenarios.csv`, `one_at_a_time.csv`).

* Test weeks only (Jul-Sep 2011, 10 weekly snapshots), calibrated probability from
  the modeling team, value `value_v`, Base scenario unless stated.
* **Unit of every margin figure:** simulated net margin over the 90 days after one
  weekly selection, averaged over the ten weekly cohorts. Cohorts overlap in time,
  so the figures are not added into a yearly profit.
* **Budget** = expected discount cost of the proposed list (Retain, K=20%):
  `d * V * (p*e + (1-p)*r)` summed per week. Each alternative takes customers in
  its own order until the same expected cost is used. Selection uses only
  expected cost; realized margin uses the actual labels.
""")
code("""
import sys, warnings
sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from policy_lib import expected_margin, row_gain, top_k_mask

BASE = dict(d=.10, e=.30, m=.40, r=.40, c=.50)
K = .20

a = pd.read_parquet("../data/results/customer_scores_actions_h90.parquet")
rec = pd.read_parquet("../data/processed/modeling_dataset_h90.parquet", columns=["customer_id", "snapshot_date", "recency_days"])
te = a[a.split == "test"].merge(rec, on=["customer_id", "snapshot_date"], how="left", validate="one_to_one")
te["churn"] = te.churn_actual
te = te.reset_index(drop=True)
weeks = te.snapshot_date.nunique()
p = te.churn_probability_calibrated
assert weeks == 10 and te.recency_days.notna().all()
print(len(te), "rows,", weeks, "weekly cohorts")
""")
md("## 13.1 Helpers")
code("""
def gain(s):
    return row_gain(te, s["d"], s["e"], s["m"], s["r"], s["c"], "value_v")

def exp_outlay(s):
    \"\"\"Expected discount cost of contacting a customer: d * V * (p*e + (1-p)*r).\"\"\"
    return s["d"] * te.value_v * (p * s["e"] + (1 - p) * s["r"])

def real_outlay(s):
    return s["d"] * te.value_v * np.where(te.churn == 1, s["e"], s["r"])

def retain_mask(s):
    em = expected_margin(p, te.value_v, s["d"], s["e"], s["m"], s["r"], s["c"])
    return top_k_mask(te.assign(em=em), "em", K) & (em > 0)

def fill_budget(score, budget_per_week, s):
    \"\"\"Per week, take customers by descending score until expected outlay reaches the budget.\"\"\"
    out = pd.Series(False, index=te.index)
    eo = exp_outlay(s)
    for _, g in te.groupby("snapshot_date"):
        order = g.assign(sc=score[g.index]).sort_values("sc", ascending=False)
        out[order.index[eo[order.index].cumsum() <= budget_per_week]] = True
    return out

def summarize(mask, s):
    g = pd.Series(gain(s), index=te.index)[mask]
    wk = g.groupby(te.snapshot_date[mask]).sum().reindex(te.snapshot_date.unique(), fill_value=0)
    outlay = real_outlay(s)[mask].sum() / weeks
    return dict(contacts_week=mask.sum() / weeks, net_margin_90d=wk.mean(), discount_outlay=outlay,
                margin_per_discount_gbp=wk.mean() / outlay if outlay else np.nan,
                margin_per_contact=g.sum() / max(mask.sum(), 1))
""")
md("""
## 13.2 Same expected discount budget (Base scenario)

Budget = the proposed list's expected weekly discount cost. All rows spend about the
same discount; they differ in how many customers that money reaches.
""")
code("""
ret = retain_mask(BASE)
B = exp_outlay(BASE)[ret].sum() / weeks
em = expected_margin(p, te.value_v, BASE["d"], BASE["e"], BASE["m"], BASE["r"], BASE["c"])
roi = em / exp_outlay(BASE)
rand = pd.Series(np.random.default_rng(0).uniform(size=len(te)), index=te.index)
rows = {
    "Proposed: highest expected margin (top 20%)": ret,
    "Same budget, random": fill_budget(rand, B, BASE),
    "Same budget, highest churn risk": fill_budget(p, B, BASE),
    "Same budget, longest since purchase": fill_budget(te.recency_days, B, BASE),
    "Same budget, highest expected margin per £ discount": fill_budget(roi.where(em > 0, -1), B, BASE),
    "Same budget, highest expected margin (no 20% cap)": fill_budget(em.where(em > 0, -1e9), B, BASE),
}
res = pd.DataFrame({k: summarize(v, BASE) for k, v in rows.items()}).T
res.insert(0, "budget_gbp_per_decision", B)
res.to_csv("../data/results/budget_comparison.csv")
res.round(2)
""")
md("""
Reading: ranking by calibrated churn risk is what creates the value (+51% over
longest-since-purchase at the same spend). Adding customer value does **not** raise
total margin at equal spend: the expected-margin list reaches fewer customers (485
vs 780) for roughly the same margin per £ of discount. It is the lighter-touch option.
""")
md("""
## 13.3 Equal-budget check in the three scenarios

Budget is recomputed per scenario from that scenario's proposed list. The weekly
win count is paired by week.
""")
code("""
rows3 = []
for name, (d, e) in {"Light": (.05, .20), "Base": (.10, .30), "Deep": (.20, .40)}.items():
    s = {**BASE, "d": d, "e": e}
    ret_s = retain_mask(s)
    B_s = exp_outlay(s)[ret_s].sum() / weeks
    risk_s = fill_budget(p, B_s, s)
    g = pd.Series(gain(s), index=te.index)
    wk = lambda mk: g[mk].groupby(te.snapshot_date[mk]).sum().reindex(te.snapshot_date.unique(), fill_value=0)
    diff = wk(ret_s) - wk(risk_s)
    rows3.append(dict(scenario=name, budget=B_s, proposed=wk(ret_s).mean(), risk_same_budget=wk(risk_s).mean(),
                      weeks_proposed_wins=int((diff > 0).sum()), weeks=len(diff),
                      risk_contacts=risk_s.sum() / weeks, proposed_contacts=ret_s.sum() / weeks))
eqb = pd.DataFrame(rows3)
eqb.to_csv("../data/results/equal_budget_scenarios.csv", index=False)
eqb.round(1)
""")
md("""
## 13.4 One parameter at a time

Vary a single parameter, others at Base. Both lists are top 20% by their rule
(equal contacts), so this complements 13.2: it shows whether the result is positive
and how the gap between the lists moves.
""")
code("""
out = []
grids = {"d": [.05, .10, .20], "e": [.20, .30, .40], "r": [.20, .40, .60], "m": [.30, .40, .50]}
for par, vals in grids.items():
    for v in vals:
        s = {**BASE, par: v}
        a_, b_ = summarize(retain_mask(s), s), summarize(top_k_mask(te, "churn_probability_calibrated", K), s)
        out.append(dict(parameter=par, value=v, proposed=a_["net_margin_90d"], risk_only=b_["net_margin_90d"],
                        proposed_outlay=a_["discount_outlay"], risk_outlay=b_["discount_outlay"],
                        proposed_per_gbp=a_["margin_per_discount_gbp"], risk_per_gbp=b_["margin_per_discount_gbp"]))
oat = pd.DataFrame(out)
oat.to_csv("../data/results/one_at_a_time.csv", index=False)
oat.round(2)
""")
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/13_equal_budget.ipynb")
