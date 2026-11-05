"""Generate notebooks/12_action_table.ipynb (run: python src/build_nb12.py)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 12. Customer Action Table (Deliverable)

Strategy 8.3-8.4. Takes the modeling team's scores
(`churn_probability_calibrated`, `churn_model_h90_calibrated`), adds customer
value `value_v`, RFM segment and the Strategy v2 decision rule, and produces:

1. the **customer-level score table with actions** (Retain / Nurture / No-offer),
2. its realized margin on the **test** weeks (scenario estimate),
3. the **current target list** for the last scored Monday (unlabeled) and the dashboard KPIs.

The calibrated probability is used as delivered. `expected_loss` from the same
file is not used: it is not an incremental-margin quantity (it ignores the
discount given to customers who would have bought anyway); a check in the
analysis showed it ranks no better than the recency rule.
""")
code("""
import sys, warnings
sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from policy_lib import *

SCEN = dict(d=0.10, e=0.30, m=0.40, r=0.40, c=0.50)       # Base scenario (Strategy 7.8)
SC = (SCEN["d"], SCEN["e"], SCEN["m"], SCEN["r"], SCEN["c"])
K = 0.20                                                  # operating target share
NURTURE_PCT = 0.80                                        # top 20% risk, not selected -> low-cost contact
MODEL_VERSION = "lr_extended_c0.01_weighted_cal_h90"

data = pd.read_parquet("../data/processed/modeling_dataset_h90.parquet")
scores = pd.read_parquet("../data/results/churn_scores_h90_calibrated.parquet")
key = ["customer_id", "snapshot_date"]
assert not scores.duplicated(key).any()
""")
md("## 12.1 Join and integrity checks")
code("""
feat = data[key + ["value_v", "rfm_segment", "recency_days", "segment"]].rename(columns={"segment": "customer_type"})
t = scores.merge(feat, on=key, how="left", validate="one_to_one", indicator=True)
# Deployment weeks (after the last labelable week) have no row in the labelable dataset only if absent from it
print(t["_merge"].value_counts().to_dict())
print(t.groupby("split").agg(rows=("customer_id", "size"), weeks=("snapshot_date", "nunique"),
      missing_value=("value_v", lambda s: s.isna().sum()), missing_rfm=("rfm_segment", lambda s: s.isna().sum())))
assert t.value_v.notna().all() and t.rfm_segment.notna().all()
assert t.model_version.nunique() == 1 and t.model_version.iloc[0] == MODEL_VERSION
assert t.churn_probability_calibrated.between(0, 1).all()
t = t.drop(columns="_merge")
""")
md("""
## 12.2 Decision rule

* `expected_margin` = E[ΔM] = p·e·V·(m−d) − (1−p)·r·V·d − c, with the calibrated `p`.
* **Retain**: within the weekly top-K by `expected_margin` and `expected_margin` > 0.
* **Nurture**: not Retain but in the top 20% of the week's calibrated risk
  (at risk, but an offer would not pay): low-cost contact, no discount.
* **No-offer**: the rest (likely to buy anyway, or low expected value).
""")
code("""
def add_actions(df, k=K, scen=SC):
    d, e, m, r, c = scen
    df = df.copy()
    df["expected_margin"] = expected_margin(df.churn_probability_calibrated, df.value_v, d, e, m, r, c)
    df["rank_by_margin"] = df.groupby("snapshot_date").expected_margin.rank(ascending=False, method="first").astype(int)
    retain = top_k_mask(df, "expected_margin", k) & (df.expected_margin > 0)
    risky = df.groupby("snapshot_date").churn_probability_calibrated.rank(pct=True) >= NURTURE_PCT
    df["action"] = np.where(retain, "Retain", np.where(risky, "Nurture", "No-offer"))
    return df

t = add_actions(t)
t.groupby(["split", "action"]).agg(rows=("customer_id", "size"), mean_p=("churn_probability_calibrated", "mean"),
                                   mean_value=("value_v", "mean"), churn=("churn", "mean")).round(3)
""")
md("""
## 12.3 Test-period evaluation (labelled weeks)

Policies on the same test weeks with the calibrated probability. Realized
scenario margin uses the actual labels; K=20%, Base scenario. £ per week.
""")
code("""
test = t[t.split == "test"].copy()
test["rand"] = np.random.default_rng(0).uniform(size=len(test))
masks = {
    "No offer (P0)": pd.Series(False, index=test.index),
    "Blanket (P1)": pd.Series(True, index=test.index),
    "Random (P2)": top_k_mask(test, "rand", K),
    "Recency (P3)": top_k_mask(test, "recency_days", K),
    "Risk-only, calibrated p (P5)": top_k_mask(test, "churn_probability_calibrated", K),
    "Value-aware, E[dM] > 0 (P6) = Retain": test.action == "Retain",
}
rows = {}
for n, mk in masks.items():
    w = weekly_margin(test, mk, SC)
    rows[n] = dict(contacts_week=mk.groupby(test.snapshot_date).sum().mean(), margin_week=w.mean(), std=w.std(),
                   worst_week=w.min(), share_to_would_be_buyers=(1 - test[mk].churn.mean()) if mk.any() else np.nan,
                   margin_per_contact=w.sum() / max(mk.sum(), 1))
policy_test = pd.DataFrame(rows).T
policy_test.to_csv("../data/results/test_policy_comparison.csv")
display(policy_test.round(2))
""")
code("""
# Same, by action group: where does the Retain margin come from? (per customer, realized, base scenario)
test["gain"] = row_gain(test, *SC)
test.groupby("action").agg(rows=("gain", "size"), churn_rate=("churn", "mean"), mean_value=("value_v", "mean"),
                           realized_margin_per_contact=("gain", "mean")).round(2)
""")
code("""
# Retain composition by RFM segment (test)
comp = test.groupby(["action", "rfm_segment"]).size().unstack(0).fillna(0)
(comp / comp.sum()).round(3)
""")
md("""
## 12.4 Current target list and dashboard KPIs

The last scored Monday has no outcome yet (unlabelled): this is the list marketing would act on.
""")
code("""
last = t[t.snapshot_date == t.snapshot_date.max()].sort_values("rank_by_margin")
print("Scoring date:", last.snapshot_date.iloc[0].date(), "| active customers:", len(last))
kpi = pd.Series({
    "Active customers": len(last),
    "Retain (offer)": int((last.action == "Retain").sum()),
    "Nurture (no discount)": int((last.action == "Nurture").sum()),
    "No-offer": int((last.action == "No-offer").sum()),
    "Revenue at risk in Retain list (sum of V x p, £)": float((last[last.action == "Retain"].value_v * last[last.action == "Retain"].churn_probability_calibrated).sum()),
    "Expected incremental margin of Retain list (Base, £)": float(last[last.action == "Retain"].expected_margin.sum()),
    "Expected share of Retain offers to would-be buyers": float(1 - last[last.action == "Retain"].churn_probability_calibrated.mean()),
})
display(kpi.round(3).to_frame("value"))
last[["customer_id", "rfm_segment", "customer_type", "churn_probability_calibrated", "value_v", "expected_margin", "action"]].head(15).round(3)
""")
md("## 12.5 Export")
code("""
cols = ["customer_id", "snapshot_date", "split", "churn_probability", "churn_probability_calibrated", "risk_score",
        "risk_band", "value_v", "rfm_segment", "customer_type", "expected_margin", "rank_by_margin", "action",
        "model_version", "churn"]
out = t[cols].rename(columns={"churn": "churn_actual"})
out["scenario"] = "Base d=10% e=30% m=40% r=40% c=0.50; K=20%"
out.to_parquet("../data/results/customer_scores_actions_h90.parquet", index=False)
print(out.shape, "-> data/results/customer_scores_actions_h90.parquet")
""")
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/12_action_table.ipynb")
