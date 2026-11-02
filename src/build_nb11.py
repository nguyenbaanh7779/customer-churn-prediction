"""Generate notebooks/11_rolling_backtest.ipynb (run: python src/build_nb11.py)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 11. Rolling-Origin Backtest

Strategy 7.5 / TODO `feat/rolling-backtest`. The official split tests only
Jul-Sep 2011, the lowest-churn quarter. Its two 90-day embargoes also leave
Dec 2010 - Jan 2011 (churn 59-63%) unused. Here the same recipe (Logistic
Regression, Extended features, C=0.01, rows weighted 1/n per customer) is
refit on an **expanding window** for four test blocks, with a **90-day embargo**
between the last training snapshot and the first test snapshot:

| Fold | Test block | Train snapshots up to |
|---|---|---|
| A | 2010-12-06 → 2011-01-24 (8 wk, peak churn) | 2010-09-06 |
| B | 2011-02-07 → 2011-03-28 (8 wk) | 2010-11-08 |
| C | 2011-04-11 → 2011-05-30 (8 wk) | 2011-01-10 |
| D | 2011-07-04 → 2011-09-05 (10 wk, official test) | 2011-04-04 |

The question: does the policy ranking from notebook 10 (value-aware beats
risk-only and recency) hold when churn is high and the model has less history?
Scenario estimates only: no treatment/control data.

Known approximations: the V cap (P99) and RFM quintile cut-offs in the dataset
were fitted on the original training period (to Nov 2010), so they are a fixed
scale for all folds; fold A's cap therefore uses slightly later information
than its own training window (a scaling constant, not a feature).
""")
code("""
import sys, warnings
sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from policy_lib import *

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e5e5e0"})
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"

H_DAYS = pd.Timedelta(days=90)
FOLDS = {"A": ("2010-12-06", "2011-01-24"), "B": ("2011-02-07", "2011-03-28"),
         "C": ("2011-04-11", "2011-05-30"), "D": ("2011-07-04", "2011-09-05")}
SCENARIOS = {"Light": (0.05, 0.20), "Base": (0.10, 0.30), "Deep": (0.20, 0.40)}
M, R, C = 0.40, 0.40, 0.50
K_GRID = [0.10, 0.20, 0.30]

data = pd.read_parquet("../data/processed/modeling_dataset_h90.parquet")
lab = data[data.labelable].copy()
""")
md("## 11.1 Folds, model refit and predictive performance")
code("""
folds = {}
rows = []
for name, (a, b) in FOLDS.items():
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    train = lab[lab.snapshot_date <= a - H_DAYS]
    test = lab[(lab.snapshot_date.dt.normalize() >= a) & (lab.snapshot_date.dt.normalize() <= b)].copy()
    assert train.snapshot_date.max() + H_DAYS <= test.snapshot_date.min() + pd.Timedelta(days=1)
    predict, lr = fit_train_only(train)
    test["p"] = predict(test)
    train = train.assign(p=predict(train))
    folds[name] = dict(train=train, test=test)
    rows.append(dict(fold=name, train_weeks=train.snapshot_date.nunique(), train_rows=len(train),
        test_weeks=test.snapshot_date.nunique(), test_rows=len(test), churn_rate=test.churn.mean(),
        mean_p=test.p.mean(), roc_auc=roc_auc_score(test.churn, test.p),
        pr_auc=average_precision_score(test.churn, test.p), brier=brier_score_loss(test.churn, test.p),
        brier_base_rate=brier_score_loss(test.churn, np.full(len(test), train.churn.mean()))))
perf = pd.DataFrame(rows).set_index("fold")
perf.round(3)
""")
md("""
`brier_base_rate` here is the Brier score of always predicting the *training*
churn rate, i.e. what you get with no model but a stale base rate.
""")
code("""
# Lift@10% per fold (weekly, mean ± std) vs recency rule
rows = {}
for name, f in folds.items():
    t = f["test"]
    out = {}
    for lab_, col in [("model", "p"), ("recency", "recency_days")]:
        lifts = []
        for w, g in t.groupby("snapshot_date"):
            k = max(int(round(len(g) * 0.1)), 1)
            lifts.append(g.nlargest(k, col).churn.mean() / g.churn.mean())
        out[f"lift10_{lab_}_mean"], out[f"lift10_{lab_}_std"] = np.mean(lifts), np.std(lifts, ddof=1)
    rows[name] = out
pd.DataFrame(rows).T.round(3)
""")
md("""
## 11.2 Policy comparison per fold

Same policies and margin model as notebook 10 (Strategy 7.7-7.9). P4b targets
the RFM segments whose per-customer realized margin was positive **on that fold's
training rows** (labels in the training windows end before the test block).
Weekly margin, mean over the fold's weeks.
""")
code("""
def masks_for(test, train, k, scen):
    d, e, m, r, c = scen
    t = test.assign(em=expected_margin(test.p, test.value_v, d, e, m, r, c),
                    rand=np.random.default_rng(0).uniform(size=len(test)))
    tr_gain = pd.Series(row_gain(train, d, e, m, r, c), index=train.index)
    good = tr_gain.groupby(train.rfm_segment).mean()
    offer_segs = good.index[good > 0]
    return t, {
        "P1 Blanket": pd.Series(True, index=t.index),
        "P2 Random": top_k_mask(t, "rand", k),
        "P3 Recency": top_k_mask(t, "recency_days", k),
        "P4b RFM segments (train-chosen)": t.rfm_segment.isin(offer_segs),
        "P5 Model risk-only": top_k_mask(t, "p", k),
        "P6 Model value-aware": top_k_mask(t, "em", k) & (t.em > 0),
    }, list(offer_segs)

def fold_table(name, k, scen):
    f = folds[name]
    t, masks, segs = masks_for(f["test"], f["train"], k, scen)
    out = {}
    for pol, mk in masks.items():
        w = weekly_margin(t, mk, scen)
        out[pol] = dict(margin_week=w.mean(), std=w.std(), worst_week=w.min(),
                        contacts_week=mk.groupby(t.snapshot_date).sum().mean(),
                        margin_per_contact=w.sum() / max(mk.sum(), 1))
    return pd.DataFrame(out).T, segs

BASE = (0.10, 0.30, M, R, C)
for name in folds:
    tab, segs = fold_table(name, 0.20, BASE)
    print(f"Fold {name}: base scenario, K=20%; P4b segments = {segs}")
    display(tab.round(2))
""")
code("""
# Summary across folds: weekly margin at K=20%, each scenario
rows = []
for sname, (d, e) in SCENARIOS.items():
    scen = (d, e, M, R, C)
    for name in folds:
        tab, _ = fold_table(name, 0.20, scen)
        rows.append(dict(scenario=sname, fold=name, **tab.margin_week.round(0).to_dict()))
summary = pd.DataFrame(rows)
display(summary)
summary.to_csv("../data/results/backtest_policy_summary.csv", index=False)
pols = ["P3 Recency", "P5 Model risk-only", "P6 Model value-aware"]
print("Folds × scenarios where P6 beats P3 recency:", int((summary["P6 Model value-aware"] > summary["P3 Recency"]).sum()), "/", len(summary))
print("Folds × scenarios where P6 beats P5 risk-only:", int((summary["P6 Model value-aware"] > summary["P5 Model risk-only"]).sum()), "/", len(summary))
print("Folds × scenarios where P6 beats blanket:", int((summary["P6 Model value-aware"] > summary["P1 Blanket"]).sum()), "/", len(summary))
print("Folds × scenarios where best of P2/P3/P5/P6 is P6:", int((summary[["P2 Random"] + pols].idxmax(axis=1) == "P6 Model value-aware").sum()), "/", len(summary))
""")
code("""
# Margin vs K per fold (base scenario)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=False)
cols = {"P3 Recency": ORANGE, "P5 Model risk-only": BLUE, "P6 Model value-aware": GREEN, "P2 Random": GREY}
for ax, name in zip(axes, folds):
    for pol, c in cols.items():
        ys = [fold_table(name, k, BASE)[0].loc[pol, "margin_week"] for k in K_GRID]
        ax.plot([int(k * 100) for k in K_GRID], ys, marker="o", color=c, label=pol)
    ax.set_title(f"Fold {name} (churn {folds[name]['test'].churn.mean():.0%})"); ax.set_xlabel("K (%)")
axes[0].set_ylabel("£ per week (base)"); axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig("../report/figures/ch11_backtest_policy_vs_k.png", dpi=150, bbox_inches="tight"); plt.show()
""")
md("""
## 11.3 Per-fold calibration and segment actions

Does the base-rate shift show up per fold, and do the segment actions from
notebook 10 (offer / no discount) hold when churn is high?
""")
code("""
seg_rows = []
for name, f in folds.items():
    t = f["test"]; g = pd.Series(row_gain(t, *BASE), index=t.index)
    s = g.groupby(t.rfm_segment).mean().round(2); s.name = name
    seg_rows.append(s)
display(pd.DataFrame(seg_rows).T.rename_axis("segment (realized £ per contact, base)"))
display(pd.DataFrame({n: f["test"].groupby("rfm_segment").churn.mean() for n, f in folds.items()}).round(2).rename_axis("segment churn rate"))
""")
md("""
## 11.4 Blanket vs targeting by season

Blanket total per week and per contact, to show when untargeted discounts
destroy margin (Deep scenario).
""")
code("""
rows = []
for sname, (d, e) in SCENARIOS.items():
    for name in folds:
        tab, _ = fold_table(name, 0.20, (d, e, M, R, C))
        rows.append(dict(scenario=sname, fold=name, blanket_week=tab.loc["P1 Blanket", "margin_week"],
                         blanket_per_contact=tab.loc["P1 Blanket", "margin_per_contact"],
                         P6_week=tab.loc["P6 Model value-aware", "margin_week"],
                         P6_per_contact=tab.loc["P6 Model value-aware", "margin_per_contact"]))
pd.DataFrame(rows).round(1)
""")
md("""
## 11.5 Break-even gross margin by fold

Gross margin at which each policy's net margin is zero (K=20%, base discount/response).
The dataset has no margin data, so this shows how safe each policy is to the
margin assumption. Lower is safer.
""")
code("""
rows = []
for name in folds:
    f = folds[name]
    t, masks, _ = masks_for(f["test"], f["train"], 0.20, BASE)
    rows.append({"fold": name, **{pol: margin_breakdown(t, mk, BASE)["break_even_margin_rate"] for pol, mk in masks.items()}})
display(pd.DataFrame(rows).set_index("fold").round(3))
f = folds["D"]; t, masks, _ = masks_for(f["test"], f["train"], 0.20, BASE)
display(pd.DataFrame({p: margin_breakdown(t, mk, BASE) for p, mk in masks.items()}).T.round(2).rename_axis("fold D decomposition (£/week)"))
""")
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/11_rolling_backtest.ipynb")
