"""Figure for brief/main.tex. Run after notebook 13: python src/make_brief_figures.py"""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "axes.grid.axis": "x", "grid.color": "#e5e5e0", "font.size": 9})
b = pd.read_csv("data/results/budget_comparison.csv", index_col=0)
order = [("Random", "Same budget, random", "#8a8a85"),
         ("Longest since last purchase", "Same budget, longest since purchase", "#eb6834"),
         ("Highest expected margin per contact", "Proposed: highest expected margin (top 20%)", "#1baf7a"),
         ("Highest churn risk", "Same budget, highest churn risk", "#2a78d6")]
fig, ax = plt.subplots(figsize=(7.2, 2.1))
for i, (lab, key, col) in enumerate(order):
    v, n = b.loc[key, "net_margin_90d"], b.loc[key, "contacts_week"]
    ax.barh(i, v, color=col)
    ax.text(v + 120, i, f"£{v:,.0f}  ({n:,.0f} contacts)", va="center", fontsize=8)
ax.set_yticks(range(len(order))); ax.set_yticklabels([o[0] for o in order], fontsize=8)
ax.set_xlim(0, b.net_margin_90d.max() * 1.45)
ax.set_xlabel("Simulated net margin over 90 days, per weekly cohort (£)")
fig.savefig("brief/figures/equal_budget.pdf", bbox_inches="tight")
