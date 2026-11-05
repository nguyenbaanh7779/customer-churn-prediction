"""Assign Lam's 8 RFM segments (output/02_rfm_segmentation.ipynb, part B) to every
(customer, snapshot) in the score table. Writes data/results/rfm8_segments.parquet.
Rules copied from the notebook: R<=90 active; M >= weekly top-10% = high value; F>=4 loyal; F 2-3 developing;
90<R<=180 Cooling; 180<R<=365 Lapsed; R>365 Dormant. Sales = Quantity>0 & Price>0, returns = 'C' invoices."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = ["Champions", "VIP at risk", "Loyal", "Developing", "One-time recent", "Cooling", "Lapsed", "Dormant"]

raw = pd.read_parquet(ROOT / "data/raw/online_retail_II.parquet").dropna(subset=["Customer ID"])
raw["is_return"] = raw.Invoice.astype(str).str.startswith("C")
raw["Revenue"] = raw.Quantity * raw.Price
sales_all = raw[(~raw.is_return) & (raw.Quantity > 0) & (raw.Price > 0)]
rets_all = raw[raw.is_return]
scores = pd.read_parquet(ROOT / "data/results/customer_scores_actions_h90.parquet", columns=["customer_id", "snapshot_date"])
out = []
for snap in sorted(scores.snapshot_date.unique()):
    snap = pd.Timestamp(snap)
    s = sales_all[sales_all.InvoiceDate < snap]; rt = rets_all[rets_all.InvoiceDate < snap]
    g = s.groupby("Customer ID")
    d = pd.DataFrame({"R": (snap - g.InvoiceDate.max()).dt.days, "F": g.Invoice.nunique(), "gross": g.Revenue.sum()})
    d["M"] = (d.gross - (-rt.groupby("Customer ID").Revenue.sum()).reindex(d.index).fillna(0)).clip(lower=0)
    hv, act = d.M >= d.M.quantile(.90), d.R <= 90
    cond = [hv & act, hv, act & (d.F >= 4), act & (d.F >= 2), act, d.R <= 180, d.R <= 365]
    d["segment8"] = np.select(cond, SEGMENTS[:-1], default=SEGMENTS[-1])
    d = d.reset_index().rename(columns={"Customer ID": "customer_id"})
    d["customer_id"] = d.customer_id.astype(int); d["snapshot_date"] = snap
    out.append(d[["customer_id", "snapshot_date", "segment8", "R", "F", "M"]])
res = pd.concat(out)
res.to_parquet(ROOT / "data/results/rfm8_segments.parquet", index=False)
print(len(res), res[res.snapshot_date == res.snapshot_date.max()].segment8.value_counts().to_dict())
