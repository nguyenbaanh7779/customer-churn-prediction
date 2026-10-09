"""
make_sample_data.py — Generate a synthetic Online Retail II transactions file
for demo/testing purposes.

Usage (from project root, with venv active):
    python data/raw/make_sample_data.py

Output: data/raw/online_retail_II.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT_PATH = Path(__file__).parent / "online_retail_II.parquet"

RNG = np.random.default_rng(42)

# ── Parameters ────────────────────────────────────────────────────────────────
N_CUSTOMERS   = 800
DATE_START    = pd.Timestamp("2010-01-01")
DATE_END      = pd.Timestamp("2011-12-31")
N_PRODUCTS    = 300
COUNTRIES     = ["United Kingdom", "Germany", "France", "Netherlands", "Spain"]
COUNTRY_PROBS = [0.80, 0.07, 0.06, 0.04, 0.03]

# ── Products ──────────────────────────────────────────────────────────────────
product_ids   = [f"{10000 + i}" for i in range(N_PRODUCTS)]
product_descs = [f"PRODUCT {i:04d}" for i in range(N_PRODUCTS)]
product_prices = RNG.uniform(0.5, 25.0, N_PRODUCTS).round(2)

# ── Build transactions ────────────────────────────────────────────────────────
rows = []
invoice_counter = 536000

total_days = (DATE_END - DATE_START).days

for cid in range(12300, 12300 + N_CUSTOMERS):
    country = RNG.choice(COUNTRIES, p=COUNTRY_PROBS)

    # Each customer has between 1 and 25 invoices
    n_invoices = int(RNG.integers(1, 26))

    # Spread invoice dates across the period
    offsets = sorted(RNG.integers(0, total_days, n_invoices))

    for offset in offsets:
        invoice_date = DATE_START + pd.Timedelta(days=int(offset))
        invoice_id   = str(invoice_counter)
        invoice_counter += 1

        # 1–8 line items per invoice
        n_lines = int(RNG.integers(1, 9))
        chosen  = RNG.choice(N_PRODUCTS, size=n_lines, replace=False)

        for idx in chosen:
            qty   = int(RNG.integers(1, 20))
            price = float(product_prices[idx])
            rows.append({
                "Invoice":     invoice_id,
                "StockCode":   product_ids[idx],
                "Description": product_descs[idx],
                "Quantity":    qty,
                "InvoiceDate": invoice_date,
                "Price":       price,
                "Customer ID": float(cid),
                "Country":     country,
            })

    # ~15% of customers get one cancellation
    if RNG.random() < 0.15:
        cancel_offset = int(RNG.integers(0, total_days))
        cancel_date   = DATE_START + pd.Timedelta(days=cancel_offset)
        cancel_id     = f"C{invoice_counter}"
        invoice_counter += 1
        idx   = int(RNG.integers(0, N_PRODUCTS))
        qty   = int(RNG.integers(1, 5))
        price = float(product_prices[idx])
        rows.append({
            "Invoice":     cancel_id,
            "StockCode":   product_ids[idx],
            "Description": product_descs[idx],
            "Quantity":    -qty,
            "InvoiceDate": cancel_date,
            "Price":       price,
            "Customer ID": float(cid),
            "Country":     country,
        })

df = pd.DataFrame(rows)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print(f"Generated {len(df):,} rows, {df['Customer ID'].nunique()} customers, "
      f"{df['Invoice'].nunique()} invoices")
print(f"Date range: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")

df.to_parquet(OUT_PATH, index=False)
print(f"Saved → {OUT_PATH}")
