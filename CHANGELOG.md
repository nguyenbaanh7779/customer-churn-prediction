# CHANGELOG

## Unreleased

### Features

- Add customer repurchase prediction LaTeX report, structured around CRISP-DM-style sections from Introduction through Conclusion ([`2fd2ad3`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/2fd2ad3))
- Add project overview `README.md` and a step-by-step `docs/guide/README.md` covering the Python environment and compiling the LaTeX report with XeLaTeX ([`bbd6c12`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/bbd6c12))
- Add Python dependencies in `requirements.txt` ([`2ec3999`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/2ec3999))
- Add initial customer data exploration notebook ([`09516ed`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/09516ed))
- Add `docs/strategy/README.md`, the source-of-truth business/analytical strategy for the 12-chapter report (Customer x Snapshot modeling unit, Training/Scoring population design, eligibility-episode cooldown, embargoed temporal split, business-impact formula chain) ([`da29207`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/da29207))
- Add `notebooks/06_data_understanding.ipynb`, implementing all 15 sections of Chapter 6 (data quality, monthly sales, new vs existing customers, repeat customer analysis, inter-purchase interval, RFM, customer lifecycle, provisional churn exploration, snapshot/design feasibility, key findings) on the real Online Retail II data ([`1d2b3b0`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/1d2b3b0))
- Add `notebooks/07_data_preparation.ipynb`, implementing all 10 sections of Chapter 7 (valid-transaction filtering, the weekly active-customer base table with training/scoring population flags, leakage-safe point-in-time feature engineering, the final churn label, a leakage check, and the embargoed train/validation/test split) on the real Online Retail II data, saving the final modeling dataset to `data/processed/modeling_dataset_h90.parquet` ([`a4459d7`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/a4459d7))

### Updates

- Ignore LaTeX and VS Code build artifacts in `.gitignore` ([`6bb166d`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/6bb166d))
- Rewrite all `latex/content/*.tex` files and the title page to match `docs/strategy/README.md` ([`da29207`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/da29207))
- Add explicit chapter/section numbers to every heading in `docs/strategy/README.md` ([`8a92da1`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/8a92da1))
- Ignore macOS `.DS_Store` files in `.gitignore` ([`3117eff`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/3117eff))
- Rewrite `latex/content/data_understanding.tex` with the real computed numbers, tables and exported figures from `notebooks/06_data_understanding.ipynb` ([`a02f826`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/a02f826))
- Rewrite `latex/content/data_preparation.tex` with the real computed numbers, tables and the embargo split timeline figure from `notebooks/07_data_preparation.ipynb`, and fix an `inf`-producing bug in the `recency_to_typical_gap` feature ([`8fa7b55`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/8fa7b55))
