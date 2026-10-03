# CHANGELOG

## Unreleased

### Features

- Add customer repurchase prediction LaTeX report, structured around CRISP-DM-style sections from Introduction through Conclusion ([`2fd2ad3`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/2fd2ad3))
- Add project overview `README.md` and a step-by-step `docs/guide/README.md` covering the Python environment and compiling the LaTeX report with XeLaTeX ([`bbd6c12`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/bbd6c12))
- Add Python dependencies in `requirements.txt` ([`2ec3999`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/2ec3999))
- Add initial customer data exploration notebook ([`09516ed`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/09516ed))
- Add `docs/strategy/README.md`, the source-of-truth business/analytical strategy for the 12-chapter report (Customer x Snapshot modeling unit, Training/Scoring population design, eligibility-episode cooldown, embargoed temporal split, business-impact formula chain) ([`da29207`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/da29207))

### Updates

- Ignore LaTeX and VS Code build artifacts in `.gitignore` ([`6bb166d`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/6bb166d))
- Rewrite all `latex/content/*.tex` files and the title page to match `docs/strategy/README.md` ([`da29207`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/da29207))
- Add explicit chapter/section numbers to every heading in `docs/strategy/README.md` ([`8a92da1`](https://github.com/nguyenbaanh7779/customer-repurchase-prediction-olist/commit/8a92da1))
