# TODO

Work plan to bring the project in line with [Strategy v2](docs/strategy/README.md) and the course requirements (BA Capstone 2026, Topic 1). Each item is one branch created from the latest `main`; after a squash-merge, start the next branch from the new `main`.

Priority: **P1** required by the course · **P2** core of the strategy · **P3** robustness and report polish. Items are listed in dependency order.

## Done

- [x] MVP pipeline: Data Understanding → Evaluation, scoring output and model export (#2)
- [x] Data Preparation chapter restructured into 7 sections (#6)
- [x] `activity_last_90d` computed by key instead of row position (#7)
- [x] `latex/` renamed to `report/`; added `brief/` (2-page executive brief) and `slide/` (10-minute presentation deck) as separate top-level LaTeX deliverables

## P1 — Required by the course

- [x] **`feat/value-and-rfm-segments`** — value proxy V and RFM segmentation (Strategy 4.7, 5.5)
  - Value concentration in notebook 06: revenue share of the top 1% / 10% of customers
  - `value_v` = average order value so far, winsorized at P99 fitted on the training period
  - RFM quintiles fitted on the training period, mapped to named segments (Champions, Loyal, Potential, At Risk, Can't Lose, Hibernating); per-segment count, revenue share, churn rate, average V
  - New columns in `modeling_dataset_h90.parquet`; update Chapters 4–5
  - Done when: segment table and V cap are in the report, cut-offs come from the training period only

- [ ] **`feat/policy-simulation`** — promotion scenarios and policy comparison (Strategy 7.6–7.10)
  - E[ΔM] per customer; realized margin M_policy on test with actual labels
  - Scenarios Light (5%, 20%), Base (10%, 30%), Deep (20%, 40%) × K ∈ {10, 20, 30, 40, 50%}; sensitivity m ∈ {30, 40, 50%}, r ∈ {20, 40, 60%}
  - Policies P0 No offer, P1 Blanket, P2 Random, P3 Recency, P4 RFM-segment, P5 Model risk-only, P6 Model value-aware
  - Replaces the current e-only scenario tables (Chapter 7.6–7.10) and their assumptions (m = 35%, r = 20%, flat £5 incentive)
  - Done when: policy table at the base scenario and operating K, full grid in an appendix, value captured@K reported

- [ ] **`docs/deliverables`** — required project package (Strategy 3.7, 9.4)
  - `README.md`: setup, run order of notebooks 06 → 09, outputs of each step
  - `docs/data_dictionary.md` + source log (URL, download date, license, filters, joins, cleaning log)
  - `docs/model_card.md` (one page)
  - Executive brief: done separately as `brief/main.tex` (not `docs/executive_brief.md`)
  - Done when: a third party can rerun the pipeline from the README

- [ ] **`feat/dashboard`** — 3–5 KPIs and one actionable view (Strategy 8.5); built from scratch
  - KPIs: active customers, revenue at risk in the target list, expected incremental margin, share of offers to likely buyers, customers per action
  - Controls for K, discount scenario and margin; target list with actions; policy comparison chart
  - Done when: it runs from the score table without re-training

## P2 — Core of the strategy

- [ ] **`feat/calibration-check`** — Platt / isotonic vs raw LR, fitted on validation (Strategy 6.6); keep only if Brier improves without hurting ranking

- [ ] **`feat/value-aware-targeting`** — recommendation layer and full score table (Strategy 8.3–8.4)
  - Actions Retain / Nurture / No-offer
  - Score table columns: `value_v`, `rfm_segment`, `expected_margin`, `rank`, `action`, `model_version`
  - Depends on: value-and-rfm-segments, policy-simulation

- [ ] **`chore/config`** — single `config.yaml` for H, R_max, C, warm-up, embargo, K, (d, e), m, r, contact cost, value cap (Strategy 2.10); notebooks read from it

## P3 — Robustness and report

- [ ] **`feat/rolling-backtest`** — 3–4 expanding-window folds covering Nov 2010 – Sep 2011, mean ± std and worst week (Strategy 7.5)
- [ ] **`feat/robustness-checks`** — with/without V cap, sensitivity grid, optional suppression run and H = 60 (Strategy 9.1)
- [ ] **`docs/report-structure`** — align the LaTeX report with Strategy v2
  - Chapter 8 → Targeting Policy & Decision Support (merge Feedback/monitoring into 8.7)
  - Chapter 9 → Robustness, Ethics & Limitations (ethics and bias section is new)
  - Chapter 10 → Conclusion & Recommendations with the recommended policy, K and discount
  - Chapters 1–2: business decision, unit of analysis, KPIs, success criteria v2, configuration table
