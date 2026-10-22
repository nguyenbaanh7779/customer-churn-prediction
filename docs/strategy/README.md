---
title: "Customer Retention & Promotion Targeting"
subtitle: "Value-aware churn prediction for an online retailer — Project Strategy & AI Coding Context"
date: "Version 2 · October 2026"
---

::: {.purpose}
**Purpose.** This document is the source of truth for the business and analytical strategy of the project. It answers the course's Topic 1 business question: *which customers should the retailer prioritize for retention or promotion to increase expected future margin without over-discounting?* The core **business problem** is customer retention. The **analytical core** is churn prediction, combined with customer value and promotion scenarios to decide **who receives an offer and at which discount**. The report must preserve the 10 main chapters listed below.

1\. Introduction · 2. Problem Statement and Approach · 3. Dataset and Data Specification · 4. Data Understanding · 5. Data Preparation · 6. Modeling · 7. Evaluation · 8. Targeting Policy & Decision Support · 9. Robustness, Ethics & Limitations · 10. Conclusion & Recommendations
:::

::: {.note}
**Required project package (course rules).** Reproducible code/notebooks with a README and run instructions; data dictionary and source log; executive brief (max 2 pages); dashboard with 3–5 decision-relevant KPIs and one actionable view; model/analysis card. Status and owners are tracked in [`TODO.md`](../../TODO.md).

**Baseline first.** Stretch work starts only after the baseline model, validation and business interpretation are complete.
:::

# 1. Introduction

## 1.1 Project Background

An e-commerce retailer earns revenue from first purchases of new customers and repeat purchases of existing customers. Customers may gradually reduce their purchasing activity or stop purchasing altogether, and identifying them early is difficult with a large customer base and long transaction histories.

Retention promotions can protect repeat revenue, but **every discount given to a customer who would have bought anyway is lost margin**. Retention therefore needs targeting: offers should go to customers who are at risk and valuable enough to justify one.

## 1.2 Business Decision

Each week, marketing decides **which active customers receive a retention offer, at which discount level, within a limited budget**. The analytics must support this decision, not only predict churn.

## 1.3 Problem Statement

- **Late identification of churn risk.** Inactive customers are often noticed only after a long period without purchases, when retention opportunities are limited.
- **Limited budget and over-discounting.** The business wants to increase future gross margin from existing customers under a limited retention budget. Untargeted promotions waste budget on loyal customers and give discounts to customers who would have bought anyway.

The project identifies customers who are both **at risk of churning** and **valuable enough to justify an offer**.

## 1.4 Research Questions

1. How can transaction history be transformed into customer-level weekly observations for churn prediction?
2. How well does purchasing behavior predict churn in a future window, compared with random and recency-rule baselines?
3. Which behavioral characteristics and RFM segments are associated with churn and customer value? (Associations, not causes.)
4. Which targeting policy (who, how many, which discount) maximizes expected incremental margin under explicit assumptions, compared with no-offer and blanket-offer policies?
5. How sensitive is the recommended policy to the assumptions on response rate, discount, margin and cannibalization?

## 1.5 Project Objectives

Build a reproducible pipeline that produces a **weekly customer-level score table** (churn probability, value, segment, expected margin, recommended action) and a **targeting policy** with a quantified, assumption-based margin impact.

## 1.6 Unit of Analysis, Target and Horizon

| Item | Definition |
|--------------------|----------------------------------------------------------|
| Unit of analysis | Customer × weekly snapshot date *t* |
| Decision time | Monday of each week, using data up to *t* only |
| Target | Churn(*i*, *t*) = 1 if no valid purchase in (*t*, *t* + H] |
| Horizon H | 90 days (candidates 60/90/120, fixed in 4.5 and 5.6) |
| Business outcome | Expected incremental gross margin of the targeting policy (scenario-based) |

## 1.7 Scope and Limitations

**In scope:** existing-customer retention, churn prediction, RFM value segmentation, weekly scoring, value-aware targeting, promotion scenario simulation, dashboard.

**Not claimed:**

- Causal uplift or observed revenue lift versus control — Online Retail II has no randomized promotion treatment.
- Real profit — margin, costs and response rates are stated assumptions with sensitivity analysis.
- A live deployment — Chapter 8 describes decision support demonstrated on historical data.

## 1.8 Report Structure

1. **Introduction** — background, business decision, problem statement, research questions, objectives, unit of analysis and scope.
2. **Problem Statement and Approach** — revenue and margin, KPIs and success criteria, the decision flow and the configuration parameters.
3. **Dataset and Data Specification** — the Online Retail II dataset, its schema and conventions, source log and data dictionary.
4. **Data Understanding** — sales, customers, repurchase intervals and value concentration, and the design decisions they support.
5. **Data Preparation** — cleaning, populations, churn labels, features (including value V and RFM segments) and the temporal split.
6. **Modeling** — baselines, feature sets, algorithms, tuning, calibration and interpretability.
7. **Evaluation** — predictive, targeting and business-scenario evaluation, including the policy comparison.
8. **Targeting Policy & Decision Support** — weekly decision cycle, recommendation layer, score table, dashboard, suppression and monitoring.
9. **Robustness, Ethics & Limitations** — robustness checks, ethics and bias, limitations and the model card.
10. **Conclusion & Recommendations** — the recommended policy, business-to-technical translation and next steps.

# 2. Problem Statement and Approach

## 2.1 Revenue and Margin

Revenue comes from first purchases of new customers and repeat purchases of existing customers. Gross margin = revenue × margin rate *m*. A retention offer with discount *d* changes margin in two ways:

- **Gain:** a would-be churner who responds buys again, earning V × (*m* − *d*).
- **Loss (over-discounting):** a customer who would have bought anyway redeems the offer, losing V × *d*.

The business goal is to maximize the gain minus the loss minus the contact cost.

## 2.2 Revenue from New and Existing Customers

New customers grow the customer base; existing customers generate repeat revenue. Chapter 4 quantifies the existing-customer share of revenue to justify the retention focus.

## 2.3 Customer Churn as a Business Problem

Existing customers who stop buying reduce future revenue. In a non-contractual setting churn is not observed directly, so it is defined as no valid purchase in the next H days.

## 2.4 Budget and Capacity Constraint

Marketing cannot contact or discount every customer. The weekly target size is limited to *K*% of the active base (or a fixed budget *B*), and the discount depth *d* is a policy choice.

## 2.5 Business Objective

Increase expected future gross margin from existing customers by targeting retention offers to customers with **high churn risk and sufficient value**, and by **withholding offers** where they mainly cannibalize margin.

## 2.6 KPIs

Business KPIs and ML metrics are kept separate.

| Type | KPIs |
|--------------|------------------------------------------------------------------|
| Business | Active customers, existing-customer revenue share, repurchase rate, revenue at risk, expected incremental margin, share of offers going to would-be buyers |
| Targeting | Precision\@K, Recall\@K, Lift\@K, value captured\@K |
| Model | PR-AUC, ROC-AUC, Brier score, calibration curve |

## 2.7 Success Criteria

Success is relative to simple alternatives, evaluated on the unseen test period.

| Level | Criterion |
|--------------|------------------------------------------------------------------|
| Predictive | Final model PR-AUC exceeds the test churn rate and the recency-rule baseline; calibration is acceptable (Brier score and reliability curve). |
| Targeting | Lift\@K and Recall\@K at the operating K (20–30%) exceed the recency rule and the RFM-segment rule, averaged over weekly test dates. |
| Stability | Weekly Top-K performance is reported as mean ± std; no collapse on single weeks or folds (7.5). |
| Business | Under the base scenario, the value-aware model policy yields higher expected incremental margin than no-offer, blanket offer, random, recency and RFM-segment policies. |
| Robustness | The policy ranking holds across at least three discount/response scenarios and the stated margin range. |

## 2.8 From Business Decision to Analytics

The decision needs three inputs per customer each week: the **churn probability** *p*, the **customer value** V and the **margin effect of an offer** at discount *d*. The analytical core is binary churn prediction; the decision layer combines it with value and scenario assumptions.

## 2.9 Decision Flow

::: {.flow}
Active customers at week *t* (Recency ≤ R~max~) → Remove suppressed customers (contacted in the last S weeks; empty in historical data) → Features at *t*, calibrated churn probability *p* → Value V and RFM segment → Expected incremental margin E[ΔM] per customer → Budget/capacity and business rules → Action: Retain (offer) / Nurture (low-cost contact, no discount) / No-offer
:::

## 2.10 Configuration Parameters

All parameters live in **one config file** (`config.yaml`) and are reported.

| Parameter | Symbol | Values | Decided in |
|--------------------|--------|------------------------------------|------------|
| Prediction horizon | H | 90 days (candidates 60/90/120) | 4.5, 5.6 |
| Activity threshold | R~max~ | 136 days (P90 of inter-purchase gap) | 4.5, 5.3 |
| Episode cooldown | C | = H | 5.3 |
| Feature warm-up | – | 6 months | 5.3 |
| Scoring cadence | – | Weekly (Monday) | 2.9 |
| Split embargo | – | = H | 5.6 |
| Suppression period | S | 4–8 weeks | 8.6 |
| Target share | K | 10 / 20 / 30 / 40 / 50% (operating 20–30%) | 7.9 |
| Discount / response pairs | (*d*, *e*) | (5%, 20%), (10%, 30%), (20%, 40%) | 7.8 |
| Gross margin | *m* | 30 / 40 / 50% (base 40%) | 7.8 |
| Cannibalization rate | *r* | 20 / 40 / 60% (base 40%) | 7.8 |
| Contact cost | c~contact~ | assumed GBP per contacted customer | 7.8 |
| Value cap | – | P99 of V (training period) | 5.5 |

## 2.11 Implementation Roadmap

- **Understand the data** (Chapter 4) — sales, customers, purchase intervals and value concentration; fixes H, R~max~ and the value cap.
- **Prepare customer snapshots** (Chapter 5) — Customer × Snapshot Date rows with point-in-time features, value V, RFM segment, churn labels and an embargoed temporal split.
- **Model churn risk** (Chapter 6) — baselines, feature sets, algorithms, calibration; select a final model.
- **Evaluate** (Chapter 7) — predictive, targeting and promotion-scenario evaluation; compare targeting policies.
- **Decide** (Chapter 8) — weekly score table, Retain/Nurture/No-offer actions and the dashboard.

# 3. Dataset and Data Specification

## 3.1 Data Source and Collection

The project uses the publicly available **Online Retail II** dataset (UCI Machine Learning Repository) as a historical extract of e-commerce transaction records, identified as such in the report. It contains transaction-level information, including invoice identifiers, customer identifiers, product codes, quantities, prices and invoice dates.

## 3.2 Dataset Characteristics and Limitations

The dataset covers the period from December 2009 to December 2011. Its relatively limited historical coverage constrains the number of customer snapshots and future observation periods available for churn labeling (see 5.6). The retailer serves both individual customers and wholesalers, so spend is highly heterogeneous and the value proxy V is capped (5.5). Seasonal demand, particularly around Christmas, influences purchasing patterns: the single test window may not contain a peak season, which motivates the rolling-origin backtest (7.5).

## 3.3 Raw Data Schema

The raw dataset is a single flat transaction table (not pre-split into separate customer, order and product tables): each row is one product line within one invoice.

| Column | Type | Description |
|--------------|----------|------------------------------------------------------------------|
| `Invoice` | string | Invoice number. A "C"-prefixed value (e.g. `C536379`) marks a cancellation of an earlier invoice; `Quantity` is negative on these rows. |
| `StockCode` | string | Product code. Most codes are numeric (e.g. `85123A`); a small set of non-numeric codes are not products at all (3.5). |
| `Description` | string | Free-text product description; can be missing or inconsistent for the same `StockCode`. |
| `Quantity` | integer | Units on this line. Negative for cancellations. |
| `InvoiceDate` | datetime | Timestamp of the transaction line. |
| `Price` | float | Unit price (GBP). |
| `Customer ID` | float | Customer identifier. Missing for a meaningful share of rows (5.2) — these cannot be attached to any customer history. |
| `Country` | string | Customer's country, self-reported at checkout. |

## 3.4 Pipeline Column Names

The raw columns are renamed to snake_case names used throughout the pipeline: `invoice_id`, `product_id`, `product_description`, `quantity`, `invoice_time`, `unit_price`, `customer_id` and `country`. Two columns are derived: `total_price` = `quantity` × `unit_price`, and `invoice_month` (`invoice_time` truncated to the first of its month, used for monthly aggregation).

## 3.5 Value Conventions

- **Cancellation invoices.** An `invoice_id` starting with "C" marks a cancellation of a previous purchase, with negative `quantity`. These rows are excluded from purchase history (5.2) but kept separately to build the cancellation features (5.5).
- **Non-product stock codes.** A small number of `product_id` values are operational/accounting entries rather than products — postage (`POST`, `DOT`), manual adjustments (`M`), carriage (`C2`), discounts (`D`), samples (`S`), bank charges and similar. They are discovered empirically (5.2) and excluded from product- and customer-level analysis.
- **Missing Customer ID.** Rows without a `customer_id` cannot be attached to any customer history and are excluded from every customer-level analysis from 4.3 onward, although they are still counted in the raw totals of 4.2.

## 3.6 Information Not Available

The dataset contains transactions only. Product cost, product category hierarchy and margin are not available, and neither is any marketing or campaign information (campaign dates, channels, treatment/control assignment, discounts, voucher or contact costs, redemptions). In this project the related quantities (margin *m*, discount *d*, response *e*, cannibalization *r*, contact cost) are explicit assumptions with sensitivity analysis. A production system would log CampaignID, customer, date, channel, discount, cost and redemption, so that *e* and *r* could later be measured instead of assumed.

## 3.7 Source Log and Data Dictionary

Both are required deliverables and live in `docs/`.

| Item | Value |
|--------------------|----------------------------------------------------------|
| Dataset | Online Retail II (UCI Machine Learning Repository) |
| Source URL | to record |
| Download date | to record |
| License / terms | to verify on the UCI page and record |
| Files / sheets | Year 2009–2010 and Year 2010–2011 sheets, concatenated |
| Filters | Valid-purchase rules (5.2) |
| Joins | None (single source); sheet overlap de-duplicated |

The **data dictionary** lists every raw field and every engineered feature: name, type, definition, source fields, time window, and whether it is known at decision time.

## 3.8 Analytical Dataset Design

Transactions are not modeled directly. Each final modeling row represents a **Customer × Snapshot Date**: the customer's historical features, value V and segment computed only from information available up to the snapshot date, the population flags, and a future churn label. This dataset is built in Chapter 5.

# 4. Data Understanding

## 4.1 Objective

Data Understanding and Data Preparation (Chapter 5) form one narrative: understand the data, establish the business case for retention, size the value at stake, and turn the findings into the strategy for building the modeling dataset — its population, label, features and value proxy. This chapter ends with the design decisions (4.8) that Chapter 5 implements.

## 4.2 Dataset Overview

Transaction count, unique customers, unique products, date range, countries, total quantity and revenue of the raw data; then the size of the valid-transaction subset (rows kept, invoices, customers). The per-rule data-quality breakdown belongs to 5.2.

## 4.3 Monthly Sales and Seasonality

Monthly customers, orders and revenue: the typical month and the seasonal pattern (peak months, low months, partial months). State the consequences for the design: infrequent buyers need a multi-month churn window, and seasonality requires a chronological split read per scoring week.

## 4.4 Existing Customers Drive Revenue

One-time versus repeat customers (counts, orders, revenue shares) as the headline, supported by the monthly shares of active customers and revenue for new versus existing customers. This directly supports the business case for retaining existing customers.

## 4.5 Repurchase Interval and the 90-Day Horizon

Retail is non-contractual, so churn is inferred from a period without purchases. Time between consecutive purchase occasions (per repurchase, not per customer): mean, median, percentiles and the distribution in 30-day bins. Present the choice of **H** as a trade-off between repurchase coverage, timeliness of detection and the training weeks left after the embargoed split (5.6), and set **R~max~** from the same distribution.

## 4.6 Customer Groups by Purchase Cadence

Group customers by purchase cadence: one-time (first purchase only) or an average inter-purchase gap of 0–30, 30–90 or 90+ days. Compare customer counts, AOV and revenue share per group, reconcile the per-repurchase and per-customer views (which explains a churn rate near one half), show the joint distribution with customer AOV, and how the mix evolves over time (with the caveat that a fixed window makes long gaps observable only as customers age).

## 4.7 Value Concentration

Customer revenue, average order value and frequency, and wholesale concentration: the revenue share of the top 1% and top 10% of customers. This sets the **cap for the value proxy V** (5.5) and motivates reporting results for wholesale-sized customers separately (7.12, 9.2).

## 4.8 Key Findings and Design Decisions

A table that links each finding to the design decision it supports in Chapter 5: valid purchase occasions, chronological split, weekly snapshots of active customers, H and R~max~, separate reporting for one-time customers, the value cap, and feature groups.

# 5. Data Preparation

## 5.1 Objective

Turn the cleaned transactions into a leakage-safe **Customer × Snapshot Date** modeling dataset: clean the raw rows (5.2), build the weekly populations (5.3), attach the churn label (5.4), point-in-time features, value V and RFM segment (5.5), then split the rows chronologically with an embargo (5.6). State the frozen configuration inherited from 4.8 (2.10).

## 5.2 Data Cleaning

The value conventions are defined in 3.5; here they become one reproducible **valid purchase** filter. A row is kept only if it has a Customer ID, is not a cancellation (invoice starting with "C"; cancellations are kept separately for cancellation features), has Quantity > 0 and Price > 0, is a real product (non-product codes such as postage, manual adjustments and bank charges are discovered empirically and excluded), and is not an exact duplicate.

Report the per-rule data-quality breakdown measured on the raw data, and the rows, customers **and revenue** kept after combining all rules (the cleaning log feeds the source log, 3.7). Valid rows are collapsed into **purchase occasions** (one per distinct invoice); frequency, inter-purchase intervals and the label are computed on occasions, not rows.

## 5.3 Population Construction

**Time window and eligibility.** Customer histories span the full data period; after the warm-up, one snapshot is taken every Monday. At each weekly date *t*, a customer is **eligible (active)** if:

1. at least one valid purchase exists on or before *t* (existing customer);
2. Recency at *t* ≤ R~max~ (activity condition).

R~max~ excludes long-lost customers who would inflate metrics without business value. One-time buyers stay in all populations, and results are also reported by segment (one-time vs repeat).

The eligible rows form **one weekly active-customer base table**; features, value, segment and labels are computed once on it, and the populations are flags on the same table:

| Population | Definition | Used for |
|--------------------|----------------------------------------|----------------------|
| Customer universe | All customers with a valid Customer ID | Data understanding |
| Active customer at *t* | At least one valid purchase on or before *t* and Recency ≤ R~max~ | Base table |
| Training Population | Active rows sampled with the eligibility-episode rule | Model training |
| Scoring Population at *t* | All active customers at *t*, minus suppressed | Validation, test, decision support |
| Target Population | Scoring Population after ranking, budget and rules | Retention offer (Chapter 8) |

| Flag | Rule |
|--------------------|----------------------------------------------------------|
| `in_scoring_population` | Row is active and the customer is not under suppression at *t*. In the historical data there are no real campaigns, so suppression is empty and every active row is in the Scoring Population (see 7.1). |
| `in_training_sample` | Row selected by the eligibility-episode rule below. Computed per customer in date order. |
| `sample_weight` | 1 / (number of weekly active rows of the customer in the training period); filled in after the split (5.6) and used by the ablation scheme (6.10). |

**Training sample — eligibility-episode rule.** Consecutive weekly snapshots of a customer are near-duplicates with overlapping label windows, so the Training Population is sampled per customer: take a row at the first active week *t*, then no further training row until the first active week ≥ *t* + C (C = H = 90 days); a customer who exceeds R~max~ and later returns starts a new episode. Training labels then have non-overlapping windows and each customer contributes a similar number of rows. The rule applies **only to training** — validation, test and scoring keep every active row. Alternative scheme: all active rows with weight 1 / n~i~ (6.10).

**Results.** Report the base-table size, the weekly all-active vs. episode-sampled row counts (rows, rows per customer), the **weekly Scoring Population size** over time — the pool from which marketing selects Top-K each week — and the sensitivity of population size and churn rate to R~max~ (P90 versus P95).

## 5.4 Churn Label Construction

Use only the prediction window (*t*, *t* + H] to determine churn, applying the same valid-purchase definition as 5.2. A row is labelable only if *t* + H ≤ data end; later snapshots can be scored but not evaluated, and the last scored week feeds the dashboard's current target list (8.5).

**Label EDA.** Report the overall churn rate early — churn may be the *majority* class, which changes how PR-AUC and Precision should be read. Compare the label balance of the training sample and the Scoring Population, and show the churn rate by snapshot month (seasonality).

## 5.5 Feature Engineering, Customer Value and RFM Segments

**Features.** Generate historical features using information available up to the snapshot date only:

| Group | Features |
|--------------------|----------------------------------------------------------|
| RFM | Recency, frequency (occasions), monetary total |
| Purchase behavior | Average order value |
| Temporal | Tenure, mean inter-purchase gap, recency / typical gap, activity in the last 90 days |
| Product | Distinct products so far |
| Cancellation | Cancellation count, cancelled value share |
| Context (reporting) | Country (UK vs non-UK), one-time vs repeat flag |

Report summary statistics, the **distribution of each feature split by churn outcome**, and churn by point-in-time frequency and monetary decile.

**Customer value V.** Baseline proxy: V~i~ = total valid revenue up to *t* ÷ purchase occasions up to *t* (average order value), **winsorized at P99 fitted on the training period**, because wholesale customers dominate raw values (4.7). Stretch: a future-spend regression over (*t*, *t* + H] evaluated with MAE/RMSE.

**RFM segmentation (required baseline).** Score Recency, Frequency and Monetary into quintiles with cut-offs **fitted on the training period**, and map score combinations to named segments (Champions, Loyal, Potential, At Risk, Can't Lose, Hibernating). Report per segment: customer count, revenue share, churn rate and average V. The segmentation provides the RFM-segment targeting rule used as a business baseline (6.2, 7.9) and the `rfm_segment` column of the score table (8.4).

## 5.6 Leakage Prevention and Temporal Split

**Leakage prevention.** No future information may be used in feature construction; the prediction window is reserved for label construction. Verify the as-of construction directly, and keep consistency checks between related features (e.g. activity in the last 90 days ≥ 1 exactly when recency ≤ 90 days). Scalers, encoders, imputers, RFM quintile cut-offs and the V cap are fitted on the training period only. A **leakage audit table** lists every feature and confirms its time window.

**Temporal split with embargo.** Split chronologically by snapshot date into train, validation and test, with an **embargo of H days** between consecutive periods:

::: {.formula}
first validation date ≥ last training date + H  
first test date ≥ last validation date + H
:::

Without the embargo, labels of late training snapshots are observed during the validation period, so the model effectively learns from a period it is later evaluated on.

| Split | Rows used | Purpose |
|------------|----------------------------------------------|----------------------------------|
| Train | `in_training_sample` rows in the train period | Fit candidate models |
| Validation | `in_scoring_population` rows in the validation period | Tuning, calibration fit, choice of K |
| Final refit | `in_training_sample` rows re-derived over train + validation | Fit the selected model |
| Test | `in_scoring_population` rows in the test period | Final evaluation and policy simulation |
| Scoring only | all active rows of the last Monday with data | Dashboard current target list (unlabeled) |

Validation and test therefore measure performance exactly as the model would be used: ranking all active customers each week.

::: {.note}
**Timeline (H = 90 days, weekly Mondays, 6-month warm-up).** Labelable snapshots run from 2010-06-07 to 2011-09-05 (66 weekly dates).

| Period | Snapshot dates | Weekly dates |
|------------|------------------------------|------------|
| Train | 2010-06-07 → 2010-11-15 | 24 |
| *Embargo* | *90 days* | *—* |
| Validation | 2011-02-14 → 2011-04-04 | 8 |
| *Embargo* | *90 days* | *—* |
| Test | 2011-07-04 → 2011-09-05 | 10 |

24 of the 66 weekly dates are consumed by embargoes. The single test window covers one season only (no Christmas peak), so the rolling-origin backtest of 7.5 checks stability across seasons.
:::

**Feasibility check:** for each candidate H, count labelable weekly scoring dates and the train/validation/test weeks left after the embargoes. The test period should contain enough scoring dates (e.g. ≥ 8) for stable Top-K estimates.

Report the rows actually used per split: episode-sampled rows for training (and for the final refit over train + validation), all active rows for validation and test.

## 5.7 Conclusion

The final modeling dataset: customer ID, snapshot date, segment flag (one-time / repeat), RFM segment, value V, population flags (`in_training_sample`, `in_scoring_population`), sample weight, historical features, churn label and split assignment, produced by reproducible transformations from one config. Close with a summary table of the result of each step (cleaning, population, label, features, split) handed to Chapter 6.

# 6. Modeling

## 6.1 Modeling Objective

A **well-calibrated** estimate of P(Churn = 1 | history at *t*) that ranks customers well. Interpretability comes before model complexity.

## 6.2 Rule Baselines

| Baseline | Role |
|--------------------|----------------------------------------------------------|
| Random ranking | Reference for Lift = 1 and PR-AUC = churn rate. |
| Recency rule | Longest time since last purchase first. A strong, business-realistic rule the models must beat. |
| RFM-segment rule | Target the At Risk and Can't Lose segments (5.5). |

## 6.3 Baseline Models (Required)

| Model | Features | Purpose |
|--------------------|--------------------|----------------------------------|
| M1 Logistic Regression | RFM only | Interpretable reference |
| M2 Logistic Regression | Full feature set (5.5) | Does richer history help? |

An intermediate extended feature set (behavioral and temporal features added to RFM) shows where the gain, if any, comes from.

## 6.4 Stretch Models (After the Baseline Is Complete)

| Model | Purpose |
|--------------------|----------------------------------------------------------|
| Random Forest, XGBoost (full features) | Non-linear gains |
| Future-spend regression | Replace the V proxy; MAE/RMSE |
| Training ablation | Episode vs all-weekly weighted training (6.10) |

## 6.5 Tuning

Tune on the validation Scoring Population while preserving temporal separation (fit on train, select by validation metrics — not a cross-validated search, which would break the temporal/episode structure).

## 6.6 Calibration

Because *p* enters the margin formula as a value (7.7), calibration is required. Fit Platt or isotonic calibration on the validation period — **mandatory for tree ensembles, checked for Logistic Regression** — and compare the Brier score and reliability curve before and after on validation. Keep the calibrator only if it improves calibration without hurting ranking.

## 6.7 Final Refit

After selection, **retrain the chosen configuration on the Training Population re-derived over train + validation** (respecting the embargo before the test period) and evaluate once on the test Scoring Population.

## 6.8 Model Selection

Select on PR-AUC, calibration and weekly Lift\@K at the operating K, and on the expected margin of the resulting policy (7.9). A more complex model is chosen only if it clearly beats M2.

## 6.9 Model Interpretability

Standardized LR coefficients for the baseline and global/local explanations (SHAP) for the selected model. Findings are associations, not causes, and feed the executive brief and model card.

## 6.10 Training Population Ablation

Compare two training schemes for the selected algorithm and feature set, both evaluated on the **same** test Scoring Population:

| Scheme | Training rows |
|--------------------|----------------------------------------------------------|
| A — Episode (main) | `in_training_sample` rows, unweighted |
| B — All weekly, weighted | All weekly active rows in the training period, weighted by `sample_weight` |

If A matches B on PR-AUC and weekly Lift\@K, the episode rule reduces training data and redundancy without losing predictive quality. If B is clearly better, report it and discuss the trade-off.

# 7. Evaluation

## 7.1 Evaluation Framework

Evaluation runs on the test Scoring Population in three layers: **predictive**, **targeting** and **business scenario**. Results are reported overall and for one-time vs repeat customers.

All evaluation uses the **Scoring Population**: every active customer at every weekly test date. Because the historical data contain no campaigns, suppression is not applied in the main evaluation; an optional sensitivity run removes customers selected in the Top-K during the previous S weeks.

The same customer appears in several consecutive test weeks, so test rows are not independent. Uncertainty is reported across weeks (mean ± std, worst week) or with a customer-level bootstrap rather than as if rows were independent.

## 7.2 Predictive Performance

ROC-AUC, PR-AUC (with the test churn rate as its random reference), Precision, Recall and F1 at the chosen threshold. **Calibration** — reliability curve and Brier score — is required, because *p* enters the margin formula.

## 7.3 Targeting Performance

Per weekly test date: rank the full Scoring Population, cut the top *K*%, and compute Precision\@K, Recall\@K, Lift\@K, cumulative gain and **value captured\@K** (share of revenue at risk among targeted churners). Summarize as mean ± std over weeks, at K = 10% and at the operating K (20–30%).

## 7.4 Risk Bands

Observed churn rate and average V by predicted-risk decile, to confirm that higher risk means higher observed churn — and to show whether risk and value are aligned.

## 7.5 Rolling-Origin Backtest

The single test window covers about 15% of the labelable weeks and one season only. Repeat the full recipe (episode training sample, refit, embargo of H days) over 3–4 expanding-window folds whose test blocks cover roughly November 2010 – September 2011, including the Christmas peak. Report predictive, targeting and policy metrics as mean ± std across folds and by month. The main test split stays the official result; the backtest checks stability.

## 7.6 Revenue at Risk

::: {.formula}
RevenueAtRisk~i~ = V~i~  (one future purchase-equivalent, capped value proxy)
:::

Computed for churners using history up to *t* only. This is exposure, not observed lost revenue.

## 7.7 Expected Incremental Margin (Decision Rule)

For customer *i* offered discount *d*:

::: {.formula}
E[ΔM~i~] = p~i~ · e(d) · V~i~ · (m − d)  −  (1 − p~i~) · r · V~i~ · d  −  c~contact~
:::

- First term: margin recovered from would-be churners who respond.
- Second term: over-discounting cost on customers who would buy anyway (cannibalization).
- *p* must be calibrated because it is used as a value (6.6).

**Policy:** rank customers by E[ΔM], target the top *K*% (or until budget *B*), and never target customers with E[ΔM] ≤ 0.

## 7.8 Promotion Scenarios

At least three discount/response scenarios, each crossed with margin *m* and cannibalization *r*:

| Scenario | Discount *d* | Response *e* | Note |
|----------|--------------|--------------|------|
| Light | 5% | 20% | Low cost, low response |
| Base | 10% | 30% | Main scenario |
| Deep | 20% | 40% | High response, high over-discount risk |

Base values: *m* = 40%, *r* = 40%, c~contact~ = assumed GBP per contacted customer. Sensitivity: *m* ∈ {30, 40, 50%}, *r* ∈ {20, 40, 60%}.

Realized expected margin on test, using the actual labels of the targeted customers *T*:

::: {.formula}
M~policy~ = Σ~i∈T, churn=1~ e · V~i~ · (m − d)  −  Σ~i∈T, churn=0~ r · V~i~ · d  −  |T| · c~contact~
:::

## 7.9 Policy Comparison

| Policy | Who is targeted |
|--------------------|----------------------------------------------------------|
| P0 No offer | Nobody (margin change = 0) |
| P1 Blanket offer | All active customers (non-targeted policy) |
| P2 Random \@K | Random K% |
| P3 Recency rule \@K | Longest recency first |
| P4 RFM-segment rule | At Risk / Can't Lose segments |
| P5 Model risk-only \@K | Highest *p* |
| P6 Model value-aware \@K | Highest E[ΔM], only if E[ΔM] > 0 (recommended) |

Report for each policy, scenario and K: customers targeted, churners captured, offers to would-be buyers, revenue at risk covered, discount cost, contact cost, expected margin, and the difference versus P0 and P1. The main result table uses the base scenario at the operating K; the full grid goes to the appendix and the dashboard.

## 7.10 Cost Saving vs Model Value

Targeting *K*% instead of everyone saves (1 − *K*) of the contact cost with **any** ranking — this comes from targeting itself, not from the model. The model's value is the extra margin of P5/P6 over P2–P4 at the same K.

## 7.11 Model Comparison

Compare candidates (including baselines) on predictive metrics, targeting metrics, calibration and policy margin. Do not define a single "best" model from classification metrics alone when the objective depends on ranking, value and cost.

## 7.12 Error Analysis

False positives (discount wasted) and false negatives (missed churners) by segment: one-time vs repeat, UK vs non-UK, wholesale-sized customers, RFM segment, and by test week (seasonality).

::: {.principle}
**Project-wide business-impact principle.** All margin figures are simulations under stated assumptions (*d*, *e*, *m*, *r*, c~contact~). Online Retail II has no treatment/control data, so incremental revenue, causal retention effect and true ROI cannot be identified.
:::

# 8. Targeting Policy & Decision Support

::: {.note}
This chapter turns scores into a weekly action list and a dashboard. It is a **proposed operating design demonstrated on historical data**, not a live system.
:::

## 8.1 Objective

Provide a recurring, reproducible weekly decision: which active customers receive which action, within the budget.

## 8.2 Weekly Decision Cycle

The training-episode cooldown is **not** part of this cycle; it exists only in training-data construction.

::: {.flow}
Latest transactions → Valid purchases → Active customers (Recency ≤ R~max~) → Remove suppressed customers → Features at *t* → Calibrated *p* → V and RFM segment → E[ΔM] → Budget and business rules → Action list
:::

## 8.3 Recommendation Layer

| Action | Rule (initial, configurable) | Rationale |
|----------|--------------------------------|--------------------------------------------|
| Retain | E[ΔM] > 0 at the chosen *d* and within Top-K | Offer justified by risk and value |
| Nurture | High *p* (top risk bands) but E[ΔM] ≤ 0 | Low-value at-risk customer: email/content, no discount |
| No-offer | Low *p* | Likely to buy anyway; a discount mainly cannibalizes margin |

## 8.4 Customer-Level Score Table (Deliverable)

| Column | Description |
|--------------------|----------------------------------------------------------|
| `customer_id`, `snapshot_date` | Keys |
| `churn_probability` | Calibrated *p* |
| `risk_score`, `risk_band` | 0–1000 score and its decile band |
| `value_v` | Capped average order value |
| `rfm_segment` | Segment from 5.5 |
| `expected_margin` | E[ΔM] at the chosen scenario |
| `rank` | By expected margin |
| `action` | Retain / Nurture / No-offer |
| `model_version` | Version tag |

## 8.5 Dashboard (Deliverable)

- **KPIs (3–5):** active customers this week; revenue at risk in the target list; expected incremental margin of the chosen policy; share of offers going to likely buyers; customers per action (Retain / Nurture / No-offer).
- **Actionable view:** controls for K (or budget), discount scenario and margin; the target list updates with actions, and a small chart compares the chosen policy with no-offer, blanket offer and the recency rule.
- **Data:** test weeks for evaluation views; the last available Monday (scored, unlabeled) for the current target list.

## 8.6 Suppression

Customers **contacted** in a campaign leave the Scoring Population for S weeks (4–8), so they are not offered repeatedly. Suppression applies only to targeted customers, is independent of the training cooldown C, and is logged. Historical evaluation has no campaigns, so suppression is empty; an optional sensitivity run removes the last S weeks' Top-K customers.

## 8.7 Monitoring and Retraining

After H days, compare predicted and actual churn; track weekly Lift\@K, calibration, risk-band behavior and feature drift (transaction volume, feature distributions, seasonal shifts in the behavior–churn relationship). Retrain with the same embargoed design and Training Population rule when performance drops. If campaigns are logged, measure *e* and *r* with a holdout control group and replace the assumptions.

# 9. Robustness, Ethics & Limitations

## 9.1 Robustness Checks

- Week-to-week and fold-to-fold stability of Lift\@K and expected margin (mean ± std, worst week; 7.5).
- Sensitivity of the policy ranking to *d*, *e*, *m*, *r* and c~contact~ (grid or tornado table).
- Sensitivity to H (60 vs 90) and R~max~ if time allows.
- Results with and without the V cap, to show the influence of wholesale customers.

## 9.2 Ethics and Bias

- **Differential pricing:** some customers receive discounts and others do not. Rules are based on behavior, not on protected attributes, and are documented.
- **Coverage:** compare target rates and precision across UK vs non-UK and customer size; check that the policy does not serve only large wholesale accounts.
- **Privacy:** Customer ID is pseudonymous; no personal data are added.
- **Customer experience:** suppression prevents repeated offers.

## 9.3 Limitations

- No treatment/control data: no causal uplift, incremental revenue or ROI.
- Margin, response and cannibalization are assumptions.
- About two years of data and the embargoes leave a short test window (about 10 weeks) without a Christmas peak; the backtest only partly compensates.
- V is a proxy for future value.
- Public teaching dataset: external validity is limited.

## 9.4 Model/Analysis Card (Deliverable)

One page: purpose and decision, unit of analysis, target and horizon, data and period, features, validation design, metrics with baselines, calibration, assumptions (*d*, *e*, *m*, *r*, c~contact~), risks, ethics/bias checks, limitations, intended and non-intended use.

# 10. Conclusion & Recommendations

## 10.1 Summary

The project links a weekly retention-offer decision to churn risk, customer value and the margin cost of discounts.

## 10.2 Recommendation (To Be Filled With Results)

State the recommended policy (P6 unless results show otherwise), the operating K, the discount level, the expected incremental margin range across scenarios, and the comparison with no-offer, blanket offer and the recency rule.

## 10.3 Business-to-Technical Translation

Business goal: more margin from existing customers with less wasted discount. Technical solution: calibrated churn model, capped value proxy, expected-margin ranking, Retain / Nurture / No-offer actions.

## 10.4 Limitations

Summarize 9.3 in plain business language.

## 10.5 Next Steps

Run a randomized pilot with a holdout control group to measure *e* and *r*; add margin and cost data; move to uplift modeling and budget optimization.
