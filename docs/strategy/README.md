---
title: "Customer Retention & Churn Prediction"
subtitle: "Data Science Project Strategy & AI Coding Context"
date: "Version 1 · September 2026"
---

::: {.purpose}
**Purpose.** This document is the source of truth for the business and analytical strategy of the Customer Retention & Churn Prediction project. The core **business problem** is customer retention; the core **analytical problem** is customer churn prediction. The report must preserve the 12 main chapters listed below.

1\. Introduction · 2. Business Understanding · 3. Analytic Approach · 4. Data Requirements · 5. Data Collection · 6. Data Understanding · 7. Data Preparation · 8. Modeling · 9. Evaluation · 10. Deployment · 11. Feedback · 12. Conclusion
:::

# Introduction

## Project Background

The project focuses on customer retention in an e-commerce business. Revenue is generated through customer purchases, while customer acquisition and retention both require business resources. The project investigates how Data Science can support the retention of existing customers by predicting churn.

## Business Context

The business generates sales revenue from new customers making first purchases and from existing customers making repeat purchases. Existing customers may stop purchasing, which creates a customer churn problem.

## Problem Statement

The business needs to increase sales revenue while operating under limited retention and marketing resources. The project therefore focuses on identifying existing customers at risk of churn so that retention resources can be prioritized.

## Research Questions

1. How can transaction history be transformed into customer-level observations for churn prediction?
2. How effectively can purchasing behavior predict future churn?
3. Can churn-risk ranking support prioritization under limited retention capacity, compared with simple rule-based targeting?
4. Which behavioral characteristics are *associated* with churn? (Associational, not causal.)

## Project Objectives

Design and evaluate a customer churn prediction system that supports the retention of existing customers by producing customer-level churn risk and supporting customer prioritization.

## Scope and Limitations

**In scope:** existing-customer retention, churn prediction, customer snapshots, weekly batch scoring, risk ranking and Top-K targeting.

**Out of scope / not claimed:**

- Causal uplift and incremental campaign revenue — the transaction data contain no randomized treatment/control information.
- A live deployment. **Chapters 10 and 11 describe a proposed operational design**; they are not evaluated on a running system.
- Profit analysis based on real margins or costs — these are not in the data and enter only as stated assumptions.

## Report Structure

The report follows the 12 chapters defined in this strategy.

# Business Understanding

## E-commerce Sales Business

The business sells products through an e-commerce transaction system, and sales generate revenue from customer purchases. At a high level:

::: {.formula}
Profit = Sales Revenue − Business Costs
:::

Costs may include product/operating costs, marketing costs and retention incentive costs.

## Revenue from New and Existing Customers

Sales revenue comes from new customers making first purchases and existing customers making repeat purchases. Existing-customer revenue is generated when previously acquired customers continue purchasing.

## Customer Churn as a Business Problem

Existing customers may stop purchasing, and churn reduces future sales revenue from the existing customer base. The business therefore needs to identify customers whose behavior indicates churn risk.

## Retention and Marketing Cost Constraint

Retention activities require resources. The company cannot necessarily apply the same retention action to every existing customer because marketing budget, campaign capacity and incentives are limited. Retention resources must be allocated selectively.

## Business Objective

The primary business objective is to increase total sales revenue by increasing repeat-purchase revenue from existing customers, while reducing inefficient retention and marketing expenditure. The Data Science project supports this by identifying and prioritizing existing customers at higher risk of churn.

## Business Success Perspective

Revenue measures include total revenue, existing-customer revenue and repeat-purchase activity. Marketing measures include target population size and retention cost where available. Model metrics assess prioritization ability; causal revenue uplift cannot be claimed without campaign outcome data.

## Success Criteria

Success is defined **relative to simple alternatives** rather than by arbitrary absolute thresholds:

| Level | Criterion (evaluated on the unseen test period) |
|--------------|------------------------------------------------------------------|
| Predictive | Final model PR-AUC exceeds the test-period churn rate (random reference) and the recency-rule baseline. |
| Targeting | Final model Lift\@K and Recall\@K at the chosen operating K (e.g. 20–30%) exceed the recency-rule baseline, averaged over weekly scoring dates. |
| Stability | Top-K performance does not collapse on individual test weeks (report mean ± standard deviation). |
| Business scenario | Under the base scenario, estimated net impact at the chosen K is positive and higher than for recency-rule and random targeting at the same K. |

# Analytic Approach

## From Business Objective to Analytical Problem

Increasing revenue requires more repeat-purchase revenue from existing customers and efficient use of retention resources. The analytical direction is therefore to identify existing customers at risk of not purchasing again, estimate churn probability, and prioritize customers for retention activities.

## Customer Retention Strategy

::: {.flow}
Active Customers → Scoring Population (minus suppressed) → Churn Prediction → Churn Probability → Risk Ranking → Capacity & Business Rules → Target Population → Retention Campaign
:::

## Churn Prediction Definition

The core technical problem is **binary customer churn prediction**: at snapshot date *t*, predict whether an eligible existing customer will make no valid purchase during a future prediction window.

## Customer Snapshot

The fundamental modeling unit is **Customer × Snapshot Date**, not the transaction. Each snapshot represents the customer's state at a point in time and contains features calculated only from information available up to that point.

Snapshots are drawn from **two distinct populations** with different purposes (3.7): a *Training Population*, sampled to limit redundant rows, and a *Scoring Population*, which mirrors weekly operations and is used for validation, test and deployment.

## Observation Window

Historical behavior is collected within an observation window ending at the snapshot date. Features may use full history up to *t* as well as fixed look-back windows (e.g. last 30, 90, 180 days). Features must be derived exclusively from information available at or before *t*.

## Prediction Window, Label and Label Availability

For customer *i* at snapshot date *t*:

::: {.formula}
Churn(i, t) = 1  if no valid purchase occurs in (t, t + H]  
Churn(i, t) = 0  otherwise
:::

*H* is the prediction horizon, informed by observed inter-purchase behavior (6.9).

**Label availability rule:** a snapshot can be labeled only if its full prediction window is observed, i.e. *t + H ≤ data end date*. Snapshots after this date can be scored but not evaluated.

## Population Design

| Population | Definition | Used for |
|------------------|----------------------------------------------------|--------------|
| Customer Universe | All customers with a valid Customer ID. | Data Understanding |
| Existing Customer at *t* | At least one valid purchase on or before *t*. | — |
| Active Customer at *t* | Existing customer with Recency ≤ R~max~ (**activity condition**). | Base for both populations below |
| **Training Population** | Active customers at weekly dates, **sampled with the eligibility-episode rule** (3.9). | Model training |
| **Scoring Population at *t*** | **All** active customers at weekly date *t*, excluding customers under **suppression** (10.5). **No episode cooldown.** | Validation, test, deployment |
| Target Population | Customers selected from the Scoring Population after ranking, marketing-capacity constraints and business rules. | Retention campaign |

**Why two populations.** The eligibility episode is a *data-sampling technique* for training; suppression is an *operational rule* for contacted customers. They solve different problems and must not be mixed:

- If the episode cooldown were applied to scoring, only about 1/13 of active customers would be scored each week (with H = 90 days and C = H), a customer scored as low risk would not be re-scored for 90 days even if their behavior deteriorated, and weekly Top-K would be chosen from a small, arbitrary subset rather than from all active customers.
- If every weekly snapshot were used for training without control, the training data would contain many near-duplicate rows with overlapping, strongly correlated labels.

Both populations are subsets of the same **weekly active-customer base table** (7.5), so they share one feature and label pipeline.

**Why the activity condition matters.** A customer whose last purchase was, for example, 400 days ago is almost certainly a churner. Including such customers adds "easy" positives that inflate ROC-AUC and Precision\@K while offering no business value, because they are already lost. R~max~ is set from Data Understanding (e.g. a high percentile of inter-purchase intervals) and must be configurable.

**One-time buyers.** Customers with exactly one purchase at *t* behave differently from repeat customers and are usually the largest group. They stay in both populations and results are also reported separately for the one-time and repeat segments.

## Serving Strategy

The primary serving strategy is **weekly batch scoring** of the full Scoring Population. Data may be collected daily or continuously, but core model scoring is weekly. Every active, non-suppressed customer is re-scored every week, so a change in behavior is reflected in the next weekly ranking.

## Training Population: Eligibility Episode and Cooldown

**Problem.** Consecutive snapshots of the same customer differ only slightly (mostly recency increases), and their prediction windows overlap, so their labels are strongly correlated. Weekly instead of daily snapshots reduces this by a factor of seven but does not remove it. Overlapping rows are not wrong in themselves; the risks are treating them as independent observations and letting long-history customers dominate training.

**Rule (training only).** A customer enters an **eligibility episode** at the first weekly date *t* at which they are active. A training snapshot is taken at *t*, and the customer produces no further training snapshot until the episode ends:

- **Default — fixed cooldown:** the episode lasts *C = H*. The next training snapshot is the first weekly date ≥ *t + C* at which the customer is still active. Each customer's training labels then have non-overlapping prediction windows, and each customer contributes a similar number of rows.
- **Alternative — reset on purchase (configurable):** if the customer makes a valid purchase before *t + C*, the episode ends early and the next training snapshot is the first weekly date after that purchase. This captures the new customer state sooner. Whether a new snapshot is created depends only on purchases before that snapshot date, so no leakage is introduced.
- A customer who exceeds R~max~ leaves the active base; if they purchase again, they re-enter and start a new episode.

**Scope.** The episode rule is applied **only when building the Training Population** — in the training period, and in the train + validation period when the final model is retrained (8.7). It is **never** applied to validation, test or deployment scoring.

**Comparison design.** As an ablation (8.10), a second training scheme uses **all weekly active snapshots with sample weights** *w = 1 / (number of training snapshots of that customer)*. Both schemes are evaluated on the same Scoring Population test set.

## Analytical Design Informed by Data Understanding

Prediction horizon, observation windows, activity threshold, scoring cadence, training-episode cooldown and suppression period are supported by Data Understanding — inter-purchase intervals, activity patterns and repeated-observation behavior — rather than arbitrary assumptions.

## Feature Engineering Strategy

Features are constructed from customer history available at the snapshot date. Main groups:

| Group | Examples |
|------------------|----------------------------------------------------------|
| RFM | Recency, frequency (purchase occasions), monetary value |
| Purchase behavior | Average order value, items per order, spend trend |
| Temporal behavior | Tenure, mean/std of inter-purchase gaps, recency ÷ typical gap, activity in last 30/90/180 days |
| Product behavior | Number of distinct products, repeat-product share |
| Cancellation behavior | Cancellation count and share of value |

## Modeling Strategy

- Baselines: random ranking and a recency rule (see 8.2).
- Algorithms: Logistic Regression, Random Forest, XGBoost.
- Feature sets: RFM → RFM + behavioral/temporal → full feature set.
- Training on the **Training Population** (episode-sampled); validation, test and scoring on the **Scoring Population** (all active, non-suppressed customers each week).
- Temporal train/validation/test split **with an embargo of H days** between consecutive periods (7.9).
- No target leakage (7.8).

## Marketing Use of Model Output

The model produces churn probability, which is used to rank customers. Marketing capacity and business rules then create the target population. The model does not decide the treatment (for example, voucher versus advertising).

## Evaluation Strategy

Evaluate the system at three levels: **predictive performance**, **targeting/ranking performance**, and **business-impact estimation**. Business-impact results are scenario estimates under explicit assumptions, not observed causal effects. Details are in Chapter 9.

## Configuration Parameters

All design parameters live in a single configuration and are reported in the final report.

| Parameter | Symbol | Initial candidate | Decided in |
|----------------------|----------|-------------------------------------|------------|
| Prediction horizon | H | 60 / 90 / 120 days | 6.9, 6.14 |
| Activity threshold | R~max~ | High percentile of inter-purchase gap (e.g. P90–P95) | 6.9, 6.14 |
| Training-episode cooldown | C | = H (fixed); alternative: reset on purchase | 6.14, 8.10 |
| Training sampling scheme | — | Episode (main) vs. all weekly + weights (ablation) | 8.10 |
| Suppression period (operations) | S | e.g. 4–8 weeks after contact | 10.5 |
| Scoring cadence | — | Weekly (Monday) | 3.8 |
| Feature warm-up | — | ≈ 6 months of history before first snapshot | 6.14 |
| Split embargo | — | = H | 7.9 |
| Target share | K | 10 / 20 / 30 / 40 / 50% | 9.6 |
| Retention effectiveness | e | 30 / 40 / 50% | 9.7 |
| Gross margin | m | Assumed, e.g. 30 / 40 / 50% | 9.9 |
| Contact cost per customer | c~contact~ | Assumed constant (GBP) | 9.8 |
| Incentive cost per redemption | c~inc~ | Assumed (GBP or % of order value) | 9.9 |

# Data Requirements

## Data Requirement Overview

Requirements derive from the business and analytical objectives: identify customers, purchases, products, revenue, customer history, temporal behavior and future outcomes.

## Customer Information

Customer ID and Country. Customer ID is required to build customer histories and snapshots.

## Transaction Information

Invoice, InvoiceDate, Customer ID, StockCode, Description, Quantity and Price. Use the actual field names of the selected dataset (Online Retail II uses `Invoice`, `Price` and `Customer ID`).

## Revenue Information

Transaction revenue is derived as Quantity × Price and can be aggregated by customer, purchase occasion and month.

## Product Information

StockCode and Description support product-level and customer-product analysis. Product cost, category hierarchy and margin are not available.

## Temporal Information

InvoiceDate is required for observation windows, prediction windows, monthly metrics, inter-purchase intervals, lifecycle analysis and temporal splits.

## Marketing and Cost Information

A production system would ideally collect CampaignID, CustomerID, CampaignDate, CampaignType, Channel, Treatment, Discount, VoucherCost, MarketingCost and campaign outcomes. None of these are in the current dataset; the related quantities in Chapter 9 are assumptions.

## Analytical Dataset Design

Each final modeling row represents **Customer × Snapshot Date**, with historical features and a future churn label.

# Data Collection

## Simulated Business Data Source

Online Retail II is described as a historical extract of data collected from an e-commerce transaction system. The narrative is not "download a dataset and analyze it" but "the analytical system receives transaction records extracted from the sales system".

## Simulated E-commerce Transaction System

The simulated architecture contains Order, Product, Customer and Transaction systems. A completed purchase produces invoice, customer, product, quantity, price, timestamp and country information.

## Data Extraction

Transaction data are assumed to be extracted from the transactional database into raw analytical storage, followed by validation and analytical transformation.

## Collection Frequency

Transactions may be generated continuously while the analytical platform ingests data daily or on another schedule. Ingestion frequency is distinct from model scoring frequency.

## Historical Dataset Representation

Online Retail II is the historical extract used for experimentation, and the report transparently identifies it as a public dataset. Properties that affect the design:

- It covers roughly **two years** (December 2009 – December 2011), which limits the number of labelable weekly snapshots (see 6.14, 7.9).
- The retailer is UK-based and many customers are **wholesalers/businesses**, so spend per customer is highly heterogeneous.
- Demand is strongly **seasonal**, with a peak before Christmas.

# Data Understanding

## Objective

Determine what the transaction data reveal about sales, customers, revenue composition, repeat purchasing, behavior, and the feasibility of churn prediction.

::: {.note}
**Iterative link with Chapter 7.** Sections 6.13–6.14 need churn labels and snapshots, which are formally built in Chapter 7. This chapter uses **provisional** labels and snapshots computed for candidate values of H, R~max~ and C. Final values are fixed in Chapter 7, following the iterative nature of the Data Science methodology.
:::

## Dataset Overview

Transaction count, unique customers, unique products, date range, countries, total quantity and revenue.

## Data Quality

Missing values (especially Customer ID), duplicate records, invalid Quantity/Price values, cancellation records, non-product stock codes and other quality issues.

## Monthly Sales Performance

Monthly customers, orders, quantity and revenue to understand sales dynamics and seasonality.

## New vs Existing Customers

Classify customers by whether the period contains their first valid purchase or a subsequent one. Compare counts and revenue contribution.

## Revenue from New and Existing Customers

New Customer Revenue, Existing Customer Revenue, Total Revenue and monthly shares. This directly supports the business case for retaining existing customers.

## Repeat Customer Analysis

One-time versus repeat customers: counts, shares, orders and revenue contribution.

## Monthly Repurchase Analysis

Eligible existing customers, repeat buyers and repurchase rate by month.

## Inter-Purchase Interval

Time between consecutive purchase occasions: mean, median, percentiles and distribution. Results inform **H** and **R~max~**.

## Customer Monetary and Behavioral Analysis

Customer revenue, order frequency, quantity, average order value and behavioral heterogeneity (including the influence of large wholesale customers).

## Customer Lifecycle

Progression from first purchase to repeat purchase, active behavior, increasing recency, at-risk behavior and potential churn.

## RFM Exploration

Recency, Frequency and Monetary distributions and their relationship with subsequent purchasing.

## Churn Exploration

Using provisional labels: churn rate over time and across recency, frequency, monetary value and other characteristics. **Report the overall churn rate early** — with many one-time buyers, churn may be the *majority* class, which changes how PR-AUC and Precision should be read.

## Snapshot Population and Design Feasibility

1. Compare daily, weekly-all and weekly episode-based snapshots: number of rows, rows per customer, share of near-duplicate rows, and class distribution. This quantifies why daily snapshots are rejected and how much the episode rule reduces training redundancy.
2. Show the effect of the activity condition (R~max~) on population size and churn rate.
3. Report the **weekly Scoring Population size** over time, which is the pool from which marketing selects Top-K each week, and contrast it with the much smaller number of customers that would be scored if the episode rule were wrongly applied to scoring.
4. **Feasibility check:** for each candidate H, count labelable weekly scoring dates after the warm-up and after the embargoes in 7.9. The test period should contain enough scoring dates (e.g. ≥ 8) for stable Top-K estimates.

Use the evidence to justify weekly batch scoring, the training-episode cooldown, R~max~ and the final H.

## Key Findings

Business-oriented findings on revenue composition, repeat purchasing, customer heterogeneity, purchase intervals, churn behavior, data limitations and the chosen snapshot/prediction-window design.

# Data Preparation

## Data Cleaning

Clean transaction records using documented, reproducible rules. Every rule reports how many rows and how much revenue it removes.

## Valid Transaction Definition

A **valid purchase** row must satisfy all of the following:

| Rule | Treatment |
|--------------------|----------------------------------------------------------|
| Customer ID present | Rows without Customer ID are excluded from customer history (still counted in 6.2 overview). |
| Not a cancellation | Invoices starting with "C" are excluded from purchases; kept separately for cancellation features. |
| Quantity > 0 and Price > 0 | Otherwise excluded. |
| Product stock code | Non-product codes (e.g. postage, manual adjustments, bank charges, marketplace fees) are excluded; the final list is documented. |
| Exact duplicates | Removed. |

A **purchase occasion** is one distinct invoice (or one customer-day, if chosen and documented). Frequency and inter-purchase intervals are computed on purchase occasions, not rows.

## Customer History Construction

Aggregate valid transaction history by customer while preserving timestamps and product information.

## Eligibility Construction

Eligibility is built in two steps.

**Step 1 — Weekly active-customer base.** At each weekly date *t*, a customer is active if:

1. at least one valid purchase exists on or before *t* (existing customer);
2. Recency at *t* ≤ R~max~ (activity condition);
3. *t* lies within the snapshot period (after warm-up; *t + H* ≤ data end for labeled rows).

**Step 2 — Population flags on the base table.**

| Flag | Rule |
|--------------------|----------------------------------------------------------|
| `in_training_sample` | Row selected by the eligibility-episode rule (3.9). Computed per customer in date order. |
| `in_scoring_population` | Row is active and the customer is not under suppression at *t*. In the historical data there are no real campaigns, so suppression is empty and every active row is in the Scoring Population (see 9.1). |
| `sample_weight` | 1 / (number of weekly active rows of the customer in the training period); used only by the ablation scheme (8.10). |

## Customer Snapshot Construction

Construct **one weekly active-customer base table** of Customer × Snapshot Date rows, then compute features and labels once for this table. The Training Population and the Scoring Population are filters on it (via the flags above), which guarantees that both use identical feature and label definitions.

## Feature Engineering

Generate historical features using information available up to the snapshot date only.

## Churn Label Construction

Use only the prediction window (t, t + H] to determine churn, applying the same valid-purchase definition as 7.2.

## Leakage Prevention

No future information may be used in feature construction. The prediction window is reserved for label construction. Aggregates that depend on the whole dataset (scalers, encoders, imputation values, percentile thresholds) are fitted on the training period only.

## Temporal Dataset Split with Embargo

Split chronologically by snapshot date into train, validation and test, with an **embargo of H days** between consecutive periods:

::: {.formula}
first validation date ≥ last training date + H  
first test date ≥ last validation date + H
:::

Without the embargo, labels of late training snapshots are observed during the validation period, so the model effectively learns from a period it is later evaluated on.

| Split | Rows used | Purpose |
|------------|----------------------------------------------|----------------------------------|
| Train | `in_training_sample` rows in the train period | Fit candidate models |
| Validation | `in_scoring_population` rows in the validation period | Tuning, model selection, calibration fit |
| Final refit | `in_training_sample` rows re-derived over train + validation | Fit the selected model |
| Test | `in_scoring_population` rows in the test period | Final, operation-like evaluation |

Validation and test therefore measure performance exactly as the model would be used: ranking all active customers each week.

::: {.note}
**Illustrative timeline (H = 90 days, weekly Mondays, ≈ 6-month warm-up).** Labelable snapshots run from 2010-06-07 to 2011-09-05 (66 weekly dates).

| Period | Snapshot dates | Weekly dates |
|------------|------------------------------|------------|
| Train | 2010-06-07 → 2010-11-15 | 24 |
| *Embargo* | *90 days* | *—* |
| Validation | 2011-02-14 → 2011-04-04 | 8 |
| *Embargo* | *90 days* | *—* |
| Test | 2011-07-04 → 2011-09-05 | 10 |

About 24 of the 66 weekly dates are consumed by embargoes. If the test period is too short, options are a shorter H, a shorter warm-up, or rolling-origin validation within the training period instead of a separate validation block. The final choice is documented.
:::

## Final Modeling Dataset

Customer ID, snapshot date, segment flag (one-time / repeat), population flags (`in_training_sample`, `in_scoring_population`), sample weight, historical features, churn label and split assignment, produced by reproducible transformations from a single configuration.

# Modeling

## Modeling Objective

Estimate P(Churn = 1 | Customer History at Snapshot).

## Baselines

| Baseline | Role |
|--------------------|----------------------------------------------------------|
| Random ranking | Reference for Lift = 1 and PR-AUC = churn rate. |
| Recency rule | Rank customers by recency (longest since last purchase first). A strong, business-realistic rule the models must beat. |
| Optional: RFM score rule | Rank by a simple combined RFM score. |

## RFM Model

A model using Recency, Frequency and Monetary features.

## Extended Feature Model

Add behavioral and temporal features to test whether richer history improves prediction.

## Full Feature Model

Add product and cancellation features where supported.

## Algorithms

Compare Logistic Regression, Random Forest and XGBoost.

## Hyperparameter Tuning

Tune on the validation period while preserving temporal separation. Validation metrics are computed on the Scoring Population. After selection, **retrain the chosen configuration on the Training Population re-derived over train + validation** (respecting the embargo before the test period) and evaluate once on the test Scoring Population.

## Model Selection

Select the final model using predictive and ranking performance, with particular attention to Top-K targeting metrics at the operating K.

## Model Interpretability

Feature importance and, where appropriate, global/local explanations (e.g. SHAP). Findings are described as associations, not causes.

## Training Population Ablation

Compare two training schemes for the selected algorithm and feature set, both evaluated on the **same** test Scoring Population:

| Scheme | Training rows |
|--------------------|----------------------------------------------------------|
| A — Episode (main) | `in_training_sample` rows, unweighted |
| B — All weekly, weighted | All weekly active rows in the training period, weighted by `sample_weight` |

If A matches B on PR-AUC and weekly Lift\@K, the episode rule reduces training data and redundancy without losing predictive quality. If B is clearly better, report it and discuss the trade-off.

# Evaluation

## Evaluation Framework

Evaluate the final system on the unseen future test period using three layers: **Predictive Performance**, **Targeting Performance** and **Business Impact Estimation**. Results are reported overall and for the one-time and repeat segments.

All evaluation uses the **Scoring Population**: every active customer at every weekly test date. Because the historical data contain no campaigns, suppression is not applied in the main evaluation. An optional sensitivity analysis can simulate suppression by removing customers selected in the Top-K during the previous S weeks.

The same customer appears in several consecutive test weeks, so test rows are not independent. This mirrors real operations and is acceptable, but uncertainty is reported across weeks (mean ± standard deviation) or with a customer-level bootstrap rather than as if rows were independent.

## Classification Performance

ROC-AUC, PR-AUC, Precision, Recall and F1. Always report the test churn rate next to PR-AUC as its random reference. **Calibration** (reliability curve, Brier score) is required whenever probabilities are used as values, e.g. expected revenue at risk; Random Forest and XGBoost may need Platt or isotonic calibration fitted on the validation period.

## Ranking Performance

Precision\@K, Recall\@K, Lift\@K and cumulative Gain. Because targeting happens **each week**, these metrics are computed **per scoring date** — rank that week's full Scoring Population, cut the top K% — and then summarized as mean ± standard deviation across test weeks. Pooled curves may be shown additionally but are not the primary result.

## Risk Band Analysis

Compare actual churn rates and customer characteristics across risk bands (e.g. deciles) to check that higher predicted risk corresponds to higher observed churn.

## Customer Revenue at Risk

Estimate potential future revenue associated with churners, using a customer-level proxy:

::: {.formula}
AvgPurchaseValue~i~ = TotalRevenue~i~ ÷ PurchaseOccasions~i~  
RevenueAtRisk~i~ = AvgPurchaseValue~i~  (one future purchase-equivalent)  
or, with enough history: RevenueAtRisk~i~ = AvgMonthlyRevenue~i~ × H ÷ 30
:::

All inputs use history up to the snapshot date only. These are estimates of revenue exposure, not observed lost revenue.

## Business Impact Scenario

For each target share K ∈ {10, 20, 30, 40, 50%}, per scoring week: customers targeted (N~K~), true churners captured (TP~K~), non-churners targeted (FP~K~), Recall\@K, and revenue at risk among targeted churners.

## Retention Effectiveness Assumption

Retention success is an assumption, not an observed result. Scenarios: **Conservative e = 30%**, **Base e = 40%**, **Optimistic e = 50%**.

::: {.formula}
RevenuePreserved = RevenueAtRisk(targeted churners) × e
:::

## Marketing Cost and the Value of the Model

If contact cost per customer is roughly constant, targeting K% instead of everyone reduces reach cost by (1 − K). This saving comes from **targeting itself**, not from the model — any ranking at the same K saves the same amount.

The model's contribution is measured by comparing, **at the same K**, the churners captured, revenue preserved and net impact of:

1. the final model,
2. the recency-rule baseline, and
3. random targeting.

## Net Business Impact

Because preserved revenue is not profit, a margin assumption converts it before subtracting costs:

::: {.formula}
GrossProfitPreserved = RevenuePreserved × m  
ContactCost = N~K~ × c~contact~  
IncentiveCost = (TP~K~ × e + FP~K~ × r) × c~inc~  
NetBusinessImpact = GrossProfitPreserved − ContactCost − IncentiveCost
:::

Here *r* is the assumed share of targeted non-churners who redeem an incentive although they would have purchased anyway (cannibalization). Optionally report GrossProfitPreserved ÷ (ContactCost + IncentiveCost).

## Scenario Analysis

Present a business-impact table across K and e, with m, c~contact~, c~inc~ and r stated explicitly:

| Target % | Customers | Churners captured | Recall\@K | Revenue at risk | Preserved (30/40/50%) | Costs | Net impact | Net impact vs recency rule |
|---|---|---|---|---|---|---|---|---|
| 10% | … | … | … | … | … | … | … | … |
| 20% | … | … | … | … | … | … | … | … |
| … | … | … | … | … | … | … | … | … |

## Model Comparison

Compare candidates (including baselines) on predictive metrics, ranking metrics, calibration, operational targeting capacity and business-impact scenarios. Do not define a single "best" model from classification metrics alone when the objective depends on ranking and cost constraints.

## Error Analysis and Limitations

Investigate false positives, false negatives, segments (one-time vs repeat, UK vs non-UK, wholesale-sized customers) and weeks where performance differs, including seasonal effects. State clearly that Online Retail II contains no campaign treatment/control information, so incremental revenue, causal retention effect and true campaign ROI cannot be identified. Business-impact results are scenario estimates.

::: {.principle}
**Project-wide business-impact principle.** The model identifies and prioritizes churn risk; the retention campaign determines treatment; actual incremental revenue requires treatment/control data. Revenue at risk, revenue preserved, cost saving and net impact are simulation outputs unless supported by observed campaign outcomes.
:::

# Deployment

::: {.note}
This chapter describes a **proposed operational design**. It is not implemented as a live system in this project.
:::

## Deployment Objective

Provide recurring churn scoring that produces a prioritized existing-customer population for retention marketing.

## Weekly Batch Pipeline

The training-episode cooldown is **not** part of this pipeline; it exists only in training-data construction.

::: {.flow}
Latest transactions → Valid transactions → Active customers (Recency ≤ R~max~) → Remove suppressed customers → Snapshots → Features → Scoring → Ranking → Capacity & business rules → Target population
:::

## Customer Risk Output

Minimum output: `customer_id`, `snapshot_date`, `churn_probability`, `risk_band`, `rank`, `segment`, `model_version`.

## Target Population

Select target customers based on churn risk, marketing capacity and business constraints.

## Suppression

Customers **contacted** in a campaign are excluded from the Scoring Population for a configurable suppression period *S* (e.g. 4–8 weeks), so they are not targeted repeatedly. Suppression:

- applies only to customers who were actually targeted, not to everyone who was scored;
- is an operational business rule, independent of the training-episode cooldown *C*;
- is logged, so that suppressed customers can be analyzed separately once campaign data exist.

## Monitoring

Data quality, feature drift, model performance, calibration, lift, risk-band behavior and business metrics where available.

# Feedback

::: {.note}
Like Chapter 10, this chapter is a **proposed design**.
:::

## Feedback Loop

After H days, observe customer purchases and compare predicted churn with actual churn.

## Model Performance Monitoring

Track classification and weekly Top-K ranking metrics over time.

## Business Outcome Monitoring

Track repurchase rate, revenue contribution, target population and campaign cost when available.

## Data Drift

Monitor transaction volume, feature distributions and customer behavior.

## Concept Drift

Monitor changes in the relationship between behavior and future churn, including seasonal shifts.

## Campaign Feedback

If treatment/control data become available, evaluate campaign response and incremental outcomes; this would also allow the assumed e, r and costs to be replaced by measured values.

## Model Retraining

Retrain periodically or when performance, drift or behavior changes justify it, using the same embargoed temporal design and the same Training Population rule.

# Conclusion

## Summary of Strategy

The project connects e-commerce revenue generation with customer retention and churn prediction.

## Business-to-Technical Translation

The business objective is to increase sales revenue through repeat-purchase revenue from existing customers while controlling retention and marketing cost. The technical solution is customer-level churn prediction and risk ranking.

## Operational Strategy

Weekly batch scoring of all active customers, customer snapshots, episode-based sampling for training, suppression of contacted customers, churn prediction, ranking and Top-K targeting.

## Limitations

The dataset lacks marketing treatment, cost, margin and control-group information, so causal uplift, incremental revenue and true ROI cannot be established. The ~2-year span and seasonality limit the size and representativeness of the test period.

## Future Work

Integrate campaign and financial data, run treatment/control experiments, apply uplift modeling and next-best-action optimization, and perform complete profitability analysis.