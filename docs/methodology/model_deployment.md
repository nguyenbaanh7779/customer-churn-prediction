# Model Deployment Methodology

> Corresponds to **Chapter 10 — Deployment** and **Chapter 11 — Feedback** of the project strategy.
>
> ⚠️ **Scope note:** The operational design described in Sections 2–8 is a proposed architecture for a live system. The **demo implementation** (Section 9) is a working Streamlit application that demonstrates the full pipeline in a single run — from raw transaction data to a ranked retention target list with churn probabilities and confidence information.

---

## The Master Plan — Philosophy and the "Why"

### Deployment is not the end — it is where the model starts earning its keep

Training and evaluation tell us the model works in the lab. Deployment is the process of making it work in the field — reliably, repeatedly, and with predictable behavior — so that a marketing team can act on its output every Monday morning without a data scientist present.

The key design decisions in this chapter each exist for a specific reason:

### Why weekly batch scoring instead of real-time

Churn is not a real-time event. A customer does not decide to churn in a single moment — it is a gradual drift in behavior (increasing recency, declining frequency). Weekly scoring is the right cadence because:

- The prediction horizon H = 90 days. Whether we score on Monday or Thursday of the same week makes no meaningful difference to the forecast.
- Weekly batch scoring is operationally simple — one run, one output file, one target list for the campaign team.
- Real-time scoring would add infrastructure complexity without improving the quality of the churn signal.

### Why we score ALL active customers every week

A simpler approach would be: "score each customer once every 90 days." This is wrong for two reasons:

1. If a customer's behavior deteriorates between their last scoring date and their next scheduled scoring date, they will not appear in the target list even though they are now at high risk.
2. The recency of a customer's last purchase changes every day they go without buying. A customer who looked safe four weeks ago may now be in the danger zone.

Scoring everyone every week ensures that risk assessments are always current. The training-episode cooldown (which prevents near-duplicate training rows) is a training-data technique only — it has no place in the scoring pipeline.

### Why suppression exists and what it is NOT

Suppression is the rule that prevents a customer from being re-targeted within S weeks of being contacted. It exists for two business reasons:

1. Repeated contact with the same customer in a short window is annoying and can damage the brand relationship.
2. A customer who received a retention offer last week should be given time to respond before being offered another one.

Suppression is **not** the same as the training-episode cooldown. The training cooldown reduces correlation in training data. Suppression is a campaign management rule. Confusing the two would result in severely under-scoring the customer base.

### Why the pipeline must be reproducible end-to-end

Every scoring run must produce the same output given the same input. This matters for:

- **Debugging:** if the marketing team questions a scoring result, we can reproduce it exactly.
- **Auditing:** regulators or internal compliance teams may ask why a specific customer was or was not targeted.
- **Retraining:** when we retrain the model, we need to reproduce historical feature calculations to compare old and new model performance on the same data.

All configuration parameters are versioned and logged alongside every scoring output.

### The demo philosophy — show the full chain in 15 minutes

The Streamlit application (Section 9) is designed around a single philosophy: **one command, the full chain, interactive results**. A 15-minute demo should show:

1. Raw transaction data loaded
2. Features computed per customer at today's snapshot date
3. Model scores assigned (churn probability)
4. Customers ranked by risk
5. Target list generated at a chosen K%
6. Revenue at risk estimated per targeted customer
7. Dashboard visualising the risk distribution and top customers

The audience for this demo is a marketing manager or business stakeholder, not a data scientist. The interface uses plain language: "High Risk," "Likely to leave," "Estimated revenue at stake" — not "predicted probability" or "decile bucket."

---

## Overview

The deployment system transforms raw customer transaction history into a weekly ranked retention target list. The pipeline runs end-to-end:

```
Customer transaction history
    → Data validation and cleaning
    → Active customer identification
    → Feature generation (per customer, per snapshot date)
    → Model scoring (churn probability)
    → Risk ranking (highest risk first)
    → Business rules and suppression
    → Target population output + dashboard
```

---

## 1. Deployment Objective

Produce a weekly ranked list of existing customers most at risk of churning, so that the retention marketing team can prioritise their limited campaign resources on the customers where intervention is most likely to preserve revenue.

The system outputs:
- A ranked customer list with churn probabilities, risk bands, and revenue-at-risk estimates.
- A Streamlit dashboard for interactive exploration and target list export.

---

## 2. Weekly Batch Pipeline — Architecture

The scoring pipeline is a linear sequence of deterministic steps. Each step has a single responsibility and a clearly defined input/output contract.

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Data Ingestion                                         │
│  Input:  raw transaction file (Excel / Parquet)                 │
│  Output: validated transaction DataFrame                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Step 2: Transaction Cleaning                                   │
│  Rules:  remove nulls, cancellations, invalid qty/price,        │
│          non-product codes, exact duplicates                    │
│  Output: clean purchase records                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Step 3: Active Customer Identification                         │
│  Rule:   Recency ≤ R_max at snapshot date                       │
│  Apply:  suppression list (customers contacted in last S weeks) │
│  Output: scoring population for this week                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Step 4: Feature Engineering                                    │
│  Features: RFM + behavioral + temporal + product + cancellation │
│  Constraint: use only history up to snapshot_date (no leakage)  │
│  Output: feature matrix (one row per customer)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Step 5: Model Scoring                                          │
│  Model:  trained Logistic Regression (Full feature set, C=0.01) │
│  Output: churn_probability ∈ [0, 1] per customer               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Step 6: Risk Ranking and Banding                               │
│  Rank:   descending by churn_probability                        │
│  Bands:  High (top 30%), Medium (30–60%), Low (bottom 40%)      │
│  Output: ranked scored DataFrame with risk_band, rank           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Step 7: Target Population Selection                            │
│  Rule:   top K% by rank, respecting marketing capacity          │
│  Apply:  additional business rules (new customer exclusion etc.)│
│  Output: target list (customer_id + all scoring fields)         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Step 8: Output + Dashboard                                     │
│  Files:  scored_all_{snapshot_date}.csv                         │
│          target_list_{snapshot_date}.csv                        │
│  App:    Streamlit dashboard (interactive exploration)           │
└─────────────────────────────────────────────────────────────────┘
```

### Scoring cadence

Weekly (e.g. every Monday). Transaction data may be ingested daily, but model scoring and target-list generation happen once per week.

### What the training-episode cooldown is NOT doing here

The cooldown rule (C = H days between training snapshots of the same customer) was a training-data construction technique to reduce correlated rows. It is explicitly **not applied** in the scoring pipeline. Every active customer is scored every week.

---

## 3. Feature Engineering in the Scoring Context

**The "Why":** Features must be computed using only information available at or before the snapshot date. Violating this rule (data leakage) would cause the model to make predictions using future information — producing artificially good scores in testing that collapse in production.

### Feature groups

| Group | Features | Why it matters |
|---|---|---|
| RFM | `recency_days`, `frequency_so_far`, `monetary_so_far` | Core churn signals — most models start here |
| Purchase behavior | `avg_order_value_so_far`, `tenure_days` | Distinguishes high-value customers from low-activity ones |
| Temporal behavior | `mean_gap_so_far`, `recency_to_typical_gap`, `activity_last_90d` | Detects acceleration or deceleration in purchase rate |
| Product behavior | `distinct_products_so_far` | Breadth of engagement — customers who buy many product types are stickier |
| Cancellation behavior | `cancel_count_so_far`, `cancel_value_share_so_far` | High cancellation share predicts unhappy, at-risk customers |

### Log-transformation

Skewed features (monetary, avg_order_value, tenure, etc.) are log1p-transformed before scaling. This is applied identically in training and scoring — the transformation is part of the feature recipe, not the model.

### Imputation and scaling

The imputer (median strategy) and scaler (StandardScaler) are fitted on the training data and saved alongside the model. In scoring, the same fitted objects are applied — no re-fitting on new data, which would introduce distribution shift.

---

## 4. Customer Risk Output Schema

Every scoring run produces one row per active customer with the following fields:

| Field | Type | Description |
|---|---|---|
| `customer_id` | string | Unique customer identifier |
| `snapshot_date` | date | Date of this scoring run |
| `churn_probability` | float [0, 1] | Model's estimated probability of churn in the next H days |
| `risk_band` | string | High / Medium / Low |
| `rank` | integer | 1 = highest churn risk in this week's Scoring Population |
| `segment` | string | `one-time` or `repeat` buyer |
| `revenue_at_risk` | float (£) | Avg purchase value — estimate of revenue at stake |
| `suppressed` | boolean | True if this customer is under active suppression |
| `in_target_list` | boolean | True if selected for this week's campaign |
| `model_version` | string | Version identifier of the model used |

### Risk bands — rationale

| Band | Score range | Business meaning |
|---|---|---|
| High | Top 30% by probability | Act now — high probability of churning within H days |
| Medium | 30–60% | Watch — elevated risk, monitor closely |
| Low | Bottom 40% | No immediate action needed |

These thresholds are configurable. The 30/30/40 split is a starting point; the marketing team may adjust based on campaign capacity.

---

## 5. Target Population Selection

**The "Why":** The scored list contains every active customer. The target list contains only those the campaign will actually contact. The selection process must respect budget constraints and business rules.

### Selection logic

```
1. Sort Scoring Population by rank (ascending, 1 = highest risk).
2. Apply suppression: remove customers in the suppression list.
3. Apply business rules:
   - Exclude customers with tenure < 30 days (too new to be at-risk by behavioral signal).
   - Exclude customers flagged for VIP treatment (separate campaign).
   - Apply any country or region exclusions.
4. Select top K% of remaining customers.
5. Cap at maximum campaign capacity (if N_max is defined).
```

### Choosing K

The evaluation in Chapter 9 showed that:
- At K = 10–30%, the recency rule produces similar or slightly higher net business impact under base assumptions.
- At K ≥ 40–50%, the final model's broader churn coverage produces materially better outcomes.
- The optimal K depends on the campaign budget and the assumed retention effectiveness.

The default recommendation is **K = 30–40%** as a balance between targeting quality and coverage, with the expectation that this will be tuned as real campaign outcome data becomes available.

---

## 6. Suppression

**The "Why":** Repeated targeting within a short window harms the customer experience and wastes incentive budget on customers who may be responding to the previous offer.

### Rules

- Suppression period: **S = 4–8 weeks** (configurable).
- A customer enters suppression the week they are added to a target list (i.e., when they are actually contacted — not just scored).
- Suppression applies to individual customers, not to the entire Scoring Population.
- Suppressed customers are still **scored** every week (their risk is tracked), but they are excluded from the target list.
- The suppression log records: `customer_id`, `contact_date`, `suppression_end_date`, `campaign_id`.
- After suppression expires, the customer re-enters the target list selection as normal.

### Why suppression is separate from the training cooldown

| Mechanism | Used in | Purpose |
|---|---|---|
| Training-episode cooldown (C) | Training data construction only | Reduces near-duplicate rows with correlated labels |
| Suppression (S) | Scoring pipeline, live operations | Prevents repeated customer contact in a short window |

Applying the cooldown in scoring would mean only ~1/13 of active customers are scored each week (with H = 90 days and C = H). That would break the system.

---

## 7. Monitoring and Feedback Loop

**The "Why":** Models degrade. Customer behavior shifts. Transaction volume changes seasonally. Without monitoring, a model that was excellent at launch can quietly become useless — and the marketing team would not know until churn rates start rising.

### 7.1 Data Quality Monitoring

Check on every pipeline run:

| Check | Alert condition |
|---|---|
| Transaction row count | < 80% of the rolling 4-week average |
| Missing Customer ID rate | > 5% of rows |
| Invalid quantity/price rows | > 2% of rows |
| Feature null rate | > 10% for any feature |

### 7.2 Feature Drift Monitoring

After each scoring run, compare the distribution of key features against the training-period distribution.

```
PSI (Population Stability Index) per feature:
PSI = Σ (actual% − expected%) × ln(actual% / expected%)

PSI < 0.10 : stable
0.10 ≤ PSI < 0.20 : minor drift — monitor
PSI ≥ 0.20 : major drift — investigate
```

Flag: `recency_days`, `frequency_so_far`, `monetary_so_far` are the three features most likely to drift seasonally.

### 7.3 Model Performance Monitoring

Labels become available H = 90 days after each scoring run. At that point, compute:

```
For each labeled scoring week:
    PR-AUC vs test-period baseline
    Lift@K (operating K) vs test-period mean
    Weekly Precision@K vs test-period mean ± 2×std
```

Alert if:
- PR-AUC drops > 10% below test-period value for 2 consecutive weeks.
- Lift@K drops below 1.3 (approaching the recency rule's performance) for 3 consecutive weeks.

### 7.4 Calibration Monitoring

Every 4 weeks, after labels are available:

```
Brier Score vs test-period baseline
Reliability curve — check whether high/medium/low bands
retain their expected churn rates
```

Recalibrate (Platt or isotonic) if the reliability curve systematically over- or under-predicts.

### 7.5 Business Metrics (once campaign data are available)

When the campaign team provides contact outcomes:

| Metric | Formula |
|---|---|
| Repurchase rate in targeted group | repurchases ÷ contacts |
| Repurchase rate in control group | if A/B test is run |
| Incremental repurchase rate | targeted rate − control rate |
| Revenue preserved (observed) | repurchases × avg order value |
| Campaign ROI | (revenue preserved × margin − costs) ÷ costs |

These observed metrics replace the estimated values from Chapter 9 and should be used to calibrate the `e` (retention effectiveness) assumption for future scenarios.

### 7.6 Model Retraining Triggers

| Trigger | Action |
|---|---|
| PR-AUC degradation (see 7.3) | Retrain on extended training window |
| PSI ≥ 0.20 on ≥ 2 key features | Investigate data source; retrain if drift is structural |
| Periodic schedule | Retrain every 6 months regardless of performance |
| Business change (new product lines, pricing model) | Immediate review and likely retrain |

Retraining must follow the **same temporal split design** (Chapter 7) with a fresh embargo before the new test period. The retrained model is evaluated on a held-out period before replacing the production model.

---

## 8. Operational Parameters

All parameters are versioned and logged in the scoring output alongside every run:

| Parameter | Symbol | Default value | Description |
|---|---|---|---|
| Activity threshold | R_max | 136 days (P90) | Maximum recency to be considered active |
| Prediction horizon | H | 90 days | Forward window for churn label |
| Suppression period | S | 6 weeks | Weeks a contacted customer is excluded |
| Scoring cadence | — | Weekly (Monday) | When the pipeline runs |
| Target share | K | 30% | Fraction of Scoring Population targeted |
| Feature warm-up | — | 6 months history | Minimum history before a customer is scored |
| Model version | — | Logged per run | Git hash or semantic version of model artefact |

---

## 9. Demo Implementation — Streamlit Scoring App

**The "Why":** The full pipeline described above is a production design. The Streamlit demo is a self-contained, interactive version that runs the complete chain in a single session — from loading raw data to displaying a ranked retention target list.

Purpose: demonstrate to a marketing stakeholder or product owner, in 15 minutes, how the system works, what the output looks like, and why the model's rankings are trustworthy.

---

### 9.1 Architecture of the Demo App

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit App                                              │
│                                                             │
│  Sidebar:                                                   │
│   - Upload transaction file (or use sample data)           │
│   - Set snapshot date (default: today or latest date)      │
│   - Set K% target (slider: 10–50%)                         │
│   - Set R_max (days)                                        │
│                                                             │
│  Main panel:                                                │
│   - Tab 1: Pipeline Summary                                 │
│   - Tab 2: Risk Overview Dashboard                          │
│   - Tab 3: Target List                                      │
│   - Tab 4: Individual Customer Lookup                       │
└─────────────────────────────────────────────────────────────┘
```

---

### 9.2 The One-Command Pipeline

The user uploads a transaction file (or uses the bundled sample) and clicks **Run Pipeline**. The app executes the full scoring chain:

```python
# Pseudocode for the pipeline function called by the app

def run_scoring_pipeline(transactions_df, snapshot_date, k_frac, r_max_days):

    # Step 1: Clean transactions
    clean = clean_transactions(transactions_df)

    # Step 2: Identify active customers at snapshot_date
    active = get_active_customers(clean, snapshot_date, r_max_days)

    # Step 3: Compute features (history up to snapshot_date only)
    features = build_features(clean, active, snapshot_date)

    # Step 4: Score with the trained model
    scores = score_customers(features, model, imputer, scaler)

    # Step 5: Rank and band
    scored = rank_and_band(scores)

    # Step 6: Select target population
    target_list = select_targets(scored, k_frac)

    # Step 7: Attach revenue-at-risk estimates
    output = attach_revenue_at_risk(target_list, clean, snapshot_date)

    return scored, output
```

Total runtime on the UCI Online Retail II dataset: approximately 5–15 seconds.

---

### 9.3 Tab 1 — Pipeline Summary

Displays a card summary of what just ran:

```
┌────────────────────────────────────────────────────────┐
│  Scoring Run: 2011-09-05                               │
│  Active customers scored:   2,425                      │
│  Overall churn rate:        40.1%                      │
│  Target list size (K=30%):  728 customers              │
│  Total revenue at risk:     £287,543                   │
│  Model version:             v1.0 (LR, Full features)   │
└────────────────────────────────────────────────────────┘
```

Followed by a step-by-step trace showing how many rows passed or were filtered at each pipeline stage (data quality, active filter, suppression, business rules).

---

### 9.4 Tab 2 — Risk Overview Dashboard

Four visualisations that give a marketing manager an at-a-glance view:

**Chart 1 — Risk Band Distribution (bar chart)**
Number of customers in High / Medium / Low bands. Annotated with the churn rate per band from the training-period decile analysis.

**Chart 2 — Churn Probability Distribution (histogram)**
Full distribution of predicted churn probabilities across all active customers. The K% cutoff is shown as a vertical line. This helps the marketing team understand whether the target list is a tight cluster of high-risk customers or a broad spread.

**Chart 3 — Revenue at Risk by Risk Band (stacked bar)**
Total estimated revenue at risk broken down by High / Medium / Low band. This answers: "Is the revenue concentrated in the High band, or are there important customers in the Medium band we might miss?"

**Chart 4 — Weekly Lift@K — Historical Test Performance**
The weekly Lift@10% chart from Chapter 9 evaluation — shows that the model consistently outperforms the recency rule across all 10 test weeks. This is the **credibility chart**: it demonstrates to the stakeholder that the rankings are based on a validated model, not a heuristic.

---

### 9.5 Tab 3 — Target List

A sortable, filterable table of all customers selected for the campaign:

| customer_id | churn_probability | risk_band | rank | segment | revenue_at_risk (£) |
|---|---|---|---|---|---|
| 12345 | 0.87 | High | 1 | repeat | 1,245 |
| 67890 | 0.84 | High | 2 | repeat | 890 |
| ... | ... | ... | ... | ... | ... |

Features:
- Filter by risk band, segment, or revenue threshold.
- Sort by any column.
- **Download as CSV** button — the file the marketing team uploads to their campaign platform.

Plain-language column names are used throughout (no jargon like "probability score" or "feature vector").

---

### 9.6 Tab 4 — Individual Customer Lookup

The user enters a customer ID to see:

- Their churn probability and risk band.
- Their rank within this week's Scoring Population.
- Key features driving the prediction:
  - Days since last purchase
  - Number of purchase occasions
  - Total spend to date
  - Average order value
  - Activity in the last 90 days
- An estimated revenue at risk.
- Whether they are in the target list for this week.

This tab is useful for a campaign manager who wants to understand why a specific customer was or was not flagged — supporting human oversight of model decisions.

---

### 9.7 Demo Script — 15 Minutes

| Minute | Action | What it shows |
|---|---|---|
| 0–1 | Open app, show the sidebar | The interface is clean, no code required |
| 1–2 | Upload transaction file (or click "Use sample data") | Data ingestion |
| 2–3 | Click Run Pipeline, watch the progress bar | The full chain runs automatically |
| 3–5 | Walk through Tab 1 — Pipeline Summary | How many customers, churn rate, target list size |
| 5–9 | Walk through Tab 2 — Risk Overview Dashboard | Risk distribution, revenue at risk, historical validation |
| 9–12 | Walk through Tab 3 — Target List | Sortable, filterable, downloadable |
| 12–14 | Tab 4 — look up a high-risk customer | Explain the key features, revenue at risk |
| 14–15 | Change K% slider, re-run | Show how the target list changes with budget |

The demo ends with the marketing manager holding a CSV file they could upload to a campaign platform — the output is actionable, not just analytical.

---

## 10. From Demo to Production — What Changes

| Aspect | Demo (Streamlit) | Production system |
|---|---|---|
| Data source | Uploaded file or local Parquet | Automated extract from transaction database |
| Scheduling | Manual (button click) | Cron job or Airflow DAG every Monday 6am |
| Suppression | Not implemented in demo | Suppression list read from campaign CRM |
| Model artefacts | Loaded from local file | Loaded from model registry (MLflow, S3) |
| Output | CSV download | Written to campaign database / CRM |
| Monitoring | Not implemented | Automated data quality + drift checks |
| Alerting | Not implemented | Email / Slack alerts on threshold breach |
| Authentication | None | SSO / role-based access |

The demo is a proof of concept that demonstrates the logic and output format. Production deployment requires the additional infrastructure listed above, but the core pipeline logic — cleaning → features → scoring → ranking → targeting — is identical.
