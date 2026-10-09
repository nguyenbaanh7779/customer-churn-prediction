# Model Monitoring Methodology

> Corresponds to **Chapter 11 — Monitoring and Feedback** of the project report.

---

## 1. Philosophy — Why Monitoring Exists

A model that was excellent at launch can quietly become useless as customer behavior shifts, transaction volumes change seasonally, or the underlying data pipeline evolves. Without monitoring, the marketing team would not know until churn rates start rising or campaign performance drops.

Monitoring serves three purposes:

1. **Detect degradation early** — catch problems before they affect campaign outcomes.
2. **Support the retraining decision** — provide objective evidence of when the model needs to be refreshed.
3. **Build trust** — give the marketing team and business stakeholders visibility into model health, not just a black box that produces a list each week.

Monitoring is separated into three layers:

| Layer | What it tracks | When data is available |
|---|---|---|
| **Data quality** | Transaction volume, missing rates, feature nulls | Every scoring run |
| **Model stability** | Score distribution, feature drift, calibration | Every scoring run (stability); 90 days later (performance with labels) |
| **Business metrics** | Campaign ROI, repurchase rates | When campaign outcome data is provided |

---

## 2. Dashboard — What It Shows

The monitoring dashboard provides a weekly snapshot of model health. It is designed for a data scientist or analytics lead, not a marketing end-user. It covers three areas:

### 2.1 Evaluation Metrics

Core performance metrics computed after labels become available (H = 90 days post-scoring):

| Metric | Chart type | Description |
|---|---|---|
| PR-AUC | Line chart over time | Weekly PR-AUC vs test-period baseline (dashed reference line) |
| Lift@K | Line chart over time | Weekly Lift at operating K% vs test-period mean ± 2 std (control band) |
| Precision@K | Line chart over time | Weekly Precision at operating K% |
| Brier Score | Line chart over time | Probability calibration quality |

Each chart shows a **reference band** (test-period mean ± 2 std). A metric falling outside the band for two consecutive weeks triggers a review alert.

Summary card at the top:

```
┌──────────────────────────────────────────────────────────┐
│  Model: Logistic Regression v1.0    Last run: 2011-12-01 │
│  PR-AUC (latest):   0.641    Reference: 0.659  ▼ −2.7%  │
│  Lift@30% (latest): 1.88     Reference: 1.92   ▼ −2.1%  │
│  Brier Score:       0.192    Reference: 0.185  ▼ −3.8%  │
│  Status: ✅ Within normal range                          │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Model Stability

Stability metrics do not require labels — they are computed on every scoring run.

**Feature drift (PSI table):**

| Feature | PSI (this week) | PSI trend (4w) | Status |
|---|---|---|---|
| recency_days | 0.04 | → stable | ✅ |
| frequency_so_far | 0.07 | ↑ rising | ⚠️ |
| monetary_so_far | 0.12 | ↑ rising | 🟡 Minor drift |
| recency_to_typical_gap | 0.03 | → stable | ✅ |
| ... | ... | ... | ... |

PSI thresholds: `< 0.10` stable · `0.10–0.20` minor drift · `≥ 0.20` major drift.

**Score distribution stability (KS statistic):**

KS statistic comparing the current week's churn probability distribution against the training-period reference distribution.

```
KS < 0.05    no material shift
KS ≥ 0.10    distribution shift — investigate
```

Displayed as a single number with a trend sparkline.

**Calibration check (reliability curve, every 4 weeks):**

Group predictions into 10 probability bins. For each bin, compare mean predicted probability against observed churn rate. Overlay training-period reliability curve for comparison. Divergence from the diagonal indicates the model is over- or under-predicting risk.

### 2.3 Risk Score Distribution

The churn probability distribution across all active customers in the current scoring run. This is the most immediate visual check of model health.

**Histogram — current week vs reference:**

Overlay of this week's churn probability distribution (solid bars) against the training-period reference distribution (outline). Shape changes indicate behavioral drift.

```
Customers
  │     ████
  │   ████████
  │ ████████████ ──── current week
  │ ░░░░░░░░░░░░ ···· reference
  └─────────────────────────────── churn probability
    0.0   0.2   0.4   0.6   0.8   1.0
```

**Risk band breakdown (this week vs 4-week rolling average):**

| Risk band | This week | 4w average | Change |
|---|---|---|---|
| 🔴 High | 728 (30%) | 735 (30%) | −7 |
| 🟡 Medium | 728 (30%) | 721 (30%) | +7 |
| 🟢 Low | 969 (40%) | 969 (40%) | 0 |

A sudden shift in band proportions (e.g., High band growing from 30% to 45%) indicates either a real deterioration in customer behavior or a data quality issue upstream.

**Mean churn probability trend (weekly):**

Line chart of mean predicted churn probability across all active customers, week by week. A sustained upward trend is a leading indicator of a population-level churn event. A sudden drop may indicate a data quality problem.

---

## 3. Data Quality Monitoring

Checked on every pipeline run before scoring begins. A failed check halts the pipeline and raises an alert.

| Check | Alert condition | Severity |
|---|---|---|
| Transaction row count | < 80% of the rolling 4-week average | 🔴 Critical |
| Missing Customer ID rate | > 5% of rows | 🔴 Critical |
| Invalid quantity / price rows | > 2% of rows | 🟡 Warning |
| Feature null rate | > 10% for any feature after engineering | 🟡 Warning |
| Zero active customers | Active population = 0 | 🔴 Critical |

---

## 4. Performance Monitoring (with Labels)

Labels become available H = 90 days after each scoring run. At that point:

**Compute per labeled week:**

```
PR-AUC
ROC-AUC
Lift@K (operating K)
Precision@K
Brier Score
Reliability curve
```

**Alert conditions:**

| Condition | Action |
|---|---|
| PR-AUC drops > 10% below test-period baseline for 2 consecutive weeks | Flag for review |
| Lift@K drops below 1.3 for 3 consecutive weeks | Trigger retraining assessment |
| Brier Score increases > 15% above test-period baseline | Recalibrate (Platt scaling or isotonic regression) |
| Reliability curve shows systematic over- or under-prediction | Recalibrate |

---

## 5. Retraining Triggers

| Trigger | Action |
|---|---|
| PR-AUC degradation (see Section 4) | Retrain on extended training window |
| PSI ≥ 0.20 on ≥ 2 key features | Investigate data source; retrain if drift is structural |
| Periodic schedule | Retrain every 6 months regardless of performance |
| Business change (new product lines, pricing model, customer base shift) | Immediate review and likely retrain |

Retraining must follow the same temporal split design as the original model. The retrained model is evaluated on a fresh held-out period before replacing the production model.

---

## 6. Business Metrics (Once Campaign Data Is Available)

When the campaign team provides contact outcomes (repurchase records matched to campaign contacts):

| Metric | Formula |
|---|---|
| Repurchase rate — targeted group | repurchases ÷ contacts |
| Repurchase rate — control group | (if A/B test is run) |
| Incremental repurchase rate | targeted rate − control rate |
| Revenue preserved (observed) | repurchases × avg order value |
| Campaign ROI | (revenue preserved × margin − costs) ÷ costs |
| Model lift on revenue | revenue_per_contact(model) ÷ revenue_per_contact(recency_rule) |

These observed metrics replace the scenario estimates from the evaluation chapter and should be used to calibrate the `e` (retention effectiveness) assumption for future business impact calculations.

---

## 7. Monitoring Cadence Summary

| Activity | Frequency | Data required |
|---|---|---|
| Data quality checks | Every scoring run (weekly) | Current transaction extract |
| Risk score distribution | Every scoring run (weekly) | Current scores |
| Feature PSI | Every scoring run (weekly) | Current features vs training reference |
| KS statistic | Every scoring run (weekly) | Current score distribution vs reference |
| Performance metrics (PR-AUC, Lift) | 90 days after each scoring run | Scored customers + churn outcomes |
| Calibration check | Every 4 weeks (after labels available) | Scored customers + churn outcomes |
| Business metrics | As campaign data becomes available | Campaign contacts + repurchase records |
| Retraining assessment | Triggered by alerts or every 6 months | All of the above |
