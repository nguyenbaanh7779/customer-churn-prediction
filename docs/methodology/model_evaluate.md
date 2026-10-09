# Model Evaluation Methodology

> Corresponds to **Chapter 9 — Evaluation** of the project report.

---

## 1. Philosophy — Why We Evaluate This Way

### The real goal is not accuracy — it is better decisions under a budget constraint

The business problem is not "predict whether a customer will churn." It is:

> "Given a fixed marketing budget, which customers should we contact to maximise the revenue retained?"

This reframes the model as a **ranking engine**, not a classifier. That changes everything about how we evaluate:

- A model with ROC-AUC = 1.0 that ranks high-value churners below low-value ones is still bad for business.
- A model with modest AUC that concentrates the highest-risk, highest-revenue customers in the top 10–20% of its list is highly useful.
- The baseline question is not "did we beat random?" — it is "did we beat the simplest sensible rule a marketer could apply without a model?"

### Three layers of evidence, each answering a different question

| Layer | Question | Why it matters |
|---|---|---|
| **Classification performance** | Can the model distinguish churners from non-churners? | Baseline credibility — if this fails, nothing else matters |
| **Targeting / ranking performance** | Does the model rank the right customers at the top each week? | This is the actual use case |
| **Business impact** | Under stated assumptions, does model-based targeting produce better outcomes than simple rules? | Connects model output to the decision-maker's language |

### Why we evaluate on the Scoring Population, not the Training Population

All evaluation uses the **Scoring Population** — every active customer at every weekly test date, exactly as the model would be used in production. The training-episode cooldown (applied only during data preparation) is not used here. In real weekly operations, every active customer is scored every week.

### Why metrics are computed per week, not pooled

The same customer appears across multiple consecutive test weeks. Pooling all test rows and computing a single metric treats repeated observations as independent — which they are not. Instead, we rank each week independently, compute the metric for that week, and summarise as **mean ± std** across test weeks. The standard deviation measures week-to-week stability.

---

## 2. Evaluation Scenarios

To ensure the model is assessed under conditions that reflect real operational use, we define three evaluation scenarios. Each varies the campaign targeting share K and the assumed retention effectiveness e.

| Scenario | K (target share) | e (retention effectiveness) | Rationale |
|---|---|---|---|
| **Tight budget** | 10% | 30% | Conservative — small list, uncertain campaign response |
| **Base case** | 30% | 40% | Recommended operating point |
| **Broad targeting** | 50% | 50% | Optimistic — larger budget, higher expected response |

All three scenarios are evaluated against four candidates: the final model, an RFM rule, a recency rule, and random targeting. This makes it clear whether the model's advantage holds across different budget assumptions.

---

## 3. Metric Suite

### 3.1 Classification Performance Metrics

These metrics use the model's raw probability scores on the pooled test set. They assess whether the model can statistically distinguish churners from non-churners at all.

#### ROC-AUC

Measures how well the model separates churners from non-churners across all thresholds.

```
ROC Curve: x-axis = False Positive Rate
           y-axis = True Positive Rate (Recall)
AUC = area under that curve ∈ [0, 1]
```

Interpretation: probability that a randomly chosen churner is ranked above a randomly chosen non-churner. Score of 0.5 = coin flip; 1.0 = perfect.

Limitation: optimistic when classes are imbalanced. A model that rarely predicts "churn" can maintain a low FPR while being useless for targeting.

**Observed result:** `0.764`

#### PR-AUC (Primary Classification Metric)

Measures the trade-off between Precision and Recall across all thresholds.

```
Precision = TP / (TP + FP)  — of flagged customers, what fraction actually churned?
Recall    = TP / (TP + FN)  — of all churners, what fraction did the model catch?
PR-AUC    = area under the Precision-Recall curve ∈ [0, 1]
```

The random baseline for PR-AUC equals the churn rate (~40%). Any useful model must score substantially above this.

**Observed result:** `0.659` (baseline: `0.401`, gap: `+0.258`)

#### Precision, Recall, F1 at Threshold 0.5

Reported at the default threshold for reference. In the actual targeting application, no fixed threshold is used — the model ranks customers and the top K% are selected.

| Metric | Result | Reading |
|---|---|---|
| Precision | 0.612 | 61% of flagged customers actually churn |
| Recall | 0.649 | Model catches 65% of all churners |
| F1 | 0.630 | Balanced — neither metric is collapsed |

#### Calibration (Reliability Curve + Brier Score)

Assesses whether the model's probability outputs are trustworthy. A miscalibrated model produces wrong revenue-at-risk estimates even if its rankings are correct.

```
Brier Score = (1/N) × Σ (predicted_probability_i − actual_outcome_i)²
Ranges 0 (perfect) to 1 (worst)
```

Reliability curve: group predictions into 10 bins and compare mean predicted probability vs observed churn rate. A perfectly calibrated model falls on the y = x diagonal.

### 3.2 Stability Metrics

Stability metrics measure whether model performance is consistent week to week. Inconsistent performance creates operational risk — marketing cannot rely on a model whose quality varies substantially from week to week.

#### Week-to-Week Standard Deviation of Lift@K

Computed across all N test weeks. Lower std = more reliable.

```
std_lift = std(Lift@K_week1, Lift@K_week2, ..., Lift@K_weekN)
```

**Observed result:** Final model std = `0.049`; Recency rule std = `0.061`. The model is more stable than the strongest rule-based baseline.

#### Population Stability Index (PSI) on Key Features

After each scoring run, the distribution of key features is compared against the training-period distribution.

```
PSI_feature = Σ (actual% − expected%) × ln(actual% / expected%)

PSI < 0.10    stable
0.10 ≤ PSI < 0.20    minor drift — monitor
PSI ≥ 0.20    major drift — investigate
```

Features flagged for PSI tracking: `recency_days`, `frequency_so_far`, `monetary_so_far`.

#### Score Distribution Stability (KS Statistic)

The Kolmogorov-Smirnov statistic measures whether the churn probability distribution has shifted across scoring periods.

```
KS = max |CDF_current(x) − CDF_reference(x)|

KS < 0.05    no material shift
KS ≥ 0.10    distribution shift — investigate model behaviour
```

### 3.3 Business Impact Metrics

These metrics translate model performance into the language of the marketing decision-maker. They are scenario estimates under stated assumptions, not observed outcomes (the dataset contains no campaign response data).

#### Targeting Funnel Metrics (per scenario)

| Metric | Formula |
|---|---|
| Customers targeted | `N × K` |
| True churners captured (TP) | churners in top K% |
| Non-churners contacted (FP) | non-churners in top K% |
| Revenue at risk covered | `Σ revenue_at_risk` for TP customers |

#### Net Business Impact

```
GrossProfitPreserved = RevenueAtRisk(targeted churners) × e × m
ContactCost          = N_K × c_contact
IncentiveCost        = (TP × e + FP × r) × c_inc
NetBusinessImpact    = GrossProfitPreserved − ContactCost − IncentiveCost
```

Assumed constants (base scenario):
- `m = 35%` gross margin
- `c_contact = £0.50` per customer (email/SMS)
- `c_inc = £5.00` per redemption
- `r = 20%` cannibalization rate (non-churners who redeem the incentive anyway)

#### Incremental Value vs Baseline

```
IncrementalValue = NetBusinessImpact(model) − NetBusinessImpact(best_rule)
```

This is the model's unique monetary contribution above the simplest sensible rule. A positive number is required to justify the additional complexity of a machine learning model over a heuristic.

---

## 4. Evaluation Strategy and Method

### 4.1 Temporal Split Design

Data is split chronologically to prevent any future information from influencing model training or evaluation.

```
[──────────── Train ────────────][── Embargo ──][── Validation ──][─ Embargo ─][── Test ──]
```

- **Training period:** historical data used to fit the model.
- **Embargo:** a gap equal to H = 90 days between training and validation. Prevents label leakage — a row cannot have its churn outcome determined partly by events in the training period.
- **Validation:** used for hyperparameter tuning and feature selection. No test data is seen at this stage.
- **Test period:** 10 consecutive weekly scoring dates. Unseen data. All final reported metrics come from here only.

The model is **refitted** on train + validation before test evaluation. This refit uses the same hyperparameters and feature set selected during validation.

### 4.2 Per-Week Evaluation Procedure

For each of the 10 test weeks:

1. Take the Scoring Population for that week (all active customers at that snapshot date).
2. Score all customers with the model.
3. Rank by predicted churn probability (descending).
4. Compute Precision@K, Recall@K, Lift@K for the operating K.
5. Record the result for that week.

Summarise: **mean ± std** across 10 weeks for each metric and each candidate.

### 4.3 Baseline Comparisons

All metrics are reported for four candidates at the same K:

| Candidate | Logic |
|---|---|
| **Final model** | Logistic Regression, full feature set, C=0.01 |
| **RFM rule** | Rank by composite RFM score (recency + frequency + monetary) |
| **Recency rule** | Rank by recency_days descending (longest inactive first) |
| **Random** | Random ranking — the floor that any useful model must beat |

Comparing at the same K is essential: the contact cost is identical for all four. Only the quality of the selected customers differs.

### 4.4 Segment-Level Evaluation

Metrics are reported separately for:
- **Repeat buyers** vs **one-time buyers** — one-time buyers have less behavioral history and are harder to predict.
- **UK** vs **Non-UK** customers — Non-UK customers often have different purchasing patterns.
- **Retail** vs **wholesale-like** accounts (top 1% by revenue) — large accounts have distinctive signals.

Segment-level results expose failure modes that aggregate metrics hide.

### 4.5 Error Analysis

At K = 10%, four outcome types are counted:

| Outcome | Proportion | Business reading |
|---|---|---|
| True Positive | 7.7% | Correctly targeted churners — good spend |
| False Positive | 2.3% | Non-churners contacted — wasted incentive |
| False Negative | 32.3% | Churners missed — lost retention opportunity |
| True Negative | 57.7% | Correctly not targeted — cost saved |

The false negative rate for one-time buyers (55.5%) is substantially higher than for repeat buyers (34.1%), reflecting the limited behavioral signal available for first-time purchasers.

---

## 5. Summary of Results

### Classification performance

| Metric | Final model | RFM rule | Recency rule | Random |
|---|---|---|---|---|
| ROC-AUC | **0.764** | 0.736 | 0.638 | 0.506 |
| PR-AUC | **0.659** | 0.624 | 0.531 | 0.404 |
| Precision (0.5) | **0.612** | — | — | — |
| Recall (0.5) | **0.649** | — | — | — |

### Ranking performance (K = 10%, mean across 10 test weeks)

| Metric | Final model | RFM rule | Recency rule | Random |
|---|---|---|---|---|
| Precision@10% | **0.769** ± 0.047 | 0.718 ± 0.047 | 0.626 ± 0.056 | 0.406 ± 0.045 |
| Recall@10% | **0.193** ± 0.005 | 0.180 ± 0.005 | 0.157 ± 0.006 | 0.102 ± 0.011 |
| Lift@10% | **1.924** ± 0.049 | 1.796 ± 0.049 | 1.565 ± 0.061 | 1.017 ± 0.110 |

### Business impact (base scenario: K = 30%, e = 40%)

| Candidate | Net impact (£) | vs Recency rule |
|---|---|---|
| **Final model** | 174,273 | −18,505 |
| RFM rule | 182,450 | −10,328 |
| Recency rule | **192,778** | — |
| Random | 41,626 | −151,152 |

The model's net business impact surpasses the recency rule at K ≥ 40–50%. At tighter budgets (K ≤ 30%), the recency rule selects higher-value churners and wins on this metric — because it concentrates on fewer, larger accounts.

**Operating recommendation:** The choice of K should be driven by the marketing budget. At K ≥ 40%, the final model produces materially better business outcomes. At K ≤ 20%, the recency rule is simpler and competitive.

---

## 6. Limitations

- **No causal identification.** The dataset contains no campaign outcome data. All business impact numbers are scenario estimates under stated assumptions, not observed incremental revenue.
- **One-time buyer weakness.** With only a single purchase in history, one-time buyers have very limited behavioral features. The false negative rate for this segment is materially higher.
- **Non-UK volatility.** Non-UK customers have a higher false positive rate and irregular purchasing patterns that the current feature set may not capture well.
- **Revenue-at-risk proxy.** Using average order value as revenue exposure is a conservative, leakage-free approximation. It does not account for customer lifetime value or the probability of multiple future purchases.
