# Model Evaluation Methodology

> Corresponds to **Chapter 9 — Evaluation** of the project strategy.

---

## The Master Plan — Philosophy and the "Why"

Before any formula, it is worth asking: **why do we evaluate a churn model the way we do?**

### The real goal is not accuracy — it is better decisions under a budget constraint

The business problem is not "predict whether a customer will churn." The business problem is:

> "Given a fixed marketing budget, which customers should we contact to maximise the revenue we retain?"

This single sentence changes everything about how we evaluate. We are not building a classifier for its own sake. We are building a **ranking engine** that helps marketing spend money in the right places. That means:

- A model that perfectly separates churners from non-churners (ROC-AUC = 1.0) but ranks high-value churners below low-value churners is still bad for business.
- A model that is slightly worse on classification metrics but concentrates the highest-risk, highest-revenue customers in the top 10–20% of its ranked list is far more useful.
- The question "did we beat random?" is not enough. The question is "did we beat the simplest sensible rule a marketer could apply without any model?"

### Three layers of evidence, each answering a different question

The evaluation is structured in three layers that build on each other:

| Layer | Question answered | Why it matters |
|---|---|---|
| Predictive Performance | Can the model distinguish churners from non-churners at all? | Baseline credibility — if this fails, nothing else matters |
| Targeting / Ranking Performance | Does the model rank the right customers at the top of the list each week? | This is the actual use case — weekly batch scoring |
| Business Impact Estimation | Under stated assumptions, does targeting by model score produce better revenue outcomes than simple rules? | Connects model output to the decision-maker's language |

### Why we use the Scoring Population, not the Training Population

All evaluation uses the **Scoring Population** — every active customer at every weekly test date, exactly as the model would be used in production. This is deliberate:

- The training population uses an episode-cooldown rule to reduce near-duplicate rows. That rule exists to make training data cleaner, not to reflect reality.
- In real weekly operations, every active customer is scored every week.
- Evaluating on the Scoring Population measures the model as it would actually be deployed, not as it was trained.

### Why we compute metrics per week instead of pooling

The same customer appears in multiple consecutive test weeks. If we pool all test rows and compute a single metric, we are treating repeated observations of the same customer as independent — which they are not. A customer who scores high in week 1 will likely score high in weeks 2 and 3 as well. Pooled lift or precision numbers can be misleading.

Instead, we rank each week's Scoring Population independently and compute the metric for that week. We then summarise across weeks as **mean ± standard deviation**. This mirrors real operations and honestly captures week-to-week stability.

### The "Why" behind comparing to baselines at the same K

The model's value is often misunderstood. Consider:

- Targeting K% of customers instead of 100% saves (1 − K) × contact cost **regardless of which model you use**. That saving comes from targeting itself, not from the model.
- The model's unique contribution is in **which** K% it picks. If the model picks a K% that contains more churners than a simple recency rule would, it generates more revenue preservation per pound spent on the campaign.
- This is why we always compare the final model against the recency rule and random targeting **at the same K**. The contact cost is identical; only the quality of the selected customers differs.

### Why causal impact cannot be measured here

The dataset contains no treatment/control information — no record of which customers received a retention offer and whether they responded. This means:

- We cannot measure incremental revenue (revenue gained because of the campaign, above what would have happened anyway).
- Every business-impact number in this document is a **scenario estimate** under stated assumptions about retention effectiveness, gross margin, and costs.
- The model proves it can rank customers better than simple rules. Whether that better ranking translates to a profitable campaign depends on the campaign itself — which is outside the scope of this dataset.

---

## Overview

The evaluation assesses the final churn prediction system at three levels:

1. **Predictive Performance** — how well the model classifies churners vs non-churners.
2. **Targeting / Ranking Performance** — how well it prioritizes customers for retention campaigns each week.
3. **Business Impact Estimation** — scenario-based estimates of revenue preserved and net impact.

All evaluation is performed on the **Scoring Population** (every active customer at each weekly test date) using data from the unseen test period only. Results are reported overall and separately for the one-time and repeat-purchase segments.

**Test period summary:**
- 24,252 test rows across 10 weekly scoring dates
- 86.7% repeat-purchase segment, 13.3% one-time segment
- Overall churn rate: 40.1% (this is the random baseline for PR-AUC)

---

## 1. Predictive Performance

**The "Why":** Before we trust the model's rankings for business decisions, we first check whether it can statistically distinguish churners from non-churners at all. If predictive metrics fail, ranking metrics are meaningless.

Predictive performance is measured on the **pooled** test set (all rows together). These metrics use the model's raw probability scores, not the weekly-ranking procedure.

---

### The four outcomes of every prediction

Every prediction has one of four outcomes:

| | Actually churned | Did NOT churn |
|---|---|---|
| **Model predicted churn** | True Positive (TP) ✓ | False Positive (FP) ✗ |
| **Model predicted no churn** | False Negative (FN) ✗ | True Negative (TN) ✓ |

**Why these matter for the business:**
- **FP (false alarm):** We spend retention budget on a customer who was not going to leave. Wasted money.
- **FN (missed churner):** A churner receives no retention effort and leaves. Lost revenue.
- The trade-off between FP and FN is controlled by the decision threshold. Rather than fixing a threshold, we use summary metrics that describe model performance across all thresholds.

---

### 1.1 ROC-AUC

**What it measures:** How well the model separates churners from non-churners across all possible decision thresholds.

```
ROC Curve: x-axis = False Positive Rate (FP / (FP + TN))
           y-axis = True Positive Rate, i.e. Recall (TP / (TP + FN))
           Plotted at every possible threshold

AUC = area under that curve, ranging from 0 to 1
```

**The "Why" — intuition:** Imagine picking one random churner and one random non-churner from the test data. ROC-AUC is the probability that the model gives the churner a higher churn score than the non-churner. A score of 0.5 means the model is no better than a coin flip. A score of 1.0 means it perfectly ranks every churner above every non-churner.

**Limitation:** ROC-AUC is optimistic when one class is large. Because non-churners are always abundant, even a weak model can maintain a low False Positive Rate just by rarely predicting "churn" — yet that model would be useless for targeting.

**Result:** `roc_auc = 0.764`
→ The model ranks a random churner above a random non-churner ~76% of the time. Solid discrimination.

---

### 1.2 PR-AUC (Primary Metric)

**What it measures:** How Precision and Recall trade off across all thresholds, summarised as a single number.

```
Precision = TP / (TP + FP)   [of flagged customers, what fraction actually churned?]
Recall    = TP / (TP + FN)   [of all churners, what fraction did the model catch?]

PR Curve: x-axis = Recall
          y-axis = Precision
          Plotted at every possible threshold

PR-AUC = area under that curve
```

**The "Why" — why PR-AUC is the primary metric here:**

With a churn rate of ~40%, churners are not rare. PR-AUC punishes a model that achieves high recall only by predicting "churn" for almost everyone — that strategy inflates recall but destroys precision, and is useless for targeted campaigns.

The **random baseline for PR-AUC equals the churn rate.** A model that predicts randomly would assign churn probability = churn rate to everyone, giving PR-AUC ≈ churn rate. Any useful model must score well above this baseline.

**Result:**
- Random baseline: `0.401` (= churn rate)
- Model PR-AUC: `0.659`
- Gap above baseline: `+0.258` — confirms the model adds substantial value beyond random targeting.

---

### 1.3 Precision, Recall, F1 at the 0.5 Threshold

These metrics require choosing a threshold. We report at the default 0.5, but note that in the targeting application we do not use a fixed threshold — we rank and cut at top K%.

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

**The "Why" — what each tells us:**
- **Precision:** Of the customers the model flags as likely to churn, how many actually do? Low precision = wasted campaign spend.
- **Recall:** Of all customers who will churn, how many does the model catch? Low recall = churners escape without retention effort.
- **F1:** A single number that penalises you if either metric is poor. Useful as a quick sanity check.

**Results (threshold = 0.5):**

| Metric | Value | Plain reading |
|---|---|---|
| Precision | 0.612 | 61% of model-flagged customers actually churn |
| Recall | 0.649 | The model catches 65% of all churners |
| F1 | 0.630 | Balanced; neither precision nor recall is collapsed |

---

### 1.4 Calibration (Reliability Curve + Brier Score)

**The "Why":** A model that ranks customers correctly but miscalibrates probabilities will produce wrong revenue-at-risk estimates. If the model outputs "70% churn probability" for a group where only 40% actually churn, every downstream business scenario calculation will be systematically off.

**Reliability curve:** Group customers into 10 bins by predicted probability. For each bin, compare the mean predicted probability (x-axis) against the actual observed churn rate (y-axis). A perfectly calibrated model would fall exactly on the diagonal y = x.

**Brier Score:**
```
Brier Score = (1/N) × Σ (predicted_probability_i − actual_outcome_i)²
```
Ranges from 0 (perfect) to 1 (worst). Lower is better. It measures the mean squared error of probability estimates.

**Result:** Calibration curve and Brier score are plotted in notebook Section 9.2. The model shows reasonable calibration without requiring additional post-processing (Platt scaling or isotonic regression).

---

### 1.5 Summary Table

| Metric | Value | Interpretation |
|---|---|---|
| Test churn rate | 0.401 | Random baseline for PR-AUC |
| ROC-AUC | 0.764 | Good overall discrimination |
| PR-AUC | 0.659 | +0.258 above random baseline — model adds real value |
| Precision (threshold 0.5) | 0.612 | 61% of flagged customers actually churn |
| Recall (threshold 0.5) | 0.649 | Model catches 65% of all churners |
| F1 (threshold 0.5) | 0.630 | Balanced |

> The 0.5 threshold metrics are for reference. The operating decision in this project is not "churn or not churn" — it is "who do we contact this week?" That question is answered by Section 2.

---

## 2. Targeting / Ranking Performance

**The "Why":** Predictive metrics tell us whether the model can distinguish churners from non-churners in aggregate. Ranking metrics tell us whether it puts the right customers at the top of the list each week — which is the actual business action.

The difference matters enormously. A model can have a good ROC-AUC but still rank a mix of churners and non-churners in the top 10%, making it no better than a simple rule. Conversely, a model with modest ROC-AUC might still concentrate churners heavily at the top, making it very useful for targeted campaigns.

---

### Why "per week" not "pooled"

The Scoring Population is scored fresh every Monday. We rank that week's customers, select the top K%, and run the campaign. The same customer may appear in weeks 1 through 10.

If we pool all 10 weeks together and compute a single Lift@10%, we are treating 10 appearances of the same customer as 10 independent data points. That inflates our confidence and hides whether any individual week was weak.

The correct approach: rank each week independently → compute the metric for that week → summarise as **mean ± std** across the 10 test weeks. This mirrors exactly how the model would be used, and the standard deviation tells us whether performance is stable or volatile.

---

### 2.1 Precision@K

**What it measures:** Of the top K% of ranked customers, what fraction actually churn?

```
Sort all N customers by predicted churn probability (highest first).
Take the top K% → that's your target list of size k = round(N × K).
Precision@K = (number of actual churners in the top k) / k
```

**The "Why":** This directly measures the quality of the target list. If you send a retention offer to the top 10% and Precision@10% = 0.77, then 77% of those offers go to customers who would have churned without intervention — your budget is well-targeted.

**Result (K = 10%, mean across 10 test weeks):**

| Candidate | Precision@10% mean | Precision@10% std |
|---|---|---|
| **Final model** | **0.769** | 0.047 |
| RFM rule | 0.718 | 0.047 |
| Recency rule | 0.626 | 0.056 |
| Random | 0.406 | 0.045 |

The final model's top 10% is ~77% true churners, versus ~63% for the recency rule and ~41% for random targeting.

---

### 2.2 Recall@K (= Gain@K)

**What it measures:** Of all actual churners in the Scoring Population, what fraction are captured in the top K%?

```
Recall@K = (churners in top k) / (total churners in Scoring Population)
```

**The "Why":** Precision@K tells you about the quality of your contact list. Recall@K tells you about the coverage — how much of the churn risk you are actually reaching. At K = 10%, you cannot expect to catch all churners (that would require Recall@K = 1), but you want to catch as many as possible given the budget constraint.

**Result (K = 10%, mean across 10 test weeks):**

| Candidate | Gain@10% mean | Gain@10% std |
|---|---|---|
| **Final model** | **0.193** | 0.005 |
| RFM rule | 0.180 | 0.005 |
| Recency rule | 0.157 | 0.006 |
| Random | 0.102 | 0.011 |

The final model captures 19.3% of all churners in the top 10%, versus 15.7% for the recency rule. That's +3.6 percentage points more churners caught with the same budget.

---

### 2.3 Lift@K

**What it measures:** How many times better is the model's target list compared to random targeting at the same K?

```
Lift@K = Precision@K / (overall churn rate)
       = Precision@K / base_rate
```

**The "Why":** Lift normalises Precision@K by the baseline so results are comparable across datasets with different churn rates. A Lift of 2.0 at K = 10% means the top 10% of ranked customers churn at twice the rate of a randomly selected 10%. This is the most intuitive metric for explaining model value to a business stakeholder.

- Lift = 1.0 → no better than random
- Lift = 2.0 → twice as many churners per campaign contact

**Result (K = 10%, mean across 10 test weeks):**

| Candidate | Lift@10% mean | Lift@10% std |
|---|---|---|
| **Final model** | **1.924** | 0.049 |
| RFM rule | 1.796 | 0.049 |
| Recency rule | 1.565 | 0.061 |
| Random | 1.017 | 0.110 |

The final model achieves a lift of ~1.92, meaning the top 10% of ranked customers churn at nearly twice the rate of a randomly selected 10%. The recency rule achieves only 1.57×. The model beats the strongest baseline by +0.36 lift points — stably across weeks (std = 0.049).

**Interpretation of stability:** The recency rule has a higher standard deviation (0.061) than the final model (0.049). This means the recency rule is more volatile week-to-week — its quality depends more on which particular week it is. The final model is not only stronger on average, it is more reliable.

---

### 2.4 Why the Final Model Beats the Recency Rule on Lift but Not on Revenue-at-Risk

The recency rule ranks by longest-recency first (customers who haven't purchased for a long time). This captures customers who are clearly drifting away. But there is a subtlety:

Customers with very long recency tend to be **small buyers** — they purchased infrequently, and their last purchase was long ago precisely because they are low-activity customers. Their average order value is modest.

The final model uses the full feature set (RFM + behavioral + product + cancellation features). It learns that some customers who appear "recent" by the recency rule are actually high-risk based on purchase frequency trends or declining order sizes — and those customers may be **high-value**. This explains why the model captures more churners in its top 10% (higher Precision@K and Lift@K), even though the revenue-at-risk among its top 10% is lower than the recency rule's.

This is not a model failure. It is a deliberate trade-off: concentrate the contact list on the highest-risk customers, even if some of them are smaller accounts, in order to maximise the number of churners reached per pound of campaign spend.

---

## 3. Risk Band Analysis

**The "Why":** Before trusting the model's probability scores for business calculations, we need a sanity check: does a higher predicted churn probability actually correspond to a higher observed churn rate? If the model's decile 10 (highest predicted risk) does not have a substantially higher actual churn rate than decile 1, the model's scores are not meaningful.

### Procedure

1. Rank all test rows by predicted churn probability.
2. Divide into 10 equal-sized bins (deciles): decile 1 = lowest risk, decile 10 = highest risk.
3. For each decile, compute the observed churn rate.

### Results

| Risk decile | Customers | Observed churn rate |
|---|---|---|
| 1 (lowest) | 2,426 | 5.1% |
| 2 | 2,425 | 13.9% |
| 3 | 2,425 | 18.6% |
| 4 | 2,425 | 29.3% |
| 5 | 2,425 | 38.8% |
| 6 | 2,425 | 47.5% |
| 7 | 2,425 | 53.8% |
| 8 | 2,425 | 53.5% |
| 9 | 2,425 | 62.8% |
| 10 (highest) | 2,426 | 77.3% |

**Reading the results:** The model correctly orders risk from 5% in the lowest decile to 77% in the highest — a 15× difference. There is a slight non-monotonicity between deciles 7 and 8 (53.8% vs 53.5%), which is expected noise and not a concern. The overall monotonic trend confirms that the model's probability scores are meaningful and can be used as a reliable basis for ranking.

---

## 4. Customer Revenue at Risk

**The "Why":** The business does not care equally about all churners. A customer who spends £3,000 per order is worth far more retention effort than one who spends £30. We need a per-customer revenue estimate to weight business-impact scenarios correctly.

We use **one purchase-equivalent** as the revenue exposure: the customer's average order value based on their history up to the snapshot date. This is a conservative and leakage-free estimate — it uses only information available at the time of scoring.

```
AvgPurchaseValue_i  = TotalRevenue_i ÷ PurchaseOccasions_i
RevenueAtRisk_i     = AvgPurchaseValue_i   [one future purchase-equivalent]
```

An alternative using monthly history is:
```
RevenueAtRisk_i  = AvgMonthlyRevenue_i × H ÷ 30
```
where H = 90 days is the prediction horizon.

**Important:** These are **estimates of revenue exposure**, not observed lost revenue. They represent the approximate revenue at stake if a customer churns and is not retained.

**Distribution across test rows:**

| Statistic | Value (£) |
|---|---|
| Mean | 395.70 |
| Median | 303.58 |
| Std | 561.61 |
| 25th percentile | 197.75 |
| 75th percentile | 438.93 |
| Max | 14,844.77 |

The high standard deviation and large maximum reflect the mix of retail customers and wholesale-like customers in the dataset. The long right tail (max £14,844) means a small number of large accounts contribute disproportionately to total revenue at risk.

---

## 5. Business Impact Estimation

**The "Why":** Lift@K tells marketing that the model's top 10% contains more churners per contact. But the marketing director's question is: "In pounds, how much more is our model worth than just using the recency rule?" This section answers that question — under stated assumptions.

The full chain of reasoning:

```
Better ranking → More churners targeted → More revenue preserved (if campaign works)
              → Lower false-positive rate → Less incentive cost wasted on non-churners
              → Net benefit = revenue preserved × margin − contact costs − incentive costs
```

Every step after "better ranking" depends on assumptions. Those assumptions are stated explicitly and varied in a scenario table so the reader can see how sensitive the conclusion is.

---

### 5.1 Targeting Funnel (per scoring week, K = 10%)

Summed across all 10 test weeks at K = 10%:

| Candidate | Targeted (N) | True churners (TP) | Non-churners (FP) | Revenue at risk (£) |
|---|---|---|---|---|
| **Final model** | 2,427 | 1,867 | 560 | 337,904 |
| RFM rule | 2,427 | 1,743 | 684 | 498,060 |
| Recency rule | 2,427 | 1,521 | 906 | 538,397 |
| Random | 2,427 | 986 | 1,441 | 330,373 |

**Reading the table:** All four candidates target exactly the same number of customers (2,427) because they all operate at K = 10%. The difference is in **who** they target. The final model captures 1,867 true churners vs 1,521 for the recency rule — 346 more churners caught with identical contact spend.

Note: The recency rule's revenue-at-risk figure (£538,397) is higher than the model's (£337,904) even though it captures fewer churners. This confirms the earlier point: the recency rule tends to select larger-value customers who happen to have long recency, whereas the model targets a broader churn-risk group that includes smaller-value accounts.

---

### 5.2 Retention Effectiveness Assumption

Retention success is an assumption — we have no campaign outcome data. Three scenarios:

| Scenario | Effectiveness (e) | Meaning |
|---|---|---|
| Conservative | 30% | 30% of correctly targeted churners are actually retained |
| Base | 40% | 40% of correctly targeted churners are retained |
| Optimistic | 50% | 50% of correctly targeted churners are retained |

```
RevenuePreserved = RevenueAtRisk(targeted churners) × e
```

---

### 5.3 Net Business Impact Formula

```
GrossProfitPreserved  = RevenuePreserved × m
ContactCost           = N_K × c_contact
IncentiveCost         = (TP_K × e  +  FP_K × r) × c_inc
NetBusinessImpact     = GrossProfitPreserved − ContactCost − IncentiveCost
```

Assumed constants (base scenario):
- `m = 35%` — gross margin on preserved revenue
- `c_contact = £0.50` per customer — constant contact cost (email/SMS)
- `c_inc = £5.00` per redemption — incentive cost (voucher, discount)
- `r = 20%` — cannibalization rate: share of non-churners who redeem the incentive anyway

---

### 5.4 Net Business Impact at K = 10%, e = 40% (Base Scenario)

| Candidate | Revenue preserved (£) | Gross profit (£) | Contact cost (£) | Incentive cost (£) | **Net impact (£)** |
|---|---|---|---|---|---|
| **Final model** | 135,161 | 47,307 | 1,214 | 4,294 | **41,799** |
| RFM rule | 199,224 | 69,728 | 1,214 | 4,170 | **64,345** |
| Recency rule | 215,359 | 75,376 | 1,214 | 3,948 | **70,214** |
| Random | 132,149 | 46,252 | 1,214 | 3,413 | **41,626** |

**Reading the result:** At K = 10%, the recency rule produces a higher net business impact than the final model under the base scenario. This appears counterintuitive given that the model has higher Lift@K. The explanation is the revenue-at-risk composition discussed above: the recency rule tends to select higher-value churners, which generates more gross profit per churner retained — even though fewer churners are captured.

This is a genuinely important finding and a core reason why **we do not select the final model from classification metrics alone**. At K = 10%, the recency rule is the better business choice under these assumptions. The model's advantage only materialises at larger K values.

---

### 5.5 Full Scenario Analysis — Final Model vs Recency Rule

| K | e | Customers | Churners captured | Recall@K | Revenue at risk (£) | Net impact (£) | vs Recency rule (£) |
|---|---|---|---|---|---|---|---|
| 10% | 30% | 2,427 | 1,867 | 0.19 | 337,904 | 30,906 | −21,225 |
| 10% | 40% | 2,427 | 1,867 | 0.19 | 337,904 | 41,799 | −28,415 |
| 10% | 50% | 2,427 | 1,867 | 0.19 | 337,904 | 52,692 | −35,605 |
| 20% | 30% | 4,850 | 3,377 | 0.35 | 794,131 | 74,420 | −24,452 |
| 20% | 40% | 4,850 | 3,377 | 0.35 | 794,131 | 100,526 | −32,792 |
| 20% | 50% | 4,850 | 3,377 | 0.35 | 794,131 | 126,632 | −41,133 |
| 30% | 30% | 7,274 | 4,697 | 0.48 | 1,356,294 | 129,151 | −13,700 |
| 30% | 40% | 7,274 | 4,697 | 0.48 | 1,356,294 | 174,273 | −18,505 |
| 30% | 50% | 7,274 | 4,697 | 0.48 | 1,356,294 | 219,395 | −23,309 |
| 40% | 30% | 9,700 | 5,993 | 0.62 | 1,878,461 | 179,692 | **+231** |
| 40% | 40% | 9,700 | 5,993 | 0.62 | 1,878,461 | 242,442 | **−6** |
| 40% | 50% | 9,700 | 5,993 | 0.62 | 1,878,461 | 305,191 | **−243** |
| 50% | 30% | 12,126 | 7,148 | 0.74 | 2,412,837 | 231,585 | **+20,362** |
| 50% | 40% | 12,126 | 7,148 | 0.74 | 2,412,837 | 312,460 | **+26,782** |
| 50% | 50% | 12,126 | 7,148 | 0.74 | 2,412,837 | 393,336 | **+33,201** |

**Key insight:** The final model's net impact **surpasses the recency rule at K ≥ 40–50%**, and the advantage grows as K increases. At lower K values, the recency rule wins because it concentrates on a smaller number of high-value churners. At higher K values, the model's broader and more accurate coverage of the full churn-risk population becomes the dominant factor.

**Operating recommendation:** The choice of K should be driven by the marketing budget. If the budget supports targeting 40–50% of active customers weekly, the final model produces materially better business outcomes than the recency rule. If the budget is very tight (K = 10–20%), the recency rule is simpler and performs better under these assumptions.

---

## 6. Model Comparison

**The "Why":** We compare four candidates — not to crown a single "winner" by one metric, but to understand each model's profile of strengths and weaknesses across all three layers of evidence.

| Candidate | ROC-AUC | PR-AUC | Lift@10% mean | Lift@10% std | Net impact @ K=10%, e=40% (£) |
|---|---|---|---|---|---|
| **Final model** | **0.764** | **0.659** | **1.924** | **0.049** | 41,799 |
| RFM rule | 0.736 | 0.624 | 1.796 | 0.049 | 64,345 |
| Recency rule | 0.638 | 0.531 | 1.565 | 0.061 | **70,214** |
| Random | 0.506 | 0.404 | 1.017 | 0.110 | 41,626 |

**Reading the comparison:**
- The final model dominates on every predictive and ranking metric.
- But the recency rule wins on net business impact at K = 10% — because it selects higher-value churners even though it selects fewer of them.
- The recency rule also has higher week-to-week variance (std = 0.061) compared to the final model (std = 0.049), meaning it is less stable.
- The model's advantage is decisive at K ≥ 40–50%.

**Conclusion:** The final model is the better choice when the campaign K is ≥ 40%. For very tight budgets (K ≤ 20%), the recency rule is a competitive alternative and should not be dismissed.

---

## 7. Error Analysis and Limitations

**The "Why":** Aggregate metrics can hide important failure modes. If the model works well for one segment but poorly for another, a campaign using average metrics will systematically under-serve or over-target certain groups.

### Error types at K = 10% (pooled test period)

| Error type | Proportion of test rows |
|---|---|
| True negative (correctly not targeted) | 57.7% |
| False negative (churner missed) | 32.3% |
| True positive (correctly targeted) | 7.7% |
| False positive (non-churner targeted) | 2.3% |

At K = 10%, the model correctly identifies 7.7% of rows as high-risk churners and misses 32.3% of churners (false negatives). This is expected — at K = 10%, the model can only target a small fraction of all churners in a single week.

### Performance by segment

| Segment | Rows | FP rate among targeted | FN rate among untargeted |
|---|---|---|---|
| Repeat buyers | 21,027 | 25.3% | 34.1% |
| One-time buyers | 3,225 | 20.6% | 55.5% |

**Key insight on one-time buyers:** The model has a much higher false negative rate for one-time buyers (55.5%) than for repeat buyers (34.1%). One-time buyers are the most common and hardest segment to predict — by definition they have only one purchase in history, so there is very little behavioral signal to work with. They churn at high rates but the model has limited features to distinguish those who will come back from those who won't.

### Performance by geography

| Geography | Rows | FP rate | FN rate |
|---|---|---|---|
| UK | 21,935 | 22.3% | 36.3% |
| Non-UK | 2,317 | 29.4% | 32.5% |

Non-UK customers have a higher false positive rate (more wasted contacts) and a slightly lower false negative rate. Non-UK customers in this dataset often behave differently — many are small wholesale accounts with irregular purchasing patterns that the model may misinterpret.

### Performance by customer size

| Customer type | Rows | FP rate | FN rate |
|---|---|---|---|
| Retail | 23,314 | 22.7% | 37.2% |
| Wholesale-like (top 1% by revenue) | 938 | n/a | 7.8% |

**Key insight on wholesale-like customers:** The model almost never misses a wholesale-like churner (FN rate = 7.8%). These large accounts have distinctive behavioral signatures that the model captures well. However, there are so few wholesale-like customers that they barely affect aggregate metrics.

---

### Core limitations

**1. No treatment/control data — causal impact cannot be measured.**
The Online Retail II dataset records transactions but contains no record of which customers received a retention offer and whether they responded. Every business-impact number in this document is a scenario estimate. The model identifies and prioritizes churn risk; the retention campaign determines treatment; actual incremental revenue requires a randomized experiment.

**2. Non-independent test observations.**
The same customer appears across multiple consecutive test weeks. Aggregate metrics treat these appearances as independent, which overstates statistical confidence. This is why metrics are reported as mean ± std across weeks rather than as a single pooled number.

**3. Revenue-at-risk is a proxy.**
`AvgPurchaseValue_i` is an estimate of what a churned customer would have spent. It ignores seasonality, purchase frequency trends, and the time value of money. It is directionally correct but not a precise accounting figure.

**4. The ~2-year dataset limits test period size.**
With a 90-day prediction horizon and embargos between splits, the test period contains only 10 weekly scoring dates. This is enough for stable mean estimates but not enough to detect slow seasonal trends or model drift.
