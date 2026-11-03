# Business Insights

This document records the business insights drawn from the report, the problems they exposed, and how each problem was handled (or is proposed to be handled). Use it as shared context when discussing targeting, business impact, or the Evaluation and Deployment chapters.

All figures are on the test period (10 scoring weeks, 2,907 customers, churn rate 40.1%) unless stated otherwise. Re-check them against [`report/content/evaluation.tex`](../../report/content/evaluation.tex) before quoting, since they change when the pipeline is rerun.

| # | Insight | Status |
|---|---|---|
| 1 | The model catches churners well, but they are mostly low-value customers, so business impact is low | Handled: Expected Loss ranking (`feat/value-aware-impact`, commit `174c16d`, 2026-10-30) |
| 2 | The top 5% of customers hold 52% of revenue; their real risk is slowing down, not leaving | Open: proposed handling only |

---

## Insight 1 — Good churn detector, weak business impact

*Raised 2026-10-28.*

### Observation

- The strongest churn signals (SHAP) are low `frequency_so_far` and low `monetary_so_far`: customers who bought once, recently started buying, buy sparsely, and have bought few products.
- The final model (LR, Extended features, C = 0.01, weighted training) is the best churn detector (ROC-AUC 0.759, PR-AUC 0.654, Lift@10% 1.888). Yet at K = 10% with the reference measurement, its net impact is the **lowest** of the non-random candidates:

  | Candidate | TP (weekly) | Revenue at risk per TP | Net impact |
  |---|---|---|---|
  | Final model | 1,831 | ~£222 | £51,506 |
  | RFM rule | 1,732 | — | £64,616 |
  | Recency rule | 1,521 | ~£354 | £70,214 |

- The churners the model is surest about are mostly **one-time buyers** with small baskets (median order £175, against £260 for the recency rule's).

### Refinements agreed in discussion

- "Low value" means low **total** spend (because frequency is low). Average order value separates churners only weakly (median £269 vs £324).
- Not every churner is low-potential. *At Risk* (churn 57.8%, above-average value) and *Potential* (new customers, churn 60.6%) are worth keeping.

### Root problem

The ranking objective (**who is most likely to leave**) does not match the business objective (**how much revenue we expect to lose**).

### Problems found while checking the business-impact calculation

The impact chain is: Revenue at risk × e (40% retention success) × m (35% margin) − £0.50 contact cost − incentive cost (TP × e + FP × r) × £5, with r = 20%. Checking how each term was computed exposed five problems:

| # | Problem | Cause | Handling | Result |
|---|---|---|---|---|
| 1 | Revenue at risk used the **uncapped** AOV | Chapter 7 used `avg_order_value_so_far`, while Section 5.5 defines `value_v` = AOV capped at the training P99 (£1,956.84) | Added `rar_v` = `value_v`. Tested the hypothesis "the recency rule wins because it catches wholesale customers" | Only partly true: four wholesale-sized rows carry £54K of the recency rule's revenue at risk. With the cap, its net impact drops from £70,214 to £63,767, below the RFM rule. But the model still loses, because its churners are mostly one-time buyers |
| 2 | Revenue at risk counted **one order** only | Undervalues regular buyers, who would lose several orders within H = 90 days | Added `rar_v_orders` = `value_v` × clip(90 / `mean_gap_so_far`, 1, 13); one-time customers count one order | Mean exposure more than doubles (£374 → £833) and spreads widely |
| 3 | Campaigns were counted **per weekly row**, not per customer | Every customer is rescored every week, so the same customer stays in the top K for several weeks | Added two counting schemes: **unique** (count each customer at first targeting) and **suppressed** (fixed weekly budget; contacted customers are excluded for 13 weeks, as in Section 8.5) | The model's 1,831 TP are only 425 distinct customers (~4.3 weeks each), so totals were inflated 3–4×. Unique counting is not budget-equal (random wins by spreading wider), so **suppressed** is the realistic scheme |
| 4 | Probabilities were **overestimated** | Weighting by 1/n raises the effective training churn rate to 62.0% (unweighted 50.0%). Multiplying P by value needs P on the right scale | Prior correction: logit_cal = logit − [logit(0.620) − logit(0.500)] = logit − 0.489. No extra data needed | Brier 0.198 → 0.195; mean predicted probability 0.468 → 0.375 (actual churn 40.1%) |
| 5 | Suppressed counting **saturates** at K = 10% | Over 10 test weeks with 13-week suppression, a 10% weekly budget contacts 83% of customers, so every ranking reaches nearly the same people | Added smaller weekly budgets K = 1%, 2%, 5% | Candidates separate again (results below) |

All three revenue measures (`rar_aov`, `rar_v`, `rar_v_orders`) and all three counting schemes (weekly, unique, suppressed) are kept side by side for analysis. The weekly + `rar_aov` combination stays as the reference measurement.

### Solution: rank by Expected Loss

**Expected Loss = calibrated churn probability × `value_v` × expected orders in 90 days**, i.e. the revenue we expect to lose if we do nothing. It uses the same fitted model; only the ranking key changes.

Net impact with suppressed counting and `rar_v_orders`:

| Ranking | K = 2% / week | K = 5% / week |
|---|---|---|
| **Expected Loss** | **£46,949** | **£65,805** |
| Recency rule | £20,120 | £41,195 |
| RFM rule | £17,963 | £38,051 |
| Random | £15,740 | £38,305 |
| Churn probability alone | £12,782 | £37,031 |

- Expected Loss is the best ranking at every K from 1% to 5%, under **all three** revenue measures. Its lead under `rar_aov` and `rar_v` is the independent evidence, since `rar_v_orders` is what it was built on.
- Ranking by churn probability alone is **below random** at K = 1–5% under `rar_v_orders`. This is the clearest quantitative form of Insight 1.
- Expected Loss catches fewer churners (TP 235 vs 363 at K = 2%), but contacting a non-churner is cheap compared with keeping one high-value churner.

**Operating rule:** score churn risk with the model, rank by Expected Loss, and contact a small weekly share with 13-week suppression.

### Where it lives

- [`notebooks/09_evaluation.ipynb`](../../notebooks/09_evaluation.ipynb): calibration, Measurement Sensitivity, Scenario Analysis, and the `churn_probability_calibrated` and `expected_loss` scoring columns
- [`report/content/evaluation.tex`](../../report/content/evaluation.tex): Sections 7.2, 7.5, Measurement Sensitivity, 7.10, 7.11
- [`report/content/deployment.tex`](../../report/content/deployment.tex): Sections 8.3–8.5
- [`brief/main.tex`](../../brief/main.tex) and [`slide/main.tex`](../../slide/main.tex): Recommendation

---

## Insight 2 — Revenue is concentrated in a few large customers

*Raised 2026-10-28.*

### Observation

- The top 5% of customers (293) hold **52.0%** of revenue; the top 1% (59) hold **32.1%** (Section 4, value concentration). Many of them look like wholesale buyers.
- These customers almost never churn under the 90-day label. In exploratory analysis (not in the report), wholesale-like customers had a false-negative rate of 7.8%, and none of them entered the model's top 10%.

### Problems

1. **The churn label misses their real risk.** A key account rarely disappears for 90 days; the realistic risk is buying **less often or less per order**. A model trained on "no purchase in 90 days" cannot see that.
2. **Discounts are the wrong tool for them.** Customers in *Can't Lose* would likely buy anyway, so a discount mostly gives away margin (cannibalization).
3. **"Getting small customers to buy more" is a growth problem** (upsell, customer lifetime value), not a churn problem. Folding it into the churn model would blow up the scope.

### Proposed handling (not implemented)

- **Monitor purchase rhythm for key accounts.** Flag a large customer when `recency_to_typical_gap` (current recency relative to their usual gap) rises well above 1, instead of waiting for the churn label.
- **Personal account care, no discount** for high-value, low-risk customers.
- **Risk × Value matrix** to assign an action to every customer:

  | | High value | Low value |
  |---|---|---|
  | **High risk** | Personal contact | Cheap automated channel |
  | **Low risk** | Keep the rhythm, no discount | Nurture / upsell |

- Keep **upsell** for low-value customers as a recommendation in the report rather than a modeling task.

Related TODO items: `feat/value-aware-targeting` (actions per customer) and `feat/policy-simulation` (policy comparison), see [`TODO.md`](../../TODO.md).

---

## Open Items

- Implement the Insight 2 handling: rhythm monitoring for key accounts and the Risk × Value action matrix.
- Validate the Expected Loss ranking in a pilot campaign with a held-out control group. The dataset has no treatment or control information, so all impact figures are scenario estimates (Section 10.5).
