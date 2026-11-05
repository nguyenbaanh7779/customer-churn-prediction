# Model Card: Customer churn probability (h = 90 days)

## Purpose and use
Estimate, for each active customer on each Monday, the probability that they make **no valid purchase in the next 90 days**. The score ranks customers for a retention offer; it does not measure whether an offer changes behaviour. Intended use is decision support for a weekly retention list, demonstrated on historical data. Not for individual credit-like or pricing decisions outside this retailer.

## Model
- Logistic regression, C = 0.01, trained with per-customer weights (1 / rows per customer) so frequent customers do not dominate.
- Features (8): recency_days, frequency_so_far, monetary_so_far, avg_order_value_so_far, tenure_days, mean_gap_so_far, recency_to_typical_gap, activity_last_90d. The skewed ones go through log1p; median imputation and standardisation follow.
- Final fit on train + validation. A logit offset (about +0.49) produces `churn_probability_calibrated`; this step was delivered by the modeling team.
- Artifacts: `models/churn_model_h90_calibrated.joblib`, scores in `data/results/churn_scores_h90_calibrated.parquet`.

## Data
Online Retail II, valid purchases only (see `docs/data_dictionary.md`). Weekly snapshots from June 2010; train 2010-06-07 to 2010-11-15, validation 2011-02-14 to 2011-04-04, test 2011-07-04 to 2011-09-05, with 90-day embargoes between periods.

## Performance (test, Jul-Sep 2011; churn rate 40.1%)
| Metric | Value |
|---|---|
| ROC-AUC | 0.759 (validation 0.780) |
| PR-AUC | 0.654 |
| Lift@10% | 1.89 |
| ROC-AUC, repeat customers | 0.75 |
| ROC-AUC, one-time customers | 0.66 |

## Use in the decision
Probabilities feed the targeting rule: expected incremental margin per customer under scenario assumptions (notebooks 10-13). At the same expected discount spend, ranking by this probability gives about 51% more simulated margin than ranking by days since last purchase (notebook 13). These are scenario results, not measured campaign returns.

## Limitations
- **Seasonality.** The churn rate moves from about 61% (Dec-Jan) to 40% (Jul-Sep) while the mean predicted probability moves much less. Absolute risk is understated in winter and overstated in summer; the ranking is more reliable than the level. The model has no calendar feature.
- **One-time buyers** are ranked less accurately (AUC 0.66) and make up 44% of the proposed offer list in the test weeks.
- **Value.** Average order value is a proxy for future spend; large wholesale-like orders can dominate.
- **No causal claim.** There are no promotions in the data. Response, redemption and margin are assumptions.
- **Calibration offset.** Which period it was fitted on has not been confirmed to exclude the test weeks.
- **Coverage and fairness.** Rules use behaviour only (no protected attributes), but results by country or customer size have not been checked.
- A single test season (Jul-Sep 2011); the rolling backtest (notebook 11) covers four periods with models refit per fold.

## Monitoring
After 90 days compare predicted and actual churn by week; track lift, calibration and drift in transaction volume; retrain with the same embargoed split when ranking quality drops. If campaigns are run, measure response and redemption with a no-offer control group.
