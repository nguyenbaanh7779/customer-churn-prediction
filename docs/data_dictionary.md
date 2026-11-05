# Data Dictionary and Source Log

## 1. Source log

| Item | Value |
|---|---|
| Dataset | Online Retail II (UCI Machine Learning Repository) |
| URL | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| Licence | CC BY 4.0 (credit required) |
| Citation | Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D |
| Content | All transactions of a UK-based, registered, non-store online retailer of all-occasion gift-ware (many customers are wholesalers), 2009-12-01 to 2011-12-09, donated 2019-09-20; 1,067,371 rows, 5,942 distinct Customer IDs; 22.8% of rows have no Customer ID |
| Loaded by | `notebooks/00_explore_data.ipynb`: reads the two Excel sheets and writes `data/raw/online_retail_II.parquet` (gitignored) |
| Retrieval date | 2 November 2026 (date the team obtained the data and converted it to parquet) |
| What the data does not contain | product cost, gross margin, past promotions, discounts offered, customer demographics |

### Cleaning log (valid-purchase filter, `notebooks/07_data_preparation.ipynb`, report Section 5.2)

A row is kept only if it:
1. has a Customer ID;
2. is not a cancellation (Invoice starting with `C`; 19,494 rows, 1.83%; kept separately for cancellation features);
3. has Quantity > 0 and Price > 0;
4. is a real product (StockCode starting with a digit; non-product codes are excluded);
5. is not an exact duplicate of an earlier row.

Rows without a Customer ID cannot be attached to any customer history and are dropped from every customer-level result.

### Joins and derived tables

| Table | Built by | Key |
|---|---|---|
| `data/processed/modeling_dataset_h90.parquet` | notebook 07 | customer_id x snapshot_date |
| `data/results/churn_scores_h90.parquet` | notebook 09 | customer_id x snapshot_date |
| `data/results/churn_scores_h90_calibrated.parquet` | **modeling team delivery, copied in; no notebook in this repo produces it** | customer_id x snapshot_date |
| `data/results/customer_scores_actions_h90.parquet` | notebook 12 (joins the calibrated scores with value, RFM segment and actions) | customer_id x snapshot_date |

## 2. Raw schema (`online_retail_II.parquet`)

| Column | Type | Meaning |
|---|---|---|
| Invoice | string | Invoice number; `C` prefix = cancellation |
| StockCode | string | Product code; some non-numeric codes are not products |
| Description | string | Free-text product name |
| Quantity | int | Units on the line; negative for cancellations |
| InvoiceDate | datetime | Transaction timestamp |
| Price | float | Unit price, GBP |
| Customer ID | float | Customer identifier (missing for 22.8% of rows) |
| Country | string | Customer country |

## 3. Modeling dataset (`modeling_dataset_h90.parquet`, 208,764 rows)

Unit: one customer on one Monday snapshot date. Features use data up to that date only. Horizon H = 90 days.

| Column | Meaning |
|---|---|
| customer_id, snapshot_date | Keys |
| segment | `repeat` or `one-time` (customer type at that date) |
| in_training_sample, in_scoring_population | Row-eligibility flags used by the split and scoring |
| sample_weight | Training weight, 1 / (number of training rows of that customer) |
| recency_days | Days since last valid purchase |
| frequency_so_far | Purchase occasions so far |
| monetary_so_far | Revenue so far, GBP |
| avg_order_value_so_far | Revenue per purchase occasion so far |
| tenure_days | Days since first purchase |
| mean_gap_so_far | Mean days between purchase occasions |
| recency_to_typical_gap | recency_days relative to the customer's typical gap |
| activity_last_90d | Purchase occasions in the last 90 days |
| distinct_products_so_far | Distinct products bought so far |
| cancel_count_so_far, cancel_value_share_so_far | Cancellation count and share of value |
| labelable | True if snapshot_date + 90 days is within the data |
| churn | 1 if no valid purchase in (t, t+90 days], else 0 (empty if not labelable) |
| split | `train`, `validation`, `test`, or `none` (embargo gaps) |
| value_v | Customer value V: average order value capped at the P99 of the training period |
| r_score, f_score, m_score | RFM quintile scores fitted on the training period |
| rfm_segment | Champions, Loyal, Potential, At Risk, Can't Lose, Hibernating (6-group rule from the pipeline; the dashboard and brief use `segment8`, see below) |

Split (embargo 90 days between periods): train 2010-06-07 to 2010-11-15; validation 2011-02-14 to 2011-04-04; test 2011-07-04 to 2011-09-05. The last scored week (2011-12-05) has no label yet.

## 4. Score table with actions (`customer_scores_actions_h90.parquet`, 60,455 rows)

Test weeks plus later unlabelled weeks (`split` = `test` or `deployment`).

| Column | Meaning |
|---|---|
| customer_id, snapshot_date, split | Keys and period |
| churn_probability | Raw model probability of no purchase in 90 days |
| churn_probability_calibrated | Calibrated probability (logit offset from the modeling team); used for all policy results |
| risk_score, risk_band | 0-1000 score and decile band |
| value_v | Capped average order value (GBP) |
| rfm_segment | Segment as above |
| customer_type | `repeat` or `one-time` |
| expected_margin | E[dM] = p*e*V*(m-d) - (1-p)*r*V*d - c, Base scenario |
| rank_by_margin | Rank within the week by expected_margin |
| action | Retain (top 20% by expected margin and E[dM] > 0), Nurture (not Retain, top 20% risk), No-offer |
| model_version | `lr_extended_c0.01_weighted_cal_h90` |
| churn_actual | Observed churn label (empty for unlabelled weeks) |
| scenario | Scenario string used |

Scenario parameters: d = discount, e = response of customers who would not otherwise buy, r = redemption by customers who would buy anyway, m = gross margin before discount, c = contact cost. Base: d 10%, e 30%, r 40%, m 40%, c 0.50 GBP. Light: d 5%, e 20%. Deep: d 20%, e 40%. All are assumptions, not data.

## Dashboard segments (`data/results/rfm8_segments.parquet`)

Built by `src/make_rfm8.py` with the 8-group rule of `output/02_rfm_segmentation.ipynb` (part B), computed at each weekly snapshot: Champions (top-10% spend, R<=90), VIP at risk (top-10% spend, R>90), Loyal (R<=90, F>=4), Developing (R<=90, F=2-3), One-time recent (R<=90, F=1), Cooling (R 91-180), Lapsed (R 181-365), Dormant (R>365). The score table only holds active customers, so Lapsed and Dormant do not appear in the dashboard. Column `segment8`.
