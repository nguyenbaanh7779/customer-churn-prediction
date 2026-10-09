# Model Deployment Methodology

> Corresponds to **Chapter 10 — Deployment** of the project report.

---

## 1. Philosophy — What Deployment Means Here

Training and evaluation prove the model works in a controlled experiment. Deployment makes it work in the field — reliably, repeatedly, and with a clear input/output contract — so the marketing team can act on its output every week without a data scientist present.

Two deployment modes are designed:

| Mode | Description | Audience |
|---|---|---|
| **Weekly batch scoring** | Scheduled pipeline run → ranked target list | Marketing team (primary) |
| **REST API** | On-demand scoring endpoint | CRM systems, campaign platforms, downstream integrations |

Both modes run the same core pipeline logic. The difference is how they are triggered and how they deliver output.

---

## 2. Weekly Batch Scoring

### 2.1 Purpose and Cadence

The model scores every active customer once per week (e.g., every Monday at 06:00). The output is a ranked retention target list that the marketing team uses for the week's campaign.

Weekly cadence is appropriate because:
- The prediction horizon is H = 90 days. Scoring on Monday vs Thursday of the same week makes no material difference to the forecast.
- Weekly batch is operationally simple — one run, one output file, one target list.
- Real-time scoring adds infrastructure complexity without improving the churn signal, which is driven by gradual behavioral drift, not moment-to-moment events.

### 2.2 Pipeline — Input / Output Contract

**Input:**

| Field | Type | Description |
|---|---|---|
| Transaction file | `.parquet` or `.xlsx` | Raw customer transaction history |
| `snapshot_date` | `date` | Scoring date (defaults to latest transaction date) |
| `k_frac` | `float` [0, 1] | Target list fraction (e.g., `0.30` = top 30%) |
| `r_max` | `int` (days) | Activity threshold — customers with recency > r_max are excluded |

**Output — scored_all_{snapshot_date}.csv (one row per active customer):**

| Field | Type | Description |
|---|---|---|
| `customer_id` | int | Unique customer identifier |
| `snapshot_date` | date | Date of this scoring run |
| `churn_probability` | float [0, 1] | Estimated probability of churn in the next H = 90 days |
| `churn_prob_lo` | float [0, 1] | Lower bound of 90% confidence interval |
| `churn_prob_hi` | float [0, 1] | Upper bound of 90% confidence interval |
| `risk_band` | string | `High` / `Medium` / `Low` |
| `rank` | int | 1 = highest risk in this week's scoring population |
| `segment` | string | `one-time` or `repeat` buyer |
| `revenue_at_risk` | float (£) | Average order value — estimate of revenue at stake |
| `in_target_list` | bool | True if selected for this week's campaign |
| `model_version` | string | Version identifier of the model used |

**Output — target_list_{snapshot_date}.csv (subset: top K% only):**

Same schema as above, filtered to `in_target_list == True`. This is the file the marketing team uploads to their campaign platform.

### 2.3 Pipeline Steps

```
Customer transaction history
    → Step 1: Data ingestion and validation
    → Step 2: Transaction cleaning
    → Step 3: Active customer identification (recency ≤ R_max)
    → Step 4: Feature engineering (history up to snapshot_date only)
    → Step 5: Model scoring (churn_probability)
    → Step 6: Confidence interval estimation (parametric bootstrap)
    → Step 7: Risk ranking and banding
    → Step 8: Target population selection (top K%)
    → Step 9: Output files + monitoring dashboard update
```

Each step has a single responsibility. Steps 4–6 are identical between training and serving — no train/serve skew.

### 2.4 Why Every Active Customer Is Scored Every Week

A simpler approach would be to score each customer once every 90 days. This is wrong:

- A customer's behavioral risk changes every week they go without buying. A customer who looked safe four weeks ago may now be overdue.
- If a customer's behavior deteriorates between scoring dates, they will not appear in the target list even though they are now high risk.

The training-episode cooldown (applied only during data preparation to reduce correlated training rows) is explicitly **not** applied in the scoring pipeline.

### 2.5 Suppression

To avoid repeatedly contacting the same customer in a short window:

- A contacted customer is suppressed for S = 4–8 weeks (configurable).
- Suppressed customers are still **scored** every week — their risk is tracked — but they are excluded from the target list.
- Suppression log fields: `customer_id`, `contact_date`, `suppression_end_date`, `campaign_id`.

Suppression is a campaign management rule. It is unrelated to the training cooldown.

---

## 3. REST API Deployment

### 3.1 Purpose

The REST API exposes the scoring pipeline as an HTTP endpoint so that other systems (CRM, campaign platforms, internal tools) can request churn scores on demand without running the full pipeline manually.

### 3.2 Endpoint Design

**POST /score**

Scores a batch of customers at a given snapshot date.

**Request body:**

```json
{
  "snapshot_date": "2011-12-01",
  "k_frac": 0.30,
  "r_max": 136,
  "data_source": "latest"
}
```

`data_source` can be `"latest"` (use the most recent transaction extract on file) or a path/reference to a specific extract.

**Response:**

```json
{
  "snapshot_date": "2011-12-01",
  "model_version": "v1.0",
  "active_customers": 2425,
  "target_list_size": 728,
  "scores": [
    {
      "customer_id": 12345,
      "churn_probability": 0.872,
      "churn_prob_lo": 0.831,
      "churn_prob_hi": 0.908,
      "risk_band": "High",
      "rank": 1,
      "segment": "repeat",
      "revenue_at_risk": 1245.00,
      "in_target_list": true
    },
    ...
  ]
}
```

**GET /score/{customer_id}**

Returns the most recent score for a single customer.

**Response:**

```json
{
  "customer_id": 12345,
  "snapshot_date": "2011-12-01",
  "churn_probability": 0.872,
  "churn_prob_lo": 0.831,
  "churn_prob_hi": 0.908,
  "risk_band": "High",
  "rank": 1,
  "segment": "repeat",
  "revenue_at_risk": 1245.00,
  "in_target_list": true,
  "model_version": "v1.0"
}
```

**GET /health**

Returns pipeline and model status.

```json
{
  "status": "ok",
  "model_version": "v1.0",
  "last_scoring_run": "2011-12-01T06:00:00",
  "active_customers_last_run": 2425
}
```

### 3.3 How Scores Are Used Downstream

| Consumer | How it uses the API |
|---|---|
| CRM system | Pulls `GET /score/{customer_id}` on customer profile page to show risk band to account manager |
| Campaign platform | Calls `POST /score` on Monday morning to get the full target list for the week's retention campaign |
| Analytics dashboard | Reads from the scored output files to populate monitoring charts |
| A/B test framework | Uses `in_target_list` field to split customers into treatment (contacted) and holdout (not contacted) groups |

---

## 4. Feature Engineering — No Train/Serve Skew

All feature transformations applied at training time must be applied identically at scoring time. The full feature recipe is:

| Feature | Transformation |
|---|---|
| `monetary_so_far` | `log1p`, clip to [0, 1e9] |
| `avg_order_value_so_far` | `log1p`, clip to [0, 1e9] |
| `tenure_days` | `log1p`, clip to [0, 1e9] |
| `mean_gap_so_far` | `log1p`, clip to [0, 1e9] |
| `recency_to_typical_gap` | clip to [0, 20], then `log1p` |
| `activity_last_90d` | `log1p`, clip to [0, 1e9] |
| `distinct_products_so_far` | `log1p`, clip to [0, 1e9] |
| `cancel_count_so_far` | `log1p`, clip to [0, 1e9] |
| `frequency_so_far` | `log1p`, clip to [0, 1e9] |
| All features | Missing values → replace `inf`/`-inf` with `NaN` → impute with training median |
| All features | Scale with training `StandardScaler` |

The imputer and scaler are fitted on training data and saved alongside the model. They are **never re-fitted** on new scoring data.

---

## 5. Operational Parameters

All parameters are versioned and logged alongside every scoring output:

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| Activity threshold | R_max | 136 days | Max recency to be considered active (P90 of inter-purchase interval) |
| Prediction horizon | H | 90 days | Forward window for churn label |
| Suppression period | S | 6 weeks | Weeks a contacted customer is excluded from the target list |
| Scoring cadence | — | Weekly (Monday) | When the batch pipeline runs |
| Target share | K | 30% | Fraction of scoring population targeted |
| Minimum history | — | 6 months | Minimum customer tenure before scoring |
| CI method | — | Parametric bootstrap | 200 samples, 90% confidence interval |
| Model version | — | Logged per run | Semantic version of the model artefact |

---

## 6. Model Artefacts

Three objects are saved at training time and loaded at scoring time:

| Artefact | File | Contents |
|---|---|---|
| Model | `model.pkl` | Fitted `LogisticRegression` (C=0.01, solver=liblinear) |
| Imputer | `imputer.pkl` | Fitted `SimpleImputer` (strategy=median) |
| Scaler | `scaler.pkl` | Fitted `StandardScaler` |

These are generated by running `python app/save_artefacts.py` once after training. In a production setup, they would be stored in a model registry (e.g., MLflow) and loaded by version tag rather than by file path.

---

## 7. From Demo to Production

The Streamlit app (`app/app.py`) is a working demo of the full pipeline. It covers Steps 1–8 above in a single interactive session. What changes when moving to production:

| Aspect | Demo (Streamlit) | Production |
|---|---|---|
| Data source | Uploaded file or local Parquet | Automated extract from transaction database |
| Scheduling | Manual (button click) | Cron job or Airflow DAG every Monday 06:00 |
| Suppression | Not implemented | Suppression list read from campaign CRM |
| Model artefacts | Local `.pkl` files | Model registry (MLflow, S3) |
| Output | CSV download in browser | Written to campaign database / CRM via API |
| Authentication | None | SSO / role-based access |

The core pipeline logic (Steps 1–8) is identical in both environments.
