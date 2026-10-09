# Deployment Options Evaluation: Streamlit vs MLflow

> Evaluating two deployment paths for the customer churn scoring pipeline.  
> Context: UCI Online Retail II project — weekly batch scoring → retention target list.

---

## The Question

We have a trained churn prediction model. We need to deploy it so that a **marketing team can use it every week** without a data scientist present. Two open-source options are on the table:

| Option | What it is |
|---|---|
| **Streamlit (local)** | Python web app framework — turn a script into an interactive dashboard |
| **MLflow (open-source)** | ML lifecycle platform — experiment tracking, model registry, model serving |

These are not the same kind of tool. The comparison is not "which is better" in the abstract — it is "which fits this project's actual constraints and goals."

---

## Understanding What Each Tool Actually Does

### Streamlit

Streamlit turns a Python script into a browser-based interactive application. It is:

- A **UI layer** — forms, sliders, charts, tables, file uploads, download buttons.
- A **presentation tool** — shows the pipeline output to a human.
- A **demo environment** — excellent for showing stakeholders how the system works.

Streamlit does **not** do:

- Model versioning or experiment tracking.
- Scheduled pipeline execution.
- REST API serving.
- Monitoring or drift detection.

### MLflow (Open Source)

MLflow is a platform for managing the ML lifecycle. Its four components are:

| Component | What it does |
|---|---|
| **Tracking** | Log metrics, parameters, and artefacts during training runs |
| **Projects** | Package ML code in a reproducible format |
| **Models** | Standardize model serialization (MLmodel format) |
| **Registry** | Version models, move them through Staging → Production → Archived |

MLflow also has a **Model Serving** feature (`mlflow models serve`) that exposes a registered model as a REST API.

MLflow does **not** do:

- Build interactive dashboards or visualisations.
- Replace the scoring pipeline logic.
- Schedule jobs (it needs an orchestrator like Airflow or a cron job alongside it).

### The key insight

These tools solve **different problems**. Streamlit is a frontend. MLflow is a backend lifecycle manager. In a complete system, you would use both — MLflow manages the model artefact, Streamlit (or another frontend) presents the results. The question is which to prioritise **first** given this project's current stage and goals.

---

## Evaluation Framework

We evaluate against five dimensions that matter for this project:

1. **Fit for purpose** — does it solve the actual problem?
2. **Ease of setup** — how long to get something running?
3. **Demo quality** — how well does it communicate value to a non-technical stakeholder?
4. **Production readiness** — how close is it to a real weekly pipeline?
5. **Maintenance burden** — what is the ongoing cost of keeping it working?

---

## Side-by-Side Evaluation

### Dimension 1 — Fit for Purpose

**What we need:**
- Marketing team can run the pipeline weekly without writing code.
- Output is a ranked customer list with churn probabilities and revenue-at-risk estimates.
- A stakeholder demo shows the model working end-to-end in ~15 minutes.

| Criteria | Streamlit | MLflow |
|---|---|---|
| Marketing team runs pipeline | ✅ Button click in browser | ❌ Requires CLI command or API call |
| Ranked customer list as output | ✅ Built-in table + CSV download | ⚠️ Possible via API, but no built-in UI |
| 15-minute stakeholder demo | ✅ Excellent — visual, interactive | ❌ Technical — shows experiment runs, not business output |
| Churn probability dashboard | ✅ Native charts, risk bands, filters | ❌ Not a dashboard tool |

**Winner: Streamlit** — it directly solves the "marketing team uses this" problem.

---

### Dimension 2 — Ease of Setup

**Streamlit:**
```bash
pip install streamlit
streamlit run app.py
```
The app opens in a browser at `localhost:8501`. Zero server configuration. Zero database. Works on any laptop.

**MLflow:**
```bash
pip install mlflow
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```
This starts the tracking server. Then you need to:
- Instrument your training code with `mlflow.log_params()`, `mlflow.log_metrics()`, `mlflow.sklearn.log_model()`.
- Register the model in the registry.
- Run `mlflow models serve -m "models:/churn_model/Production" --port 5001` to expose a REST endpoint.
- Write a separate scoring script that calls the endpoint.

**Verdict:**

| | Streamlit | MLflow |
|---|---|---|
| Time to first working demo | 1–2 hours | 4–8 hours |
| Dependencies added to project | 1 package | 3–5 packages (mlflow, sqlalchemy, alembic, gunicorn) |
| Server infrastructure needed | None | SQLite or PostgreSQL backend |
| Works on a laptop without Docker | ✅ Yes | ⚠️ Yes, but fragile |

**Winner: Streamlit** — dramatically simpler first-time setup.

---

### Dimension 3 — Demo Quality

**Streamlit demo (15 minutes):**
- Upload transaction file → click Run → see a risk distribution chart, target list table, individual customer lookup.
- A marketing manager understands immediately: "these are my 728 at-risk customers this week."
- Download button → CSV file ready for the campaign platform.

**MLflow demo (15 minutes):**
- Open MLflow UI → show experiment runs → compare metrics across training runs.
- This is compelling for a data science audience — you can see how Logistic Regression compares to XGBoost on PR-AUC, view feature importance plots, inspect training parameters.
- For a marketing manager, this is opaque. "What do these metrics mean? Where is my customer list?"

**Verdict:**

| Audience | Streamlit | MLflow |
|---|---|---|
| Marketing manager | ✅ Immediately useful | ❌ Confusing |
| Data scientist | ⚠️ Shows output, not training process | ✅ Excellent |
| Engineering manager | ✅ Shows working pipeline | ✅ Shows model governance |

**Winner: Streamlit** for stakeholder demos. **MLflow** if the audience is technical.

---

### Dimension 4 — Production Readiness

This is where the comparison reverses.

**What production requires beyond a demo:**

| Requirement | Streamlit | MLflow |
|---|---|---|
| Scheduled weekly run (no human needed) | ❌ Cannot self-schedule | ⚠️ Needs cron/Airflow alongside |
| Model versioning (v1, v2, rollback) | ❌ No built-in versioning | ✅ Full registry: Staging → Production → Archived |
| Reproducible model artefacts | ⚠️ You manage files yourself | ✅ MLmodel format + artefact storage |
| A/B model comparison in production | ❌ Manual | ✅ Native via registry stages |
| REST API for other systems to call | ❌ Not a backend tool | ✅ `mlflow models serve` |
| Audit trail (who changed what model, when) | ❌ None | ✅ Registry tracks all transitions |
| Multi-user access control | ❌ None | ⚠️ Basic (full RBAC needs MLflow Enterprise) |

**Winner: MLflow** — it is designed for production model governance.

---

### Dimension 5 — Maintenance Burden

**Streamlit:**
- Low maintenance when working.
- Fragile for production: if the person running the laptop forgets to update the transaction file, nothing runs.
- No separation between "this is the tested model version" and "the model I'm experimenting with."
- As the project grows (more models, more features), there is no structure to manage model versions.

**MLflow:**
- Higher initial setup cost.
- Once the server is running and the model is registered, maintenance is low.
- Model updates are controlled: you promote a new model to Production, the old one moves to Archived, and everything that calls the registry gets the new version automatically.
- Requires keeping the MLflow server alive (process management, backups of the SQLite/PostgreSQL database).

---

## Decision Matrix

| Dimension | Weight | Streamlit score | MLflow score |
|---|---|---|---|
| Fit for purpose (marketing team + demo) | 35% | 9 | 4 |
| Ease of setup | 20% | 9 | 5 |
| Demo quality (stakeholder) | 20% | 9 | 3 |
| Production readiness | 15% | 3 | 8 |
| Maintenance burden (long-term) | 10% | 6 | 7 |
| **Weighted total** | | **7.90** | **5.05** |

**For this project at its current stage: Streamlit wins clearly.**

---

## Recommendation

### Recommended approach: Streamlit now, MLflow later

This is not a permanent either/or choice. The recommended path is:

```
Stage 1 — Demo / Pilot (now)
    ┌─────────────────────────────────────────────────────┐
    │  Streamlit local app                                │
    │  → One-command pipeline: raw data → scored list     │
    │  → Dashboard: risk distribution, target list, CSV   │
    │  → Validates the concept with marketing stakeholders │
    └─────────────────────────────────────────────────────┘

Stage 2 — Operational (when team commits to weekly use)
    ┌─────────────────────────────────────────────────────┐
    │  Add MLflow tracking to training notebooks          │
    │  → Log all experiment runs and metrics              │
    │  → Register the production model in MLflow registry │
    │  Keep Streamlit as the UI                           │
    │  → Load model from MLflow registry instead of file  │
    │  Add cron job / simple Airflow DAG for weekly run   │
    └─────────────────────────────────────────────────────┘

Stage 3 — Scale (if team and data volume grow)
    ┌─────────────────────────────────────────────────────┐
    │  MLflow serves the model as a REST API              │
    │  → CRM / campaign platform reads scores directly    │
    │  Streamlit becomes internal analytics dashboard     │
    │  Full monitoring and alerting pipeline              │
    └─────────────────────────────────────────────────────┘
```

---

## Why NOT MLflow-first for this project

Three specific reasons based on this project's constraints:

**1. The current stack has no MLflow dependency.**

The project uses: `pandas`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `pyarrow`, `openpyxl`. Adding MLflow means adding `mlflow`, `sqlalchemy`, `alembic`, `gunicorn` and their transitive dependencies — a significant increase in complexity for a project that is currently clean and minimal.

**2. The primary deliverable is a stakeholder demo, not a production system.**

The project strategy explicitly states: *"Chapters 10 and 11 describe a proposed operational design; they are not evaluated on a running system."* A Streamlit app directly supports this — it demonstrates the proposed design interactively. MLflow's value is in managing an operational system over time, which is future scope.

**3. MLflow without a UI is incomplete for a marketing team.**

Even if you register the model in MLflow and serve it via REST API, the marketing team still needs a way to see the ranked customer list and download it. You would end up building a separate UI anyway — which is exactly what Streamlit is. MLflow without a frontend solves the backend problem but leaves the user-facing problem unsolved.

---

## Why NOT Streamlit-forever

If this project moves from demo to operational weekly pipeline, Streamlit becomes the bottleneck:

- Someone has to manually start the app and run the pipeline every Monday.
- There is no safety net if the transaction file format changes — the app crashes with an unclear error, and there is no automatic alerting.
- When you train a new model version, there is no formal process for promoting it — you just overwrite the `.pkl` file, with no record of what changed or why.
- If two people run the app simultaneously with different data, there is no coordination or locking.
- There is no rollback capability if a new model performs worse in production.

These are not Streamlit's fault — they are inherent limitations of any local application that was not designed as a production service.

---

## Implementation Plan — Stage 1: Streamlit App

### Project structure

```
customer-churn-prediction/
├── app/
│   ├── app.py                    ← Streamlit entry point
│   ├── pipeline.py               ← Core scoring logic (clean → features → score → rank)
│   ├── features.py               ← Feature engineering functions
│   ├── charts.py                 ← Reusable Matplotlib/Plotly chart components
│   └── model_artefacts/
│       ├── model.pkl             ← Trained LogisticRegression (saved with joblib)
│       ├── imputer.pkl           ← Fitted SimpleImputer (median)
│       └── scaler.pkl            ← Fitted StandardScaler
├── data/
│   └── raw/                      ← Transaction data (gitignored)
├── requirements.txt              ← existing project requirements
└── requirements_app.txt          ← app-specific additions
```

### Additional requirements

```
# requirements_app.txt  (additions to existing requirements.txt)
streamlit==1.40.0
joblib==1.4.2       # already bundled with scikit-learn, explicit pin for clarity
```

No new ML dependencies. `joblib` is already installed as part of `scikit-learn`.

### Saving model artefacts (run once after training)

```python
# app/save_artefacts.py
# Run this once after 08_modeling.ipynb finishes training

import joblib

# These objects are created in 09_evaluation.ipynb (refit section)
# Export them here for the app to use
joblib.dump(final_model,  "app/model_artefacts/model.pkl")
joblib.dump(imputer,      "app/model_artefacts/imputer.pkl")
joblib.dump(scaler,       "app/model_artefacts/scaler.pkl")

print("Artefacts saved.")
```

### Running the app

```bash
# Install app requirements
pip install -r requirements_app.txt

# Save artefacts (once)
python app/save_artefacts.py

# Launch the app
streamlit run app/app.py
# → opens at http://localhost:8501
```

### Core pipeline function (pipeline.py sketch)

```python
# app/pipeline.py

import numpy as np
import pandas as pd
import joblib
from features import build_features, clean_transactions

MODEL    = joblib.load("app/model_artefacts/model.pkl")
IMPUTER  = joblib.load("app/model_artefacts/imputer.pkl")
SCALER   = joblib.load("app/model_artefacts/scaler.pkl")

FULL_FEATURES = [
    "recency_days", "frequency_so_far", "monetary_so_far",
    "avg_order_value_so_far", "tenure_days", "mean_gap_so_far",
    "recency_to_typical_gap", "activity_last_90d",
    "distinct_products_so_far", "cancel_count_so_far",
    "cancel_value_share_so_far",
]

LOG_FEATURES = [
    "monetary_so_far", "avg_order_value_so_far", "tenure_days",
    "mean_gap_so_far", "recency_to_typical_gap", "activity_last_90d",
    "distinct_products_so_far", "cancel_count_so_far", "frequency_so_far",
]

def run_pipeline(transactions_df, snapshot_date, k_frac=0.30, r_max_days=136):
    # 1. Clean
    clean = clean_transactions(transactions_df)

    # 2. Active customers
    active = get_active_customers(clean, snapshot_date, r_max_days)
    if active.empty:
        return None, None

    # 3. Features
    features_df = build_features(clean, active, snapshot_date)

    # 4. Score
    X = features_df[FULL_FEATURES].copy()
    for col in LOG_FEATURES:
        X[col] = np.log1p(X[col])
    X_scaled = SCALER.transform(IMPUTER.transform(X))
    features_df["churn_probability"] = MODEL.predict_proba(X_scaled)[:, 1]

    # 5. Rank + band
    features_df = features_df.sort_values("churn_probability", ascending=False)
    features_df["rank"] = range(1, len(features_df) + 1)
    features_df["risk_band"] = pd.cut(
        features_df["rank"],
        bins=[0, len(features_df)*0.3, len(features_df)*0.6, len(features_df)],
        labels=["High", "Medium", "Low"]
    )

    # 6. Revenue at risk
    features_df["revenue_at_risk"] = features_df["avg_order_value_so_far"]

    # 7. Target list
    k = max(int(round(len(features_df) * k_frac)), 1)
    target_list = features_df.head(k).copy()
    target_list["in_target_list"] = True

    return features_df, target_list
```

---

## Implementation Plan — Stage 2: Adding MLflow

When the project moves to Stage 2, add MLflow with **minimal changes to existing code**:

### Step 1 — Instrument training (08_modeling.ipynb)

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")   # local SQLite, no server needed yet
mlflow.set_experiment("churn-prediction")

with mlflow.start_run(run_name="LR_full_features_C0.01"):

    # --- existing training code, unchanged ---
    final_model.fit(X_train, y_train)
    # -----------------------------------------

    # Log parameters
    mlflow.log_params({
        "algorithm":   "LogisticRegression",
        "C":           0.01,
        "feature_set": "full",
        "H_days":      90,
        "cooldown":    90,
    })

    # Log metrics
    mlflow.log_metrics({
        "pr_auc":      pr_auc,
        "roc_auc":     roc_auc,
        "lift_at_10":  lift_at_10_mean,
        "brier_score": brier,
    })

    # Log model + preprocessing artefacts together
    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model",
        registered_model_name="churn_model",
    )
    mlflow.sklearn.log_model(imputer, "imputer")
    mlflow.sklearn.log_model(scaler,  "scaler")
```

### Step 2 — Promote to Production in the registry

```bash
# View runs
mlflow ui

# In the MLflow UI: navigate to Models → churn_model → version 1
# Click "Stage: None" → "Transition to Production"
```

Or programmatically:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="churn_model",
    version=1,
    stage="Production"
)
```

### Step 3 — Update pipeline.py to load from registry

```python
# app/pipeline.py — Stage 2 version
# One-line change: replace joblib.load with mlflow.sklearn.load_model

import mlflow.sklearn

MODEL   = mlflow.sklearn.load_model("models:/churn_model/Production")
IMPUTER = mlflow.sklearn.load_model("models:/churn_imputer/Production")
SCALER  = mlflow.sklearn.load_model("models:/churn_scaler/Production")

# Everything else in pipeline.py stays identical
```

The Streamlit app does not change at all. The UI is decoupled from the model artefact source.

### Benefits gained in Stage 2

```
Before (Stage 1):         After (Stage 2):
model.pkl (opaque file)   churn_model v1 (tracked run, logged metrics)
                          churn_model v2 (retrain attempt, better PR-AUC)
                          churn_model v3 → promoted to Production
                          churn_model v2 → Archived

No history               Full audit trail: who promoted what, when, why
Manual rollback          One click in MLflow UI to roll back to v2
```

---

## Summary

| | Streamlit (local) | MLflow (open-source) |
|---|---|---|
| **Best for** | Stakeholder demos, marketing team UI | Experiment tracking, model versioning, production governance |
| **Setup time** | 1–2 hours | 4–8 hours |
| **New dependencies** | 1 package (`streamlit`) | 3–5 packages |
| **Schedules jobs** | ❌ No | ❌ No (needs cron/Airflow) |
| **Interactive dashboard** | ✅ Yes | ❌ No |
| **Model versioning** | ❌ No | ✅ Yes |
| **When to use** | **Now — demo and pilot** | Add in Stage 2 when operational |
| **Can they coexist?** | ✅ Yes | ✅ Yes |

**Bottom line:** Deploy with Streamlit now. It fits the project's current goals — a working demo that a marketing team can understand and use. Add MLflow in Stage 2 to gain model versioning, experiment tracking, and an audit trail when the system goes from demo to operational. The two tools are complementary, not competing.
