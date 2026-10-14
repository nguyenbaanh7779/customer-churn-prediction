# Customer Churn Prediction — Evaluation Methodology Report

**Course**: Master of Data Science — Predictive Analytics  
**Dataset**: Online Retail II (UCI Machine Learning Repository)  
**Model**: Logistic Regression · Full Feature Set · H = 90 days · C = 0.01  
**Evaluation Period**: 2011-07-04 → 2011-09-05 (10 weeks, 24,252 scoring rows)  
**Deployment Period**: 2011-09-12 → 2011-12-05 (13 weeks, 36,203 scoring rows)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Experimental Setup](#2-experimental-setup)
3. [Classification Performance](#3-classification-performance)
4. [Targeting Performance — Ranking Metrics](#4-targeting-performance--ranking-metrics)
5. [Risk Band Analysis](#5-risk-band-analysis)
6. [Customer Revenue at Risk](#6-customer-revenue-at-risk)
7. [Business Impact Analysis](#7-business-impact-analysis)
   - 7.1 [Retention Effectiveness Assumption](#71-retention-effectiveness-assumption)
   - 7.2 [Marketing Cost Structure](#72-marketing-cost-structure)
   - 7.3 [Value of the Model](#73-value-of-the-model)
   - 7.4 [Net Business Impact Formula](#74-net-business-impact-formula)
   - 7.5 [Scenario Analysis](#75-scenario-analysis)
8. [Deployment Scoring Output](#8-deployment-scoring-output)
9. [Error Analysis & Limitations](#9-error-analysis--limitations)
10. [Key Findings & Conclusions](#10-key-findings--conclusions)
11. [Glossary for Beginners](#11-glossary-for-beginners)

---

## 1. Project Overview

### What is Customer Churn?

**Customer churn** (hay còn gọi là *khách hàng rời bỏ*) xảy ra khi một khách hàng ngừng mua hàng từ một doanh nghiệp trong một khoảng thời gian nhất định. Trong dự án này, một khách hàng được gán nhãn **churn = 1** nếu họ không thực hiện bất kỳ giao dịch nào trong vòng **H = 90 ngày** tính từ ngày snapshot.

**Churn definition (formal):**

> A customer observed active at snapshot date *t* is labelled **churned** if they make no purchase in the horizon window *(t, t + H]*, where H = 90 days.

### Why Does It Matter?

Giữ chân khách hàng hiện tại rẻ hơn từ 5–25 lần so với thu hút khách hàng mới (theo nghiên cứu của Harvard Business Review). Bằng cách dự đoán *ai* có khả năng rời bỏ *trước khi* họ thực sự rời bỏ, doanh nghiệp có thể triển khai các chiến dịch retention có mục tiêu (targeted retention campaigns), tối ưu hóa ngân sách marketing.

### Three-Layer Evaluation Framework

Dự án này đánh giá mô hình theo **ba lớp phân tích**, từ kỹ thuật đến kinh doanh:

```
Layer 1: Predictive Performance  →  How accurately does the model assign probabilities?
                                     (ROC-AUC, PR-AUC, Calibration)

Layer 2: Targeting Performance   →  When we target the top-K% riskiest customers,
                                     how many true churners do we capture?
                                     (Precision@K, Lift@K, Gain@K)

Layer 3: Business Impact         →  If we act on the model's output, what is the
                                     estimated net financial return?
                                     (Revenue Preserved, Costs, Net Impact)
```

---

## 2. Experimental Setup

### 2.1 Dataset

| Property | Value |
|---|---|
| Source | Online Retail II — UCI ML Repository |
| Period | December 2010 – December 2011 |
| Transactions | ~500,000 invoice lines |
| Unique customers | ~4,300 active in scoring population |
| Target variable | Binary churn (H = 90 days) |
| Churn rate (test) | **40.1%** |

### 2.2 Temporal Split

Việc chia dữ liệu theo **thời gian** (temporal split) là cực kỳ quan trọng trong bài toán churn. Nếu dùng random split, thông tin từ tương lai sẽ "rò rỉ" vào tập huấn luyện, dẫn đến kết quả đánh giá không trung thực.

```
─────────────────────────────────────────────────────────────────────────▶ Time
│                TRAIN                │   VALIDATION   │      TEST       │
│  2010-06 ────────── 2011-03         │ 2011-04─06     │ 2011-07─09     │
│  12,019 training rows               │ 21,514 rows    │ 24,252 rows    │
│  (after episode-cooldown C=90d)     │                │                │
─────────────────────────────────────────────────────────────────────────
                                                        DEPLOYMENT ──────▶
                                                        2011-09 → 2011-12
                                                        36,203 rows
```

> **Key concept — Episode Cooldown:** Để tránh đưa các snapshot gần nhau của cùng một khách hàng vào tập train (chúng rất tương tự nhau và sẽ làm mô hình bị *overfit*), chỉ một snapshot mỗi C = 90 ngày được chọn cho mỗi khách hàng trong tập train.

### 2.3 Features Used

| Feature Group | Features | Role |
|---|---|---|
| **RFM** | `recency_days`, `frequency_so_far`, `monetary_so_far` | Core behavioral signals |
| **Extended** | `avg_order_value_so_far`, `tenure_days`, `mean_gap_so_far`, `recency_to_typical_gap`, `activity_last_90d` | Temporal purchase rhythm |
| **Full** | + `distinct_products_so_far`, `cancel_count_so_far`, `cancel_value_share_so_far` | Product breadth & cancellations |

**Final model uses the Full feature set (11 features total).**  
Skewed features (`monetary_so_far`, `tenure_days`, etc.) are log1p-transformed before scaling.

### 2.4 Final Model Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Algorithm | Logistic Regression | Outperforms RF and XGBoost on this small training set (~12,000 rows) |
| Feature set | Full (11 features) | Retained for completeness; strong regularization absorbs noise |
| Regularization C | **0.01** (strong L2) | Selected by validation PR-AUC; suppresses noise from extra features |
| Solver | `liblinear` | Efficient for small–medium datasets |
| Training rows | 12,019 | Re-derived over train + validation using episode cooldown |

### 2.5 Baselines

Các baseline không cần học từ dữ liệu — chúng là các heuristic đơn giản mà mô hình ML phải **vượt qua** để xứng đáng được triển khai:

| Baseline | Rule | Interpretation |
|---|---|---|
| **Random** | Uniform random score | Reference for Lift = 1 (no better than chance) |
| **Recency Rule** | Score = `recency_days` (↑ = riskier) | Target customers who haven't purchased recently |
| **RFM Rule** | Score = z(recency) − z(frequency) − z(monetary) | High recency + low frequency/monetary = high risk |

---

## 3. Classification Performance

### 3.1 What These Metrics Measure

Các chỉ số phân loại đánh giá khả năng mô hình phân biệt giữa khách hàng rời bỏ và không rời bỏ ở cấp độ toàn tập test.

| Metric | What It Measures | Range | Interpretation |
|---|---|---|---|
| **ROC-AUC** | Probability that model ranks a random churner above a random non-churner | 0–1 (0.5 = random) | Higher is better |
| **PR-AUC** | Area under Precision-Recall curve; sensitive to class imbalance | 0–1 (= churn rate if random) | Higher is better |
| **Precision@0.5** | Of customers predicted to churn (p ≥ 0.5), what % actually churn? | 0–1 | Higher = fewer false alarms |
| **Recall@0.5** | Of all true churners, what % does the model flag? | 0–1 | Higher = fewer missed churners |
| **F1@0.5** | Harmonic mean of Precision and Recall | 0–1 | Balanced trade-off |

### 3.2 Results on Test Set

| Metric | Final Model | Interpretation |
|---|---|---|
| Test Churn Rate | 0.401 | 40.1% of test customers churn — random reference for PR-AUC |
| **ROC-AUC** | **0.764** | Model ranks a random churner above a random non-churner 76.4% of the time |
| **PR-AUC** | **0.659** | 64.5% above the random reference of 0.401 |
| **Precision@0.5** | **0.612** | 61.2% of flagged customers actually churn |
| **Recall@0.5** | **0.649** | 64.9% of all churners are correctly flagged |
| **F1@0.5** | **0.630** | Balanced score between precision and recall |

> **For beginners:** Imagine the model assigns a churn probability to every customer. ROC-AUC asks: *"If I randomly pick one customer who churned and one who didn't, how often does the churner have a higher score?"* A score of 0.764 means this happens 76.4% of the time — significantly better than random chance (50%).

### 3.3 Calibration

**Calibration** measures whether predicted probabilities are *well-calibrated* — i.e., if the model says "70% churn probability", do roughly 70% of such customers actually churn?

- **Brier Score = 0.193** (lower is better; 0 = perfect, 0.25 = random for 50% base rate)
- The reliability curve tracks the diagonal closely, with slight over-prediction near p ≈ 0.65 (predicted 0.65, observed 0.57)
- **Implication:** Probabilities are usable as-is for ranking; a Platt or isotonic calibration step would improve accuracy if raw probabilities are used directly in revenue calculations

---

## 4. Targeting Performance — Ranking Metrics

### 4.1 The Operational Context

Trong thực tế, chiến dịch retention không thể tiếp cận **tất cả** khách hàng — ngân sách và nhân lực có giới hạn. Mô hình được dùng để **xếp hạng** khách hàng theo mức độ rủi ro và chỉ liên hệ với **top K%** rủi ro nhất.

> **Key insight:** Ranking metrics matter more than classification metrics for operational use. A model that ranks churners at the top (high Lift@K) is more valuable for a targeted campaign than a model with perfect binary accuracy.

### 4.2 Metric Definitions

```
Precision@K  =  (True Churners in Top K%) / (Customers in Top K%)

Lift@K       =  Precision@K / Overall Churn Rate
             =  How many times better than random targeting is the model?

Gain@K       =  (True Churners in Top K%) / (All True Churners)
             =  What fraction of all churners are captured in the top K%?
```

### 4.3 Results at K = 10% (Per Week, Mean ± Std)

| Candidate | Precision@10% | Lift@10% | Gain@10% |
|---|---|---|---|
| **Final Model** | **0.769 ± 0.047** | **1.924 ± 0.049** | **0.193 ± 0.005** |
| RFM Rule | 0.718 ± 0.047 | 1.796 ± 0.049 | 0.180 ± 0.005 |
| Recency Rule | 0.626 ± 0.056 | 1.565 ± 0.061 | 0.157 ± 0.006 |
| Random | 0.406 ± 0.045 | 1.017 ± 0.110 | 0.102 ± 0.011 |

> **How to read this table:**  
> — The Final Model achieves **Lift@10% = 1.924**, meaning targeting the top 10% riskiest customers yields **nearly 2× more churners** than random selection.  
> — The low standard deviation (0.049) shows this advantage is **consistent week-over-week**, not a lucky average.  
> — The Final Model leads **all 10 test weeks without exception** — statistical stability matters for operational confidence.

### 4.4 Why Weekly Metrics?

Targeting happens **each week** — every Monday, the model scores all active customers and the marketing team contacts the top K%. Therefore, metrics must be computed **per scoring date** (rank that week's customers, cut the top K%) and then summarised as mean ± standard deviation.

Pooling all 10 weeks into one ranked list would produce misleading metrics because:
1. The same customer appears in multiple weeks (not independent observations)
2. Base churn rates shift slightly between weeks (seasonal effects)

---

## 5. Risk Band Analysis

### 5.1 What Are Risk Bands?

The model's continuous churn probability (0–100%) is converted into a **scorecard-style integer risk score** and then bucketed into **10 risk bands** for operational simplicity.

**Score conversion formula:**
```
RiskScore = base_score + (PDO / ln 2) × log-odds(churn)

Parameters:
  base_score = 500   (score at churn odds 1:1, i.e. p = 0.5)
  PDO = 50           (Points to Double the churn Odds)
  Range: [0, 1000]
```

- **Band 1** = lowest risk (score ~24–300)
- **Band 10** = highest risk (score ~600–674)
- Every step up in band reflects a meaningful increase in predicted churn risk

### 5.2 Observed Churn Rate by Band (Test Period)

| Risk Band | Customers | Churners | Churn Rate | Revenue at Risk (£) |
|---|---|---|---|---|
| 1 (Lowest) | 3,582 | 239 | **5.1%** | £2,283,341 |
| 2 | 2,886 | 502 | 17.4% | £2,256,258 |
| 3 | 2,614 | 669 | 25.6% | £2,106,797 |
| 4 | 2,526 | 924 | 36.6% | £2,058,661 |
| 5 | 2,629 | 1,227 | 46.7% | £2,334,500 |
| 6 | 2,470 | 1,304 | 52.8% | £1,624,354 |
| 7 | 1,939 | 1,042 | 53.7% | £1,095,077 |
| 8 | 2,072 | 1,200 | 57.9% | £748,893 |
| 9 | 1,679 | 1,135 | 67.6% | £451,077 |
| 10 (Highest) | 1,855 | 1,472 | **79.4%** | £369,195 |

**Key observation:** Churn rate rises almost monotonically from 5.1% (Band 1) to 79.4% (Band 10) — a **15× spread** confirming the risk score is informative.

### 5.3 The Revenue-Risk Paradox

Một phát hiện quan trọng: **khách hàng có rủi ro churn cao nhất (Band 10) lại có doanh thu thấp nhất** (£369,195), trong khi **khách hàng ít rủi ro nhất (Band 1) lại có doanh thu cao nhất** (£2,283,341).

```
Revenue at Risk (£)
  Band 1  ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■  £2,283,341  (low risk, high value)
  Band 5  ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■  £2,334,500  (mid risk, high value)
  Band 10 ■■■■■■                            £369,195   (high risk, low value)
```

This has profound implications for business impact — explored in Section 7.

---

## 6. Customer Revenue at Risk

### 6.1 Definition

**Revenue at Risk** ước tính doanh thu tiềm năng bị mất nếu một khách hàng ngừng mua. Đây là **ước tính**, không phải doanh thu thực sự quan sát được.

**Proxy formula used:**

```
RevenueAtRisk_i = monetary_so_far_i
                = Total historical spend of customer i up to snapshot date
```

> **Why this proxy?** The dataset spans only ~2 years; there is insufficient post-snapshot history to compute a stable monthly revenue estimate. `monetary_so_far` (total cumulative spend) serves as the best available approximation of lifetime value exposure.

### 6.2 Distribution Statistics (Test Period Churners)

| Statistic | Value |
|---|---|
| Mean revenue at risk | £395.70 |
| Median revenue at risk | £303.58 |
| Standard deviation | £561.61 |
| Minimum | £2.90 |
| Maximum | £14,844.77 |

The distribution is **highly right-skewed**: a small number of high-value customers (e.g., wholesale accounts) contribute disproportionately large revenue at risk. The mean (£395.70) is 30% higher than the median (£303.58), confirming skewness.

> **Implication for targeting:** A model that captures *more* churners is not automatically better for revenue recovery — capturing the *highest-value* churners matters more. This is why the model's business impact can differ from its predictive accuracy ranking.

---

## 7. Business Impact Analysis

> **⚠ Important Disclaimer**  
> All business impact figures are **scenario estimates under stated assumptions**, not measured causal outcomes. The Online Retail II dataset contains no campaign treatment/control information. True incremental revenue from a retention campaign cannot be identified without an A/B test design.

### 7.1 Retention Effectiveness Assumption

**Retention effectiveness (e)** là xác suất giả định rằng nếu tiếp cận một khách hàng sắp rời bỏ, họ thực sự sẽ ở lại và mua hàng.

```
RevenuePreserved = RevenueAtRisk(targeted churners) × e
```

Three scenarios are modelled:

| Scenario | Value of e | Rationale |
|---|---|---|
| **Conservative** | e = 30% | Cautious estimate; most churners leave regardless of outreach |
| **Base** | e = 40% | Moderate effectiveness; industry-typical for email/voucher campaigns |
| **Optimistic** | e = 50% | High effectiveness; strong personalization or high-value incentives |

> **Why not 100%?** Even the best retention campaigns cannot retain every at-risk customer. Some have already decided to leave, switched to competitors, or no longer need the product. Assuming e < 1 is essential to avoid overestimating campaign value.

### 7.2 Marketing Cost Structure

Two cost components are modelled:

| Cost Component | Parameter | Default Value | Description |
|---|---|---|---|
| **Contact cost** | c_contact | £0.50 / customer | Email, SMS, or direct mail per customer contacted |
| **Incentive cost** | c_inc | £5.00 / redemption | Voucher, discount, or free shipping per redemption |
| **Cannibalization rate** | r | 20% | Share of non-churners (false positives) who redeem the incentive anyway |

**Total marketing cost formula:**
```
ContactCost  = N_K × c_contact
IncentiveCost = (TP_K × e + FP_K × r) × c_inc

Where:
  N_K   = total customers targeted (true churners + non-churners in top K%)
  TP_K  = true churners targeted (benefit from retention)
  FP_K  = non-churners targeted (false positives; some redeem anyway = cannibalization)
```

> **Cannibalization** là một khái niệm quan trọng: khi bạn gửi voucher cho khách hàng không có nguy cơ rời bỏ (false positives), một số người họ sẽ dùng voucher đó để mua hàng mà họ đã định mua — điều này làm tăng chi phí incentive mà không tạo ra doanh thu mới.

### 7.3 Value of the Model

**"Value of the Model"** so sánh Net Business Impact của mô hình ML với các baseline *tại cùng K%* và cùng giả định chi phí. Đây là câu hỏi thực sự quan trọng: *"Model có xứng đáng hơn một heuristic đơn giản không?"*

#### Results at K = 20%, e = 40% (Base), m = 35%, c_contact = £0.50, c_inc = £5.00, r = 20%

| Candidate | Recall@20% | Revenue at Risk Captured | Net Impact (£) |
|---|---|---|---|
| **Final Model** | **34.8%** | £1,270,399 | **£167,204** |
| Recency Rule | 28.9% | *higher* | £510,740 |
| RFM Rule | 32.6% | *moderate* | £304,352 |
| Random | 20.0% | £ variable | £437,441 |

#### The Counter-Intuitive Finding

> **The Final Model has higher Recall but lower Net Impact than the baselines at K ≤ 30%.**

**Why does this happen?**

The churn prediction model optimizes *churn probability ranking* — it identifies customers most likely to churn. However, customers with the **highest churn probability** (Band 10) tend to be lower-revenue customers (one-time buyers, low-frequency purchasers), as shown in Section 5.3.

The **Recency Rule** targets customers who haven't purchased recently — this includes many churners, but crucially, it also includes high-value loyal customers who are just slow to return. Among the churners it does capture, they tend to have **higher average order values**, generating more revenue at risk.

```
Final Model at K=10%:
  Captures: 1,867 churners (most churners caught!)
  Revenue at risk: £337,904   ← LOWER (captures many low-value churners)

Recency Rule at K=10%:
  Captures: 1,521 churners (fewer churners)
  Revenue at risk: £538,397   ← HIGHER (captures higher-value churners)
```

**This does not mean the model is bad.** It means:
1. For **churn prediction accuracy**, the model is clearly superior
2. For **revenue-maximizing targeting**, a revenue-weighted objective would be needed
3. The model's advantage in net business impact only materialises at **K ≥ 40%**

### 7.4 Net Business Impact Formula

```
GrossProfitPreserved = RevenuePreserved × m
                     = (RevenueAtRisk × e) × m

ContactCost          = N_K × c_contact
IncentiveCost        = (TP_K × e + FP_K × r) × c_inc

NetBusinessImpact    = GrossProfitPreserved − ContactCost − IncentiveCost
```

**Parameters:**

| Symbol | Parameter | Default | Range in Dashboard |
|---|---|---|---|
| K | Top-K % targeted | 20% | 10%–50% |
| e | Retention effectiveness | 40% | 30%–60% |
| m | Gross margin | 35% | 20%–60% |
| c_contact | Contact cost per customer | £0.50 | £0.10–£5.00 |
| c_inc | Incentive cost per redemption | £5.00 | £1.00–£20.00 |
| r | Cannibalization rate | 20% | 0%–50% |

#### Worked Example (Final Model, K=20%, e=40%)

```
Period: 10 test weeks
Customers targeted (N_K):       4,850
True churners caught (TP_K):    3,377
False positives (FP_K):         1,473
Revenue at risk:                £1,270,399

Revenue preserved   = £1,270,399 × 0.40 = £508,160
Gross profit        = £508,160   × 0.35 = £177,856
Contact cost        = 4,850      × £0.50 = £2,425
Incentive cost      = (3,377×0.40 + 1,473×0.20) × £5.00
                    = (1,351 + 295) × £5.00 = £8,228
Net impact          = £177,856 − £2,425 − £8,228 = £167,203
```

### 7.5 Scenario Analysis

Phân tích kịch bản cho phép đánh giá độ nhạy cảm (sensitivity) của kết quả theo các tham số khác nhau.

#### Net Business Impact (£) — Final Model vs Recency Rule Baseline

| Target K% | Final Model Net (£) | Recency Rule Net (£) | Model vs Baseline |
|---|---|---|---|
| 10% | £41,799 | £70,214 | **−£28,415** ← Model worse |
| 20% | £100,526 | £133,318 | **−£32,792** ← Model worse |
| 30% | £174,273 | £192,778 | **−£18,505** ← Model worse |
| 40% | £242,442 | £242,447 | **−£6** ≈ Tie |
| **50%** | **£312,460** | **£285,679** | **+£26,782** ← **Model wins** |

**Key insight:** The Final Model's net business advantage over the recency rule only materialises at **K ≥ 40%**. For typical retention campaigns targeting 10–20% of customers, the simple recency rule generates comparable or superior revenue returns — despite the model's superior predictive accuracy.

This is a central finding of this evaluation: **the best classifier is not automatically the best business decision** at a fixed operating point. It highlights the need for **revenue-weighted model objectives** in future work.

---

## 8. Deployment Scoring Output

### 8.1 What Is Batch Scoring?

**Batch scoring** là quá trình áp dụng mô hình đã huấn luyện để tính toán xác suất churn cho *tất cả* khách hàng đang hoạt động vào một thời điểm cụ thể (mỗi tuần một lần). Không có nhãn churn thực tế trong giai đoạn deployment vì chúng ta đang dự đoán *tương lai*.

### 8.2 Deployment Configuration

| Parameter | Value |
|---|---|
| Scoring cadence | Weekly (every Monday) |
| Deployment window | 2011-09-12 → 2011-12-05 (13 weeks) |
| Model version | `lr_full_c0.01_h90` |
| Customers scored | 2,416 → 3,181 per week (growing customer base) |
| Output per customer | `churn_probability`, `risk_score`, `risk_band`, `rank` |

### 8.3 Weekly Scoring Population

| Week | Customers Scored | Avg Churn Prob. | High Risk (Bands 8–10) |
|---|---|---|---|
| 2011-09-12 | 2,416 | 42.3% | 515 (21.3%) |
| 2011-09-19 | 2,465 | 42.4% | 528 (21.4%) |
| 2011-09-26 | 2,514 | 42.2% | 517 (20.6%) |
| 2011-10-03 | 2,579 | 42.2% | 533 (20.7%) |
| 2011-10-10 | 2,656 | 42.3% | 555 (20.9%) |
| 2011-10-17 | 2,718 | 42.4% | 582 (21.4%) |
| 2011-10-24 | 2,771 | 42.5% | 611 (22.1%) |
| 2011-10-31 | 2,831 | 42.5% | 609 (21.5%) |
| 2011-11-07 | 2,902 | 42.5% | 642 (22.1%) |
| 2011-11-14 | 2,980 | 42.4% | 653 (21.9%) |
| 2011-11-21 | 3,062 | 42.3% | 667 (21.8%) |
| 2011-11-28 | 3,128 | 42.4% | 703 (22.5%) |
| 2011-12-05 | 3,181 | 42.2% | 712 (22.4%) |

**Observations:**
- Customer base **grows steadily** from 2,416 to 3,181 (+31.7%) over the deployment period
- Average churn probability remains **stable** around 42.3–42.5% — no model drift detected
- High-risk proportion (~21%) also stable, confirming model consistency

### 8.4 How to Use Scoring Output

```
Weekly Deployment Workflow:
─────────────────────────────────────────────────────
Monday morning:
  1. Extract all active customers (no purchase in last 7–90 days)
  2. Compute features as of Sunday (snapshot_date = Monday)
  3. Apply model → churn_probability for each customer
  4. Assign risk_score (500 ± PDO scaling) and risk_band (1–10)
  5. Rank by churn_probability (rank = 1 is highest risk)
  6. Extract top-K% → send to CRM for campaign execution
─────────────────────────────────────────────────────
```

---

## 9. Error Analysis & Limitations

### 9.1 Confusion Matrix at K = 10% (Final Model, Test Period)

| | Actually Churned | Did Not Churn |
|---|---|---|
| **Targeted (Top 10%)** | TP: 1,867 (7.7%) | FP: 560 (2.3%) |
| **Not Targeted (Bottom 90%)** | FN: 7,881 (32.5%) | TN: 13,944 (57.5%) |

> **For beginners:**  
> — **True Positive (TP)**: We correctly identified a churner and contacted them ✓  
> — **False Positive (FP)**: We contacted a loyal customer who wasn't going to churn (wasted incentive)  
> — **False Negative (FN)**: A churner we missed — they left without receiving any retention offer  
> — **True Negative (TN)**: A loyal customer we correctly left alone ✓

At K = 10%, the model only catches 19.3% of all churners (Gain@10%) — the other 80.7% are missed. This is expected: a 10% net cannot catch most of a 40% base rate.

### 9.2 Segment-Level Error Rates

| Group | FP Rate (among targeted) | FN Rate (among non-targeted) |
|---|---|---|
| **One-time customers** | 20.6% | **55.5%** |
| **Repeat customers** | 25.3% | **34.1%** |
| UK customers | 22.3% | 36.3% |
| Non-UK customers | **29.4%** | 32.5% |
| Retail customers | 22.7% | 37.2% |
| Wholesale-like customers | 0% | **7.8%** |

**Analysis of segment differences:**

1. **One-time customers are missed at a much higher rate (55.5% FN).** These customers have only one purchase in their history — the model has limited behavioral signal to work with. This is a fundamental limitation of feature-based churn models for new or infrequent customers.

2. **Non-UK customers have a higher false positive rate (29.4%).** The model is trained predominantly on UK behavior patterns. Non-UK purchasing rhythms differ, reducing generalization quality.

3. **Wholesale-like customers are easiest to model (7.8% FN, 0% FP).** Their high-frequency, high-volume purchase patterns produce clear churn signals — when a wholesale customer suddenly stops, it's unambiguous.

### 9.3 Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| No treatment/control data | Cannot measure *causal* retention effect | Frame all results as scenario estimates |
| Revenue proxy (monetary_so_far) | Overestimates revenue at risk for churners who return as low-frequency buyers | Use avg_order_value × expected_orders as alternative |
| No seasonal adjustment | Test period (Jul–Sep) and deployment period (Sep–Dec) have different churn dynamics | Monitor churn rate drift weekly |
| Static weekly features | Customer behavior evolves; model not retrained after deployment | Schedule quarterly retraining |
| One-time customer weakness | 55.5% FN rate for this segment | Build a separate model for new customers |

---

## 10. Key Findings & Conclusions

### 10.1 Summary of Results

```
┌─────────────────────────────────────────────────────────────────────┐
│              FINAL MODEL PERFORMANCE SUMMARY                         │
│                                                                       │
│  PREDICTIVE (Test Set, 24,252 rows, 10 weeks):                       │
│    ROC-AUC:       0.764   ████████████████████░░░░░  (>0.7 = good)  │
│    PR-AUC:        0.659   ████████████████░░░░░░░░░  (vs 0.401 random)│
│    F1@0.5:        0.630   ████████████████░░░░░░░░░                  │
│                                                                       │
│  RANKING (Per Week, Top 10%):                                        │
│    Lift@10%:      1.924   ██ (2× better than random = excellent)     │
│    Gain@10%:      19.3%   Captures 1 in 5 churners in top 10%       │
│    Stability:     ±0.049  Low week-to-week variation                 │
│                                                                       │
│  BUSINESS (K=20%, e=40% Base):                                       │
│    Net Impact:  £167,204  (10-week test period estimate)             │
│    Vs Recency:  −£32,792  Model trails at K≤30%                     │
│    Vs Recency:  +£26,782  Model wins at K=50%                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.2 Does the Model Meet the Success Criteria?

| Success Criterion | Result | Met? |
|---|---|---|
| ROC-AUC > recency rule baseline (0.638) | 0.764 > 0.638 | ✅ Yes |
| PR-AUC > recency rule baseline (0.531) | 0.659 > 0.531 | ✅ Yes |
| Lift@10% > recency rule (1.565) every week | 1.924, all 10 weeks | ✅ Yes |
| Net business impact > recency rule at K=10% | £41,799 < £70,214 | ❌ Not at K≤30% |
| Net business impact > recency rule at K=50% | £312,460 > £285,679 | ✅ Yes at K≥40% |

### 10.3 Recommendations

1. **For predictive quality:** The Final Model (LR, Full features, C=0.01) is the clear choice — it outperforms all baselines on every statistical metric, consistently across all 10 test weeks.

2. **For business impact at narrow K (10–30%):** Consider a **revenue-weighted scoring objective** that explicitly targets high-value churners, or **apply the model's probability as a ranking signal within high-revenue customer segments** rather than across the full population.

3. **For deployment:** The stable average churn probability (~42%) and consistent high-risk proportion (~21%) across the 13 deployment weeks suggest no model drift. However, **quarterly retraining** is recommended as customer behavior evolves.

4. **For one-time customers:** A **segment-specific model** with richer cold-start features would significantly improve the 55.5% false-negative rate.

---

## 11. Glossary for Beginners

| Term | Simple Explanation |
|---|---|
| **Churn** | A customer who stops buying from the business within a defined period (H = 90 days here) |
| **Snapshot date** | The weekly date on which the model scores all active customers |
| **Scoring Population** | All customers who are "active" at a snapshot date (have purchased at least once in their history) |
| **ROC-AUC** | How well the model separates churners from non-churners overall; 0.5 = random, 1.0 = perfect |
| **PR-AUC** | Same as ROC-AUC but more sensitive to imbalanced classes (when churners are a minority) |
| **Precision@K** | Of the top K% riskiest customers we target, what fraction are actual churners? |
| **Recall@K** | Of ALL actual churners, what fraction did we manage to include in our top K% target list? |
| **Lift@K** | How much better is Precision@K than random targeting? Lift = 2 means 2× better than random |
| **Risk Band** | A group of customers (1–10) ordered by predicted churn risk; Band 10 = highest risk |
| **Revenue at Risk** | Estimated revenue the business would lose if a churner is not retained |
| **Retention Effectiveness (e)** | Assumed probability that a contacted at-risk customer actually stays |
| **Cannibalization** | When a voucher/incentive is redeemed by customers who would have purchased anyway (wasted cost) |
| **Gross Margin (m)** | The percentage of revenue that becomes profit after subtracting cost of goods; 35% here |
| **Net Business Impact** | Gross profit from retention minus all campaign costs; the "bottom line" value of the campaign |
| **Scenario Estimate** | A calculated projection under stated assumptions — not an observed outcome from a real campaign |
| **Baseline** | A simple, easy-to-implement strategy that the ML model must outperform to justify its complexity |
| **False Positive (FP)** | A customer we thought would churn but didn't — wasted campaign spend |
| **False Negative (FN)** | A customer who churned but we didn't target — missed retention opportunity |
| **Episode Cooldown** | A rule to limit how often the same customer appears in training data (once per 90 days) |
| **Regularization (C)** | A parameter that prevents the model from memorizing noise in training data; smaller C = stronger regularization |
| **SHAP Value** | A method to explain how much each feature contributed to a single prediction |
| **Batch Scoring** | Applying the trained model to all customers at once on a regular schedule (weekly) |

---

## Appendix: Dashboard Navigation Guide

The interactive dashboard (`streamlit run app.py`) presents the above analysis across four tabs:

| Tab | Content | Key Actions |
|---|---|---|
| **🎯 Classification Performance** | ROC/PR curves, calibration, weekly lift vs all baselines | Verify model superiority over all 4 candidates |
| **📊 Risk Band Analysis** | Band-by-band churn rates, revenue distribution, segment mix | Identify which bands to prioritize for high-value retention |
| **💰 Business Impact Scenarios** | Value of the Model comparison, waterfall, scenario matrix, weekly trends | Adjust K%, effectiveness, costs to explore campaign ROI |
| **🗓️ Scoring Output** | 13-week deployment snapshots, batch scoring summary, per-customer table | Explore live scoring results; filter by week, segment, band |

---

*Report generated from: `notebooks/09_evaluation.ipynb` · Dashboard: `app.py` · Data: `model/churn_scores_h90.parquet`, `model/modeling_dataset_h90.parquet`*
