# Telco Customer Churn Analysis

End-to-end data analysis project identifying why customers churn and predicting 
which customers are at risk — using EDA, SQL, and machine learning.

---

## Business Problem

A telecom company is losing 26.6% of its customers annually. This project 
analyzes 7,032 customer records to identify the root causes of churn and build 
a predictive model to flag at-risk customers before they leave.

---

## Key Business Findings

### 1. Contract Type is the Biggest Lever
| Contract | Churn Rate |
|----------|-----------|
| Month-to-month | 42.7% |
| One year | 11.3% |
| Two year | 2.8% |

**Recommendation:** Aggressively incentivize annual contracts. Offer first-month 
discounts or loyalty rewards to convert month-to-month customers.

### 2. The First 12 Months Are Critical
- New customers (0-12 months) churn at **47%**
- Customers past 5 years churn at only **7%**
- The rolling churn rate doesn't drop below 30% until month 19

**Recommendation:** Invest in onboarding. A dedicated 90-day success program 
for new customers would have outsized impact on retention.

### 3. The Company Is Losing Its Highest-Paying Customers
- Fiber optic customers pay the most (~$80-100/month) AND churn at 42%
- Month-to-month + Fiber optic + Electronic check = **$68,281 lost every month**
- That's approximately **$819,000 in annual recurring revenue** walking out the door

**Recommendation:** Immediate investigation into fiber optic service quality 
and pricing. Exit surveys for this segment are critical.

### 4. Electronic Check Is a Strong Churn Signal
- Electronic check customers churn at **45%** vs 15-17% for all other methods
- Customers on auto-pay (credit card, bank transfer) are significantly more loyal

**Recommendation:** Offer a small discount for switching to automatic payment 
methods to increase payment commitment and reduce churn signal.

---

## Machine Learning Results

Three models trained on SMOTE-balanced data to handle class imbalance (26.6% churn):

| Model | AUC | Churn Recall | Churn F1 |
|-------|-----|-------------|----------|
| Logistic Regression | **0.8105** | 0.63 | 0.59 |
| Random Forest | 0.8057 | 0.60 | 0.58 |
| XGBoost | 0.8002 | 0.62 | 0.58 |

**Best model: Logistic Regression (AUC 0.81)**

At 0.81 AUC, the model correctly identifies at-risk customers 81% of the time — 
enabling proactive retention outreach before churn occurs.

### Top Churn Drivers (Feature Importance)
1. **Monthly Charges** (0.147) — Higher bills = higher churn risk
2. **Total Charges** (0.143) — Correlated with tenure and billing history
3. **Tenure** (0.139) — Newer customers far more likely to churn
4. **Electronic Check Payment** (0.102) — Strong behavioral churn signal
5. **Fiber Optic Internet** (0.059) — Service dissatisfaction indicator

---

## Project Structure
```
telco-churn-analysis/
├── data/
│   ├── raw/                  # Original Kaggle dataset
│   └── processed/            # Cleaned data
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory analysis + visualizations
│   ├── 02_SQL_analysis.ipynb # SQL cohort + revenue analysis
│   └── 03_churn_prediction.ipynb # ML models
├── visuals/                  # All saved charts
└── README.md
```

## Tech Stack
- **Python** — pandas, numpy, scikit-learn, xgboost, imbalanced-learn
- **SQL** — SQLite with window functions and cohort analysis
- **Visualization** — matplotlib, seaborn

## Data Source
[IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,032 customers, 21 features
