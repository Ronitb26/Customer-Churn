# E-Commerce Customer Churn Prediction

Predicting which customers on an e-commerce platform are likely to churn, using behavioral, transactional, and engagement data. The final model is an **XGBoost classifier** tuned for **recall**, since catching potential churners is more valuable than being conservative when the cost of a false positive (a retention offer to a loyal customer) is much lower than the cost of a false negative (losing a customer silently).

## Results

| Metric | Score (test set) |
|---|---|
| Recall | 0.893 |
| Precision | 0.909 |
| F1 | 0.901 |
| Accuracy | 0.967 |

Best cross-validated recall during hyperparameter tuning: **0.875** (5-fold CV).

## Dataset

- ~5,073 rows after de-duplication, 19 features, binary target (`Churn`).
- Split 80/20 (train/test), stratified on churn rate (~16.6% in both splits) to keep the class imbalance consistent across the split.
- Features span device/app usage, order behavior (order count, coupons used, days since last order), satisfaction score, cashback amount, complaint flag, and demographic fields (city tier, marital status, gender).

## Approach

**1. Data cleaning**
- Dropped `CustomerID` (unique identifier, no predictive signal).
- Standardized inconsistent category labels found during EDA — e.g. `PreferredLoginDevice` had `Mobile Phone` / `Phone` / `Mobile` all referring to the same thing, and payment mode had `CC` vs `Credit Card`, `COD` vs `Cash on Delivery`.
- Removed exact duplicate rows.

**2. Train/test split before any statistics-dependent step**
The split happens *before* outlier capping and imputation, and all thresholds (quantiles, medians, modes) are computed on the training set only and applied to both — to avoid leaking test-set information into preprocessing.

**3. Outlier handling**
Capped extreme values (1st/99th or 95th percentile depending on the feature) in `CashbackAmount`, `WarehouseToHome`, `CouponUsed`, `OrderCount`, and `DaySinceLastOrder`, based on distributions observed in EDA.

**4. Missing value imputation**
- Median imputation for skewed continuous fields (`Tenure`, `OrderAmountHikeFromlastYear`).
- Mode imputation for low-cardinality count fields (`CouponUsed`, `OrderCount`).
- `KNNImputer` (on standardized features) for continuous, behaviorally-correlated fields (`HourSpendOnApp`, `WarehouseToHome`, `DaySinceLastOrder`), where a neighbor-based estimate is more appropriate than a single central value.

**5. Model comparison**
Compared 9 classifiers (Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes, Gradient Boosting, XGBoost, LightGBM) across both Ordinal and OneHot encoding, using 10-fold CV recall as the selection metric.

| Model | Recall (Ordinal) | Recall (OneHot) |
|---|---|---|
| **XGBoost** | **0.851** | **0.849** |
| LightGBM | 0.827 | 0.810 |
| Decision Tree | 0.819 | 0.808 |
| Random Forest | 0.792 | 0.772 |
| Gradient Boosting | 0.618 | 0.622 |
| Naive Bayes | 0.536 | 0.740 |
| SVM | 0.490 | 0.531 |
| KNN | 0.487 | 0.508 |
| Logistic Regression | 0.424 | 0.517 |

Tree ensembles led by a wide margin, and encoding choice barely moved the top models — expected, since tree splits don't depend on encoded category order the way linear/distance-based models do. XGBoost was carried forward as the final model with Ordinal Encoding, since it keeps the feature space smaller without hurting tree-based performance.

**6. Hyperparameter tuning**
`GridSearchCV` (5-fold, scoring = recall) over `n_estimators`, `max_depth`, `min_child_weight`, `reg_alpha`, and `scale_pos_weight` (set relative to the class imbalance ratio to bias the model toward catching churners).

Best parameters:
```
max_depth: 8
min_child_weight: 3
n_estimators: 200
reg_alpha: 0.3
scale_pos_weight: 7
```

**7. Feature importance**

Top drivers of churn, by XGBoost feature importance:

1. **Tenure** — newer customers churn the most.
2. **Complain** — a logged complaint is a strong churn signal.
3. **PreferedOrderCat**, **SatisfactionScore**, **NumberOfAddress** — mid-tier importance.
4. **DaySinceLastOrder**, **CashbackAmount** — recency and incentive-related signals.

### Business takeaways
- Low tenure + a recent complaint are the strongest churn signals — worth proactive outreach in a customer's first few months and immediately after any complaint is logged.
- Customers going quiet (`DaySinceLastOrder`) or receiving less `CashbackAmount` show elevated risk — cashback/loyalty incentives may be worth targeting at-risk segments.
- `WarehouseToHome` distance correlates with churn, consistent with delivery experience affecting retention.

## Repo contents

| File | Description |
|---|---|
| `churn_prediction.ipynb` | Full notebook: EDA, cleaning, preprocessing, model comparison, tuning, evaluation |
| `pipeline.pkl` | Final fitted `sklearn` Pipeline (preprocessing + tuned XGBoost model) |
| `test_data.csv` | Held-out test set, kept for reproducibility |

## Tech stack

`pandas`, `numpy`, `scikit-learn`, `XGBoost`, `LightGBM`, `seaborn`/`matplotlib`

## Usage

```python
import pickle
import pandas as pd

with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

df_new = pd.read_csv('test_data.csv').drop(columns=['Churn'])
predictions = pipeline.predict(df_new)
```
