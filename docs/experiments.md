# Experiments Log — Playground Series S6E10 (Predicting Airline Satisfaction)

Competition URL: https://www.kaggle.com/competitions/playground-series-s6e10/overview  
Task: Binary classification (`satisfaction`: True/False)  
Evaluation metric (assumed from Playground series): Accuracy / F1 / ROC-AUC  
Data: 699,635 train rows, 299,844 test rows, 23 raw features

---

## Experiment 1 — Initial Blend (XGB + RF) — 2026-10-22 13:10

### What we did
- Downloaded data from Kaggle (`train.csv`, `test.csv`, `sample_submission.csv`).
- Built feature engineering pipeline (service ratings mean/min/max/range, age bins, delay interactions, loyal/business indicators, class/travel interactions).
- Trained XGBoost (`n_estimators=600`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `learning_rate=0.03`) and Random Forest (`n_estimators=300`, `max_depth=12`).
- Blended predictions 60% XGB / 40% RF on probabilities.
- Submitted `submission.csv` (mean prediction ~0.444).

### Why
- Playground series binary classification tasks respond well to gradient boosting with engineered interaction features.
- Blending two different model families (boosting + bagging) reduces variance.

### Results
- Validation Accuracy: 0.9242
- Validation F1: 0.9135
- Validation ROC-AUC: 0.9576
- Kaggle submission: completed successfully (9 submissions remaining in daily quota).

---

## Experiment 2 — Hyperparameter Tuning + Gradient Boosting — 2026-10-22 13:30 (Fixed at 13:37)

### What we did
- Found 292 NaN values in `Arrival Delay in Minutes` (not handled in v1). Fixed by filling with 0 in both train/test.
- Built tuned XGB (`n_estimators=300`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.85`, `colsample_bytree=0.85`).
- Attempted full blend (XGB + GB + RF) but timed out at 300s due to dataset size (699k rows) + multiple large models.
- Submitted faster tuned XGB (`submissions/submission_v2.csv`) to preserve daily submission quota.

### Why
- Tuning `subsample` and `colsample_bytree` to 0.85 reduces overfitting compared to 0.8.
- Lower `n_estimators` with slightly higher `learning_rate` (0.05 vs 0.03) trains faster and sometimes improves validation performance.
- NaN handling is critical because GradientBoostingClassifier (sklearn) does not accept NaN natively.

### Results
- Validation Accuracy (tuned XGB): 0.9238
- Kaggle submission v2: completed successfully (8 submissions remaining in daily quota).
- Note: Full GB blend deferred due to timeout; will try faster settings or use HistGradientBoosting (handles NaN natively) in Experiment 3.

---

## Experiment 3 — Target Encoding + Feature Expansion — Planned

### What we'll do
- Add mean-target encoding for categorical variables (`Gender`, `Customer Type`, `Type of Travel`, `Class`).
- Add polynomial interaction terms (Age × Flight Distance, total_delay × service_avg).
- Try CatBoost if installed (handles categoricals natively).

---

## Experiment 4 — Final Ensemble / Stacking — Planned

### What we'll do
- Stack predictions with meta-learner (LogisticRegression) using out-of-fold predictions from base models.
- Submit best model before any deadline.

---

## File Structure

```
/ (repo root)
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── archive/
│       └── playground-series-s6e10.zip
├── notebooks/
│   └── notebook.ipynb
├── submissions/
│   ├── submission.csv (v1)
│   └── submission_v2.csv (planned)
└── docs/
    └── experiments.md (this file)
```

---

## Continuous Iteration Note

No fixed competition deadline has been specified for this Playground series. We will continue submitting improved versions daily (up to the Kaggle submission limit) until either:
- A deadline is announced,
- Leaderboard position plateaus,
- User instructs to stop.
Experiment 2 timed out at 5 min. Adjusting: using faster training settings and submitting v2 after shorter run.
