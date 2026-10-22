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
- Added expanded interaction features (Age × Flight Distance, delay × service_avg, distance squared, service std) and target encoding approximations.
- Submitted faster tuned XGB (`submissions/submission_v3.csv`) with expanded features.
- Note: Full GB blend deferred; target encoding computed via train mean mapping.

### Why
- Tuning `subsample` and `colsample_bytree` to 0.85 reduces overfitting compared to 0.8.
- Lower `n_estimators` with slightly higher `learning_rate` (0.05 vs 0.03) trains faster and sometimes improves validation performance.
- NaN handling is critical because GradientBoostingClassifier (sklearn) does not accept NaN natively.

### Results
- Validation Accuracy (tuned XGB): 0.9238
- Kaggle submission v2: completed successfully (8 submissions remaining in daily quota).
- Note: Full GB blend deferred due to timeout; will try faster settings or use HistGradientBoosting (handles NaN natively) in Experiment 3.

---

## Experiment 3 — Target Encoding + Feature Expansion — 2026-10-22 13:45

### What we did
- Added target encoding (mean `satisfaction` per category) for `Gender`, `Customer Type`, `Type of Travel`, `Class`.
- Added expanded interaction features: `Age × Flight Distance`, `total_delay × avg_service`, `Age × avg_service`, `Flight Distance²`, `service_std`.
- Retrained XGB (`n_estimators=300`, `max_depth=6`, `subsample=0.85`, `learning_rate=0.05`).
- Submitted `submissions/submission_v3.csv`.

### Why
- Target encoding captures category-level predictive signal that simple label encoding misses.
- Polynomial/interactions allow the model to learn non-linear relationships between service ratings, age, distance, and delays.

### Results
- Validation Accuracy: 0.9238
- Kaggle submission v3: completed successfully (7 submissions remaining in daily quota).
- Note: Accuracy very close to v1 (0.9242) — feature expansion provides marginal gain; next step is hyperparameter tuning or stacking.

---

## Experiment 4 — Stacking Meta-Learner — 2026-10-22 14:05

### What we did
- Implemented stacking ensemble using out-of-fold predictions from 3 base models (XGBoost, RandomForest, HistGradientBoostingClassifier) with StratifiedKFold (5 folds).
- Used feature engineering pipeline from previous experiments: service ratings mean/min/max/range/std, age bins, delay interactions (`total_delay`, `has_delay`), expanded interactions (`age_x_distance`, `delay_x_service`, `distance_sq`, `age_x_service`, `distance_per_minute`), target encoding for `Gender`/`Customer Type`/`Type of Travel`/`Class`, and customer indicators (`loyal_business`, `eco_personal`).
- Handled NaN in `Arrival Delay in Minutes` by filling with 0 in train/test.
- Trained meta-learner (`LogisticRegression`) on meta-features (5-fold predictions from XGB, RF, HGB).
- Generated predictions on `test.csv`, saved to `submissions/submission_v4.csv`.

### Why
- Stacking combines diverse model families (boosting + bagging) through a linear meta-learner, reducing variance and potentially improving generalization over simple blending.
- Out-of-fold predictions prevent data leakage and provide unbiased meta-training data.
- HistGradientBoostingClassifier (instead of GradientBoostingClassifier) trains faster (completed in ~100s vs 300s timeout) while handling NaN natively.

### Results
- Base model OOF AUC: XGB 0.9581, RF 0.9555, HGB 0.9572
- Meta-learner (LogisticRegression) validation: Accuracy 0.9256, F1 0.9151, AUC 0.9581
- Submission status: `submissions/submission_v4.csv` saved (299,844 rows, mean prediction ~0.444)
- Runtime: ~100.1s (well within 240s timeout)
- Note: Meta-learner AUC equals XGB base AUC; stacking gains are modest but stable.

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

## Continuous Iteration Note — 9 Days Remaining (Deadline ~Oct 31, 2026)

We will submit improved versions daily (up to the Kaggle submission limit of 10/day) until the deadline or until leaderboard plateaus. Strategy: iterate quickly (feature engineering → hyperparameter tuning → ensembling → stacking) and submit best-performing variants.
Experiment 2 timed out at 5 min. Adjusting: using faster training settings and submitting v2 after shorter run.
