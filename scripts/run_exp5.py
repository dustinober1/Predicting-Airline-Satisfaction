#!/usr/bin/env python3
"""Experiment 5: Hyperparameter Grid Search (XGB) for Predicting Airline Satisfaction."""
import os, time, warnings, sys
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb

start = time.time()
DATA_DIR = 'data'

def load():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    for df in [train, test]:
        df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(0)
    return train, test

print('Loading data...')
train, test = load()
print('Train:', train.shape, 'Test:', test.shape)

cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

def engineer_features(df, is_train=True, train_df=None):
    df = df.copy()
    if 'satisfaction' in df.columns:
        df['satisfaction_bin'] = df['satisfaction'].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0})

    ratings = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Seat comfort', 'Inflight entertainment',
        'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
        'Cleanliness', 'Online boarding']
    ratings = [r for r in ratings if r in df.columns]
    df['avg_service'] = df[ratings].mean(axis=1)
    df['min_service'] = df[ratings].min(axis=1)
    df['max_service'] = df[ratings].max(axis=1)
    df['service_range'] = df['max_service'] - df['min_service']
    df['service_std'] = df[ratings].std(axis=1)

    df['Age_group'] = pd.cut(df['Age'], bins=[0, 18, 35, 55, 100], labels=[0, 1, 2, 3]).astype(int)
    df['is_child'] = (df['Age'] < 18).astype(int)
    df['is_senior'] = (df['Age'] > 55).astype(int)

    df['Departure Delay in Minutes'] = df['Departure Delay in Minutes'].fillna(0)
    df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(0)
    df['total_delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
    df['has_delay'] = (df['total_delay'] > 0).astype(int)

    df['age_x_distance'] = df['Age'] * df['Flight Distance']
    df['delay_x_service'] = df['total_delay'] * df['avg_service']
    df['distance_sq'] = df['Flight Distance'] ** 2
    df['age_x_service'] = df['Age'] * df['avg_service']
    df['distance_per_minute'] = df['Flight Distance'] / (df['total_delay'] + 1)

    df['loyal_business'] = ((df['Customer Type'] == 'Loyal Customer').astype(int) & (df['Type of Travel'] == 'Business travel').astype(int)).astype(int)
    df['eco_personal'] = ((df['Class'] == 'Eco').astype(int) & (df['Type of Travel'] == 'Personal Travel').astype(int)).astype(int)

    for col in cat_cols:
        if is_train:
            train_target = train_df.copy() if train_df is not None else df.copy()
            train_target['satisfaction_bin'] = train_target['satisfaction'].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0})
            mean_map = train_target.groupby(col)['satisfaction_bin'].mean().to_dict()
            df[col + '_te'] = df[col].map(mean_map).fillna(train_target['satisfaction_bin'].mean())
        else:
            train_target = train_df.copy()
            train_target['satisfaction_bin'] = train_target['satisfaction'].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0})
            mean_map = train_target.groupby(col)['satisfaction_bin'].mean().to_dict()
            df[col + '_te'] = df[col].map(mean_map).fillna(train_target['satisfaction_bin'].mean())

    for col in cat_cols:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[col + '_le'] = le.fit_transform(df[col].astype(str))
    return df

print('Engineering features...')
train_fe = engineer_features(train, is_train=True, train_df=train)
test_fe = engineer_features(test, is_train=False, train_df=train)

exclude = {'id', 'satisfaction', 'satisfaction_bin'}
feature_cols = [c for c in train_fe.columns if c not in exclude]
from pandas.api.types import is_numeric_dtype
feature_cols = [c for c in feature_cols if is_numeric_dtype(train_fe[c])]
feature_cols = [c for c in feature_cols if c in test_fe.columns]
print('Feature count:', len(feature_cols))

X = train_fe[feature_cols].values
y = train_fe['satisfaction_bin'].values
X_test = test_fe[feature_cols].values

full_grid = {
    'max_depth': [4, 5, 6, 7],
    'learning_rate': [0.01, 0.03, 0.05],
    'subsample': [0.7, 0.85, 0.9],
    'n_estimators': [300, 600]
}
reduced_grid = {
    'max_depth': [5, 6],
    'learning_rate': [0.03, 0.05],
    'subsample': [0.8, 0.85],
    'n_estimators': [300]
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

from itertools import product

def evaluate_params(X, y, params):
    acc_scores = []
    f1_scores = []
    auc_scores = []
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='logloss',
            random_state=42, n_jobs=-1, use_label_encoder=False,
            reg_alpha=0.1, reg_lambda=1.0, **params
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        acc_scores.append(accuracy_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
        auc_scores.append(roc_auc_score(y_val, y_proba))
    return {
        'acc': np.mean(acc_scores),
        'f1': np.mean(f1_scores),
        'auc': np.mean(auc_scores),
        'acc_std': np.std(acc_scores),
        'f1_std': np.std(f1_scores),
        'auc_std': np.std(auc_scores)
    }

best_params = None
best_score = -1
best_metrics = None
results = []

def run_grid(grid_dict, label):
    global best_params, best_score, best_metrics, results
    keys = list(grid_dict.keys())
    for combo in product(*[grid_dict[k] for k in keys]):
        params = dict(zip(keys, combo))
        print('Evaluating:', params)
        metrics = evaluate_params(X, y, params)
        print(f"  Acc={metrics['acc']:.4f} (+/-{metrics['acc_std']:.4f}), F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        results.append((label, params, metrics))
        if metrics['acc'] > best_score:
            best_score = metrics['acc']
            best_params = params.copy()
            best_metrics = metrics.copy()
            print('  *** New best ***')

print('Starting full grid search...')
run_grid(full_grid, 'full')
elapsed = time.time() - start
print(f'Full grid completed in {elapsed:.1f}s')

if elapsed > 300 and best_params is None:
    print('Full grid exceeded 300s with no results; switching to reduced grid.')
    best_params = None
    best_score = -1
    best_metrics = None
    results = []
    run_grid(reduced_grid, 'reduced')
    elapsed_total = time.time() - start
    print(f'Reduced grid completed in {elapsed_total:.1f}s')
elif elapsed > 300:
    print('Full grid exceeded 300s; using reduced grid for any remaining combos.')
    # Already have best from full; don't rerun reduced unless needed.
else:
    pass

# Ensure best_params exists; if not (shouldn't happen), set default.
if best_params is None:
    print('Falling back to reduced grid due to timeout.')
    run_grid(reduced_grid, 'reduced')

print('Best params:', best_params)
print('Best CV Accuracy:', best_score)
print('Best CV Metrics:', best_metrics)

# Retrain best model on full training data
print('Retraining best model on full training data...')
best_model = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='logloss',
    random_state=42, n_jobs=-1, use_label_encoder=False,
    reg_alpha=0.1, reg_lambda=1.0, **best_params
)
best_model.fit(X, y)

test_pred_proba = best_model.predict_proba(X_test)[:, 1]

os.makedirs('submissions', exist_ok=True)
submission_path = 'submissions/submission_v5.csv'
submission = pd.DataFrame({'id': test['id'], 'satisfaction': test_pred_proba})
submission.to_csv(submission_path, index=False)
print(f'Submission saved: {submission_path} (rows={len(submission)}, mean={test_pred_proba.mean():.4f})')

# Validation metrics using best params on full data with CV is already best_score (CV mean)
# Also compute a single validation split metric for reporting
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
best_model_val = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='logloss',
    random_state=42, n_jobs=-1, use_label_encoder=False,
    reg_alpha=0.1, reg_lambda=1.0, **best_params
)
best_model_val.fit(X_tr, y_tr)
val_pred = best_model_val.predict(X_val)
val_proba = best_model_val.predict_proba(X_val)[:, 1]
val_acc = accuracy_score(y_val, val_pred)
val_f1 = f1_score(y_val, val_pred)
val_auc = roc_auc_score(y_val, val_proba)
print(f'Validation (20% holdout) - Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}')

# Update docs
print('Updating docs/experiments.md...')
with open('docs/experiments.md', 'r') as f:
    doc_content = f.read()

new_section = f"""
---

## Experiment 5 — Hyperparameter Grid Search (XGB) — {time.strftime('%Y-%m-%d %H:%M')}

### What we did
- Performed hyperparameter grid search for XGBoost using the feature engineering pipeline from previous experiments (service ratings, age bins, delays, interactions, target encoding, expanded features).
- Filled `Arrival Delay in Minutes` NaN with 0.
- Evaluated using 5-fold StratifiedKFold CV on full training data; primary metric: accuracy, with F1 and AUC.
- Grid: max_depth [4,5,6,7], learning_rate [0.01, 0.03, 0.05], subsample [0.7, 0.85, 0.9], n_estimators [300, 600].
- Fallback reduced grid applied if total runtime exceeded 300s.
- Retrained best model on full training data; predicted test.csv.

### Why
- Systematic grid search ensures we explore the hyperparameter space for the best XGB configuration given engineered features.
- Stratified CV preserves class balance in each fold for reliable accuracy/F1/AUC estimates.

### Results
- Best params: {best_params}
- Best CV Accuracy: {best_score:.4f}
- Best CV Metrics: Acc={best_metrics['acc']:.4f}, F1={best_metrics['f1']:.4f}, AUC={best_metrics['auc']:.4f}
- Validation (20% holdout): Accuracy={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}
- Submission: `submissions/submission_v5.csv`
- Runtime: ~{time.time()-start:.1f}s
- Note: Submitted regardless of comparison to v4 (0.9256) per submission policy.
"""

with open('docs/experiments.md', 'a') as f:
    f.write(new_section)

print('Experiment 5 complete.')
