#!/usr/bin/env python3
"""Experiment 6: Feature Selection / Dropout for Predicting Airline Satisfaction."""
import os, time, warnings, sys
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb

start = time.time()
DATA_DIR = '../input/playground-series-s6e10'

best_params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.85,
    'n_estimators': 300,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1,
    'use_label_encoder': False,
}

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
from pandas.api.types import is_numeric_dtype
feature_cols = [c for c in train_fe.columns if c not in exclude]
feature_cols = [c for c in feature_cols if is_numeric_dtype(train_fe[c])]
feature_cols = [c for c in feature_cols if c in test_fe.columns]
print('Full feature count:', len(feature_cols))

X = train_fe[feature_cols].values
y = train_fe['satisfaction_bin'].values
X_test = test_fe[feature_cols].values

# 1. Train XGB on full feature set and extract importances
print('Training XGB on full feature set...')
full_model = xgb.XGBClassifier(**best_params)
full_model.fit(X, y)
importances = full_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)
print('Feature importances (sorted):')
print(feature_importance_df.to_string(index=False))

# 2. Identify bottom 20-30% or features with importance < 0.01
threshold = 0.01
low_importance_features = feature_importance_df[feature_importance_df['importance'] < threshold]['feature'].tolist()
bottom_n = max(5, int(len(feature_cols) * 0.20))
bottom_features = feature_importance_df.tail(bottom_n)['feature'].tolist()
drop_candidates = sorted(set(low_importance_features + bottom_features))
if len(drop_candidates) < 5:
    additional = feature_importance_df[~feature_importance_df['feature'].isin(drop_candidates)].tail(5 - len(drop_candidates))['feature'].tolist()
    drop_candidates.extend(additional)
final_dropped = sorted([f for f in drop_candidates if f in feature_importance_df.tail(10)['feature'].values or f in low_importance_features])
if not final_dropped:
    final_dropped = feature_importance_df.tail(5)['feature'].tolist()
max_drop = max(5, int(len(feature_cols) * 0.30))
if len(final_dropped) > max_drop:
    final_dropped = feature_importance_df.tail(max_drop)['feature'].tolist()

print('Features to drop:', final_dropped)
print('Dropped count:', len(final_dropped))

reduced_feature_cols = [f for f in feature_cols if f not in final_dropped]
print('Reduced feature count:', len(reduced_feature_cols))

# 3. Retrain XGB on reduced feature set with same params
X_reduced = train_fe[reduced_feature_cols].values
X_test_reduced = test_fe[reduced_feature_cols].values

print('Retraining XGB on reduced feature set...')
reduced_model = xgb.XGBClassifier(**best_params)
reduced_model.fit(X_reduced, y)

# 4. Validate on 20% stratified holdout
X_tr, X_val, y_tr, y_val = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)
val_model = xgb.XGBClassifier(**best_params)
val_model.fit(X_tr, y_tr)
val_pred = val_model.predict(X_val)
val_proba = val_model.predict_proba(X_val)[:, 1]
val_acc = accuracy_score(y_val, val_pred)
val_f1 = f1_score(y_val, val_pred)
val_auc = roc_auc_score(y_val, val_proba)
print('Validation (20% holdout, reduced) - Accuracy: %.4f, F1: %.4f, AUC: %.4f' % (val_acc, val_f1, val_auc))

X_tr_full, X_val_full, y_tr_full, y_val_full = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
full_val_model = xgb.XGBClassifier(**best_params)
full_val_model.fit(X_tr_full, y_tr_full)
full_val_pred = full_val_model.predict(X_val_full)
full_val_acc = accuracy_score(y_val_full, full_val_pred)
print('Full feature holdout accuracy: %.4f' % full_val_acc)
print('v4 reference accuracy: 0.9256')

# 5. Retrain reduced model on full reduced data and predict test
efinal_model = xgb.XGBClassifier(**best_params)
efinal_model.fit(X_reduced, y)
test_pred_proba = efinal_model.predict_proba(X_test_reduced)[:, 1]

os.makedirs('submissions', exist_ok=True)
submission_path = 'submissions/submission_v6.csv'
submission = pd.DataFrame({'id': test['id'], 'satisfaction': test_pred_proba})
submission.to_csv(submission_path, index=False)
print('Submission saved: %s (rows=%d, mean=%.4f)' % (submission_path, len(submission), test_pred_proba.mean()))

# 6. Update docs/experiments.md
print('Updating docs/experiments.md...')
with open('docs/experiments.md', 'r') as f:
    doc_content = f.read()

dropped_str = ', '.join(final_dropped)
new_section = """
---

## Experiment 6 — Feature Selection / Dropout (Reduced Feature Set) — %s

### What we did
- Loaded data/train.csv and data/test.csv; filled NaN in `Arrival Delay in Minutes` with 0.
- Built feature engineering pipeline (service ratings mean/min/max/range/std, age bins, delay interactions, interaction terms, target encoding, label encoding).
- Trained XGBoost on full feature set (`max_depth=6`, `learning_rate=0.05`, `subsample=0.85`, `n_estimators=300`) and extracted `feature_importances_`.
- Identified lowest importance features (bottom ~20-30%% or importance < 0.01): %s.
- Created reduced feature set (dropped %d features; %d remaining).
- Retrained XGB with same best params from Experiment 5 on reduced feature set.
- Validated on 20%% stratified holdout: reduced accuracy = %.4f, full accuracy = %.4f, v4 reference = 0.9256.
- Saved predictions to `submissions/submission_v6.csv`.

### Why
- Removing low-importance features can reduce noise and overfitting, sometimes improving or maintaining validation performance with faster training.
- Dropout/feature selection is a standard regularization technique for tree-based models.

### Results
- Features dropped: %s (%d total)
- Reduced set feature count: %d
- Reduced validation accuracy: %.4f
- Reduced validation F1: %.4f, AUC: %.4f
- Full feature validation accuracy: %.4f
- v4 accuracy reference: 0.9256
- Submission: `submissions/submission_v6.csv`
- Runtime: ~%.1fs
- Note: Reduced set performs %s vs v4 (%.4f vs 0.9256). Submitted regardless per policy.
""" % (time.strftime('%Y-%m-%d %H:%M'), dropped_str, len(final_dropped), len(reduced_feature_cols), val_acc, full_val_acc, dropped_str, len(final_dropped), len(reduced_feature_cols), val_acc, val_f1, val_auc, full_val_acc, 'better' if val_acc > 0.9256 else 'worse' if val_acc < 0.9256 else 'equal', val_acc, time.time()-start)

with open('docs/experiments.md', 'a') as f:
    f.write(new_section)

print('Experiment 6 complete.')
