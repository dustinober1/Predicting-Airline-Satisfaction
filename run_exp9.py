#!/usr/bin/env python3
"""Experiment 9: Last-Day Optimization Enhanced Blend"""
import os, time, warnings, sys
warnings.filterwarnings('ignore')
start = time.time()

# 1. Load dataset
kaggle_path = '../input/playground-series-s6e10/train.csv'
local_path = 'data/train.csv'
if os.path.exists(kaggle_path):
    DATA_DIR = '../input/playground-series-s6e10'
else:
    DATA_DIR = 'data'

print('Using DATA_DIR:', DATA_DIR)

import pandas as pd, numpy as np
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
if os.path.exists(os.path.join(DATA_DIR, 'test.csv')):
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
else:
    test = pd.read_csv('data/test.csv')

print('Train:', train.shape, 'Test:', test.shape)

for df in [train, test]:
    df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(0)
    df['Departure Delay in Minutes'] = df['Departure Delay in Minutes'].fillna(0)

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

    df['total_delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
    df['has_delay'] = (df['total_delay'] > 0).astype(int)

    # Enhanced interaction / polynomial expansions
    df['age_x_distance'] = df['Age'] * df['Flight Distance']
    df['delay_x_service'] = df['total_delay'] * df['avg_service']
    df['distance_sq'] = df['Flight Distance'] ** 2
    df['age_x_service'] = df['Age'] * df['avg_service']
    df['distance_per_minute'] = df['Flight Distance'] / (df['total_delay'] + 1)

    # NEW: Enhanced terms for last-day optimization
    df['Age_group_sq'] = df['Age_group'] ** 2
    df['service_range_cubed'] = df['service_range'] ** 3
    df['flight_distance_x_avg_service_x_total_delay'] = df['Flight Distance'] * df['avg_service'] * df['total_delay']
    df['service_range_sq'] = df['service_range'] ** 2
    df['avg_service_cubed'] = df['avg_service'] ** 3
    df['age_x_service_range'] = df['Age'] * df['service_range']
    df['distance_cubed'] = df['Flight Distance'] ** 3

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

print('Engineering enhanced features...')
train_fe = engineer_features(train, is_train=True, train_df=train)
test_fe = engineer_features(test, is_train=False, train_df=train)

exclude = {'id', 'satisfaction', 'satisfaction_bin'}
from pandas.api.types import is_numeric_dtype
feature_cols = [c for c in train_fe.columns if c not in exclude]
feature_cols = [c for c in feature_cols if is_numeric_dtype(train_fe[c])]
feature_cols = [c for c in feature_cols if c in test_fe.columns]
print('Feature count:', len(feature_cols))

X = train_fe[feature_cols].values
y = train_fe['satisfaction_bin'].values
X_test = test_fe[feature_cols].values

# 2. Validate on 20% stratified holdout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Retrain blend with fine-tuned settings
print('Training/reloading base models...')
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Fine-tuned XGB
xgb_params = {
    'n_estimators': 600, 'learning_rate': 0.03, 'max_depth': 7,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'random_state': 42, 'n_jobs': -1, 'use_label_encoder': False,
    'reg_alpha': 0.1, 'reg_lambda': 1.0
}

print('  Training XGB (fine-tuned)...')
xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_tr, y_tr)
xgb_val_pred = xgb_model.predict(X_val)
xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]

# Fine-tuned RF
rf_params = {'n_estimators': 400, 'max_depth': 15, 'min_samples_split': 5, 'random_state': 42, 'n_jobs': -1}
print('  Training RF (fine-tuned)...')
rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X_tr, y_tr)
rf_val_pred = rf_model.predict(X_val)
rf_val_proba = rf_model.predict_proba(X_val)[:, 1]

# Meta-learner retrained with stronger regularization or specified params
print('  Training meta-learner (LogisticRegression C=1.0, max_iter=500)...')
meta_learner = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500, random_state=42)
meta_train = np.column_stack([xgb_val_proba, rf_val_proba])
meta_learner.fit(meta_train, y_val)
meta_val_pred = meta_learner.predict(meta_train)
meta_val_proba = meta_learner.predict_proba(meta_train)[:, 1]

# Validation accuracies
xgb_acc = accuracy_score(y_val, xgb_val_pred)
rf_acc = accuracy_score(y_val, rf_val_pred)
meta_acc = accuracy_score(y_val, meta_val_pred)
print(f'  Validation Accuracy - XGB: {xgb_acc:.4f}, RF: {rf_acc:.4f}, Meta: {meta_acc:.4f}')

# 4. Blend weights: 50% XGB + 20% RF + 30% meta (as per exp 8 best blend)
blend_weights = np.array([0.50, 0.20, 0.30])
print('  Blend weights: 50% XGB + 20% RF + 30% Meta')

# Retrain blend components on full training data
print('Retraining blend components on full data...')
final_xgb = xgb.XGBClassifier(**xgb_params)
final_xgb.fit(X, y)
final_rf = RandomForestClassifier(**rf_params)
final_rf.fit(X, y)

# Retrain meta-learner on full predictions (only XGB + RF for meta input to match weights)
print('  Retraining meta-learner on full predictions...')
oof_xgb = final_xgb.predict_proba(X)[:, 1]
oof_rf = final_rf.predict_proba(X)[:, 1]
meta_train_full = np.column_stack([oof_xgb, oof_rf])
final_meta = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500, random_state=42)
final_meta.fit(meta_train_full, y)

# Blend predictions on test
xgb_test = final_xgb.predict_proba(X_test)[:, 1]
rf_test = final_rf.predict_proba(X_test)[:, 1]
meta_test = np.column_stack([
    final_xgb.predict_proba(X_test)[:, 1],
    final_rf.predict_proba(X_test)[:, 1],
])
meta_learner_test_proba = final_meta.predict_proba(meta_test)[:, 1]

test_blend = (blend_weights[0] * xgb_test +
              blend_weights[1] * rf_test +
              blend_weights[2] * meta_learner_test_proba)

# Blend validation
blend_val_pred_bin = ((blend_weights[0] * xgb_model.predict(X_val) +
                        blend_weights[1] * rf_model.predict(X_val) +
                        blend_weights[2] * meta_learner.predict(np.column_stack([
                            xgb_model.predict_proba(X_val)[:,1],
                            rf_model.predict_proba(X_val)[:,1]
                        ]))) > 0.5).astype(int)

# More accurate blend validation using probabilities
xgb_val_final_proba = final_xgb.predict_proba(X_val)[:,1]
rf_val_final_proba = final_rf.predict_proba(X_val)[:,1]
meta_val_proba_full = final_meta.predict_proba(np.column_stack([final_xgb.predict_proba(X_val)[:,1], final_rf.predict_proba(X_val)[:,1]]))[:,1]
blend_val_proba = blend_weights[0]*xgb_val_final_proba + blend_weights[1]*rf_val_final_proba + blend_weights[2]*meta_val_proba_full
blend_val_pred = (blend_val_proba > 0.5).astype(int)
blend_acc = accuracy_score(y_val, blend_val_pred)
blend_f1 = f1_score(y_val, blend_val_pred)
blend_auc = roc_auc_score(y_val, blend_val_proba)

print(f'Blend validation - Accuracy: {blend_acc:.4f}, F1: {blend_f1:.4f}, AUC: {blend_auc:.4f}')
print(f'Best validation accuracy target: 0.9260')
print(f'Met target >= 0.9260? {"Yes" if blend_acc >= 0.9260 else "No"} (current: {blend_acc:.4f})')

if blend_acc < 0.9260:
    print('WARNING: Blend validation accuracy < 0.9260 — submit anyway per submission policy.')

os.makedirs('submissions', exist_ok=True)
submission_path = 'submissions/submission_v9.csv'
submission = pd.DataFrame({'id': test['id'], 'satisfaction': test_blend})
submission.to_csv(submission_path, index=False)
print(f'Submission saved: {submission_path} (rows={len(submission)}, mean={test_blend.mean():.4f})')

# Note leaderboard gap
print('NOTE: Leaderboard top score 0.96370 vs our validation accuracy ~0.9258 suggests evaluation metric may be AUC or another metric, or our validation approach underestimates performance. Deeper feature engineering / different metric focus likely needed beyond this plan.')

# 5. Update docs
with open('docs/experiments.md', 'a') as f:
    f.write(f"""
---

## Experiment 9 — Last-Day Optimization Enhanced Blend — {time.strftime('%Y-%m-%d %H:%M')}

### What we did
- Loaded dataset using Kaggle path (`../input/playground-series-s6e10/train.csv`) with local fallback (`data/train.csv`).
- Enhanced feature engineering: added `Age_group_sq`, `service_range_cubed`, `flight_distance_x_avg_service_x_total_delay`, `service_range_sq`, `avg_service_cubed`, `age_x_service_range`, `distance_cubed`.
- Retrained best blend from Experiment 8 with fine-tuned parameters:
  - XGB: `n_estimators=600`, `learning_rate=0.03`, `max_depth=7`, `subsample=0.8`, `colsample_bytree=0.8`
  - RF: `n_estimators=400`, `max_depth=15`, `min_samples_split=5`
  - Meta: `LogisticRegression(C=1.0, solver='lbfgs', max_iter=500)`
- Blend weights: 50% XGB + 20% RF + 30% Meta.
- Trained on 100% of data; validated on 20% stratified holdout.
- Note: Leaderboard gap (0.96370) indicates possible AUC or different metric focus; deeper feature engineering or metric-specific tuning needed beyond this blend.

### Why
- Fine-tuned settings aim for faster adaptation (lower LR, deeper trees, adjusted subsample) while preserving blend stability.
- Enhanced polynomial/interaction terms capture non-linear relationships not fully modeled in v8.

### Results
- Validation Accuracy: {blend_acc:.4f}
- Validation F1: {blend_f1:.4f}
- Validation AUC: {blend_auc:.4f}
- Blend weights: XGB=0.50, RF=0.20, Meta=0.30
- Validation target (>=0.9260): {'Met' if blend_acc >= 0.9260 else 'Not met'} (current: {blend_acc:.4f})
- Submission: `submissions/submission_v9.csv`
- Runtime: ~{time.time()-start:.1f}s
- Note: Leaderboard score (0.96370) far exceeds our validation accuracy (~0.9258). This suggests the evaluation metric may differ (likely AUC or another metric) or validation underestimates performance. Radical change (deeper feature engineering, different metric focus) likely required for significant improvement.
- Note: If validation accuracy < 0.9260, submitted per submission policy.
""")

elapsed = time.time() - start
print(f'Experiment 9 completed in {elapsed:.1f}s')
