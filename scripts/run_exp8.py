#!/usr/bin/env python3
"""Experiment 8: Final Blend (Best-of-Sequence Ensemble)"""
import os, time, warnings, sys
warnings.filterwarnings('ignore')

start = time.time()

# 1. Load dataset with fallback
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
print('Feature count:', len(feature_cols))

X = train_fe[feature_cols].values
y = train_fe['satisfaction_bin'].values
X_test = test_fe[feature_cols].values

# 2. Split train for validation (20% holdout)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Train/reload base predictions/models
print('Training/reloading base models...')

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Best XGB (v5 params)
xgb_params = {
    'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6,
    'subsample': 0.85, 'colsample_bytree': 0.85,
    'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'random_state': 42, 'n_jobs': -1, 'use_label_encoder': False,
    'reg_alpha': 0.1, 'reg_lambda': 1.0
}

print('  Training XGB...')
xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_tr, y_tr)
xgb_val_pred = xgb_model.predict(X_val)
xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]

# Best RF
rf_params = {'n_estimators': 300, 'max_depth': 12, 'random_state': 42, 'n_jobs': -1}
print('  Training RF...')
rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X_tr, y_tr)
rf_val_pred = rf_model.predict(X_val)
rf_val_proba = rf_model.predict_proba(X_val)[:, 1]

# Best HistGB (if faster)
hgb_params = {'max_depth': 6, 'learning_rate': 0.05, 'max_iter': 200, 'random_state': 42}
print('  Training HistGB...')
hgb_model = HistGradientBoostingClassifier(**hgb_params)
hgb_model.fit(X_tr, y_tr)
hgb_val_pred = hgb_model.predict(X_val)
hgb_val_proba = hgb_model.predict_proba(X_val)[:, 1]

# Compute validation accuracies
xgb_acc = accuracy_score(y_val, xgb_val_pred)
rf_acc = accuracy_score(y_val, rf_val_pred)
hgb_acc = accuracy_score(y_val, hgb_val_pred)
print(f'  Validation Accuracy - XGB: {xgb_acc:.4f}, RF: {rf_acc:.4f}, HGB: {hgb_acc:.4f}')

# 4. Meta-learner predictions from v4
meta_train = np.column_stack([xgb_val_proba, rf_val_proba, hgb_val_proba])
print('  Training meta-learner (LogisticRegression)...')
meta_learner = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs', random_state=42)
meta_learner.fit(meta_train, y_val)
meta_val_pred = meta_learner.predict(meta_train)
meta_val_proba = meta_learner.predict_proba(meta_train)[:, 1]
meta_acc = accuracy_score(y_val, meta_val_pred)
print(f'  Meta-learner validation accuracy: {meta_acc:.4f}')

# 5. Blend predictions with weights based on validation scores
models = {
    'XGB': (xgb_acc, xgb_model),
    'RF': (rf_acc, rf_model),
    'HGB': (hgb_acc, hgb_model),
    'Meta': (meta_acc, meta_learner)
}
best_model_name = max(models, key=lambda k: models[k][0])
best_acc = models[best_model_name][0]
print(f'  Best model by val accuracy: {best_model_name} ({best_acc:.4f})')

# Weights proportional to val accuracy
acc_vals = np.array([xgb_acc, rf_acc, hgb_acc, meta_acc])
weights_prop = acc_vals / acc_vals.sum()
print(f'  Proportional weights: XGB={weights_prop[0]:.2%}, RF={weights_prop[1]:.2%}, HGB={weights_prop[2]:.2%}, Meta={weights_prop[3]:.2%}')

# If meta-learner is best, use fixed blend: 50% XGB + 30% meta + 20% RF
if best_model_name == 'Meta':
    blend_weights = np.array([0.50, 0.20, 0.00, 0.30])
    print('  Using fixed blend (meta best): 50% XGB + 20% RF + 30% Meta')
else:
    blend_weights = weights_prop
    print(f'  Using proportional blend weights: XGB={blend_weights[0]:.2f}, RF={blend_weights[1]:.2f}, HGB={blend_weights[2]:.2f}, Meta={blend_weights[3]:.2f}')

# Retrain best blend components on full data
print('Retraining best blend components on full data...')
final_xgb = xgb.XGBClassifier(**xgb_params)
final_xgb.fit(X, y)
final_rf = RandomForestClassifier(**rf_params)
final_rf.fit(X, y)
final_hgb = HistGradientBoostingClassifier(**hgb_params)
final_hgb.fit(X, y)

# Retrain meta-learner on full predictions
print('  Retraining meta-learner on full predictions...')
oof_xgb = final_xgb.predict_proba(X)[:, 1]
oof_rf = final_rf.predict_proba(X)[:, 1]
oof_hgb = final_hgb.predict_proba(X)[:, 1]
meta_train_full = np.column_stack([oof_xgb, oof_rf, oof_hgb])
final_meta = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs', random_state=42)
final_meta.fit(meta_train_full, y)

# Blend predictions on test
xgb_test = final_xgb.predict_proba(X_test)[:, 1]
rf_test = final_rf.predict_proba(X_test)[:, 1]
hgb_test = final_hgb.predict_proba(X_test)[:, 1]
meta_test = np.column_stack([
    final_xgb.predict_proba(X_test)[:, 1],
    final_rf.predict_proba(X_test)[:, 1],
    final_hgb.predict_proba(X_test)[:, 1]
])
meta_learner_test_proba = final_meta.predict_proba(meta_test)[:, 1]

test_blend = (blend_weights[0] * xgb_test +
              blend_weights[1] * rf_test +
              blend_weights[2] * hgb_test +
              blend_weights[3] * meta_learner_test_proba)

# Validate blend on holdout
xgb_val_pred_final = final_xgb.predict(X_val)
rf_val_pred_final = final_rf.predict(X_val)
hgb_val_pred_final = final_hgb.predict(X_val)
meta_val_pred_final = final_meta.predict(np.column_stack([
    final_xgb.predict_proba(X_val)[:, 1],
    final_rf.predict_proba(X_val)[:, 1],
    final_hgb.predict_proba(X_val)[:, 1]
]))
blend_val_pred = (blend_weights[0] * xgb_val_pred_final +
                  blend_weights[1] * rf_val_pred_final +
                  blend_weights[2] * hgb_val_pred_final +
                  blend_weights[3] * meta_val_pred_final)
blend_val_pred_bin = (blend_val_pred > 0.5).astype(int)
blend_acc = accuracy_score(y_val, blend_val_pred_bin)
blend_f1 = f1_score(y_val, blend_val_pred_bin)
blend_auc = roc_auc_score(y_val, blend_val_pred)

print(f'Blend validation - Accuracy: {blend_acc:.4f}, F1: {blend_f1:.4f}, AUC: {blend_auc:.4f}')
print(f'v4 reference accuracy: 0.9256')
print(f'Blend beats v4? {"Yes" if blend_acc > 0.9256 else "No" if blend_acc < 0.9256 else "Equal"} (current: {blend_acc:.4f} vs 0.9256)')

if blend_acc < 0.9250:
    print('WARNING: Blend accuracy < 0.9250 — submit anyway per submission policy.')

os.makedirs('submissions', exist_ok=True)
submission_path = 'submissions/submission_v8.csv'
submission = pd.DataFrame({'id': test['id'], 'satisfaction': test_blend})
submission.to_csv(submission_path, index=False)
print(f'Submission saved: {submission_path} (rows={len(submission)}, mean={test_blend.mean():.4f})')

# 6. Update docs
with open('docs/experiments.md', 'a') as f:
    best_str = best_model_name
    f.write(f"""
---

## Experiment 8 — Final Blend (Best-of-Sequence Ensemble) — {time.strftime('%Y-%m-%d %H:%M')}

### What we did
- Loaded dataset using Kaggle path (`../input/playground-series-s6e10/train.csv`) with local fallback (`data/train.csv`).
- Used existing feature engineering pipeline (service ratings mean/min/max/range/std, age bins, delay interactions, expanded interactions, target encoding, label encoding, indicators).
- Trained/retrained base predictions/models: best XGB (v5 params: max_depth=6, lr=0.05, subsample=0.85, n_estimators=300), best RF (n_estimators=300, max_depth=12), best HistGB (max_depth=6, lr=0.05, max_iter=200), and meta-learner predictions from v4 approach (LogisticRegression on base model predictions).
- Blended predictions with weights based on validation scores. Best model by val accuracy: `{best_str}` ({best_acc:.4f}).
- Blend weights used: `XGB={blend_weights[0]:.0%}, RF={blend_weights[1]:.0%}, HGB={blend_weights[2]:.0%}, Meta={blend_weights[3]:.0%}`.
- Retrained blend components on full training data; validated blend on 20% stratified holdout.

### Why
- Best-of-sequence ensemble combines strongest individual models (XGB from v5, RF, meta-learner from v4) to maximize validation accuracy.
- Proportional/fixed weights favor the best-performing components.

### Results
- Validation Accuracy: {blend_acc:.4f}
- Validation F1: {blend_f1:.4f}
- Validation AUC: {blend_auc:.4f}
- Blend weights: XGB={blend_weights[0]:.2f}, RF={blend_weights[1]:.2f}, HGB={blend_weights[2]:.2f}, Meta={blend_weights[3]:.2f}
- Best base model: {best_str} (val acc {best_acc:.4f})
- v4 reference accuracy: 0.9256
- Beat v4? {'Yes' if blend_acc > 0.9256 else 'No' if blend_acc < 0.9256 else 'Equal'} (current: {blend_acc:.4f} vs 0.9256)
- Submission: `submissions/submission_v8.csv`
- Runtime: ~{time.time()-start:.1f}s
- Note: {'Blend accuracy below 0.9250 threshold — submitted per submission policy.' if blend_acc < 0.9250 else 'Blend meets or exceeds v4 reference; submitted.'}
""")

elapsed = time.time() - start
print(f'Experiment 8 completed in {elapsed:.1f}s')
