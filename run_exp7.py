#!/usr/bin/env python3
"""Experiment 7: Alternative Algorithms (HistGB / LightGBM / CatBoost)"""
import os, time, warnings
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

start = time.time()

kaggle_path = '../input/playground-series-s6e10/train.csv'
local_path = 'data/train.csv'
if os.path.exists(kaggle_path):
    DATA_DIR = '../input/playground-series-s6e10'
else:
    DATA_DIR = 'data'

print('Using DATA_DIR:', DATA_DIR)
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv')) if os.path.exists(os.path.join(DATA_DIR, 'test.csv')) else pd.read_csv('data/test.csv')

for df in [train, test]:
    df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(0)
    df['Departure Delay in Minutes'] = df['Departure Delay in Minutes'].fillna(0)

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

# 3. Train HistGradientBoostingClassifier
print('Training HistGradientBoostingClassifier...')
hgb = HistGradientBoostingClassifier(max_iter=100, max_depth=6, learning_rate=0.05, random_state=42)
hgb.fit(X, y)
algo_used = 'HistGradientBoostingClassifier (sklearn)'
final_model = hgb

# Check LightGBM
lightgbm_available = False
try:
    import lightgbm
    lightgbm_available = True
    print('LightGBM available; training quick model...')
    import lightgbm as lgb
    lgbm = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
    lgbm.fit(X, y)
    algo_used = 'LightGBM (n_estimators=200, max_depth=6, lr=0.05)'
    final_model = lgbm
except Exception as e:
    print('LightGBM not available or failed:', e)

if not lightgbm_available:
    catboost_available = False
    try:
        import catboost
        catboost_available = True
        print('CatBoost available; training quick model...')
        from catboost import CatBoostClassifier
        cat_model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, random_seed=42, verbose=False)
        cat_model.fit(X, y)
        algo_used = 'CatBoost (iterations=200, depth=6, lr=0.05)'
        final_model = cat_model
    except Exception as e:
        print('CatBoost not available or failed:', e)
        catboost_available = False
else:
    catboost_available = False
    try:
        import catboost
        catboost_available = True
    except:
        pass

# 5. Validate on 20% stratified holdout
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
final_model.fit(X_tr, y_tr)
val_pred = final_model.predict(X_val)
val_proba = final_model.predict_proba(X_val)[:, 1] if hasattr(final_model, 'predict_proba') else np.ones(len(y_val)) * val_pred.mean()
val_acc = accuracy_score(y_val, val_pred)
val_f1 = f1_score(y_val, val_pred)
val_auc = roc_auc_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else 0.5
print('Validation (20%% holdout) - Accuracy: %.4f, F1: %.4f, AUC: %.4f' % (val_acc, val_f1, val_auc))

# Retrain on full data
final_model.fit(X, y)
test_pred_proba = final_model.predict_proba(X_test)[:, 1]

os.makedirs('submissions', exist_ok=True)
submission_path = 'submissions/submission_v7.csv'
submission = pd.DataFrame({'id': test['id'], 'satisfaction': test_pred_proba})
submission.to_csv(submission_path, index=False)
print('Submission saved:', submission_path, '(rows=%d, mean=%.4f)' % (len(submission), test_pred_proba.mean()))

# 6. Update docs/experiments.md
with open('docs/experiments.md', 'a') as f:
    f.write(f"""
---

## Experiment 7 — Alternative Algorithms (HistGB / LightGBM / CatBoost) — {time.strftime('%Y-%m-%d %H:%M')}

### What we did
- Loaded dataset using Kaggle path (`../input/playground-series-s6e10/train.csv`) with local fallback (`data/train.csv`).
- Used existing feature engineering pipeline: service ratings mean/min/max/range/std, age bins, delays (`total_delay`, `has_delay`), expanded interactions (`age_x_distance`, `delay_x_service`, `distance_sq`, `age_x_service`, `distance_per_minute`), target encoding (`_te`), label encoding (`_le`), `loyal_business` / `eco_personal`.
- Filled NaN in delay columns with 0 (HistGB handles NaN natively but filled for consistency).
- Trained primary model: `{algo_used}`.
- LightGBM available: {lightgbm_available}, CatBoost available: {catboost_available}.
- Validated on 20%% stratified holdout.
- Generated predictions and saved to `submissions/submission_v7.csv`.

### Why
- Testing faster/native-NaN boosting implementations vs previous XGB to see if algorithm swap improves validation metrics.
- Keeps feature pipeline identical for fair comparison.

### Results
- Algorithm used: {algo_used}
- LightGBM available: {lightgbm_available}
- CatBoost available: {catboost_available}
- Validation Accuracy: {val_acc:.4f}
- Validation F1: {val_f1:.4f}
- Validation AUC: {val_auc:.4f}
- v4 reference accuracy: 0.9256
- Beat v4? {'Yes' if val_acc > 0.9256 else 'No' if val_acc < 0.9256 else 'Equal'} (current: {val_acc:.4f} vs 0.9256)
- Submission: `submissions/submission_v7.csv`
- Runtime: ~{time.time()-start:.1f}s
- Note: Submitted regardless of comparison to v4 (0.9256) per submission policy.
""")

print('Experiment 7 complete. Runtime: %.1fs' % (time.time()-start))
