#!/usr/bin/env python3
"""Experiment 4: Stacking meta-learner for Predicting Airline Satisfaction."""
import os, time, warnings
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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

    # Target encoding using train mean per category
    for col in cat_cols:
        if is_train:
            train_target = train_df.copy() if train_df is not None else df.copy()
            train_target['satisfaction_bin'] = train_target['satisfaction'].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0})
            mean_map = train_target.groupby(col)['satisfaction_bin'].mean().to_dict()
            df[col + '_te'] = df[col].map(mean_map).fillna(train_target['satisfaction_bin'].mean())
        else:
            # Use precomputed mean_map from train; passed via train_df
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

xgb_params = {
    'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6,
    'subsample': 0.85, 'colsample_bytree': 0.85,
    'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'random_state': 42, 'n_jobs': -1, 'use_label_encoder': False,
    'reg_alpha': 0.1, 'reg_lambda': 1.0
}
rf_params = {'n_estimators': 300, 'max_depth': 12, 'random_state': 42, 'n_jobs': -1}
hgb_params = {'max_depth': 6, 'learning_rate': 0.05, 'max_iter': 200, 'random_state': 42}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_xgb = np.zeros(X.shape[0])
oof_rf = np.zeros(X.shape[0])
oof_hgb = np.zeros(X.shape[0])

test_xgb = np.zeros(X_test.shape[0])
test_rf = np.zeros(X_test.shape[0])
test_hgb = np.zeros(X_test.shape[0])

print('Starting out-of-fold stacking (5 folds)...')
for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'Fold {fold_idx}/5...')
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr, y_tr)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    test_xgb += xgb_model.predict_proba(X_test)[:, 1] / 5

    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_tr, y_tr)
    oof_rf[val_idx] = rf_model.predict_proba(X_val)[:, 1]
    test_rf += rf_model.predict_proba(X_test)[:, 1] / 5

    hgb_model = HistGradientBoostingClassifier(**hgb_params)
    hgb_model.fit(X_tr, y_tr)
    oof_hgb[val_idx] = hgb_model.predict_proba(X_val)[:, 1]
    test_hgb += hgb_model.predict_proba(X_test)[:, 1] / 5

    val_pred_xgb = (oof_xgb[val_idx] > 0.5).astype(int)
    print(f'  Fold {fold_idx} XGB AUC: {roc_auc_score(y_val, oof_xgb[val_idx]):.4f}')

meta_train = np.column_stack([oof_xgb, oof_rf, oof_hgb])
print('Base model OOF AUCs:')
for i, name in enumerate(['XGB', 'RF', 'HGB']):
    auc = roc_auc_score(y, meta_train[:, i])
    print(f'  {name}: {auc:.4f}')

print('Training meta-learner...')
meta_learner = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs', random_state=42)
meta_learner.fit(meta_train, y)

meta_val_pred = meta_learner.predict(meta_train)
meta_val_proba = meta_learner.predict_proba(meta_train)[:, 1]
print('Meta-learner validation:')
print(f'  Accuracy: {accuracy_score(y, meta_val_pred):.4f}')
print(f'  F1: {f1_score(y, meta_val_pred):.4f}')
print(f'  AUC: {roc_auc_score(y, meta_val_proba):.4f}')

meta_test = np.column_stack([test_xgb, test_rf, test_hgb])
test_pred = meta_learner.predict(meta_test)
test_proba = meta_learner.predict_proba(meta_test)[:, 1]

os.makedirs('submissions', exist_ok=True)
submission_path = 'submissions/submission_v4.csv'
submission = pd.DataFrame({'id': test['id'], 'satisfaction': test_proba})
submission.to_csv(submission_path, index=False)
print(f'Submission saved: {submission_path} (rows={len(submission)}, mean={test_proba.mean():.4f})')

elapsed = time.time() - start
print(f'Experiment 4 completed in {elapsed:.1f}s')
