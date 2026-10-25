#!/usr/bin/env python3
"""5-Fold XGBoost with AUC Optimization (eval_metric='auc')"""
import os, time, warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
start_time = time.time()

print("=" * 60)
print("5-Fold XGBoost CV (AUC Optimized)")
print("=" * 60)

DATA_DIR = 'data'
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Delay features
for df in [train, test]:
    df['Departure Delay in Minutes'] = df['Departure Delay in Minutes'].fillna(0)
    df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(0)
    df['total_delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
    df['has_delay'] = (df['total_delay'] > 0).astype(int)

cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
for c in cat_cols:
    le = LabelEncoder()
    train[c] = le.fit_transform(train[c].astype(str))
    test[c] = le.transform(test[c].astype(str))

feature_cols = [c for c in train.columns if c not in ['id', 'satisfaction']]
X = train[feature_cols].values
y = train['satisfaction'].astype(int).values
X_test = test[feature_cols].values

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_aucs = []

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 7,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'n_estimators': 800,
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

print(f"\nStarting {n_splits}-Fold Stratified Cross-Validation...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    f_start = time.time()
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[val_idx], y[val_idx]

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )

    va_preds = model.predict_proba(X_va)[:, 1]
    oof_preds[val_idx] = va_preds
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

    fold_auc = roc_auc_score(y_va, va_preds)
    fold_aucs.append(fold_auc)
    print(f"  Fold {fold}/{n_splits} AUC: {fold_auc:.5f} ({time.time()-f_start:.1f}s)")

overall_auc = roc_auc_score(y, oof_preds)
print(f"Overall OOF ROC-AUC: {overall_auc:.5f}")

os.makedirs('models_oof', exist_ok=True)
np.save('models_oof/oof_xgb.npy', oof_preds)
np.save('models_oof/test_xgb.npy', test_preds)

os.makedirs('submissions', exist_ok=True)
sub = pd.DataFrame({'id': test['id'], 'satisfaction': test_preds})
sub.to_csv('submissions/submission_xgb_auc.csv', index=False)
print(f"XGB CV completed in {time.time()-start_time:.1f}s")
