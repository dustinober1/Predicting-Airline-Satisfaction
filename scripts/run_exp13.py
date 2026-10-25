#!/usr/bin/env python3
"""Experiment 13: Route-Profile Feature Engineered LightGBM 5-Fold CV"""
import os, time, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

warnings.filterwarnings('ignore')
start_time = time.time()

print("=" * 60)
print("Experiment 13: Route-Profile LightGBM 5-Fold CV (AUC Optimized)")
print("=" * 60)

DATA_DIR = 'data'
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print(f"Loaded train: {train.shape}, test: {test.shape}")

# Delay features
for df in [train, test]:
    df['Departure Delay in Minutes'] = df['Departure Delay in Minutes'].fillna(0)
    df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(0)
    df['total_delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
    df['has_delay'] = (df['total_delay'] > 0).astype(int)

ratings = [
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Cleanliness'
]

cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

# Distance / Route Profile Features
dist_col = 'Flight Distance'
route_counts = train[dist_col].value_counts()
route_mean = train.groupby(dist_col)[ratings].mean()
global_mean = train[ratings].mean()

for df in [train, test]:
    df['route_count'] = df[dist_col].map(route_counts).fillna(0).astype('float32')
    # Route average satisfaction proxy features
    for r in ['Inflight wifi service', 'Online boarding', 'Seat comfort', 'Cleanliness']:
        r_mean = df[dist_col].map(route_mean[r]).fillna(global_mean[r]).astype('float32')
        df[f'route_{r}_mean'] = r_mean
        df[f'route_{r}_diff'] = (df[r] - r_mean).astype('float32')

for c in cat_cols:
    train[c] = train[c].astype('category')
    test[c] = test[c].astype('category')

feature_cols = [c for c in train.columns if c not in ['id', 'satisfaction']]
X = train[feature_cols]
y = train['satisfaction'].astype(int)
X_test = test[feature_cols]

print(f"Features ({len(feature_cols)}): {feature_cols}")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_aucs = []

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.03,
    'num_leaves': 127,
    'max_depth': -1,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'n_estimators': 1500,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

print(f"\nStarting {n_splits}-Fold Stratified Cross-Validation...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    f_start = time.time()
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    va_preds = model.predict_proba(X_va)[:, 1]
    oof_preds[val_idx] = va_preds
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

    fold_auc = roc_auc_score(y_va, va_preds)
    fold_aucs.append(fold_auc)
    print(f"  Fold {fold}/{n_splits} AUC: {fold_auc:.5f} | Best iter: {model.best_iteration_} ({time.time()-f_start:.1f}s)")

overall_auc = roc_auc_score(y, oof_preds)
oof_binary = (oof_preds > 0.5).astype(int)
overall_acc = accuracy_score(y, oof_binary)
overall_f1 = f1_score(y, oof_binary)

print("\n" + "=" * 60)
print(f"Overall OOF ROC-AUC: {overall_auc:.5f} (Mean Fold: {np.mean(fold_aucs):.5f} +/- {np.std(fold_aucs):.5f})")
print(f"Overall OOF Accuracy: {overall_acc:.5f} | F1: {overall_f1:.5f}")
print("=" * 60)

# Save OOF and test predictions for stacking
os.makedirs('models_oof', exist_ok=True)
np.save('models_oof/oof_exp13_route_lgb.npy', oof_preds)
np.save('models_oof/test_exp13_route_lgb.npy', test_preds)

# Save submission
os.makedirs('submissions', exist_ok=True)
sub_path = 'submissions/submission_v13.csv'
sub = pd.DataFrame({'id': test['id'], 'satisfaction': test_preds})
sub.to_csv(sub_path, index=False)
print(f"Saved submission to {sub_path} (rows={len(sub)}, mean_prob={test_preds.mean():.4f})")

runtime = time.time() - start_time
print(f"Exp 13 completed in {runtime:.1f}s")
