#!/usr/bin/env python3
"""Experiment 25: Route-Hamming & Original-Lookup Augmented LightGBM 5-Fold CV"""
import os, time, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

warnings.filterwarnings('ignore')
start_time = time.time()

print("=" * 60)
print("Experiment 25: Route-Hamming + OG-Lookup LightGBM 5-Fold CV")
print("=" * 60)

DATA_DIR = 'data'
FEAT_DIR = 'data/features'

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Load engineered route features and OG features
par_tr = pd.read_csv(os.path.join(FEAT_DIR, 'par_train.csv'))
par_te = pd.read_csv(os.path.join(FEAT_DIR, 'par_test.csv'))
ogp_tr = pd.read_csv(os.path.join(FEAT_DIR, 'ogp_train.csv'))
ogp_te = pd.read_csv(os.path.join(FEAT_DIR, 'ogp_test.csv'))
look_tr = pd.read_csv(os.path.join(FEAT_DIR, 'lookup_train.csv'))
look_te = pd.read_csv(os.path.join(FEAT_DIR, 'lookup_test.csv'))

# Delay features
for df in [train, test]:
    df['Departure Delay in Minutes'] = df['Departure Delay in Minutes'].fillna(0)
    df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(0)
    df['total_delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
    df['has_delay'] = (df['total_delay'] > 0).astype(int)

# Concatenate all features
X_tr_full = pd.concat([train.drop(columns=['id', 'satisfaction']), par_tr, ogp_tr, look_tr], axis=1)
X_te_full = pd.concat([test.drop(columns=['id']), par_te, ogp_te, look_te], axis=1)
y = train['satisfaction'].astype(int)

cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
for c in cat_cols:
    X_tr_full[c] = X_tr_full[c].astype('category')
    X_te_full[c] = X_te_full[c].astype('category')

feature_cols = X_tr_full.columns.tolist()
print(f"Total features ({len(feature_cols)}): {feature_cols[:10]}... + {len(feature_cols)-10} more")

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
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_full, y), 1):
    f_start = time.time()
    X_tr, y_tr = X_tr_full.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X_tr_full.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    va_preds = model.predict_proba(X_va)[:, 1]
    oof_preds[val_idx] = va_preds
    test_preds += model.predict_proba(X_te_full)[:, 1] / n_splits

    fold_auc = roc_auc_score(y_va, va_preds)
    fold_aucs.append(fold_auc)
    print(f"  Fold {fold}/{n_splits} AUC: {fold_auc:.5f} | Best iter: {model.best_iteration_} ({time.time()-f_start:.1f}s)")

overall_auc = roc_auc_score(y, oof_preds)
oof_binary = (oof_preds > 0.5).astype(int)
overall_acc = accuracy_score(y, oof_binary)
overall_f1 = f1_score(y, oof_binary)

print("\n" + "=" * 60)
print(f"Experiment 25 OOF ROC-AUC: {overall_auc:.5f} (Exp 11 Baseline was 0.95894 -> Delta: {overall_auc - 0.95894:+.5f})")
print(f"OOF Accuracy: {overall_acc:.5f} | F1: {overall_f1:.5f}")
print("=" * 60)

# Save OOF and test predictions
os.makedirs('models_oof', exist_ok=True)
np.save('models_oof/oof_exp25_route_lgb.npy', oof_preds)
np.save('models_oof/test_exp25_route_lgb.npy', test_preds)

# Save submission
os.makedirs('submissions', exist_ok=True)
sub_path = 'submissions/submission_v25.csv'
sub = pd.DataFrame({'id': test['id'], 'satisfaction': test_preds})
sub.to_csv(sub_path, index=False)
print(f"Saved submission to {sub_path} (mean prob: {test_preds.mean():.4f})")
print(f"Exp 25 completed in {time.time()-start_time:.1f}s")
