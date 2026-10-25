#!/usr/bin/env python3
"""Experiment 12: Native CatBoost 5-Fold CV with AUC Optimization"""
import os, time, warnings
import pandas as pd
import numpy as np
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

warnings.filterwarnings('ignore')
start_time = time.time()

print("=" * 60)
print("Experiment 12: Native CatBoost 5-Fold CV (AUC Optimized)")
print("=" * 60)

DATA_DIR = 'data'
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print(f"Loaded train: {train.shape}, test: {test.shape}")

# Feature Preprocessing
for df in [train, test]:
    df['Departure Delay in Minutes'] = df['Departure Delay in Minutes'].fillna(0)
    df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(0)
    df['total_delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
    df['has_delay'] = (df['total_delay'] > 0).astype(int)

cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
for c in cat_cols:
    train[c] = train[c].astype(str)
    test[c] = test[c].astype(str)

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

cb_params = {
    'iterations': 1000,
    'learning_rate': 0.08,
    'depth': 7,
    'eval_metric': 'AUC',
    'cat_features': cat_cols,
    'random_seed': 42,
    'thread_count': -1,
    'verbose': 0
}

print(f"\nStarting {n_splits}-Fold Stratified Cross-Validation...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    f_start = time.time()
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    model = cb.CatBoostClassifier(**cb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=0
    )

    va_preds = model.predict_proba(X_va)[:, 1]
    oof_preds[val_idx] = va_preds
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

    fold_auc = roc_auc_score(y_va, va_preds)
    fold_aucs.append(fold_auc)
    print(f"  Fold {fold}/{n_splits} AUC: {fold_auc:.5f} | Best iter: {model.get_best_iteration()} ({time.time()-f_start:.1f}s)")

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
np.save('models_oof/oof_exp12_cb.npy', oof_preds)
np.save('models_oof/test_exp12_cb.npy', test_preds)

# Save submission
os.makedirs('submissions', exist_ok=True)
sub_path = 'submissions/submission_v12.csv'
sub = pd.DataFrame({'id': test['id'], 'satisfaction': test_preds})
sub.to_csv(sub_path, index=False)
print(f"Saved submission to {sub_path} (rows={len(sub)}, mean_prob={test_preds.mean():.4f})")

runtime = time.time() - start_time
print(f"Exp 12 completed in {runtime:.1f}s")
