#!/usr/bin/env python3
"""Experiment 14: Tri-GBDT Stacking Meta-Learner (LGBM + CatBoost + XGBoost)"""
import os, time, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

warnings.filterwarnings('ignore')
start_time = time.time()

print("=" * 60)
print("Experiment 14: Tri-GBDT Stacking Meta-Learner (LGBM + CB + XGB)")
print("=" * 60)

DATA_DIR = 'data'
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), usecols=['id', 'satisfaction'])
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), usecols=['id'])
y = train['satisfaction'].astype(int).values

# Load OOF and test predictions
print("Loading model predictions from models_oof/...")
oof_lgb = np.load('models_oof/oof_exp11_lgb.npy')
test_lgb = np.load('models_oof/test_exp11_lgb.npy')

oof_cb = np.load('models_oof/oof_exp12_cb.npy')
test_cb = np.load('models_oof/test_exp12_cb.npy')

# Also check for XGB if available, else route_lgb
models = {'LightGBM': (oof_lgb, test_lgb), 'CatBoost': (oof_cb, test_cb)}

if os.path.exists('models_oof/oof_xgb.npy'):
    models['XGBoost'] = (np.load('models_oof/oof_xgb.npy'), np.load('models_oof/test_xgb.npy'))
if os.path.exists('models_oof/oof_exp13_route_lgb.npy'):
    models['Route_LGB'] = (np.load('models_oof/oof_exp13_route_lgb.npy'), np.load('models_oof/test_exp13_route_lgb.npy'))

print(f"Models in stack: {list(models.keys())}")
for name, (oof, _) in models.items():
    print(f"  {name} OOF AUC: {roc_auc_score(y, oof):.5f}")

# Convert probabilities to clipped logits (Bayesian log-odds space)
def to_logit(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))

oof_list = [to_logit(m[0]) for m in models.values()]
test_list = [to_logit(m[1]) for m in models.values()]

X_oof_logits = np.column_stack(oof_list)
X_test_logits = np.column_stack(test_list)

# 5-fold meta-stacking with L2 regularized Logistic Regression
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_oof = np.zeros(len(y))
meta_test = np.zeros(len(test))
coefs = []

print("\nFitting Logistic Regression Meta-Learner (C=0.01)...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_oof_logits, y), 1):
    lr = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
    lr.fit(X_oof_logits[tr_idx], y[tr_idx])
    
    val_preds = lr.predict_proba(X_oof_logits[val_idx])[:, 1]
    meta_oof[val_idx] = val_preds
    meta_test += lr.predict_proba(X_test_logits)[:, 1] / 5
    coefs.append(lr.coef_[0])
    
    print(f"  Fold {fold}/5 AUC: {roc_auc_score(y[val_idx], val_preds):.5f}")

overall_auc = roc_auc_score(y, meta_oof)
mean_coefs = np.mean(coefs, axis=0)
print("\n" + "=" * 60)
print(f"Tri-GBDT Stacking OOF ROC-AUC: {overall_auc:.5f}")
for name, c in zip(models.keys(), mean_coefs):
    print(f"  Weight for {name}: {c:.4f}")
print("=" * 60)

# Save OOF and test
os.makedirs('models_oof', exist_ok=True)
np.save('models_oof/oof_exp14_stack.npy', meta_oof)
np.save('models_oof/test_exp14_stack.npy', meta_test)

# Save submission
os.makedirs('submissions', exist_ok=True)
sub_path = 'submissions/submission_v14.csv'
sub = pd.DataFrame({'id': test['id'], 'satisfaction': meta_test})
sub.to_csv(sub_path, index=False)
print(f"Saved submission to {sub_path} (rows={len(sub)}, mean_prob={meta_test.mean():.4f})")

runtime = time.time() - start_time
print(f"Exp 14 completed in {runtime:.1f}s")
