#!/usr/bin/env python3
"""Experiment 26: 13-Model Multi-Paradigm Meta-Stack (TFMs + Route LightGBM + CatBoost + TabPFN)"""
import os, time, warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
start_time = time.time()

print("=" * 60)
print("Experiment 26: 13-Model Multi-Paradigm Meta-Stack")
print("=" * 60)

DATA_DIR = 'data'
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), usecols=['id'])
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), usecols=['satisfaction'])
y = train['satisfaction'].astype(int).values

models = ['tabpfn35', 'kumolarge', 'kumosmall', 'realmlp', 'xgboost', 'tabfm', 'exaone', 'limix2', 'causilo', 'tabicl2']
X_oof = []
X_test = []

for m in models:
    oof = np.load(f'/tmp/cdeotte_tfm/oof_{m}.npy').astype(np.float64)
    tst = np.load(f'/tmp/cdeotte_tfm/test_{m}.npy').astype(np.float64)
    oof = np.clip(oof, 1e-6, 1.0 - 1e-6)
    tst = np.clip(tst, 1e-6, 1.0 - 1e-6)
    X_oof.append(np.log(oof / (1.0 - oof)))
    X_test.append(np.log(tst / (1.0 - tst)))

# 11. Our Route-LightGBM Exp 25
r_lgb_oof = np.load('models_oof/oof_exp25_route_lgb.npy').astype(np.float64)
r_lgb_tst = np.load('models_oof/test_exp25_route_lgb.npy').astype(np.float64)
r_lgb_oof = np.clip(r_lgb_oof, 1e-6, 1.0 - 1e-6)
r_lgb_tst = np.clip(r_lgb_tst, 1e-6, 1.0 - 1e-6)
X_oof.append(np.log(r_lgb_oof / (1.0 - r_lgb_oof)))
X_test.append(np.log(r_lgb_tst / (1.0 - r_lgb_tst)))
models.append('route_lgb_exp25')

# 12. Our CatBoost Exp 12
cb_oof = np.load('models_oof/oof_exp12_cb.npy').astype(np.float64)
cb_tst = np.load('models_oof/test_exp12_cb.npy').astype(np.float64)
cb_oof = np.clip(cb_oof, 1e-6, 1.0 - 1e-6)
cb_tst = np.clip(cb_tst, 1e-6, 1.0 - 1e-6)
X_oof.append(np.log(cb_oof / (1.0 - cb_oof)))
X_test.append(np.log(cb_tst / (1.0 - cb_tst)))
models.append('catboost_exp12')

# 13. Goodpjw TabPFN-te4
te4_oof = pd.read_csv('/tmp/tabpfn_dataset/oof_tabpfn_te4.csv')['pred'].values.astype(np.float64)
te4_tst = pd.read_csv('/tmp/tabpfn_dataset/test_tabpfn_te4.csv')['pred'].values.astype(np.float64)
te4_oof = np.clip(te4_oof, 1e-6, 1.0 - 1e-6)
te4_tst = np.clip(te4_tst, 1.0 - 1e-6, 1e-6) if False else np.clip(te4_tst, 1e-6, 1.0 - 1e-6)
X_oof.append(np.log(te4_oof / (1.0 - te4_oof)))
X_test.append(np.log(te4_tst / (1.0 - te4_tst)))
models.append('tabpfn_te4')

X_oof = np.column_stack(X_oof)
X_test = np.column_stack(X_test)

print(f"Constructed feature matrices: X_oof {X_oof.shape}, X_test {X_test.shape}")

# Train 5-fold regularized meta-learner (C=0.005)
C = 0.005
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y))
test_preds = np.zeros(len(test))
coefs = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_oof, y), 1):
    lr = LogisticRegression(C=C, max_iter=1000)
    lr.fit(X_oof[tr_idx], y[tr_idx])
    oof_preds[val_idx] = lr.predict_proba(X_oof[val_idx])[:, 1]
    test_preds += lr.predict_proba(X_test)[:, 1] / 5.0
    coefs.append(lr.coef_[0])
    fold_auc = roc_auc_score(y[val_idx], oof_preds[val_idx])
    print(f"  Fold {fold} Meta AUC: {fold_auc:.6f}")

overall_auc = roc_auc_score(y, oof_preds)
print("\n" + "=" * 60)
print(f"Experiment 26 Meta-Stack OOF ROC-AUC: {overall_auc:.6f}")
print("=" * 60)

mean_coefs = np.mean(coefs, axis=0)
imp_df = pd.DataFrame({'model': models, 'mean_weight': mean_coefs}).sort_values('mean_weight', ascending=False)
print("\nMeta-Model Weights:")
print(imp_df.to_string(index=False))

# Save OOF and test predictions
os.makedirs('models_oof', exist_ok=True)
np.save('models_oof/oof_exp26_stack.npy', oof_preds)
np.save('models_oof/test_exp26_stack.npy', test_preds)

# Save submission
os.makedirs('submissions', exist_ok=True)
sub_path = 'submissions/submission_v26.csv'
sub = pd.DataFrame({'id': test['id'], 'satisfaction': test_preds})
sub.to_csv(sub_path, index=False)
print(f"\nSaved submission to {sub_path} (mean_prob: {test_preds.mean():.4f})")
print(f"Exp 26 completed in {time.time()-start_time:.1f}s")
