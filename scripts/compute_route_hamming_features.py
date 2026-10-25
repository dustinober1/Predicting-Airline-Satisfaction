#!/usr/bin/env python3
"""Compute nearest-original route Hamming features for train and test datasets."""
import os, time, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
start_time = time.time()

print("=" * 60)
print("Computing Nearest-Original Route Hamming Features")
print("=" * 60)

DATA_DIR = 'data'
orig_path = '/tmp/og_data/original_aligned.csv'
train_path = os.path.join(DATA_DIR, 'train.csv')
test_path = os.path.join(DATA_DIR, 'test.csv')

orig = pd.read_csv(orig_path)
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

CATS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
FEATURES_RAW = [c for c in orig.columns if c not in ['id', 'satisfaction', 'is_original']]
CMP_COLS = [c for c in FEATURES_RAW if c != 'Flight Distance']
codes = {c: {v: i for i, v in enumerate(sorted(orig[c].unique()))} for c in CATS}

def encode(df):
    X = df[CMP_COLS].copy()
    for c in CATS:
        X[c] = X[c].map(codes[c])
    return X.fillna(-1).round().astype(np.int16).to_numpy()

X_orig = encode(orig)
y_orig = orig['satisfaction'].astype(np.float32).to_numpy()
route_orig = orig['Flight Distance'].to_numpy()
by_route = pd.Series(route_orig).groupby(route_orig).indices

def nearest_originals(X, routes, alphas=(0.5, 1, 2)):
    names = ['dist', 'n_closest', 'label_closest', 'label_within1'] + [f'label_w{a}' for a in alphas] + ['density']
    out = np.full((len(X), len(names)), np.nan, dtype=np.float32)
    for r, rows in pd.Series(routes).groupby(routes).indices.items():
        ref = by_route.get(r)
        if ref is None:
            continue
        y_ref = y_orig[ref]
        for s in range(0, len(rows), 2048):
            q = rows[s:s + 2048]
            d = (X[q, None, :] != X_orig[None, ref, :]).sum(-1).astype(np.float32)
            d0 = d.min(1, keepdims=True)
            closest, within1 = d == d0, d <= d0 + 1
            cols = [d0[:, 0], closest.sum(1), closest @ y_ref / closest.sum(1), within1 @ y_ref / within1.sum(1)]
            for a in alphas:
                w = np.exp(-a * (d - d0))
                cols.append(w @ y_ref / w.sum(1))
            cols.append(np.exp(-d).sum(1))
            out[q] = np.column_stack(cols)
    return pd.DataFrame(out, columns=[f'par_{n}' for n in names])

print("Encoding and computing features for train...")
t0 = time.time()
par_train = nearest_originals(encode(train), train['Flight Distance'].to_numpy())
print(f"Train computed in {time.time()-t0:.1f}s, shape: {par_train.shape}")

print("Encoding and computing features for test...")
t0 = time.time()
par_test = nearest_originals(encode(test), test['Flight Distance'].to_numpy())
print(f"Test computed in {time.time()-t0:.1f}s, shape: {par_test.shape}")

# Save features as npz and csv
out_dir = 'data/features'
os.makedirs(out_dir, exist_ok=True)
np.savez_compressed(os.path.join(out_dir, 'par_features.npz'),
                    train=par_train.values,
                    test=par_test.values,
                    columns=par_train.columns.values)
par_train.to_csv(os.path.join(out_dir, 'par_train.csv'), index=False)
par_test.to_csv(os.path.join(out_dir, 'par_test.csv'), index=False)

print(f"Saved features to {out_dir}/par_features.npz and csv files")
print(f"Total time: {time.time()-start_time:.1f}s")
