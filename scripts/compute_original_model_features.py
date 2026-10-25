#!/usr/bin/env python3
"""Compute original model logits and lookup table features for train and test datasets."""
import os, sys, time, warnings
import pandas as pd
import numpy as np

sys.path.append('/tmp/og_data')
from original_model import score

warnings.filterwarnings('ignore')
start_time = time.time()

print("=" * 60)
print("Computing Original Model Logits & Lookup Features")
print("=" * 60)

DATA_DIR = 'data'
out_dir = 'data/features'
os.makedirs(out_dir, exist_ok=True)

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print("Scoring original models on train...")
t0 = time.time()
ogp_train = score(train)
print(f"Train scored in {time.time()-t0:.1f}s, shape: {ogp_train.shape}")

print("Scoring original models on test...")
t0 = time.time()
ogp_test = score(test)
print(f"Test scored in {time.time()-t0:.1f}s, shape: {ogp_test.shape}")

# Also compute the lookup features from lookup_long.csv
print("Computing lookup rate features...")
L = pd.read_csv('/tmp/og_data/lookup_long.csv', dtype={"value": str}, keep_default_na=False)
for df, name in [(train, 'train'), (test, 'test')]:
    lookup_df = pd.DataFrame(index=df.index)
    for f, t in L[L.feature.isin(df.columns)].groupby("feature"):
        lookup_df["orig_rate__" + f] = df[f].astype(str).fillna("nan").map(dict(zip(t.value, t.rate))).astype('float32')
    # Save lookup dataframe
    lookup_df.to_csv(os.path.join(out_dir, f'lookup_{name}.csv'), index=False)
    print(f"Computed lookup_{name}.csv, columns: {lookup_df.shape[1]}")

ogp_train.to_csv(os.path.join(out_dir, 'ogp_train.csv'), index=False)
ogp_test.to_csv(os.path.join(out_dir, 'ogp_test.csv'), index=False)

print(f"All features computed and saved in {time.time()-start_time:.1f}s")
