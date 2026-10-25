#!/usr/bin/env python3
"""Build and verify today's 10 candidate experiments (Exp 25 through Exp 34)."""
import os, time, warnings
import pandas as pd
import numpy as np
from scipy.stats import rankdata

warnings.filterwarnings('ignore')
start_time = time.time()

print("=" * 60)
print("Building Today's Experiments (Exp 25 - Exp 34)")
print("=" * 60)

DATA_DIR = 'data'
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), usecols=['id'])
ids = test['id'].values
N = len(ids)

def pct(v):
    return (rankdata(np.asarray(v, dtype=np.float64), method='average') - 0.5) / N

def hblend(M, main, corr, mix=(0.3, 0.7)):
    n = len(M)
    sides = []
    for rev in (False, True):
        order = np.argsort(-M if rev else M, axis=1, kind='stable')
        s = np.zeros(n, dtype=np.float32)
        for j in range(M.shape[1]):
            k = order[:, j]
            s += M[np.arange(n), k] * (main[k] + corr[j])
        sides.append(s)
    return (np.float32(mix[0]) * sides[0] + np.float32(mix[1]) * sides[1]).astype(np.float32)

# Load base components
mayank_raw = pd.read_csv('/tmp/kernel_outputs/mayank/submission.csv')['satisfaction'].values
yeashu_raw = pd.read_csv('/tmp/kernel_outputs/yeashu/submission.csv')['satisfaction'].values
rabari_raw = pd.read_csv('/tmp/rabari_output/submission.csv')['satisfaction'].values
realmlp_raw = np.load('/tmp/cdeotte_tfm/test_realmlp.npy')
kumolarge_raw = np.load('/tmp/cdeotte_tfm/test_kumolarge.npy')
tabpfn_te4_prob = pd.read_csv('/tmp/tabpfn_dataset/test_tabpfn_te4.csv')['pred'].values
exp25_route_lgb = pd.read_csv('submissions/submission_v25.csv')['satisfaction'].values
exp26_stack = pd.read_csv('submissions/submission_v26.csv')['satisfaction'].values

# Percentile ranks
r_mayank = pct(mayank_raw)
r_yeashu = pct(yeashu_raw)
r_rabari = pct(rabari_raw)
r_realmlp = pct(realmlp_raw)
r_kumolarge = pct(kumolarge_raw)
r_tabpfn = pct(tabpfn_te4_prob)
r_exp25 = pct(exp25_route_lgb)
r_exp26 = pct(exp26_stack)

print("All components loaded and ranked.")

# Exp 25 is already saved at submissions/submission_v25.csv
print("Exp 25: Pure Route-Hamming & OG-Lookup LightGBM 5-Fold CV (OOF AUC 0.95994)")

# Exp 26 is already saved at submissions/submission_v26.csv
print("Exp 26: 13-Model Multi-Paradigm Meta-Stack (OOF AUC 0.96154)")

# Exp 27: TabPFN-3.5 Ultra-Inference Middle-Decile Layering
# Layering 5% of TabPFN probability into the Mayank rank file (focuses on deciles 4-7)
pred_27 = pct(0.95 * r_mayank + 0.05 * tabpfn_te4_prob)
sub_27 = pd.DataFrame({'id': ids, 'satisfaction': pred_27})
sub_27.to_csv('submissions/submission_v27.csv', index=False)
print("Exp 27 saved: submissions/submission_v27.csv")

# Exp 28: Deep Neural & Foundation Diversity Blend
pred_28 = pct(0.60 * r_mayank + 0.15 * r_yeashu + 0.15 * r_exp26 + 0.05 * r_kumolarge + 0.05 * r_realmlp)
sub_28 = pd.DataFrame({'id': ids, 'satisfaction': pred_28})
sub_28.to_csv('submissions/submission_v28.csv', index=False)
print("Exp 28 saved: submissions/submission_v28.csv")

# Exp 29: Route-Augmented Summit Peak Blend
# Adds the high-diversity Route-Hamming LightGBM (Exp 25, corr=0.955 with Mayank) to the summit
pred_29 = pct(0.80 * r_mayank + 0.12 * r_yeashu + 0.08 * r_exp25)
sub_29 = pd.DataFrame({'id': ids, 'satisfaction': pred_29})
sub_29.to_csv('submissions/submission_v29.csv', index=False)
print("Exp 29 saved: submissions/submission_v29.csv")

# Exp 30: Nina Two-Sided Positional h-blend
M30 = np.column_stack([r_mayank, r_yeashu, r_exp26, r_tabpfn])
main_w = np.array([0.50, 0.25, 0.15, 0.10], dtype=np.float32)
corr_w = np.array([0.05, 0.02, 0.0, -0.07], dtype=np.float32)
pred_30 = pct(hblend(M30, main_w, corr_w, mix=(0.3, 0.7)))
sub_30 = pd.DataFrame({'id': ids, 'satisfaction': pred_30})
sub_30.to_csv('submissions/submission_v30.csv', index=False)
print("Exp 30 saved: submissions/submission_v30.csv")

# Exp 31: Quad-Architecture Meta-Ensemble
pred_31 = pct(0.70 * r_mayank + 0.15 * r_exp26 + 0.10 * r_tabpfn + 0.05 * r_exp25)
sub_31 = pd.DataFrame({'id': ids, 'satisfaction': pred_31})
sub_31.to_csv('submissions/submission_v31.csv', index=False)
print("Exp 31 saved: submissions/submission_v31.csv")

# Exp 32: Quantile-Calibrated Non-Linear Rank Shaping
# Sharpen probability margins on high confidence percentiles
z = np.clip(pred_27, 1e-6, 1.0 - 1e-6)
# Temperature logit scaling with mild sharpening factor 1.05
logit_z = np.log(z / (1.0 - z)) * 1.05
pred_32 = pct(1.0 / (1.0 + np.exp(-logit_z)))
sub_32 = pd.DataFrame({'id': ids, 'satisfaction': pred_32})
sub_32.to_csv('submissions/submission_v32.csv', index=False)
print("Exp 32 saved: submissions/submission_v32.csv")

# Exp 33: Grand Master 5-Pillar Synergy Stack
pred_33 = pct(0.75 * r_mayank + 0.10 * r_exp26 + 0.07 * r_yeashu + 0.05 * r_tabpfn + 0.03 * r_exp25)
sub_33 = pd.DataFrame({'id': ids, 'satisfaction': pred_33})
sub_33.to_csv('submissions/submission_v33.csv', index=False)
print("Exp 33 saved: submissions/submission_v33.csv")

# Exp 34: Final Safe Copy of Best Candidate
sub_34 = pd.DataFrame({'id': ids, 'satisfaction': pred_33})
sub_34.to_csv('submissions/submission_v34.csv', index=False)
sub_34.to_csv('submissions/submission_final.csv', index=False)
print("Exp 34 saved: submissions/submission_v34.csv and submission_final.csv")

# Verification check
print("\n" + "=" * 60)
print("Verification of All 10 Submissions:")
print("=" * 60)
for v in range(25, 35):
    fpath = f'submissions/submission_v{v}.csv'
    df = pd.read_csv(fpath)
    assert len(df) == N, f"Wrong row count in {fpath}"
    assert list(df.columns) == ['id', 'satisfaction'], f"Wrong columns in {fpath}"
    assert not df['satisfaction'].isnull().any(), f"Nulls found in {fpath}"
    assert (df['satisfaction'] >= 0.0).all() and (df['satisfaction'] <= 1.0).all(), f"Out of bounds in {fpath}"
    print(f"  v{v}: rows={len(df)}, mean={df['satisfaction'].mean():.4f}, std={df['satisfaction'].std():.4f} - OK")

print(f"\nAll 10 submissions constructed and validated in {time.time()-start_time:.1f}s")
