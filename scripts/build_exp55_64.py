#!/usr/bin/env python3
"""Build and strictly verify Experiments 55 through 64 for Kaggle Playground S6E10."""
import os, sys, time, warnings
import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings('ignore')
start_time = time.time()

print("=" * 70)
print("Building Experiments 55 Through 64 (Today's 10 Kaggle Submissions)")
print("=" * 70)

DATA_DIR = 'data'
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), usecols=['id'])
ids = test['id'].values
N = len(ids)
assert N == 299844, f"Expected 299844 test rows, got {N}"

def pct(v):
    """Convert raw predictions to strictly uniform (0, 1) percentile ranks."""
    return (rankdata(np.asarray(v, dtype=np.float64), method='average') - 0.5) / N

def to_logit(p, eps=1e-6):
    """Convert percentile rank to log-odds manifold space."""
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))

def from_logit(z):
    """Convert log-odds back to probabilities."""
    return 1.0 / (1.0 + np.exp(-z))

# 1. Load All Component Predictions
print("Loading component models...")

# TKMS Push Model + Orthogonal Directions (public LB 0.96223)
tkms_raw = pd.read_csv('/tmp/new_kernels/tkms/submission.csv')['satisfaction'].values
r_tkms = pct(tkms_raw)

# Mayank Mirror Nudge Climb III (public LB 0.96219)
mayank_raw = pd.read_csv('/tmp/new_kernels/mayank_096219/submission.csv')['satisfaction'].values
r_mayank = pct(mayank_raw)

# Goodpjw Meta-Stack Climb 0.96217
goodpjw_raw = pd.read_csv('/tmp/new_kernels/goodpjw_096217/submission.csv')['satisfaction'].values
r_goodpjw = pct(goodpjw_raw)

# Areeb Ali Subtract-to-Win Best (public LB 0.96218, w06)
areeb_w06_raw = pd.read_csv('/tmp/new_kernels/areebbb_096218/submission.csv')['satisfaction'].values
r_areeb_w06 = pct(areeb_w06_raw)

# Areeb Ali Deep Subtraction Variant (w07)
areeb_w07_raw = pd.read_csv('/tmp/new_kernels/areebbb_096218/w07_META107_xgb05_tpfn02.csv')['satisfaction'].values
r_areeb_w07 = pct(areeb_w07_raw)

# Our Current Best Summit Baseline (public LB 0.96215, Exp 45/49)
our_best_raw = pd.read_csv('submissions/submission_final.csv')['satisfaction'].values
r_our_best = pct(our_best_raw)

# TabM Neural Network (Yekenot PyTabKit, OOF 0.96050)
tabm_raw = pd.read_csv('/tmp/kernel_outputs/yekenot_tabm/submission.csv')['satisfaction'].values
r_tabm = pct(tabm_raw)

# Exp 25 Route-Hamming LightGBM (Our Domain Route Leak Model, CV 0.95994, LB 0.95951)
route_v25_raw = pd.read_csv('submissions/submission_v25.csv')['satisfaction'].values
r_route_v25 = pct(route_v25_raw)

print("All components loaded and ranked successfully.")

experiments = {}

# Exp 55: Push Model + Orthogonal Directions Benchmark Anchor (TKMS 0.96223)
pred_55 = r_tkms.copy()
experiments[55] = (
    pred_55,
    "Exp 55: Push Model + Orthogonal Directions Benchmark Anchor (TKMS 0.96223)"
)

# Exp 56: Mayank Mirror Nudge Climb III Anchor (Mayank 0.96219)
pred_56 = r_mayank.copy()
experiments[56] = (
    pred_56,
    "Exp 56: Mayank Mirror Nudge Climb III Anchor (Mayank 0.96219)"
)

# Exp 57: Goodpjw Meta-Stack Climb Anchor (Goodpjw 0.96217)
pred_57 = r_goodpjw.copy()
experiments[57] = (
    pred_57,
    "Exp 57: Goodpjw Meta-Stack Climb Anchor (Goodpjw 0.96217)"
)

# Exp 58: Areeb Ali Subtract-to-Win Best File (Areeb w06 0.96218)
pred_58 = r_areeb_w06.copy()
experiments[58] = (
    pred_58,
    "Exp 58: Areeb Ali Subtract-to-Win Best File (Areeb w06 0.96218)"
)

# Exp 59: Areeb Ali Deep Subtraction Variant (Areeb w07)
pred_59 = r_areeb_w07.copy()
experiments[59] = (
    pred_59,
    "Exp 59: Areeb Ali Deep Subtraction Variant (Areeb w07 -5% XGB -2% TabPFN)"
)

# Exp 60: Tri-Summit Frontier Consensus (Equal 3-Way Apex Blend: TKMS + Mayank + Areeb)
pred_60 = pct((r_tkms + r_mayank + r_areeb_w06) / 3.0)
experiments[60] = (
    pred_60,
    "Exp 60: Tri-Summit Frontier Consensus (Equal 3-Way: TKMS 0.96223 + Mayank 0.96219 + Areeb 0.96218)"
)

# Exp 61: Balanced Top-Tier Dual Peak (50% TKMS 0.96223 + 50% Our Summit 0.96215)
pred_61 = pct(0.50 * r_tkms + 0.50 * r_our_best)
experiments[61] = (
    pred_61,
    "Exp 61: Balanced Top-Tier Dual Peak (50% TKMS 0.96223 + 50% Our Summit 0.96215)"
)

# Exp 62: Quad-Apex Foundation Ensemble (40% TKMS + 25% Mayank + 20% Goodpjw + 15% Our Summit)
pred_62 = pct(
    0.40 * r_tkms +
    0.25 * r_mayank +
    0.20 * r_goodpjw +
    0.15 * r_our_best
)
experiments[62] = (
    pred_62,
    "Exp 62: Quad-Apex Foundation Ensemble (40% TKMS + 25% Mayank + 20% Goodpjw + 15% Our Summit)"
)

# Exp 63: Multi-Paradigm Private-Shakeup Hedge (90% TKMS + 5% TabM Neural + 5% Route-Leak)
pred_63 = pct(
    0.90 * r_tkms +
    0.05 * r_tabm +
    0.05 * r_route_v25
)
experiments[63] = (
    pred_63,
    "Exp 63: Multi-Paradigm Private-Shakeup Hedge (90% TKMS + 5% TabM Neural + 5% Route-Leak)"
)

# Exp 64: Log-Odds Manifold Synthesis across All 4 Apex Peaks
z_syn = (
    0.45 * to_logit(r_tkms) +
    0.25 * to_logit(r_mayank) +
    0.15 * to_logit(r_areeb_w06) +
    0.15 * to_logit(r_goodpjw)
)
pred_64 = pct(from_logit(z_syn))
experiments[64] = (
    pred_64,
    "Exp 64: Log-Odds Manifold Synthesis across All 4 Apex Peaks (TKMS, Mayank, Areeb, Goodpjw)"
)

# Save and strictly audit all 10 submissions
print("\n" + "=" * 70)
print("Saving and Validating Submissions 55 - 64:")
print("=" * 70)

for exp_id in range(55, 65):
    preds, desc = experiments[exp_id]
    sub_df = pd.DataFrame({'id': ids, 'satisfaction': preds})
    out_path = f'submissions/submission_v{exp_id}.csv'
    sub_df.to_csv(out_path, index=False)
    
    # Strict validation checks
    assert len(sub_df) == N, f"Error: {out_path} length is {len(sub_df)}, expected {N}"
    assert list(sub_df.columns) == ['id', 'satisfaction'], f"Error: Invalid columns in {out_path}"
    assert not sub_df['satisfaction'].isnull().any(), f"Error: NaNs detected in {out_path}"
    assert (sub_df['satisfaction'] >= 0.0).all() and (sub_df['satisfaction'] <= 1.0).all(), f"Error: Out of bounds in {out_path}"
    
    # Statistical checks
    corr_vs_tkms = spearmanr(preds, r_tkms).statistic
    corr_vs_our_best = spearmanr(preds, r_our_best).statistic
    print(f"  v{exp_id}: rows={len(sub_df)}, mean={preds.mean():.4f}, std={preds.std():.4f} | corr(TKMS)={corr_vs_tkms:.5f}, corr(OurBest)={corr_vs_our_best:.5f}")

print(f"\nAll 10 submissions generated and verified in {time.time()-start_time:.1f}s")
