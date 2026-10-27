#!/usr/bin/env python3
"""Build and strictly verify Experiments 35 through 44 for Kaggle Playground S6E10."""
import os, sys, time, warnings
import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings('ignore')
start_time = time.time()

print("=" * 70)
print("Building Experiments 35 Through 44 (Today's 10 Submissions)")
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

def hblend(M, main, corr, mix=(0.3, 0.7)):
    """Two-sided positional sorting fusion."""
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

# 1. Load All Diverse Architectural Components
print("Loading diverse architectural components...")

# Anchor / Frontier: Yuvraj Lamba (0.96276)
yuvraj_raw = pd.read_csv('/tmp/kernel_outputs/yuvraj/submission.csv')['satisfaction'].values
r_yuvraj = pct(yuvraj_raw)

# Summit Baseline: Mayank Summit Composer (0.96213)
mayank_raw = pd.read_csv('/tmp/kernel_outputs/mayank/submission.csv')['satisfaction'].values
r_mayank = pct(mayank_raw)

# High OOF CV Anchor: Masum Najiama (CV 0.96221, LB 0.96187)
masum_raw = np.load('/tmp/kernel_outputs/masum/test_najiama.npy')
r_masum = pct(masum_raw)

# TabM Neural Network: Yekenot PyTabKit (OOF 0.96050)
tabm_raw = pd.read_csv('/tmp/kernel_outputs/yekenot_tabm/submission.csv')['satisfaction'].values
r_tabm = pct(tabm_raw)

# Goodpjw Stack245 Multi-Family Meta
goodpjw_raw = pd.read_csv('/tmp/kernel_outputs/goodpjw/submission_ours_245.csv')['satisfaction'].values
r_goodpjw = pct(goodpjw_raw)

# Exp 25: Route-Hamming & CC0 Original Lookup LightGBM (Our Route Leak Model)
exp25_raw = pd.read_csv('submissions/submission_v25.csv')['satisfaction'].values
r_exp25 = pct(exp25_raw)

# Exp 26: 13-Model Multi-Paradigm Meta-Stack
exp26_raw = pd.read_csv('submissions/submission_v26.csv')['satisfaction'].values
r_exp26 = pct(exp26_raw)

# TabPFN In-Context Model (te4 probability)
tabpfn_prob = pd.read_csv('/tmp/tabpfn_dataset/test_tabpfn_te4.csv')['pred'].values
r_tabpfn = pct(tabpfn_prob)

# Deep Tabular Models (Chris Deotte TFM)
realmlp_raw = np.load('/tmp/cdeotte_tfm/test_realmlp.npy')
r_realmlp = pct(realmlp_raw)

kumolarge_raw = np.load('/tmp/cdeotte_tfm/test_kumolarge.npy')
r_kumolarge = pct(kumolarge_raw)

print("All components loaded successfully.")

# Dictionary to store all candidate predictions
experiments = {}

# Exp 35: The New Frontier Direct Anchor (Yuvraj 0.96276 Synthesis Benchmark)
pred_35 = r_yuvraj.copy()
experiments[35] = (
    pred_35,
    "Exp 35: New Frontier Benchmark Anchor (0.96276+ Multi-Family GPU XGBoost & Foundation Synthesis)"
)

# Exp 36: TabM Neural Diversity Injection into Frontier Anchor
# 93% Yuvraj Frontier + 7% TabM PyTabKit (Deep tabular network with piecewise linear embeddings)
pred_36 = pct(0.93 * r_yuvraj + 0.07 * r_tabm)
experiments[36] = (
    pred_36,
    "Exp 36: TabM Neural Diversity Injection (93% Frontier Anchor + 7% Yekenot TabM PyTabKit)"
)

# Exp 37: Route-Leak Hybrid on the New Frontier
# 92% Yuvraj Frontier + 5% Exp 25 Route-LightGBM + 3% Exp 26 Meta-Stack
pred_37 = pct(0.92 * r_yuvraj + 0.05 * r_exp25 + 0.03 * r_exp26)
experiments[37] = (
    pred_37,
    "Exp 37: Route-Leak Hybrid Frontier (92% Frontier + 5% Exp 25 Route-LightGBM + 3% Exp 26 Meta-Stack)"
)

# Exp 38: Tri-Pillar Foundation Consensus
# 75% Yuvraj Frontier + 15% Mayank Summit + 10% Exp 26 13-Model Stack
pred_38 = pct(0.75 * r_yuvraj + 0.15 * r_mayank + 0.10 * r_exp26)
experiments[38] = (
    pred_38,
    "Exp 38: Tri-Pillar Foundation Consensus (75% Frontier + 15% Summit Composer + 10% 13-Model Stack)"
)

# Exp 39: Deep Tabular Neural Fusion (TabM + RealMLP + KumoLarge + Frontier)
# 84% Yuvraj Frontier + 6% TabM + 5% RealMLP + 5% KumoLarge
pred_39 = pct(0.84 * r_yuvraj + 0.06 * r_tabm + 0.05 * r_realmlp + 0.05 * r_kumolarge)
experiments[39] = (
    pred_39,
    "Exp 39: Deep Tabular Neural Fusion (84% Frontier + 6% TabM + 5% RealMLP + 5% KumoLarge)"
)

# Exp 40: Log-Odds / Logit Manifold Synthesis across Top 5 Paradigms
# Logit synthesis with bounded weights
z_syn = (
    0.55 * to_logit(r_yuvraj) +
    0.20 * to_logit(r_mayank) +
    0.10 * to_logit(r_masum) +
    0.08 * to_logit(r_exp25) +
    0.07 * to_logit(r_tabm)
)
pred_40 = pct(from_logit(z_syn))
experiments[40] = (
    pred_40,
    "Exp 40: Log-Odds Manifold Synthesis (55% Frontier + 20% Summit + 10% Najiama + 8% Route-LGB + 7% TabM)"
)

# Exp 41: Middle-Decile Layering on the 0.96276 Frontier
# Layering 5% TabPFN extreme probabilities into the 0.96276 rank
pred_41 = pct(0.95 * r_yuvraj + 0.05 * tabpfn_prob)
experiments[41] = (
    pred_41,
    "Exp 41: TabPFN-3.5 Middle-Decile Layering on Frontier (95% Frontier Rank + 5% TabPFN Probability)"
)

# Exp 42: Two-Sided Positional Nina h-blend on Frontier Quad
M42 = np.column_stack([r_yuvraj, r_mayank, r_goodpjw, r_exp25])
main_w = np.array([0.60, 0.20, 0.12, 0.08], dtype=np.float32)
corr_w = np.array([0.04, 0.01, -0.01, -0.04], dtype=np.float32)
pred_42 = pct(hblend(M42, main_w, corr_w, mix=(0.3, 0.7)))
experiments[42] = (
    pred_42,
    "Exp 42: Two-Sided Positional Nina h-blend (Frontier, Summit, Stack245, Route-LightGBM)"
)

# Exp 43: Grand Champion 6-Family Syndicate Ensemble
# 70% Yuvraj + 12% Mayank + 8% Goodpjw + 4% Exp 25 + 3% TabM + 3% TabPFN
pred_43 = pct(
    0.70 * r_yuvraj +
    0.12 * r_mayank +
    0.08 * r_goodpjw +
    0.04 * r_exp25 +
    0.03 * r_tabm +
    0.03 * r_tabpfn
)
experiments[43] = (
    pred_43,
    "Exp 43: Grand Champion 6-Family Syndicate (70% Frontier + 12% Summit + 8% Stack245 + 4% Route-LGB + 3% TabM + 3% TabPFN)"
)

# Exp 44: Temperature-Calibrated Frontier Sharpener (Conservative Final Peak)
# Temperature scaling in logit space on Exp 43 (tau = 1.03)
z_43 = to_logit(pred_43) * 1.03
pred_44 = pct(from_logit(z_43))
experiments[44] = (
    pred_44,
    "Exp 44: Temperature-Calibrated Frontier Sharpener (Logit Temp 1.03 on 6-Family Syndicate)"
)

# Save and strictly audit all 10 submissions
print("\n" + "=" * 70)
print("Saving and Validating Submissions 35 - 44:")
print("=" * 70)

for exp_id in range(35, 45):
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
    corr_vs_yuvraj = spearmanr(preds, r_yuvraj).statistic
    corr_vs_mayank = spearmanr(preds, r_mayank).statistic
    print(f"  v{exp_id}: rows={len(sub_df)}, mean={preds.mean():.4f}, std={preds.std():.4f} | corr(Yuvraj)={corr_vs_yuvraj:.5f}, corr(Mayank)={corr_vs_mayank:.5f}")

print(f"\nAll 10 submissions generated and verified in {time.time()-start_time:.1f}s")
