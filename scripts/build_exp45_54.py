#!/usr/bin/env python3
"""Build and strictly verify Experiments 45 through 54 for Kaggle Playground S6E10."""
import os, sys, time, warnings
import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings('ignore')
start_time = time.time()

print("=" * 70)
print("Building Experiments 45 Through 54 (Today's 10 Kaggle Submissions)")
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

# Anchor / Frontier: Goodpjw2008 Meta-Stack Probe E (verified LB 0.96215)
goodpjw_e_raw = pd.read_csv('/tmp/goodpjw_meta/submission.csv')['satisfaction'].values
r_goodpjw_e = pct(goodpjw_e_raw)

# Pinned Best Summit Baseline: Mayank + Yeashu Rank Blend (our verified LB 0.96213)
our_best_raw = pd.read_csv('submissions/submission_final.csv')['satisfaction'].values
r_our_best = pct(our_best_raw)

# Orange Lab Independent Blend: Secrets V14 + Orange V15 (verified LB 0.96213)
orange_raw = pd.read_csv('/tmp/orange/submission.csv')['satisfaction'].values
r_orange = pct(orange_raw)

# Areeb Ali Super-Rank Mean across 7 Top Public Summits (Daoviet, Goodpjw 3x, Mayank, Nina, Rabari)
areebbb_r09_raw = pd.read_csv('/tmp/areebbb/r09_rankmean_top.csv')['satisfaction'].values
r_areebbb_r09 = pct(areebbb_r09_raw)

# Areeb Ali Inverse-TabPFN Sharpener (105% Goodpjw Meta - 5% TabPFN-3.5)
areebbb_p08_raw = pd.read_csv('/tmp/areebbb/p08_META105_tpfn_m05.csv')['satisfaction'].values
r_areebbb_p08 = pct(areebbb_p08_raw)

# Anhad Mahajan 10-Fold CatBoost + XGBoost Orthogonal Tree Model
anhad_rank_raw = pd.read_csv('/tmp/anhad/submission_rank_percentile.csv')['satisfaction'].values
r_anhad_rank = pct(anhad_rank_raw)

# Exp 25: Route-Hamming & CC0 Original Lookup LightGBM (Our Domain Route Leak Model, CV 0.95994, LB 0.95951)
route_v25_raw = pd.read_csv('submissions/submission_v25.csv')['satisfaction'].values
r_route_v25 = pct(route_v25_raw)

# TabM Neural Network: Yekenot PyTabKit Piecewise Linear Deep Tabular Model (OOF 0.96050)
tabm_raw = pd.read_csv('/tmp/kernel_outputs/yekenot_tabm/submission.csv')['satisfaction'].values
r_tabm = pct(tabm_raw)

# Mayank Summit Composer V54 (LB 0.96210)
mayank_raw = pd.read_csv('/tmp/kernel_outputs/mayank/submission.csv')['satisfaction'].values
r_mayank = pct(mayank_raw)

# Rabari Rank 10 (LB 0.96209)
rabari_raw = pd.read_csv('/tmp/rabari/submission.csv')['satisfaction'].values
r_rabari = pct(rabari_raw)

print("All components loaded successfully.")

# Dictionary to store all candidate predictions: {exp_id: (pred_array, description_str)}
experiments = {}

# Exp 45: New Frontier Benchmark Anchor (Goodpjw Meta-Stack Probe E 0.96215)
pred_45 = r_goodpjw_e.copy()
experiments[45] = (
    pred_45,
    "Exp 45: New Frontier Benchmark Anchor (Goodpjw Meta-Stack Probe E 0.96215)"
)

# Exp 46: Top-Tier Dual-Summit Consensus (50% Goodpjw 0.96215 + 50% Summit 0.96213)
pred_46 = pct(0.50 * r_goodpjw_e + 0.50 * r_our_best)
experiments[46] = (
    pred_46,
    "Exp 46: Top-Tier Dual-Summit Consensus (50% Goodpjw 0.96215 + 50% Summit 0.96213)"
)

# Exp 47: Tri-Apex Summit Fusion (40% Goodpjw 0.96215 + 30% Summit + 30% Orange Lab 0.96213)
pred_47 = pct(0.40 * r_goodpjw_e + 0.30 * r_our_best + 0.30 * r_orange)
experiments[47] = (
    pred_47,
    "Exp 47: Tri-Apex Summit Fusion (40% Goodpjw 0.96215 + 30% Summit + 30% Orange Lab 0.96213)"
)

# Exp 48: Super-Rank Ensemble of Top-7 Public Summits (Daoviet + Goodpjw + Mayank + Nina + Rabari)
pred_48 = r_areebbb_r09.copy()
experiments[48] = (
    pred_48,
    "Exp 48: Super-Rank Ensemble of Top-7 Public Summits (Daoviet + Goodpjw + Mayank + Nina + Rabari)"
)

# Exp 49: Inverse-TabPFN Extreme Boundary Sharpener (105% Goodpjw Meta - 5% TabPFN-3.5)
pred_49 = r_areebbb_p08.copy()
experiments[49] = (
    pred_49,
    "Exp 49: Inverse-TabPFN Extreme Boundary Sharpener (105% Goodpjw Meta - 5% TabPFN-3.5)"
)

# Exp 50: Orthogonal 10-Fold Tree Diversity Injection (92% Goodpjw 0.96215 + 8% Anhad 10-Fold GBDT)
pred_50 = pct(0.92 * r_goodpjw_e + 0.08 * r_anhad_rank)
experiments[50] = (
    pred_50,
    "Exp 50: Orthogonal 10-Fold Tree Diversity Injection (92% Goodpjw 0.96215 + 8% Anhad 10-Fold GBDT)"
)

# Exp 51: Deep Neural (TabM) + Route-Leak Regularization (90% Goodpjw + 5% TabM + 5% Route-LGB)
pred_51 = pct(0.90 * r_goodpjw_e + 0.05 * r_tabm + 0.05 * r_route_v25)
experiments[51] = (
    pred_51,
    "Exp 51: Deep Neural (TabM) + Route-Leak Regularization (90% Goodpjw + 5% TabM + 5% Route-LGB)"
)

# Exp 52: Log-Odds Manifold Synthesis (50% Goodpjw + 25% Summit + 15% Orange + 5% Route + 5% TabM)
z_syn = (
    0.50 * to_logit(r_goodpjw_e) +
    0.25 * to_logit(r_our_best) +
    0.15 * to_logit(r_orange) +
    0.05 * to_logit(r_route_v25) +
    0.05 * to_logit(r_tabm)
)
pred_52 = pct(from_logit(z_syn))
experiments[52] = (
    pred_52,
    "Exp 52: Log-Odds Manifold Synthesis (50% Goodpjw + 25% Summit + 15% Orange + 5% Route + 5% TabM)"
)

# Exp 53: Two-Sided Positional Nina h-blend (Goodpjw 0.96215, Summit 0.96213, Orange 0.96213, Top-7)
M53 = np.column_stack([r_goodpjw_e, r_our_best, r_orange, r_areebbb_r09])
main_w53 = np.array([0.40, 0.25, 0.20, 0.15], dtype=np.float32)
corr_w53 = np.array([0.03, 0.01, -0.01, -0.03], dtype=np.float32)
pred_53 = pct(hblend(M53, main_w53, corr_w53, mix=(0.3, 0.7)))
experiments[53] = (
    pred_53,
    "Exp 53: Two-Sided Positional Nina h-blend (Goodpjw 0.96215, Summit 0.96213, Orange 0.96213, Top-7)"
)

# Exp 54: Grand Apex 7-Pillar Consortium (Goodpjw, Summit, Orange, Top-7, 10-Fold GBDT, Route-Leak, TabM)
pred_54 = pct(
    0.45 * r_goodpjw_e +
    0.25 * r_our_best +
    0.12 * r_orange +
    0.08 * r_areebbb_r09 +
    0.04 * r_anhad_rank +
    0.03 * r_route_v25 +
    0.03 * r_tabm
)
experiments[54] = (
    pred_54,
    "Exp 54: Grand Apex 7-Pillar Consortium (Goodpjw, Summit, Orange, Top-7, 10-Fold GBDT, Route-Leak, TabM)"
)

# Save and strictly audit all 10 submissions
print("\n" + "=" * 70)
print("Saving and Validating Submissions 45 - 54:")
print("=" * 70)

for exp_id in range(45, 55):
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
    corr_vs_goodpjw = spearmanr(preds, r_goodpjw_e).statistic
    corr_vs_our_best = spearmanr(preds, r_our_best).statistic
    print(f"  v{exp_id}: rows={len(sub_df)}, mean={preds.mean():.4f}, std={preds.std():.4f} | corr(Goodpjw)={corr_vs_goodpjw:.5f}, corr(OurBest)={corr_vs_our_best:.5f}")

print(f"\nAll 10 submissions generated and verified in {time.time()-start_time:.1f}s")
