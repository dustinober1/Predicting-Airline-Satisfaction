#!/usr/bin/env python3
"""Build advanced ensembles blending our models with top verified foundation and stack models."""
import os, sys, warnings
import pandas as pd
import numpy as np
from scipy.stats import rankdata

warnings.filterwarnings('ignore')

def to_rank(probs):
    """Normalized average rank in [0, 1]."""
    return (rankdata(probs, method='average') - 0.5) / len(probs)

def to_logit(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))

def from_logit(z):
    return 1.0 / (1.0 + np.exp(-z))

DATA_DIR = 'data'
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), usecols=['id'])
ids = test['id'].values

# Load our predictions
our_lgb = pd.read_csv('submissions/submission_v11.csv')['satisfaction'].values
our_exp4 = pd.read_csv('submissions/submission_v4.csv')['satisfaction'].values

# Load public foundation / stack predictions
yekenot_realmlp = pd.read_csv('/tmp/kernel_outputs/yekenot/submission.csv')['satisfaction'].values
evgen_lgb = pd.read_csv('/tmp/kernel_outputs/evgen/submission.csv')['satisfaction'].values
yeashu_stack = pd.read_csv('/tmp/kernel_outputs/yeashu/submission.csv')['satisfaction'].values
goodpjw_stack = pd.read_csv('/tmp/kernel_outputs/goodpjw/submission.csv')['satisfaction'].values
mayank_summit = pd.read_csv('/tmp/kernel_outputs/mayank/submission.csv')['satisfaction'].values
rabari_stack = pd.read_csv('/tmp/rabari_output/submission.csv')['satisfaction'].values

print("Loaded all component predictions successfully.")

# Blend A (Exp 17): RealMLP (Neural) + Our GBDT (Trees)
# RealMLP is 0.983 correlated with summit, but from a neural architecture (PyTabKit)
rank_our_trees = 0.6 * to_rank(our_lgb) + 0.4 * to_rank(our_exp4)
v17_blend = 0.40 * to_rank(yekenot_realmlp) + 0.35 * to_rank(evgen_lgb) + 0.25 * rank_our_trees
sub_v17 = pd.DataFrame({'id': ids, 'satisfaction': v17_blend})
sub_v17.to_csv('submissions/submission_v17.csv', index=False)
print("Saved submissions/submission_v17.csv (RealMLP + GBDT Hybrid)")

# Blend B (Exp 18): High-Diversity Multi-Family Stack (GBDTs + RealMLP + 201-model Yeashu Stack)
v18_blend = 0.50 * to_rank(yeashu_stack) + 0.25 * to_rank(yekenot_realmlp) + 0.15 * to_rank(evgen_lgb) + 0.10 * to_rank(our_lgb)
sub_v18 = pd.DataFrame({'id': ids, 'satisfaction': v18_blend})
sub_v18.to_csv('submissions/submission_v18.csv', index=False)
print("Saved submissions/submission_v18.csv (Multi-Family 201-Stack Blend)")

# Blend C (Exp 19): Summit Composer (0.96210 LB Peak) + Diverse Trees & RealMLP
v19_blend = 0.65 * to_rank(mayank_summit) + 0.15 * to_rank(rabari_stack) + 0.10 * to_rank(yekenot_realmlp) + 0.10 * to_rank(our_lgb)
sub_v19 = pd.DataFrame({'id': ids, 'satisfaction': v19_blend})
sub_v19.to_csv('submissions/submission_v19.csv', index=False)
print("Saved submissions/submission_v19.csv (Summit Composer Peak Blend)")

# Blend D (Exp 20): Grand Champion Optimal Weighted Blend
v20_blend = (
    0.40 * to_rank(mayank_summit) +
    0.25 * to_rank(yeashu_stack) +
    0.15 * to_rank(rabari_stack) +
    0.10 * to_rank(yekenot_realmlp) +
    0.05 * to_rank(evgen_lgb) +
    0.05 * to_rank(our_lgb)
)
sub_v20 = pd.DataFrame({'id': ids, 'satisfaction': v20_blend})
sub_v20.to_csv('submissions/submission_v20.csv', index=False)
print("Saved submissions/submission_v20.csv (Grand Champion Final Blend)")
