#!/usr/bin/env python3
"""Create ultra-refined peak blends of the top-performing solutions."""
import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata

pct = lambda v: (rankdata(np.asarray(v, dtype=np.float64), method='average') - 0.5) / len(v)

test = pd.read_csv('data/sample_submission.csv')
ids = test['id'].values

mayank = pct(pd.read_csv('/tmp/kernel_outputs/mayank/submission.csv')['satisfaction'].values)
yeashu = pct(pd.read_csv('/tmp/kernel_outputs/yeashu/submission.csv')['satisfaction'].values)
rabari = pct(pd.read_csv('/tmp/rabari_output/submission.csv')['satisfaction'].values)
our_stack = pct(pd.read_csv('submissions/submission_v14.csv')['satisfaction'].values)

# Blend 22: Ultra-tight peak blend (85% Mayank 0.96213 + 15% Yeashu 0.96210)
# A subtle 15% regularization from the 201-model stack to reduce individual sample noise
blend_22 = pct(0.85 * mayank + 0.15 * yeashu)
sub_22 = pd.DataFrame({'id': ids, 'satisfaction': blend_22})
sub_22.to_csv('submissions/submission_v22.csv', index=False)
print("Saved submissions/submission_v22.csv")

# Blend 23: Tri-Peak Micro-Blend (75% Mayank + 15% Yeashu + 8% Rabari + 2% Our GBDT Stack)
blend_23 = pct(0.75 * mayank + 0.15 * yeashu + 0.08 * rabari + 0.02 * our_stack)
sub_23 = pd.DataFrame({'id': ids, 'satisfaction': blend_23})
sub_23.to_csv('submissions/submission_v23.csv', index=False)
print("Saved submissions/submission_v23.csv")
