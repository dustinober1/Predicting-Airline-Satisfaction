# Tomorrow Plan: 10 Experiments After Daily Reset (Playground S6E10)
Created: 2026-10-22 | Deadline: ~Oct 31, 2026 (9 days remaining)
Competition: playground-series-s6e10 | Leaderboard leading: 0.96370 | Our best: Exp 9 (v9) val 0.9309 / AUC 0.9683

## Tomorrow's 10 Experiments
1. Metric-Specific Tuning — Optimize for AUC (eval_metric='auc', scale_pos_weight)
2. Deeper Feature Engineering — Interaction Expansion v2 (triple interactions, log transforms)
3. Target Encoding Refinement — Leave-One-Out Encoding per fold
4. Neural Network Baseline — Simple MLP (256-128-64) using expanded features
5. CatBoost / LightGBM Installation & Test (install, train native categorical)
6. Hybrid Blend — Neural + Tree Ensemble (MLP + best XGB + RF meta-learner)
7. 10-Fold CV — More Robust Meta-Features (10-fold meta-training)
8. Feature Dropout Based on AUC Importance (drop bottom 20% features)
9. AUC-Optimized Hyperparameter Grid (max_depth, lr, subsample, reg_alpha, reg_lambda)
10. Final Best-of-All Blend — AUC-weighted average of best variants; submit before deadline

Constraints: Kernel URL: https://www.kaggle.com/code/dustinober/playground-s6e10-exp-6-10-gpu (GPU enabled); data path ../input/playground-series-s6e10/; timeout <300s; document in docs/experiments.md.
