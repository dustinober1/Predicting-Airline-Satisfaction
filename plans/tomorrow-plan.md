# Competition Daily Plan & Results (Playground S6E10)
Created: 2026-10-22 | Last Updated: 2026-10-24 | Deadline: ~Oct 31, 2026 (7 days remaining)
Competition: playground-series-s6e10 | Best Public Score: **0.96213** | Current Rank: **#50** (Top 1.7% of 2,893 teams)

---

## Today's 10 Experiments & Results (2026-10-24) — ALL 10 SUBMITTED

| Exp # | Description | Public Score | Rank Impact | Status |
|---|---|---|---|---|
| **Exp 11** | Native LightGBM 5-Fold CV (AUC Optimized, Clean Features) | 0.95835 | #2395 → #1889 | ✅ Complete |
| **Exp 12** | Native CatBoost 5-Fold CV (AUC Optimized, cat_features) | OOF 0.95863 | - | ✅ Complete |
| **Exp 14** | Tri-GBDT Stacking Meta-Learner (LGBM + CatBoost on Logits) | 0.95856 | Local GBDT Peak | ✅ Complete |
| **Exp 17** | Tabular Foundation (RealMLP PyTabKit) + GBDT Ensemble | 0.96100 | #1889 → #970 | ✅ Complete |
| **Exp 18** | Multi-Family 201-Model Stack Blend with Native Trees & RealMLP | 0.96184 | #970 → #462 | ✅ Complete |
| **Exp 19** | Summit Composer Peak Ensemble + Foundation Models & Native GBDT | 0.96202 | #462 → #234 | ✅ Complete |
| **Exp 20** | Grand Champion Weighted Rank Ensemble across Top Model Families | 0.96206 | #234 → #204 | ✅ Complete |
| **Exp 21** | Summit Composer V54 Pinned Baseline | 0.96213 | #204 → #102 | ✅ Complete |
| **Exp 22** | Ultra-Tight Peak Rank Blend (85% Mayank Summit + 15% Yeashu 201-Stack) | 0.96213 | Solidified Top 100 | ✅ Complete |
| **Exp 23** | Tri-Peak Micro-Blend (75% Mayank + 15% Yeashu + 8% Rabari + 2% Native Stack) | 0.96212 | Robustness Check | ✅ Complete |
| **Exp 24** | Final Selected Submission Safe Copy (`submission_final.csv`) | **0.96213** | **#50 IN THE WORLD** | 🏆 Complete |

---

## Next Steps for Tomorrow (After Next Daily Reset)
1. **Route ID Matching & Exact Lookup Table**: Extract unique routes via `Flight Distance` and investigate exact matches between train/test and external original airline dataset (`dariushafshar/s6e10-original-data-aligned`).
2. **TabPFN v3.5 Integration**: Run TabPFN local inference or stack with community TabPFN chunked predictions.
3. **Pseudo-labeling with High-Confidence Filtering**: Use predictions with p > 0.99 or p < 0.01 to fine-tune our LightGBM.
4. **Post-Processing Calibration**: Isotonic regression and probability calibration on extreme satisfaction probability percentiles.
