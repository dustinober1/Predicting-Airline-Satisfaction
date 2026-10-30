# Predicting Airline Passenger Satisfaction
**Kaggle Playground Series — Season 6, Episode 10 (S6E10)**

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Playground%20S6E10-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/playground-series-s6e10)
[![Evaluation Metric](https://img.shields.io/badge/Metric-ROC--AUC-success)]()
[![Best Public Score](https://img.shields.io/badge/Best%20Public%20Score-0.96223-brightgreen)]()
[![Current Rank](https://img.shields.io/badge/Current%20Rank-Top%201.0%25%20(%2338%20Global)-gold)]()

---

## 📌 Overview

This repository contains the complete competitive machine learning pipeline and solution for **Kaggle Playground Series S6E10: Predicting Airline Satisfaction**.

The goal is to predict whether an airline passenger was satisfied with their flight experience based on survey responses, flight distance, delay metrics, and passenger demographics.

* **Task**: Binary Classification (`satisfaction`: 0 / 1)
* **Metric**: ROC-AUC
* **Dataset Scale**: 699,635 training rows, 299,844 test rows, 23 raw features
* **Peak Public Leaderboard Score**: **0.96223** (Top 1.0% / #38 in the world out of 3,556 teams)

---

## 📂 Repository Structure

```text
├── README.md                      # Project overview, methodology & reproduction guide
├── .gitignore                     # Git ignore rules for ML artifacts, caches, and logs
├── kernel-metadata.json           # Kaggle kernel push/submission metadata
├── data/                          # Dataset directory (train, test, sample_submission)
│   └── features/                  # Precomputed route-Hamming & original lookup features
├── docs/
│   └── experiments.md             # Complete experiment log (Exps 1 through 34 with CV & LB)
├── logs/                          # Training and pipeline execution logs
├── models_oof/                    # Out-Of-Fold (OOF) and test prediction arrays (.npy)
├── notebooks/                     # Exploratory analysis and Kaggle kernel notebooks
├── plans/
│   └── tomorrow-plan.md           # Daily experiment logs, leaderboard tracking, and future roadmap
├── scripts/                       # Training, feature engineering, and ensembling scripts
│   ├── compute_route_hamming_features.py    # Vectorized route-leak nearest-neighbor engine
│   ├── compute_original_model_features.py   # CC0 original dataset lookup & pre-trained model scorer
│   ├── run_exp25.py               # Route-Hamming + OG-Lookup augmented 5-fold LightGBM CV
│   ├── run_exp26.py               # 13-model multi-paradigm meta-stack (TFMs + Route LGB + CB + TabPFN)
│   ├── build_today_experiments.py # Pipeline generating and validating candidate submissions
│   ├── create_peak_blends.py      # Regularized peak rank blending
│   ├── ensemble_experiments.py    # Foundation model rank ensembling
│   ├── run_exp11.py               # Native LightGBM 5-fold CV baseline
│   ├── run_exp12.py               # Native CatBoost 5-fold CV baseline
│   ├── run_exp13.py               # Route profile feature engineered LightGBM
│   ├── run_exp14.py               # Tri-GBDT stacking meta-learner
│   └── run_xgb_cv.py              # XGBoost 5-fold CV pipeline
└── submissions/                   # Validated submission CSV files (submission_final.csv selected)
```

---

## 🔬 Core Innovations & Methodology

### 1. Reverse-Engineering the Synthetic Generator (Route Hamming Leak)
Playground S6E10 data was generated using deep generative modeling (CTGAN) conditioned on the CC0 original airline survey dataset. 
- Over **99.999%** of both train and test rows share their exact `Flight Distance` with an original survey respondent.
- By computing the **vectorized Hamming distance** to the nearest original passengers on that exact route ([`scripts/compute_route_hamming_features.py`](file:///Users/dustinober/Projects/Data_Science/Predicting%20Airline%20Satisfaction/scripts/compute_route_hamming_features.py)), we extract exponential-decay weighted route satisfaction proxies (`par_label_w0.5`, `par_label_w1`).
- The `par_label_w1` feature alone achieves an individual **solo AUC of 0.91928**, and boosting with these features improved our pure LightGBM model from **0.95835 to 0.95951 (+0.00116 LB)**.

### 2. Multi-Paradigm Tabular Foundation Stacking
Rather than ensembling only tree-based models, our architecture integrates four distinct model paradigms:
- **Gradient Boosted Decision Trees**: Custom LightGBM, CatBoost, and XGBoost with native categorical handling.
- **In-Context Tabular Transformers**: TabPFN-3.5 and TabICL-2 operating directly on row context.
- **Deep Tabular Architectures**: KumoLarge Tabular Transformer and PyTabKit RealMLP.
- **Pretrained Transfer Logits**: Models trained exclusively on the CC0 original source rows.

A 5-fold meta-learner trained with regularized logistic regression on logits achieved **0.96154 OOF AUC** and **0.96110 on Public LB** without community ensembling.

### 3. Non-Linear Middle-Decile Layering & Positional Sorting
- **Middle-Decile Probability Layering**: TabPFN predicts with 86% of its probability mass at the extremes (<0.1 or >0.9). Layering 5% of TabPFN probability into the rank base selectively sharpens ambiguous middle deciles (deciles 4–7) without disturbing confident predictions.
- **F.A. Nina Two-Sided Positional `hblend`**: Two-sided positional sorting fusion across multi-family anchors to maximize ranking stability against private leaderboard shakeup.

---

## 🚀 Quickstart & Reproduction

### Prerequisites
```bash
pip install numpy pandas scipy scikit-learn lightgbm catboost xgboost
```

### 1. Compute Route Leak and Original Lookup Features
```bash
python3 scripts/compute_route_hamming_features.py
python3 scripts/compute_original_model_features.py
```

### 2. Train Route-Augmented 5-Fold LightGBM
```bash
python3 scripts/run_exp25.py
```

### 3. Run Multi-Paradigm 13-Model Stacking
```bash
python3 scripts/run_exp26.py
```

### 4. Build and Validate Top Ensembles
```bash
python3 scripts/build_today_experiments.py
```

---

## 📊 Experiment History Highlights

| Phase | Milestone | Best OOF AUC | Public Score | Key Advancement |
|---|---|---|---|---|
| **Phase 1** | Baseline XGB / RF / HistGB | 0.9581 | 0.95772 | Clean baseline feature pipeline & handling NaN delays |
| **Phase 2** | Native LightGBM & CatBoost (Exp 11–14) | 0.95909 | 0.95856 | Native categorical handling + AUC-targeted loss |
| **Phase 3** | Tabular Foundation & PyTabKit RealMLP (Exp 17–19) | 0.96120 | 0.96202 | Integration of deep tabular neural models |
| **Phase 4** | Summit Composer & Multi-Stack Blends (Exp 20–24) | - | **0.96213** | Ultra-tight rank blending; broke into **Top 50 in the world** |
| **Phase 5** | Route-Hamming Leak & 13-TFM Meta-Stack (Exp 25–34) | **0.96154** | **0.96213** | Solved route generator fingerprint (+0.00116 single-model boost) |

For complete per-experiment logs, CV fold splits, and parameter configs, see [`docs/experiments.md`](file:///Users/dustinober/Projects/Data_Science/Predicting%20Airline%20Satisfaction/docs/experiments.md) and [`plans/tomorrow-plan.md`](file:///Users/dustinober/Projects/Data_Science/Predicting%20Airline%20Satisfaction/plans/tomorrow-plan.md).

---

## 🏆 Final Submission Selection
The active designated competition submission is [`submissions/submission_final.csv`](file:///Users/dustinober/Projects/Data_Science/Predicting%20Airline%20Satisfaction/submissions/submission_final.csv) (Public Score: **0.96213**).
