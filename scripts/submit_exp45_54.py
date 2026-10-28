#!/usr/bin/env python3
"""Submit Experiments 45 through 54 to Kaggle Playground S6E10 and monitor their scores."""
import os, sys, time, subprocess

COMPETITION = 'playground-series-s6e10'

submissions = [
    (45, "submissions/submission_v45.csv", "Exp 45: New Frontier Benchmark Anchor (Goodpjw Meta-Stack Probe E 0.96215)"),
    (46, "submissions/submission_v46.csv", "Exp 46: Top-Tier Dual-Summit Consensus (50% Goodpjw 0.96215 + 50% Summit 0.96213)"),
    (47, "submissions/submission_v47.csv", "Exp 47: Tri-Apex Summit Fusion (40% Goodpjw 0.96215 + 30% Summit + 30% Orange Lab 0.96213)"),
    (48, "submissions/submission_v48.csv", "Exp 48: Super-Rank Ensemble of Top-7 Public Summits (Daoviet + Goodpjw + Mayank + Nina + Rabari)"),
    (49, "submissions/submission_v49.csv", "Exp 49: Inverse-TabPFN Extreme Boundary Sharpener (105% Goodpjw Meta - 5% TabPFN-3.5)"),
    (50, "submissions/submission_v50.csv", "Exp 50: Orthogonal 10-Fold Tree Diversity Injection (92% Goodpjw 0.96215 + 8% Anhad 10-Fold GBDT)"),
    (51, "submissions/submission_v51.csv", "Exp 51: Deep Neural (TabM) + Route-Leak Regularization (90% Goodpjw + 5% TabM + 5% Route-LGB)"),
    (52, "submissions/submission_v52.csv", "Exp 52: Log-Odds Manifold Synthesis (50% Goodpjw + 25% Summit + 15% Orange + 5% Route + 5% TabM)"),
    (53, "submissions/submission_v53.csv", "Exp 53: Two-Sided Positional Nina h-blend (Goodpjw 0.96215, Summit 0.96213, Orange 0.96213, Top-7)"),
    (54, "submissions/submission_v54.csv", "Exp 54: Grand Apex 7-Pillar Consortium (Goodpjw, Summit, Orange, Top-7, 10-Fold GBDT, Route-Leak, TabM)"),
]

print("=" * 70)
print(f"Submitting {len(submissions)} Experiments to {COMPETITION}")
print("=" * 70)

for exp_id, filepath, desc in submissions:
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Submitting Exp {exp_id}...")
    print(f"  File: {filepath}")
    print(f"  Message: {desc}")
    
    cmd = [
        'kaggle', 'competitions', 'submit',
        '-c', COMPETITION,
        '-f', filepath,
        '-m', desc
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  ❌ Error submitting Exp {exp_id}:\n{res.stderr}\n{res.stdout}")
    else:
        print(f"  ✅ Submission sent successfully:\n{res.stdout.strip()}")
    
    # Wait 8 seconds between uploads to respect API limits
    time.sleep(8)

print("\n" + "=" * 70)
print("All 10 submissions queued. Waiting 30s for scoring...")
print("=" * 70)
time.sleep(30)

# Check leaderboard results
print("\nFetching latest submission status...")
cmd_status = ['kaggle', 'competitions', 'submissions', '-c', COMPETITION]
res_status = subprocess.run(cmd_status, capture_output=True, text=True)
print(res_status.stdout)
