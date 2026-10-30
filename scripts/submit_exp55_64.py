#!/usr/bin/env python3
"""Submit Experiments 55 through 64 to Kaggle Playground S6E10 and monitor their scores."""
import os, sys, time, subprocess

COMPETITION = 'playground-series-s6e10'

submissions = [
    (55, "submissions/submission_v55.csv", "Exp 55: Push Model + Orthogonal Directions Benchmark Anchor (TKMS 0.96223)"),
    (56, "submissions/submission_v56.csv", "Exp 56: Mayank Mirror Nudge Climb III Anchor (Mayank 0.96219)"),
    (57, "submissions/submission_v57.csv", "Exp 57: Goodpjw Meta-Stack Climb Anchor (Goodpjw 0.96217)"),
    (58, "submissions/submission_v58.csv", "Exp 58: Areeb Ali Subtract-to-Win Best File (Areeb w06 0.96218)"),
    (59, "submissions/submission_v59.csv", "Exp 59: Areeb Ali Deep Subtraction Variant (Areeb w07 -5% XGB -2% TabPFN)"),
    (60, "submissions/submission_v60.csv", "Exp 60: Tri-Summit Frontier Consensus (Equal 3-Way: TKMS 0.96223 + Mayank 0.96219 + Areeb 0.96218)"),
    (61, "submissions/submission_v61.csv", "Exp 61: Balanced Top-Tier Dual Peak (50% TKMS 0.96223 + 50% Our Summit 0.96215)"),
    (62, "submissions/submission_v62.csv", "Exp 62: Quad-Apex Foundation Ensemble (40% TKMS + 25% Mayank + 20% Goodpjw + 15% Our Summit)"),
    (63, "submissions/submission_v63.csv", "Exp 63: Multi-Paradigm Private-Shakeup Hedge (90% TKMS + 5% TabM Neural + 5% Route-Leak)"),
    (64, "submissions/submission_v64.csv", "Exp 64: Log-Odds Manifold Synthesis across All 4 Apex Peaks (TKMS, Mayank, Areeb, Goodpjw)"),
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
