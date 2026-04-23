#!/usr/bin/env python3
"""Single entry point for the supervised learning ablation."""
import subprocess, sys, os, time, json

DIR = os.path.dirname(os.path.abspath(__file__))

def run(script):
    print(f"\n{'='*60}\nRunning {script}\n{'='*60}")
    r = subprocess.run([sys.executable, os.path.join(DIR, script)], cwd=DIR)
    if r.returncode != 0:
        print(f"ERROR: {script} failed with code {r.returncode}")
        sys.exit(1)

t0 = time.time()

# Step 1: generate data if needed
if not (os.path.exists(os.path.join(DIR, "X.npy")) and os.path.exists(os.path.join(DIR, "y.npy"))):
    run("generate_data.py")
else:
    print("Data already generated (X.npy, y.npy found). Skipping.")

# Step 2: train and evaluate
run("train_eval.py")

# Step 3: generate output table
run("output_table.py")

print(f"\n{'='*60}")
print(f"Ablation complete in {(time.time()-t0)/60:.1f} minutes.")
print("Output files:")
print("  ablation_results.json  — raw numbers")
print("  ablation_table.tex     — LaTeX table for paper")
print("  ablation_paragraph.txt — interpretation paragraph")
print(f"{'='*60}")
