#!/usr/bin/env python3
"""
PAVO Master Experiment Runner for Lambda Labs H100.
Runs all experiments sequentially and saves results.

Usage: python run_all_experiments.py [--hf-token YOUR_TOKEN]
"""

import argparse
import json
import os
import sys
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token for upload")
    parser.add_argument("--skip-e2e", action="store_true",
                        help="Skip E2E experiments (if already done)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip PPO training")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip HuggingFace upload")
    args = parser.parse_args()

    results_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(results_dir, "outputs"), exist_ok=True)

    total_start = time.time()

    # ========================================================
    # Experiment 1: E2E Pipeline (500 samples)
    # ========================================================
    if not args.skip_e2e:
        print("\n" + "="*60)
        print("EXPERIMENT 1: E2E Pipeline Latency (500 samples)")
        print("="*60)
        from exp1_e2e_pipeline import run_e2e_experiment
        e2e_results = run_e2e_experiment(
            n_samples=500,
            output_path=os.path.join(results_dir, "outputs/e2e_results_500.json")
        )
        print(f"E2E done. Cloud P95: {e2e_results['cloud_premium']['e2e_latency_ms']['p95']:.0f}ms")
    else:
        print("Skipping E2E experiments")

    # ========================================================
    # Experiment 2: Expanded Coupling Calibration (n=200)
    # ========================================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: Coupling Calibration (n=200 per WER level)")
    print("="*60)
    from exp2_coupling_calibration import run_coupling_experiment
    coupling_results = run_coupling_experiment(
        n_per_level=200,
        output_path=os.path.join(results_dir, "outputs/coupling_results_200.json")
    )
    print(f"Coupling done. WER levels tested: {len(coupling_results['wer_levels'])}")

    # ========================================================
    # Experiment 3: Train PPO Meta-Controller
    # ========================================================
    if not args.skip_training:
        print("\n" + "="*60)
        print("EXPERIMENT 3: PPO Meta-Controller Training")
        print("="*60)
        from exp3_train_ppo import train_meta_controller
        training_results = train_meta_controller(
            data_path=os.path.join(os.path.dirname(results_dir), "tier3_50k_train.jsonl"),
            output_dir=os.path.join(results_dir, "outputs/"),
            n_steps=100000,
            n_profiles=48
        )
        print(f"Training done. Final reward: {training_results['final_reward']:.4f}")
    else:
        print("Skipping PPO training")

    # ========================================================
    # Experiment 4: Real Ablation on GPU
    # ========================================================
    print("\n" + "="*60)
    print("EXPERIMENT 4: Real Component Ablation")
    print("="*60)
    from exp4_real_ablation import run_ablation
    ablation_results = run_ablation(
        n_samples=200,
        model_weights_path=os.path.join(results_dir, "outputs/meta_controller.pt"),
        output_path=os.path.join(results_dir, "outputs/ablation_results_real.json")
    )
    print(f"Ablation done. PAVO-Full latency: {ablation_results['pavo_full']['mean_latency_ms']:.0f}ms")

    # ========================================================
    # Experiment 5: Upload to HuggingFace
    # ========================================================
    if not args.skip_upload:
        print("\n" + "="*60)
        print("EXPERIMENT 5: Upload to HuggingFace")
        print("="*60)
        token = args.hf_token
        if token is None:
            token = input("Enter HuggingFace token: ").strip()
        from exp5_upload import upload_all
        upload_all(
            results_dir=results_dir,
            token=token
        )
        print("Upload complete!")
    else:
        print("Skipping upload")

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE in {total_time/3600:.1f} hours")
    print(f"Results saved to: {os.path.join(results_dir, 'outputs/')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
