#!/usr/bin/env python3
"""
Experiment 5: Upload all artifacts to HuggingFace.
Uploads: 50K dataset, model weights, training logs, all experiment results.
"""

import os
import json


def upload_all(results_dir, token):
    from huggingface_hub import HfApi, upload_file, upload_folder

    api = HfApi()
    repo_id = "vnmoorthy/pavo-bench"

    parent_dir = os.path.dirname(results_dir)
    outputs_dir = os.path.join(results_dir, "outputs")

    print(f"  Uploading to {repo_id}...")

    # 1. Upload 50K dataset files (train + test)
    for fname in ["tier3_50k_train.jsonl", "tier3_50k_test.jsonl", "tier3_50k_summary.json"]:
        fpath = os.path.join(parent_dir, fname)
        if os.path.exists(fpath):
            print(f"    Uploading {fname}...")
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f"data/{fname}",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
        else:
            print(f"    WARNING: {fname} not found at {fpath}")

    # 2. Upload original experiment results
    for fname in os.listdir(parent_dir):
        if fname.endswith(".json") and not fname.startswith("tier3_50k"):
            fpath = os.path.join(parent_dir, fname)
            print(f"    Uploading {fname}...")
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f"results/{fname}",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )

    # 3. Upload new experiment outputs
    if os.path.exists(outputs_dir):
        for fname in os.listdir(outputs_dir):
            fpath = os.path.join(outputs_dir, fname)
            if os.path.isfile(fpath):
                print(f"    Uploading outputs/{fname}...")
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=f"outputs/{fname}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                )

    # 4. Upload model weights specifically to a models/ directory
    for model_file in ["meta_controller.pt", "meta_controller_best.pt"]:
        model_path = os.path.join(outputs_dir, model_file)
        if os.path.exists(model_path):
            print(f"    Uploading model: {model_file}...")
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=f"models/{model_file}",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )

    # 5. Upload training log
    log_path = os.path.join(outputs_dir, "training_log.json")
    if os.path.exists(log_path):
        print(f"    Uploading training_log.json...")
        api.upload_file(
            path_or_fileobj=log_path,
            path_in_repo=f"training/training_log.json",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

    print("  Upload complete!")
    print(f"  View at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    import sys
    token = sys.argv[1] if len(sys.argv) > 1 else input("HuggingFace token: ").strip()
    results_dir = os.path.dirname(os.path.abspath(__file__))
    upload_all(results_dir, token)
