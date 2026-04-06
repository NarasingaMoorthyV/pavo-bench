#!/usr/bin/env python3
"""
Experiment 2: Expanded coupling calibration.
Inject WER at controlled levels and measure LLM response quality.
n=200 per WER level (up from n=10).
"""

import json
import os
import random
import re
import subprocess
import time
import numpy as np
from collections import defaultdict


# Factual QA pairs for coupling calibration
FACTUAL_QA = [
    ("What is the capital of France?", "Paris"),
    ("What year did World War II end?", "1945"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the speed of light in km/s?", "299792"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("Who discovered penicillin?", "Fleming"),
    ("What is the square root of 144?", "12"),
    ("What is the currency of Japan?", "yen"),
    ("Who was the first president of the United States?", "Washington"),
    ("What element has atomic number 1?", "hydrogen"),
    ("What is the longest river in the world?", "Nile"),
    ("How many continents are there?", "7"),
    ("What is the freezing point of water in Fahrenheit?", "32"),
    ("Who developed the theory of relativity?", "Einstein"),
    ("What is the chemical formula for water?", "H2O"),
    ("What planet is closest to the sun?", "Mercury"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the capital of Japan?", "Tokyo"),
    ("Who invented the telephone?", "Bell"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("What year did the Berlin Wall fall?", "1989"),
    ("What is the atomic number of carbon?", "6"),
    ("Who wrote the Odyssey?", "Homer"),
    ("What is Pi to two decimal places?", "3.14"),
    ("What is the tallest mountain in the world?", "Everest"),
    ("What gas do plants absorb from the atmosphere?", "carbon dioxide"),
    ("How many bones are in the adult human body?", "206"),
    ("What is the capital of Australia?", "Canberra"),
    ("Who composed the Four Seasons?", "Vivaldi"),
    ("What is the smallest prime number?", "2"),
    ("What year was the Declaration of Independence signed?", "1776"),
    ("What is the hardest natural substance?", "diamond"),
    ("Who was the first person to walk on the moon?", "Armstrong"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("How many planets are in our solar system?", "8"),
    ("What is the capital of Brazil?", "Brasilia"),
    ("Who painted Starry Night?", "Van Gogh"),
]


def inject_wer(text, target_wer):
    """Inject word errors at a target WER rate."""
    if target_wer == 0:
        return text
    words = text.split()
    n_errors = max(1, int(len(words) * target_wer / 100))
    error_words = list(words)
    indices = random.sample(range(len(words)), min(n_errors, len(words)))
    substitutions = ["the", "a", "is", "was", "and", "or", "but", "for", "not", "with",
                     "from", "that", "this", "have", "had", "been", "were", "are"]
    for idx in indices:
        error_words[idx] = random.choice(substitutions)
    return " ".join(error_words)


def query_ollama(model_name, prompt, timeout=30):
    """Query ollama and return response + latency."""
    start = time.perf_counter()
    try:
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True, text=True, timeout=timeout
        )
        elapsed = (time.perf_counter() - start) * 1000
        return result.stdout.strip(), elapsed
    except subprocess.TimeoutExpired:
        elapsed = (time.perf_counter() - start) * 1000
        return "[TIMEOUT]", elapsed


def check_answer(response, correct_answer):
    """Check if the LLM response contains the correct answer."""
    response_lower = response.lower()
    answer_lower = correct_answer.lower()
    # Check for exact match or containment
    return answer_lower in response_lower


def compute_quality_score(response, correct_answer, original_question):
    """Compute a quality score [0,1] for an LLM response."""
    if "[TIMEOUT]" in response:
        return 0.0

    # Exact match component (0 or 1)
    exact = 1.0 if check_answer(response, correct_answer) else 0.0

    # Length-based relevance (penalize very short or very long)
    words = len(response.split())
    length_score = min(1.0, words / 5) if words > 0 else 0.0

    # Combined: 70% exact match, 30% relevance
    return 0.7 * exact + 0.3 * length_score


def run_coupling_experiment(n_per_level=200, output_path="outputs/coupling_results_200.json"):
    """Run coupling calibration at multiple WER levels."""

    wer_levels = [0, 1, 2, 3, 5, 8, 10, 15, 20]
    models = ["llama3.1:8b", "gemma2:2b"]

    results = {
        "wer_levels": wer_levels,
        "n_per_level": n_per_level,
        "models": {},
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_qa_pairs": len(FACTUAL_QA),
            "gpu": "NVIDIA H100 SXM5",
        }
    }

    for model_name in models:
        print(f"\n  Testing {model_name}...")
        model_results = {}

        for wer in wer_levels:
            print(f"    WER={wer}%: ", end="", flush=True)
            scores = []
            exact_matches = []
            latencies = []
            responses_sample = []

            for i in range(n_per_level):
                # Cycle through QA pairs
                qa_idx = i % len(FACTUAL_QA)
                question, answer = FACTUAL_QA[qa_idx]

                # Inject WER into the question (simulating ASR errors)
                corrupted_q = inject_wer(question, wer)

                # Query LLM
                prompt = f"Answer this question in one sentence: {corrupted_q}"
                response, latency_ms = query_ollama(model_name, prompt)

                # Score
                score = compute_quality_score(response, answer, question)
                exact = check_answer(response, answer)
                scores.append(score)
                exact_matches.append(exact)
                latencies.append(latency_ms)

                if i < 5:
                    responses_sample.append({
                        "question": question,
                        "corrupted": corrupted_q,
                        "response": response[:200],
                        "correct": answer,
                        "exact_match": exact,
                        "score": score
                    })

                if (i + 1) % 50 == 0:
                    print(f"{i+1}", end=" ", flush=True)

            model_results[f"wer_{wer}"] = {
                "wer_percent": wer,
                "n_samples": len(scores),
                "quality_mean": float(np.mean(scores)),
                "quality_std": float(np.std(scores)),
                "quality_ci95_low": float(np.percentile(scores, 2.5)),
                "quality_ci95_high": float(np.percentile(scores, 97.5)),
                "exact_match_rate": float(np.mean(exact_matches)),
                "exact_match_ci95": [
                    float(np.mean(exact_matches) - 1.96 * np.sqrt(np.mean(exact_matches) * (1 - np.mean(exact_matches)) / len(exact_matches))),
                    float(np.mean(exact_matches) + 1.96 * np.sqrt(np.mean(exact_matches) * (1 - np.mean(exact_matches)) / len(exact_matches)))
                ],
                "latency_mean_ms": float(np.mean(latencies)),
                "latency_std_ms": float(np.std(latencies)),
                "individual_scores": [float(s) for s in scores],
                "individual_exact_matches": [bool(e) for e in exact_matches],
                "sample_responses": responses_sample
            }
            print(f"  exact_match={np.mean(exact_matches):.2f}, quality={np.mean(scores):.3f}")

        results["models"][model_name] = model_results

    # Compute coupling threshold
    for model_name in models:
        m = results["models"][model_name]
        baseline_acc = m["wer_0"]["exact_match_rate"]
        threshold_wer = None
        for wer in wer_levels:
            if wer == 0:
                continue
            acc = m[f"wer_{wer}"]["exact_match_rate"]
            if acc < 0.70:  # Accuracy drops below 70%
                threshold_wer = wer
                break
        results["models"][model_name]["coupling_threshold_wer"] = threshold_wer
        results["models"][model_name]["baseline_accuracy"] = baseline_acc

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_path}")

    return results


if __name__ == "__main__":
    run_coupling_experiment()
