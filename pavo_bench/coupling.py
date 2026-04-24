"""reproduce_coupling_cliff — minimal reproduction of the Tier-1 coupling
calibration on an arbitrary (ASR, LLM) pair.

Usage:

    from pavo_bench import reproduce_coupling_cliff

    def my_llm(prompt: str) -> str:
        # call your LLM here
        return ...

    results = reproduce_coupling_cliff(
        llm_fn=my_llm,
        wer_levels=(0, 1, 2, 3, 5, 8, 10, 15, 20),
        n_queries_per_wer=10,
    )
    # results is a dict shaped like tier1_coupling_results.json
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Tuple


_DEFAULT_QUERIES = [
    "What time is it in Tokyo?",
    "Summarize this document in one sentence.",
    "Remind me to call mom at six.",
    "Translate 'good morning' into French.",
    "How long is the flight from JFK to LAX?",
    "What's the weather like tomorrow?",
    "Set an alarm for seven.",
    "Play some jazz music.",
    "Define 'photosynthesis' in one sentence.",
    "What's 48 divided by 6?",
]


def _inject_wer(text: str, wer_pct: float, rng: random.Random) -> str:
    """Inject synthetic ASR errors at approximately wer_pct% word-error rate.

    We substitute, delete, or insert tokens uniformly at random. This matches
    the protocol used by experiments/exp2_coupling_calibration.py.
    """
    tokens = re.findall(r"\S+|\s+", text)
    target_changes = int(len(tokens) * wer_pct / 100.0)
    for _ in range(target_changes):
        if not tokens:
            break
        i = rng.randrange(len(tokens))
        op = rng.choice(["sub", "del", "ins"])
        if op == "sub":
            tokens[i] = "".join(
                rng.choice("abcdefghijklmnopqrstuvwxyz")
                for _ in range(max(len(tokens[i]), 1))
            ) if tokens[i].strip() else tokens[i]
        elif op == "del":
            tokens.pop(i)
        elif op == "ins":
            tokens.insert(
                i,
                " " + "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(4)) + " ",
            )
    return "".join(tokens)


def _score_answer(response: str, clean_reference: str) -> float:
    """Tiny heuristic quality score in [0, 1].

    For a faithful reproduction install `bert-score` and replace this.
    """
    ref_tokens = set(re.findall(r"[a-z0-9']+", clean_reference.lower()))
    out_tokens = set(re.findall(r"[a-z0-9']+", response.lower()))
    if not ref_tokens:
        return 1.0 if out_tokens else 0.0
    return len(ref_tokens & out_tokens) / len(ref_tokens)


def reproduce_coupling_cliff(
    llm_fn: Callable[[str], str],
    *,
    wer_levels: Iterable[int] = (0, 1, 2, 3, 5, 8, 10, 15, 20),
    n_queries_per_wer: int = 10,
    queries: Iterable[str] = _DEFAULT_QUERIES,
    seed: int = 42,
) -> Dict:
    """Reproduce the Tier-1 coupling cliff calibration on your LLM.

    Args:
        llm_fn: a callable that takes a prompt string and returns the LLM's
                response. Wrap your own inference engine.
        wer_levels: integer percentages of word error to inject.
        n_queries_per_wer: 10 matches the paper's Tier-1; 200 matches the
                extended experiment (tier1_coupling_results.json).

    Returns:
        A dict shaped like tier1_coupling_results.json so you can diff your
        result against the committed one.
    """
    rng = random.Random(seed)
    queries = list(queries)
    results: Dict[str, Dict] = {}

    for wer in wer_levels:
        scores: List[float] = []
        for i in range(n_queries_per_wer):
            q = queries[i % len(queries)]
            perturbed = _inject_wer(q, wer, rng) if wer > 0 else q
            try:
                response = llm_fn(perturbed)
            except Exception as e:  # noqa: BLE001
                response = ""
            scores.append(_score_answer(response, q))
        mean = sum(scores) / max(len(scores), 1)
        var = sum((s - mean) ** 2 for s in scores) / max(len(scores), 1)
        results[f"wer_{wer}"] = {
            "wer_pct": wer,
            "mean_quality": mean,
            "std_quality": var ** 0.5,
            "n_queries": len(scores),
            "quality_scores": scores,
        }

    return {
        "experiment": "Coupling Constraint Validation (reproduction)",
        "n_queries_per_wer": n_queries_per_wer,
        "wer_levels_tested": list(wer_levels),
        "results": results,
    }
