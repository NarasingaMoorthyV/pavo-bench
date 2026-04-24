"""PAVO-Bench turn loader.

Each turn is a lightweight dataclass with the per-turn features the router sees
and the reference strings for quality scoring.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass(frozen=True)
class PAVOBenchTurn:
    """One voice-turn record from PAVO-Bench."""

    index: int
    complexity: int            # 1..5, higher = harder
    snr_db: float              # environment SNR
    noise_type: str            # 'babble', 'traffic', 'white', ...
    cpu_util: float            # 0..1
    battery: float             # 0..1
    rtt_ms: float              # network round-trip
    ctx_tokens: int            # dialogue context depth
    user_input: str
    reference_response: str
    source: str = "synthetic"
    audio_idx: Optional[int] = None
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_json(cls, obj: dict) -> "PAVOBenchTurn":
        return cls(
            index=int(obj["index"]),
            complexity=int(obj["complexity"]),
            snr_db=float(obj["snr_db"]),
            noise_type=str(obj.get("noise_type", "clean")),
            cpu_util=float(obj["cpu_util"]),
            battery=float(obj["battery"]),
            rtt_ms=float(obj["rtt_ms"]),
            ctx_tokens=int(obj["ctx_tokens"]),
            user_input=str(obj.get("user_input", "")),
            reference_response=str(obj.get("reference_response", "")),
            source=str(obj.get("source", "synthetic")),
            audio_idx=obj.get("audio_idx"),
            raw=obj,
        )


def _resolve_split_path(split: str, repo_root: Optional[Path] = None) -> Path:
    """Resolve the jsonl path for 'train' or 'test', checking common locations."""
    fname = {"train": "tier3_50k_train.jsonl", "test": "tier3_50k_test.jsonl"}.get(split)
    if fname is None:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    # Priority order:
    # 1. explicit repo_root
    # 2. $PAVO_BENCH_ROOT/fname
    # 3. ./fname (cwd)
    # 4. ../fname (common Colab layout: notebooks/ cwd, data next to it)
    # 5. HuggingFace download
    candidates: List[Path] = []
    if repo_root is not None:
        candidates.append(Path(repo_root) / fname)
    if env := os.environ.get("PAVO_BENCH_ROOT"):
        candidates.append(Path(env) / fname)
    candidates.append(Path.cwd() / fname)
    candidates.append(Path.cwd().parent / fname)

    for p in candidates:
        if p.exists():
            return p

    # Fall back to HuggingFace download.
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise FileNotFoundError(
            f"Could not find {fname} locally. Install huggingface_hub "
            "(`pip install huggingface_hub`) or set PAVO_BENCH_ROOT to the repo root."
        ) from e
    return Path(hf_hub_download(
        repo_id="vnmoorthy/pavo-bench",
        filename=fname,
        repo_type="dataset",
    ))


def load_dataset(split: str = "test",
                 repo_root: Optional[str | Path] = None,
                 limit: Optional[int] = None) -> List[PAVOBenchTurn]:
    """Load PAVO-Bench turns for the given split.

    Args:
        split:      'train' (40K turns) or 'test' (10K turns).
        repo_root:  path to the pavo-bench repo checkout. If None, we check
                    $PAVO_BENCH_ROOT, cwd, and then fall back to downloading
                    from huggingface.co/datasets/vnmoorthy/pavo-bench.
        limit:      optional cap on number of turns (useful for smoke tests).

    Returns:
        A list of PAVOBenchTurn.
    """
    path = _resolve_split_path(split, Path(repo_root) if repo_root else None)
    turns: List[PAVOBenchTurn] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            turns.append(PAVOBenchTurn.from_json(json.loads(line)))
            if limit is not None and len(turns) >= limit:
                break
    return turns


def iter_dataset(split: str = "test", **kwargs) -> Iterator[PAVOBenchTurn]:
    """Same as load_dataset but returns an iterator (constant memory)."""
    path = _resolve_split_path(split, kwargs.get("repo_root"))
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield PAVOBenchTurn.from_json(json.loads(line))
