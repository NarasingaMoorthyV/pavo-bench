"""Load the released MetaController checkpoint.

Handles both formats that ship in this repo:
  - experiments/outputs/meta_controller_best.pt  — raw state_dict
  - experiments/outputs/meta_controller.pt       — dict with 'model_state_dict'
    and 'architecture' keys.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch

from .model import MetaController


_DEFAULT_CANDIDATES = (
    "experiments/outputs/meta_controller_best.pt",
    "experiments/outputs/meta_controller.pt",
)


def _find_checkpoint(repo_root: Optional[Path]) -> Path:
    search: list[Path] = []
    if repo_root is not None:
        for rel in _DEFAULT_CANDIDATES:
            search.append(Path(repo_root) / rel)
    if env := os.environ.get("PAVO_BENCH_ROOT"):
        for rel in _DEFAULT_CANDIDATES:
            search.append(Path(env) / rel)
    for rel in _DEFAULT_CANDIDATES:
        search.append(Path.cwd() / rel)
        search.append(Path.cwd().parent / rel)

    for p in search:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find meta_controller.pt. Pass repo_root= or set "
        "PAVO_BENCH_ROOT to the pavo-bench repo checkout."
    )


def load_pretrained(
    repo_root: Optional[str | Path] = None,
    device: str = "cpu",
) -> Tuple[MetaController, dict]:
    """Load the released meta-controller.

    Returns:
        (model, info) where `info` is a dict with keys that may include
        'architecture', 'n_params', 'training_steps', etc. — whatever the
        released checkpoint bundles.
    """
    ckpt_path = _find_checkpoint(Path(repo_root) if repo_root else None)
    blob = torch.load(ckpt_path, map_location=device)

    if isinstance(blob, dict) and "model_state_dict" in blob:
        arch = blob.get("architecture", {}) or {}
        model = MetaController(
            state_dim=int(arch.get("state_dim", 12)),
            hidden=int(arch.get("hidden", 256)),
            n_profiles=int(arch.get("n_profiles", 48)),
        )
        model.load_state_dict(blob["model_state_dict"])
        info = {k: v for k, v in blob.items() if k != "model_state_dict"}
    else:
        model = MetaController()
        model.load_state_dict(blob)
        info = {"source": str(ckpt_path), "format": "raw_state_dict"}

    model.to(device).eval()
    info["checkpoint_path"] = str(ckpt_path)
    info["n_params_loaded"] = model.count_params()
    return model, info
