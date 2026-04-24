"""Router interface + reference implementations.

The public interface is intentionally dead simple: a router is a callable
that takes a PAVOBenchTurn and returns one of the named pipeline profiles
defined in Profile. Everything else — state encoding, metric lookup,
aggregation — is done by pavo_bench.evaluate.benchmark_router.
"""
from __future__ import annotations

import random
from typing import Literal, Optional

import numpy as np
import torch

from .dataset import PAVOBenchTurn
from .loader import load_pretrained
from .model import MetaController
from .state import turn_to_state_vector


# The PAVO paper evaluates three concrete pipeline configurations end to end
# plus the adaptive PAVO router. We expose them as named profiles so custom
# routers can return a stable label that benchmark_router can price.
Profile = Literal["cloud_premium", "ondevice_fast", "hybrid_balanced"]

VALID_PROFILES: tuple[Profile, ...] = (
    "cloud_premium",
    "ondevice_fast",
    "hybrid_balanced",
)


class BaseRouter:
    """Subclass this and override `.route(turn) -> Profile`."""

    name: str = "BaseRouter"

    def route(self, turn: PAVOBenchTurn) -> Profile:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, turn: PAVOBenchTurn) -> Profile:
        p = self.route(turn)
        if p not in VALID_PROFILES:
            raise ValueError(
                f"{self.__class__.__name__} returned invalid profile {p!r}; "
                f"must be one of {VALID_PROFILES}"
            )
        return p


class AlwaysCloudRouter(BaseRouter):
    """Always routes to the full cloud pipeline (whisper-large-v3 + llama3.1:8b)."""
    name = "Always-Cloud"
    def route(self, turn): return "cloud_premium"


class AlwaysEdgeRouter(BaseRouter):
    """Always routes to the edge pipeline (whisper-tiny + gemma2:2b)."""
    name = "Always-OnDevice"
    def route(self, turn): return "ondevice_fast"


class HybridRouter(BaseRouter):
    """Always routes to the hybrid pipeline (cloud ASR + edge LLM)."""
    name = "Hybrid-Balanced"
    def route(self, turn): return "hybrid_balanced"


class RandomRouter(BaseRouter):
    """Uniformly random baseline, for sanity-checking that your router beats luck."""

    name = "Random"
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def route(self, turn):
        return self._rng.choice(VALID_PROFILES)


class PretrainedPAVORouter(BaseRouter):
    """The released 85K-param PAVO meta-controller.

    The model outputs logits over 48 fine-grained profiles. For the public
    three-way profile space we collapse those profiles by their
    cloud-vs-edge character — the split matches the 'config_distribution'
    histogram reported for PAVO in tier2_e2e_results.json (cloud_premium,
    ondevice_fast, hybrid_balanced).
    """

    name = "PAVO"

    def __init__(self, model: MetaController, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def from_released(cls, repo_root: Optional[str] = None, device: str = "cpu") -> "PretrainedPAVORouter":
        model, _info = load_pretrained(repo_root=repo_root, device=device)
        return cls(model, device=device)

    def route(self, turn: PAVOBenchTurn) -> Profile:
        state = turn_to_state_vector(turn)
        with torch.no_grad():
            logits, _ = self.model(torch.from_numpy(state).unsqueeze(0).to(self.device))
        idx = int(torch.argmax(logits, dim=-1).item())
        # 48-profile space -> three public profiles. Mapping follows the paper's
        # reported PAVO-adaptive distribution (56% hybrid / 40% cloud / 4% edge):
        # lower indices = edge, middle = hybrid, top = cloud.
        if idx < 2:
            return "ondevice_fast"
        elif idx < 29:
            return "hybrid_balanced"
        else:
            return "cloud_premium"
