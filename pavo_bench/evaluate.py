"""benchmark_router — aggregate a router's per-turn choices into a metrics dict."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Iterable, List

import numpy as np

from .dataset import PAVOBenchTurn
from .routers import BaseRouter, VALID_PROFILES
from . import _profile_costs as pc


@dataclass
class BenchmarkResult:
    """Aggregate metrics for a router over a PAVO-Bench split."""

    router: str
    n_turns: int
    latency_ms_mean: float
    latency_ms_std:  float
    latency_ms_p50:  float
    latency_ms_p95:  float
    quality_mean:    float
    cost_usd_mean:   float
    energy_mj_mean:  float
    coupling_violations: int
    infeasible_pct: float
    profile_distribution: dict

    def as_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult({self.router}, n={self.n_turns}, "
            f"P95 latency={self.latency_ms_p95:.0f} ms, "
            f"quality={self.quality_mean:.3f}, "
            f"energy={self.energy_mj_mean:.1f} mJ, "
            f"profiles={self.profile_distribution})"
        )


def _sample_latency(profile: str, rng: random.Random) -> float:
    """Sample a per-turn latency from the profile's committed mean/std."""
    prior = pc.LATENCY_MS[profile]
    # Truncated Gaussian, clamped at 0.
    x = rng.gauss(prior["mean"], prior["std"])
    return max(x, 0.0)


def benchmark_router(
    router: BaseRouter,
    turns: Iterable[PAVOBenchTurn],
    *,
    seed: int = 0,
) -> BenchmarkResult:
    """Evaluate a router over a PAVO-Bench split.

    The simulator samples per-turn latency from the committed
    tier2_e2e_results.json distributions. Quality, cost, and energy are
    looked up per profile from component_ablation_results.json. This means
    results line up with the paper's headline numbers but do NOT require
    running the actual ASR/LLM stack — good for CI, education, and
    bakeoffs against your own router.

    For hardware-exact numbers, run experiments/run_all_experiments.py.
    """
    rng = random.Random(seed)
    latencies: List[float] = []
    quality: List[float] = []
    cost:    List[float] = []
    energy:  List[float] = []
    infeasible = 0
    violations = 0
    dist = {p: 0 for p in VALID_PROFILES}

    turns = list(turns)
    for turn in turns:
        profile = router(turn)
        dist[profile] += 1

        if pc.infeasible_for_turn(profile, turn.complexity):
            infeasible += 1
            violations += 1

        latencies.append(_sample_latency(profile, rng))
        quality.append(pc.QUALITY[profile])
        cost.append(pc.COST_USD[profile])
        energy.append(pc.ENERGY_MJ[profile])

    arr = np.asarray(latencies, dtype=np.float64)
    dist_norm = {k: v / max(len(turns), 1) for k, v in dist.items()}

    return BenchmarkResult(
        router=router.name,
        n_turns=len(turns),
        latency_ms_mean=float(arr.mean()),
        latency_ms_std=float(arr.std()),
        latency_ms_p50=float(np.percentile(arr, 50)),
        latency_ms_p95=float(np.percentile(arr, 95)),
        quality_mean=float(np.mean(quality)),
        cost_usd_mean=float(np.mean(cost)),
        energy_mj_mean=float(np.mean(energy)),
        coupling_violations=int(violations),
        infeasible_pct=100.0 * infeasible / max(len(turns), 1),
        profile_distribution=dist_norm,
    )
