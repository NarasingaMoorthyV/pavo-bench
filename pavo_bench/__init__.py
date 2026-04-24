"""
PAVO-Bench: a 50K-turn voice pipeline benchmark + pretrained router.

Minimal usage:

    from pavo_bench import (
        load_dataset, PretrainedPAVORouter, AlwaysCloudRouter, benchmark_router,
    )

    turns = load_dataset(split="test")          # 10K test turns
    pavo   = PretrainedPAVORouter.from_released()
    cloud  = AlwaysCloudRouter()

    print(benchmark_router(pavo,  turns))
    print(benchmark_router(cloud, turns))

Evaluate your own routing strategy by subclassing BaseRouter:

    from pavo_bench import BaseRouter, Profile, benchmark_router

    class MyRouter(BaseRouter):
        def route(self, turn) -> Profile:
            # decide per-turn; return one of: "cloud_premium", "ondevice_fast",
            # "hybrid_balanced".
            return "hybrid_balanced" if turn.complexity >= 3 else "ondevice_fast"

    print(benchmark_router(MyRouter(), turns))
"""

from .dataset import PAVOBenchTurn, load_dataset
from .state import turn_to_state_vector
from .model import MetaController
from .loader import load_pretrained
from .routers import (
    BaseRouter,
    Profile,
    AlwaysCloudRouter,
    AlwaysEdgeRouter,
    HybridRouter,
    RandomRouter,
    PretrainedPAVORouter,
)
from .evaluate import benchmark_router, BenchmarkResult
from .coupling import reproduce_coupling_cliff

__all__ = [
    "PAVOBenchTurn",
    "load_dataset",
    "turn_to_state_vector",
    "MetaController",
    "load_pretrained",
    "BaseRouter",
    "Profile",
    "AlwaysCloudRouter",
    "AlwaysEdgeRouter",
    "HybridRouter",
    "RandomRouter",
    "PretrainedPAVORouter",
    "benchmark_router",
    "BenchmarkResult",
    "reproduce_coupling_cliff",
]

__version__ = "1.0.0"
