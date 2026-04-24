"""Per-profile latency/quality/cost/energy priors extracted from the committed
tier2_e2e_results.json and component_ablation_results.json.

benchmark_router uses these to turn a router's per-turn profile choice into an
aggregate metrics dict without requiring the user to spin up an LLM + ASR
stack. For exact end-to-end numbers, users should run
experiments/run_all_experiments.py on real hardware.
"""

# E2E latency (ms) — mean and std, from tier2_e2e_results.json.
LATENCY_MS = {
    "cloud_premium":   {"mean": 1152.75, "std": 398.27, "p95": 1619.78},
    "ondevice_fast":   {"mean":  993.19, "std": 215.62, "p95": 1449.05},
    "hybrid_balanced": {"mean": 1119.83, "std": 341.70, "p95": 1650.64},
}

# Per-turn quality / cost / energy, derived from component_ablation_results.json
# (quality/cost/energy are consistent across "Always-*" rows in the ablation).
QUALITY = {
    "cloud_premium":   0.8746,   # Always-Cloud
    "ondevice_fast":   0.6276,   # Always-OnDevice
    "hybrid_balanced": 0.7511,   # mean of the two, used as stand-in
}
COST_USD = {
    "cloud_premium":   0.025,
    "ondevice_fast":   0.005,
    "hybrid_balanced": 0.015,
}
ENERGY_MJ = {
    "cloud_premium":   185.0,
    "ondevice_fast":   365.0,    # edge LLMs on-battery are surprisingly energy-
                                 # hungry; see the paper for discussion.
    "hybrid_balanced": 275.0,
}

# "Coupling violation" — when downstream LLM got handed a transcript it can't
# recover from. From the ablation, only NoCoupling / infeasible edge paths
# violate. For the public three-way space we approximate infeasibility as
# "edge selected on high-complexity turn (complexity >= 4)".
def infeasible_for_turn(profile: str, complexity: int) -> bool:
    return profile == "ondevice_fast" and complexity >= 4
