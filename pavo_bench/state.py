"""12-dimensional state encoding used by the PAVO meta-controller.

The layout matches exp3_train_ppo.py::RoutingEnvironment:
    [SNR/50, speaking_rate/10, pitch_variance, WADA_SNR_proxy,
     cpu_util, ram_fraction, battery, gpu_util,
     RTT/200, bandwidth_proxy,
     complexity/5, context_tokens/2000]

Only four of the twelve values vary per turn in PAVO-Bench (acoustic/hardware
proxies). The remaining eight are held at their training-time defaults.
"""
from __future__ import annotations

import numpy as np

from .dataset import PAVOBenchTurn

_DEFAULT_SPEAKING_RATE = 4.0 / 10.0
_DEFAULT_PITCH_VAR = 0.5
_DEFAULT_RAM_FRACTION = 0.8
_DEFAULT_GPU_UTIL = 0.3
_DEFAULT_BANDWIDTH = 0.5


def turn_to_state_vector(turn: PAVOBenchTurn) -> np.ndarray:
    """Convert a PAVOBenchTurn to the 12-dim state vector the router expects."""
    snr = max(min(turn.snr_db, 50.0), 0.0)
    return np.asarray([
        snr / 50.0,
        _DEFAULT_SPEAKING_RATE,
        _DEFAULT_PITCH_VAR,
        snr / 50.0,                  # WADA proxy (same as training default)
        turn.cpu_util,
        _DEFAULT_RAM_FRACTION,
        turn.battery,
        _DEFAULT_GPU_UTIL,
        turn.rtt_ms / 200.0,
        _DEFAULT_BANDWIDTH,
        turn.complexity / 5.0,
        turn.ctx_tokens / 2000.0,
    ], dtype=np.float32)
