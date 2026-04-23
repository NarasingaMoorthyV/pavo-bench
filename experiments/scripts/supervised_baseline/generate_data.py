#!/usr/bin/env python3
"""
Generate 100,000 synthetic state vectors with oracle routing labels.
Matches the demand vector distribution from the PAVO paper (Section 4).
Oracle labels are feature-dependent: SNR drives ASR choice, CPU util drives
on-device vs cloud LLM, battery drives energy weighting, context length drives
output-length decision.
"""
import time
import numpy as np

SEED = 42
N_SAMPLES = 100_000
OUTPUT_X = "X.npy"
OUTPUT_Y = "y.npy"

# ── Routing profile definitions (60 = 4 ASR × 5 LLM × 3 TTS) ────────────────
# ASR: (name, latency_ms, wer_pct_baseline)
ASR_CONFIGS = [
    ("Parakeet_FP16",  65,  1.9),
    ("Parakeet_INT8",  48,  3.1),
    ("Parakeet_INT4",  38,  4.2),
    ("Conformer_INT8", 31,  6.8),
]
# LLM: (name, lat_80tok_ms, lat_15tok_ms, bertscore)
LLM_CONFIGS = [
    ("Llama70B",      4200,  800,  0.921),
    ("Llama8B",       2900,  550,  0.893),
    ("Gemma12B",      2100,  400,  0.876),
    ("Gemma4B_INT8", 18000, 3400,  0.844),
    ("Gemma4B_INT4",  9500, 1800,  0.821),
]
# TTS: (name, latency_ms, mos)
TTS_CONFIGS = [
    ("Cloud",   210, 4.3),
    ("MeloTTS", 310, 4.0),
    ("Kokoro",  680, 3.9),
]

CLOUD_LLMS = {"Llama70B", "Llama8B", "Gemma12B"}
ONDEVICE_QUANT_LLMS = {"Gemma4B_INT8", "Gemma4B_INT4"}

# Fixed-Cloud baseline for normalisation (Parakeet_FP16 + Llama70B + Cloud)
FC_LAT_MS  = ASR_CONFIGS[0][1] + LLM_CONFIGS[0][1] + TTS_CONFIGS[0][1]  # 4475 ms
FC_ENERGY  = FC_LAT_MS / 1000 * 20   # J
FC_QUALITY = (0.4 * (1 - ASR_CONFIGS[0][2] / 100)
              + 0.4 * LLM_CONFIGS[0][3]
              + 0.2 * (TTS_CONFIGS[0][2] / 5))  # 0.9328

# Build profile list
PROFILES = []
for asr in ASR_CONFIGS:
    for llm in LLM_CONFIGS:
        for tts in TTS_CONFIGS:
            PROFILES.append((asr, llm, tts))
N_CLASSES = len(PROFILES)   # 60


def compute_cost(asr, llm, tts, x) -> float:
    """
    Feature-dependent cost J for one routing profile.
    Returns inf if infeasible.

    Features used (indices into x):
      x[2]  snr_db        (5 – 45)
      x[4]  cpu_util      (0 – 1)
      x[6]  battery       (0 – 1)
      x[8]  rtt_ms        (10 – 500)
      x[11] context_tokens (50 – 8000)
    """
    asr_name, asr_lat_ms, asr_wer_base = asr
    llm_name, llm_lat80, llm_lat15, bs  = llm
    tts_name, tts_lat_ms, mos           = tts

    snr         = float(x[2])
    cpu_util    = float(x[4])
    battery     = float(x[6])
    rtt_ms      = float(x[8])
    ctx_tokens  = float(x[11])

    # ── SNR degrades ASR WER ────────────────────────────────────────────────
    if snr < 15:
        snr_mult = 1.0 + (15 - snr) / 15 * 2.0   # up to 3× at SNR=5
    elif snr < 25:
        snr_mult = 1.0 + (25 - snr) / 25 * 0.5   # up to 1.5× at SNR=15
    else:
        snr_mult = 1.0
    effective_wer = min(asr_wer_base * snr_mult, 60.0)

    # ── Coupling constraint ─────────────────────────────────────────────────
    if effective_wer > 2.0 and llm_name in ONDEVICE_QUANT_LLMS:
        return float("inf")

    # ── Output length: long for complex (high context) queries ─────────────
    use_short = ctx_tokens < 300
    llm_lat_ms = llm_lat15 if use_short else llm_lat80

    # ── On-device LLM: CPU contention slows inference ──────────────────────
    if llm_name not in CLOUD_LLMS:
        llm_lat_ms = llm_lat_ms * (1.0 + cpu_util * 3.0)

    # ── Network RTT added to cloud components ──────────────────────────────
    if llm_name in CLOUD_LLMS:
        llm_lat_ms += rtt_ms
    if tts_name == "Cloud":
        tts_lat_ms_eff = tts_lat_ms + rtt_ms
    else:
        tts_lat_ms_eff = tts_lat_ms

    total_lat_s = (asr_lat_ms + llm_lat_ms + tts_lat_ms_eff) / 1000.0
    energy_J    = total_lat_s * 20.0  # 20 W TDP proxy

    quality = (0.4 * (1 - effective_wer / 100)
               + 0.4 * bs
               + 0.2 * (mos / 5))

    L_hat = total_lat_s / (FC_LAT_MS / 1000.0)
    E_hat = energy_J    / FC_ENERGY
    Q     = quality     / FC_QUALITY

    # ── Adaptive weights ────────────────────────────────────────────────────
    # Battery low → heavier energy penalty
    e_w = 0.15 + 0.20 * (1.0 - battery)   # [0.15, 0.35]
    l_w = 0.35                              # fixed latency weight
    m_w = 0.10                              # fixed memory proxy
    q_w = 1.0 - l_w - e_w - m_w           # [0.20, 0.40]

    return l_w * L_hat + e_w * E_hat + m_w - q_w * Q


def generate():
    rng = np.random.default_rng(SEED)
    t0  = time.time()

    X = np.zeros((N_SAMPLES, 12), dtype=np.float32)
    y = np.zeros(N_SAMPLES,       dtype=np.int32)

    # ── Sample features ─────────────────────────────────────────────────────
    X[:, 0]  = np.clip(rng.normal(4.5, 1.2, N_SAMPLES), 1, 10)            # speaking_rate
    X[:, 1]  = np.clip(rng.lognormal(4.5, 1.0, N_SAMPLES), 1, 5000)       # pitch_variance
    X[:, 2]  = rng.uniform(5, 45, N_SAMPLES)                               # snr_db
    X[:, 3]  = np.clip(rng.lognormal(1.5, 0.6, N_SAMPLES), 0.5, 15)       # duration_s
    X[:, 4]  = rng.beta(2, 5, N_SAMPLES)                                   # cpu_util
    X[:, 5]  = rng.beta(5, 2, N_SAMPLES)                                   # ram_avail
    X[:, 6]  = rng.uniform(0.05, 1.0, N_SAMPLES)                           # battery (down to 5%)
    X[:, 7]  = np.zeros(N_SAMPLES)                                          # gpu_util (N/A)
    X[:, 8]  = np.clip(rng.lognormal(3.4, 0.8, N_SAMPLES), 10, 500)        # rtt_ms (wider spread)
    X[:, 9]  = np.clip(rng.lognormal(3.9, 0.4, N_SAMPLES), 5, 200)         # bandwidth
    X[:, 10] = rng.integers(1, 51, N_SAMPLES).astype(np.float32)            # turn_index
    X[:, 11] = np.clip(rng.exponential(600, N_SAMPLES), 50, 8000)           # context_tokens

    # ── Oracle labels ────────────────────────────────────────────────────────
    for i in range(N_SAMPLES):
        costs = [compute_cost(p[0], p[1], p[2], X[i]) for p in PROFILES]
        y[i]  = int(np.argmin(costs))

        if (i + 1) % 10_000 == 0:
            pct = (i + 1) / N_SAMPLES * 100
            print(f"  {i+1:,}/{N_SAMPLES:,} ({pct:.0f}%) — {time.time()-t0:.1f}s elapsed")

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)
    print(f"\nSaved {OUTPUT_X} shape={X.shape}, {OUTPUT_Y} shape={y.shape}")
    n_unique = len(np.unique(y))
    print(f"Unique classes used: {n_unique} / {N_CLASSES}")
    label_dist = np.bincount(y, minlength=N_CLASSES)
    top5 = np.argsort(label_dist)[::-1][:5]
    print("Top-5 routing profiles:")
    for idx in top5:
        asr, llm, tts = PROFILES[idx]
        print(f"  [{idx:2d}] {asr[0]}+{llm[0]}+{tts[0]}: {label_dist[idx]:,} samples")


if __name__ == "__main__":
    print(f"Generating {N_SAMPLES:,} synthetic state vectors...")
    generate()
    print("Done.")
