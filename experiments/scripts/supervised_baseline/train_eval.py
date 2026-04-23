#!/usr/bin/env python3
"""
Train and evaluate routing methods: Logistic Regression, Random Forest,
XGBoost, MLP (cross-entropy), MLP (PPO). 80/20 split, seed=42.
"""
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Load data ────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X = np.load("X.npy").astype(np.float32)
y_raw = np.load("y.npy").astype(np.int64)   # profile indices (may be non-contiguous)

# Map to 0-indexed labels for XGBoost / PyTorch
le = LabelEncoder()
y_enc = le.fit_transform(y_raw).astype(np.int64)  # 0..N_CLASSES-1
N_CLASSES = len(le.classes_)                         # actual unique classes

PROFILE_IDX = le.classes_   # mapping: encoded label i -> profile index PROFILE_IDX[i]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=SEED
)
# Keep raw labels for cost computation
_, _, y_train_raw, y_test_raw = train_test_split(
    X, y_raw, test_size=0.2, random_state=SEED
)
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}  Classes: {N_CLASSES}")

# ── Cost function (same as generate_data.py) ─────────────────────────────────
ASR_CONFIGS = [
    ("Parakeet_FP16", 65,  1.9),
    ("Parakeet_INT8", 48,  3.1),
    ("Parakeet_INT4", 38,  4.2),
    ("Conformer_INT8", 31, 6.8),
]
LLM_CONFIGS = [
    ("Llama70B",      4200,  800, 0.921),
    ("Llama8B",       2900,  550, 0.893),
    ("Gemma12B",      2100,  400, 0.876),
    ("Gemma4B_INT8", 18000, 3400, 0.844),
    ("Gemma4B_INT4",  9500, 1800, 0.821),
]
TTS_CONFIGS = [
    ("Cloud",   210, 4.3),
    ("MeloTTS", 310, 4.0),
    ("Kokoro",  680, 3.9),
]
PROFILES = [(asr, llm, tts) for asr in ASR_CONFIGS for llm in LLM_CONFIGS for tts in TTS_CONFIGS]

FC_LAT = (ASR_CONFIGS[0][1] + LLM_CONFIGS[0][1] + TTS_CONFIGS[0][1]) / 1000
FC_E   = FC_LAT * 20
FC_Q   = 0.4 * (1 - ASR_CONFIGS[0][2]/100) + 0.4 * LLM_CONFIGS[0][3] + 0.2 * (TTS_CONFIGS[0][2]/5)

def profile_cost(profile_idx: int, use_short: bool) -> float:
    asr, llm, tts = PROFILES[profile_idx]
    if asr[2] > 2.0 and llm[0] in ("Gemma4B_INT8", "Gemma4B_INT4"):
        return float("inf")
    llm_lat = llm[2] if use_short else llm[1]
    lat = (asr[1] + llm_lat + tts[1]) / 1000
    e   = lat * 20
    q   = 0.4*(1-asr[2]/100) + 0.4*llm[3] + 0.2*(tts[2]/5)
    return 0.25*(lat/FC_LAT) + 0.25*(e/FC_E) + 0.25 - 0.25*(q/FC_Q)

def mean_cost_raw(preds_raw: np.ndarray, X_arr: np.ndarray) -> float:
    """Compute mean routing cost given raw (non-encoded) profile indices."""
    total = 0.0
    for i, p in enumerate(preds_raw):
        use_short = (X_arr[i, 0] > 5.5) and (X_arr[i, 3] < 4.0)
        c = profile_cost(int(p), use_short)
        total += c if c != float("inf") else 5.0
    return total / len(preds_raw)

def decode_preds(preds_enc: np.ndarray) -> np.ndarray:
    """Convert encoded (0-indexed) predictions back to profile indices."""
    return PROFILE_IDX[preds_enc]

def mean_cost(preds_enc: np.ndarray, X_arr: np.ndarray) -> float:
    return mean_cost_raw(decode_preds(preds_enc), X_arr)

def oracle_cost(X_arr: np.ndarray) -> float:
    return mean_cost_raw(y_test_raw, X_arr)

def top3_accuracy(probs: np.ndarray, y_true_enc: np.ndarray) -> float:
    """Top-3 accuracy: probs has shape (n, N_CLASSES) with 0-indexed columns."""
    if probs.shape[1] < 3:
        # Fewer than 3 classes: top-k = top-all
        top3 = np.argsort(probs, axis=1)
    else:
        top3 = np.argsort(probs, axis=1)[:, -3:]
    return float(np.mean([y_true_enc[i] in top3[i] for i in range(len(y_true_enc))]))

results = {}
oc = oracle_cost(X_test)   # computed once from ground-truth labels

# ─────────────────────────────────────────────────────────────────────────────
# 1. Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Logistic Regression...")
from sklearn.linear_model import LogisticRegression
t0 = time.time()
lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
lr.fit(X_train, y_train)
train_time = time.time() - t0
preds = lr.predict(X_test)
probs = lr.predict_proba(X_test)
acc   = float(np.mean(preds == y_test))
top3  = top3_accuracy(probs, y_test)
mc    = mean_cost(preds, X_test)
gap   = (mc - oc) / abs(oc) * 100 if oc != 0 else 0.0
# inference latency
t_inf = time.perf_counter_ns()
for _ in range(1000): lr.predict(X_test[:1])
lat_us = (time.perf_counter_ns() - t_inf) / 1000 / 1000
results["Logistic Reg."] = dict(acc=acc, top3=top3, cost_gap=gap, lat_us=lat_us, train_s=train_time)
print(f"  Acc={acc:.3f}  Top3={top3:.3f}  CostGap={gap:.2f}%  Lat={lat_us:.1f}µs  Train={train_time:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Random Forest
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Random Forest...")
from sklearn.ensemble import RandomForestClassifier
t0 = time.time()
rf = RandomForestClassifier(n_estimators=300, max_depth=25, random_state=SEED, n_jobs=4)
rf.fit(X_train, y_train)
train_time = time.time() - t0
preds = rf.predict(X_test)
probs = rf.predict_proba(X_test)
acc   = float(np.mean(preds == y_test))
top3  = top3_accuracy(probs, y_test)
mc    = mean_cost(preds, X_test)
gap   = (mc - oc) / abs(oc) * 100
t_inf = time.perf_counter_ns()
rf.set_params(n_jobs=1)   # single-threaded for latency measurement
for _ in range(1000): rf.predict(X_test[:1])
lat_us = (time.perf_counter_ns() - t_inf) / 1000 / 1000
results["Random Forest"] = dict(acc=acc, top3=top3, cost_gap=gap, lat_us=lat_us, train_s=train_time)
print(f"  Acc={acc:.3f}  Top3={top3:.3f}  CostGap={gap:.2f}%  Lat={lat_us:.1f}µs  Train={train_time:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 3. XGBoost  (run in subprocess to isolate potential segfaults)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] XGBoost...")
import subprocess, sys, tempfile, pickle

_xgb_script = """
import numpy as np, json, time, sys, pickle
from xgboost import XGBClassifier
SEED = 42
X_train = np.load("X_train_tmp.npy").astype(np.float32)
X_test  = np.load("X_test_tmp.npy").astype(np.float32)
y_train = np.load("y_train_tmp.npy").astype(np.int64)
y_test  = np.load("y_test_tmp.npy").astype(np.int64)
t0 = time.time()
xgb = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                    random_state=SEED, n_jobs=-1,
                    eval_metric="mlogloss", verbosity=0)
xgb.fit(X_train, y_train)
train_time = time.time() - t0
preds = xgb.predict(X_test)
probs = xgb.predict_proba(X_test)
acc = float(np.mean(preds == y_test))
top3_idx = np.argsort(probs, axis=1)[:, -3:] if probs.shape[1] >= 3 else np.argsort(probs, axis=1)
top3 = float(np.mean([y_test[i] in top3_idx[i] for i in range(len(y_test))]))
t_inf = time.perf_counter_ns()
for _ in range(1000): xgb.predict(X_test[:1])
lat_us = (time.perf_counter_ns() - t_inf) / 1000 / 1000
# mean cost: decode preds back to profile indices, compute cost
import sys; sys.path.insert(0, '.')
from train_eval_cost import mean_cost_raw, decode_preds_fn, PROFILE_IDX
preds_raw = decode_preds_fn(preds)
mc = mean_cost_raw(preds_raw)
print(json.dumps({"acc":acc,"top3":top3,"train_s":train_time,"lat_us":lat_us,"mc":mc}))
"""

# Save tmp arrays and a cost helper module
np.save("X_train_tmp.npy", X_train)
np.save("X_test_tmp.npy",  X_test)
np.save("y_train_tmp.npy", y_train)
np.save("y_test_tmp.npy",  y_test)
np.save("PROFILE_IDX_tmp.npy", PROFILE_IDX)

cost_helper = f"""
import numpy as np
PROFILES_DATA = {repr([(list(a), list(l), list(t)) for a, l, t in PROFILES])}
PROFILE_IDX = np.load("PROFILE_IDX_tmp.npy")
FC_LAT = {FC_LAT}; FC_E = {FC_E}; FC_Q = {FC_Q}
def decode_preds_fn(preds):
    return PROFILE_IDX[preds]
def profile_cost_fn(prof_idx, use_short):
    asr, llm, tts = PROFILES_DATA[prof_idx]
    if asr[2] > 2.0 and llm[0] in ("Gemma4B_INT8","Gemma4B_INT4"):
        return float("inf")
    llm_lat = llm[2] if use_short else llm[1]
    lat = (asr[1] + llm_lat + tts[1]) / 1000
    e   = lat * 20
    q   = 0.4*(1-asr[2]/100) + 0.4*llm[3] + 0.2*(tts[2]/5)
    return 0.25*(lat/FC_LAT) + 0.25*(e/FC_E) + 0.25 - 0.25*(q/FC_Q)
def mean_cost_raw(preds_raw):
    X_test = np.load("X_test_tmp.npy")
    total = 0.0
    for i, p in enumerate(preds_raw):
        use_short = (X_test[i,0] > 5.5) and (X_test[i,3] < 4.0)
        c = profile_cost_fn(int(p), use_short)
        total += c if c != float("inf") else 5.0
    return total / len(preds_raw)
"""
with open("train_eval_cost.py", "w") as f:
    f.write(cost_helper)

try:
    r = subprocess.run(
        [sys.executable, "-c", _xgb_script],
        capture_output=True, text=True, timeout=300, cwd="."
    )
    if r.returncode == 0:
        d = json.loads(r.stdout.strip().split("\n")[-1])
        mc = d["mc"]
        gap = (mc - oc) / abs(oc) * 100 if oc != 0 else 0.0
        results["XGBoost"] = dict(acc=d["acc"], top3=d["top3"], cost_gap=gap,
                                   lat_us=d["lat_us"], train_s=d["train_s"])
        print(f"  Acc={d['acc']:.3f}  Top3={d['top3']:.3f}  CostGap={gap:.2f}%  "
              f"Lat={d['lat_us']:.1f}µs  Train={d['train_s']:.1f}s")
    else:
        print(f"  XGBoost subprocess failed (rc={r.returncode}): {r.stderr[:200]}")
        results["XGBoost"] = None
except Exception as e:
    print(f"  XGBoost error: {e}")
    results["XGBoost"] = None

# Cleanup tmp files
for f in ["X_train_tmp.npy","X_test_tmp.npy","y_train_tmp.npy","y_test_tmp.npy",
          "PROFILE_IDX_tmp.npy", "train_eval_cost.py"]:
    try: __import__("os").remove(f)
    except: pass

# ─────────────────────────────────────────────────────────────────────────────
# 4. MLP (Cross-Entropy)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] MLP (Cross-Entropy)...")

class MLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, n_out),
        )
    def forward(self, x): return self.net(x)

def train_mlp_ce(X_tr, y_tr, epochs=100, batch=512, lr=3e-4):
    ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model = MLP(X_tr.shape[1], N_CLASSES)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    for epoch in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        if (epoch+1) % 25 == 0:
            print(f"    epoch {epoch+1}/{epochs}  t={time.time()-t0:.1f}s")
    return model, time.time()-t0

model_ce, train_time = train_mlp_ce(X_train, y_train)
model_ce.eval()
with torch.no_grad():
    logits = model_ce(torch.tensor(X_test))
    preds  = logits.argmax(1).numpy()
    probs  = torch.softmax(logits, 1).numpy()
acc   = float(np.mean(preds == y_test))
top3  = top3_accuracy(probs, y_test)
mc    = mean_cost(preds, X_test)
gap   = (mc - oc) / abs(oc) * 100
xt = torch.tensor(X_test[:1])
t_inf = time.perf_counter_ns()
for _ in range(1000):
    with torch.no_grad(): model_ce(xt)
lat_us = (time.perf_counter_ns() - t_inf) / 1000 / 1000
results["MLP (CE)"] = dict(acc=acc, top3=top3, cost_gap=gap, lat_us=lat_us, train_s=train_time)
print(f"  Acc={acc:.3f}  Top3={top3:.3f}  CostGap={gap:.2f}%  Lat={lat_us:.1f}µs  Train={train_time:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MLP (PPO)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] MLP (PPO)...")

# Pre-compute feasibility mask over the N_CLASSES encoded labels
# PROFILE_IDX[i] gives the actual profile index for encoded class i
feasible = np.ones(N_CLASSES, dtype=bool)
for enc_i, prof_idx in enumerate(PROFILE_IDX):
    asr, llm, tts = PROFILES[prof_idx]
    if asr[2] > 2.0 and llm[0] in ("Gemma4B_INT8", "Gemma4B_INT4"):
        feasible[enc_i] = False
FEASIBLE_MASK = torch.tensor(feasible, dtype=torch.bool)
INF_COST = 5.0

def compute_reward(action_enc: int, x_row: np.ndarray, prev_action: int) -> float:
    """action_enc is 0-indexed; decode to profile index first."""
    prof_idx = int(PROFILE_IDX[action_enc])
    use_short = (x_row[0] > 5.5) and (x_row[3] < 4.0)
    c = profile_cost(prof_idx, use_short)
    if c == float("inf"):
        c = INF_COST
    switch_pen = 0.02 if (prev_action != action_enc and prev_action >= 0) else 0.0
    return -c - switch_pen

class PPOBuffer:
    def __init__(self):
        self.states, self.actions, self.rewards = [], [], []
        self.log_probs, self.values = [], []

    def clear(self):
        self.states.clear(); self.actions.clear(); self.rewards.clear()
        self.log_probs.clear(); self.values.clear()

class ActorCritic(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(n_in, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.actor  = nn.Linear(256, n_out)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def act(self, x, feasible_mask=None):
        logits, value = self(x)
        if feasible_mask is not None:
            logits = logits.masked_fill(~feasible_mask, -1e9)
        dist  = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

CLIP_EPS = 0.2; GAMMA = 0.95; LAM = 0.95
N_STEPS = 100_000; BATCH_SIZE = 512; N_EPOCHS_PPO = 4; LR_PPO = 3e-4

ac_model  = ActorCritic(X.shape[1], N_CLASSES)
optimizer = optim.Adam(ac_model.parameters(), lr=LR_PPO)
buf = PPOBuffer()

X_tr_t = torch.tensor(X_train)
idx_pool = list(range(len(X_train)))

t0 = time.time()
prev_action = -1
step = 0

while step < N_STEPS:
    # Collect a mini-rollout
    buf.clear()
    sample_idx = np.random.choice(idx_pool, size=BATCH_SIZE, replace=True)
    states_b   = X_tr_t[sample_idx]

    with torch.no_grad():
        actions, lp, vals = ac_model.act(states_b, FEASIBLE_MASK)

    rewards = torch.tensor([
        compute_reward(int(actions[i]), X_train[sample_idx[i]], prev_action)
        for i in range(BATCH_SIZE)
    ], dtype=torch.float32)
    prev_action = int(actions[-1])

    # PPO update (single-step returns, no GAE across unrelated samples)
    returns = rewards  # single-step; no bootstrapping needed
    advantages = returns - vals.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(N_EPOCHS_PPO):
        new_logits, new_vals = ac_model(states_b)
        new_logits = new_logits.masked_fill(~FEASIBLE_MASK, -1e9)
        new_dist   = torch.distributions.Categorical(logits=new_logits)
        new_lp     = new_dist.log_prob(actions)

        ratio      = (new_lp - lp.detach()).exp()
        surr1      = ratio * advantages
        surr2      = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss= nn.functional.mse_loss(new_vals.squeeze(-1), returns)
        loss       = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(ac_model.parameters(), 0.5)
        optimizer.step()

    step += BATCH_SIZE
    if step % 10_000 == 0:
        print(f"    PPO step {step:,}/{N_STEPS:,}  loss={loss.item():.4f}  t={time.time()-t0:.1f}s")

train_time_ppo = time.time() - t0
print(f"  PPO training done in {train_time_ppo:.1f}s")

# Evaluate PPO
ac_model.eval()
with torch.no_grad():
    logits_ppo, _ = ac_model(torch.tensor(X_test))
    logits_ppo = logits_ppo.masked_fill(~FEASIBLE_MASK, -1e9)
    preds_ppo  = logits_ppo.argmax(1).numpy()
    probs_ppo  = torch.softmax(logits_ppo, 1).numpy()

acc_ppo  = float(np.mean(preds_ppo == y_test))
top3_ppo = top3_accuracy(probs_ppo, y_test)
mc_ppo   = mean_cost(preds_ppo, X_test)
gap_ppo  = (mc_ppo - oc) / abs(oc) * 100

xt = torch.tensor(X_test[:1])
t_inf = time.perf_counter_ns()
for _ in range(1000):
    with torch.no_grad(): ac_model(xt)
lat_us_ppo = (time.perf_counter_ns() - t_inf) / 1000 / 1000
results["MLP (PPO)"] = dict(acc=acc_ppo, top3=top3_ppo, cost_gap=gap_ppo,
                             lat_us=lat_us_ppo, train_s=train_time_ppo)
print(f"  Acc={acc_ppo:.3f}  Top3={top3_ppo:.3f}  CostGap={gap_ppo:.2f}%  Lat={lat_us_ppo:.1f}µs  Train={train_time_ppo:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────
with open("ablation_results.json", "w") as f:
    json.dump({"oracle_cost": oc, "models": results}, f, indent=2)
print("\nSaved ablation_results.json")
