#!/usr/bin/env python3
"""
Experiment 3: Train the PPO meta-controller.
79K-parameter MLP [12, 256, 256, 48] with multi-objective PPO.
"""

import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class MetaController(nn.Module):
    """79K-parameter MLP meta-controller."""
    def __init__(self, state_dim=12, hidden=256, n_profiles=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_profiles),
        )
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        logits = self.net(state)
        value = self.value_head(state)
        return logits, value

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RoutingEnvironment:
    """Simulates the voice pipeline routing environment using real data."""

    def __init__(self, data_path, n_profiles=48):
        self.n_profiles = n_profiles
        self.data = []
        self.idx = 0

        # Load training data
        print(f"  Loading training data from {data_path}...")
        with open(data_path) as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    self.data.append(record)
                except:
                    continue
        print(f"  Loaded {len(self.data)} training turns")

        # Define routing profiles (k-means clustered from config space)
        # Each profile: (latency_factor, energy_factor, quality_factor, feasible_complexity_range)
        np.random.seed(42)
        self.profiles = []
        for i in range(n_profiles):
            lat = 0.5 + (i / n_profiles) * 2.0  # latency multiplier 0.5x - 2.5x
            energy = 0.3 + (i / n_profiles) * 1.5
            quality = 1.0 - (i / n_profiles) * 0.3  # quality drops as we go cheaper
            max_complexity = 1 + int((i / n_profiles) * 5)  # cheaper = only simple queries
            self.profiles.append({
                "latency_factor": lat,
                "energy_factor": energy,
                "quality_factor": quality,
                "max_complexity": min(max_complexity, 5),
            })

        # Coupling threshold
        self.theta_wer = 0.02  # 2% WER threshold

        # Objective weights (balanced)
        self.w_L = 0.25
        self.w_E = 0.25
        self.w_M = 0.25
        self.w_Q = 0.25

        # Reference latencies (from real H100 measurements)
        self.L_ref = 1153.0  # cloud premium mean
        self.E_ref = 6.82    # cloud premium energy
        self.M_ref = 1.0     # memory fraction reference

    def get_state(self, record):
        """Extract 12-dimensional state from a data record."""
        snr = record.get("snr_db", record.get("snr", 20.0))
        cpu = record.get("cpu_util", 0.5)
        battery = record.get("battery", 0.8)
        rtt = record.get("rtt_ms", 50.0)
        ctx_tokens = record.get("ctx_tokens", record.get("context_tokens", 200))
        complexity = record.get("complexity", 3)

        # 12-dim state: [acoustic(4), hardware(4), network(2), context(2)]
        state = np.array([
            snr / 50.0,                    # normalized SNR
            4.0 / 10.0,                    # speaking rate (default)
            0.5,                            # pitch variance (default)
            min(max(snr, 0), 50) / 50.0,   # WADA-SNR proxy
            cpu,                            # CPU utilization
            0.8,                            # RAM fraction
            battery,                        # battery level
            0.3,                            # GPU utilization
            rtt / 200.0,                    # normalized RTT
            0.5,                            # bandwidth proxy
            complexity / 5.0,               # turn complexity
            ctx_tokens / 2000.0,            # context depth
        ], dtype=np.float32)

        return state

    def get_feasible_mask(self, state):
        """Return mask of feasible profiles given current state."""
        complexity = int(state[10] * 5)
        snr = state[0] * 50.0
        mask = np.ones(self.n_profiles, dtype=np.float32)

        for i, profile in enumerate(self.profiles):
            # Coupling constraint: cheap profiles can't handle complex queries
            if complexity > profile["max_complexity"]:
                mask[i] = 0.0
            # Low SNR further restricts cheap ASR
            if snr < 10 and profile["quality_factor"] < 0.8:
                mask[i] = 0.0

        # Ensure at least one profile is feasible (cloud fallback)
        if mask.sum() == 0:
            mask[-1] = 1.0  # Last profile = cloud premium (always feasible)

        return mask

    def step(self, state, action, record):
        """Execute action and return reward."""
        profile = self.profiles[action]
        complexity = record.get("complexity", 3)

        # Base latency depends on complexity and profile
        base_lat = 800 + complexity * 400
        latency = base_lat * profile["latency_factor"]
        energy = (latency / 1000.0) * profile["energy_factor"]
        quality = profile["quality_factor"]
        memory = 0.3 + 0.1 * profile["latency_factor"]

        # Coupling violation check
        violation = 0
        if complexity <= 2 and profile["quality_factor"] < 0.85:
            violation = 1
            quality *= 0.7  # Quality degrades under violation

        # Normalized metrics
        L_hat = latency / self.L_ref
        E_hat = energy / self.E_ref
        M_hat = memory / self.M_ref

        # Reward (Equation 5 from paper)
        reward = -self.w_L * L_hat - self.w_E * E_hat - self.w_M * M_hat + self.w_Q * quality
        reward -= 0.5 * violation  # coupling violation penalty

        return reward, latency, energy, quality, violation

    def sample_batch(self, batch_size):
        """Sample a batch of training turns."""
        indices = np.random.randint(0, len(self.data), batch_size)
        return [self.data[i] for i in indices]


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    advantages = np.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    returns = advantages + values
    return advantages, returns


def train_meta_controller(data_path, output_dir, n_steps=100000, n_profiles=48):
    """Train the PPO meta-controller."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training on {device}")

    # Initialize
    env = RoutingEnvironment(data_path, n_profiles=n_profiles)
    model = MetaController(state_dim=12, hidden=256, n_profiles=n_profiles).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps // 512)

    print(f"  Model parameters: {model.count_params()}")

    # PPO hyperparameters
    clip_eps = 0.2
    kl_coeff = 0.01
    batch_size = 512
    n_epochs = 4
    gamma = 0.99
    lam = 0.95
    switch_penalty = 0.02

    # Training loop
    training_log = []
    best_reward = -float("inf")
    step = 0
    prev_actions = None

    print(f"  Training for {n_steps} steps...")
    start_time = time.time()

    while step < n_steps:
        # Collect rollout
        states, actions, rewards, values, log_probs, dones, masks_list = [], [], [], [], [], [], []

        records = env.sample_batch(batch_size)

        for i, record in enumerate(records):
            state = env.get_state(record)
            feasible_mask = env.get_feasible_mask(state)

            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            mask_t = torch.FloatTensor(feasible_mask).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, value = model(state_t)
                # Mask infeasible actions
                logits = logits + (mask_t - 1) * 1e9
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            action_int = action.item()
            reward, lat, energy, quality, violation = env.step(state, action_int, record)

            # Switch penalty
            if prev_actions is not None and i < len(prev_actions):
                if action_int != prev_actions[i]:
                    reward -= switch_penalty

            states.append(state)
            actions.append(action_int)
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item())
            dones.append(0)
            masks_list.append(feasible_mask)

        prev_actions = actions.copy()

        # GAE
        rewards_arr = np.array(rewards)
        values_arr = np.array(values)
        dones_arr = np.array(dones)
        advantages, returns = compute_gae(rewards_arr, values_arr, dones_arr, gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states)).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        old_log_probs_t = torch.FloatTensor(log_probs).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)
        returns_t = torch.FloatTensor(returns).to(device)
        masks_t = torch.FloatTensor(np.array(masks_list)).to(device)

        # PPO update epochs
        for epoch in range(n_epochs):
            logits, values_new = model(states_t)
            logits = logits + (masks_t - 1) * 1e9
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns_t - values_new.squeeze()).pow(2).mean()
            kl_loss = (old_log_probs_t - new_log_probs).mean()

            loss = policy_loss + 0.5 * value_loss + kl_coeff * kl_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        scheduler.step()
        step += batch_size

        # Log
        mean_reward = float(np.mean(rewards))
        violations = sum(1 for r in records if
                         env.profiles[actions[records.index(r)]]["quality_factor"] < 0.85
                         and r.get("complexity", 3) <= 2) if len(records) > 0 else 0

        log_entry = {
            "step": step,
            "mean_reward": mean_reward,
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "violations_per_batch": violations,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        training_log.append(log_entry)

        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save(model.state_dict(), os.path.join(output_dir, "meta_controller_best.pt"))

        if step % 5000 == 0 or step <= 1000:
            elapsed = time.time() - start_time
            print(f"    Step {step:>6d}/{n_steps} | reward={mean_reward:.4f} | "
                  f"best={best_reward:.4f} | loss={float(loss.item()):.4f} | "
                  f"entropy={float(entropy.item()):.3f} | {elapsed:.0f}s")

    # Save final model
    final_path = os.path.join(output_dir, "meta_controller.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "architecture": {"state_dim": 12, "hidden": 256, "n_profiles": n_profiles},
        "n_params": model.count_params(),
        "training_steps": n_steps,
        "final_reward": best_reward,
        "hyperparameters": {
            "clip_eps": clip_eps, "kl_coeff": kl_coeff, "batch_size": batch_size,
            "n_epochs": n_epochs, "gamma": gamma, "lam": lam, "lr": 3e-4,
            "switch_penalty": switch_penalty,
        }
    }, final_path)
    print(f"  Model saved to {final_path} ({model.count_params()} parameters)")

    # Save training log
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "training_log": training_log,
            "final_reward": best_reward,
            "total_steps": n_steps,
            "n_params": model.count_params(),
            "training_time_seconds": time.time() - start_time,
            "device": str(device),
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "gpu": "NVIDIA H100 SXM5",
            }
        }, f, indent=2)
    print(f"  Training log saved to {log_path}")

    return {"final_reward": best_reward, "n_params": model.count_params()}


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "../tier3_50k_train.jsonl"
    train_meta_controller(data_path, output_dir="outputs/", n_steps=100000)
