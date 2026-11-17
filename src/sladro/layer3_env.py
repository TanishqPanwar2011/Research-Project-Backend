"""Layer 3 (DQN task scheduling): environment, features and numpy-only inference.

No torch in this file, so the API can import it. Training lives in layer3.py.

Design
------
The scheduler sees tasks one at a time. For every VM it builds a small feature vector, and ONE shared
network scores each (task, VM) pair: Q(state, "send task to VM m") = net(features_m). The action is the
VM with the highest score. Because the network scores VMs one by one, the same weights work for any number
of VMs (the UI allows 2 to 20) without retraining.

The agent only sees the ESTIMATED task length (Layer 2's forecast, or the average when Layer 2 is off).
The environment charges the TRUE length. So a better predictor really does give a better schedule.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MIPS = np.array([1000, 1500, 2000, 3000, 4000], dtype=float)
POWER = np.array([60, 80, 110, 150, 200], dtype=float)
N_FEATURES = 6
MAX_VMS = 20
REWARD_SCALE = 3.0   # seconds: roughly one typical task, keeps rewards near -1..0
EXEC_WEIGHT = 0.25   # small cost for running time, so the agent prefers fast VMs when makespan ties


def to_len(v, scale: float) -> np.ndarray:
    """Layer 2 CPU value -> task length in million instructions (same formula as the API and UI)."""
    return 800 + 6000 * np.clip(np.asarray(v, dtype=float) / scale, 0, 1)


def vm_specs(n_vms: int):
    idx = np.arange(n_vms) % 5
    return MIPS[idx], POWER[idx]


def vm_features(f: np.ndarray, est: float, mips: np.ndarray, power: np.ndarray) -> np.ndarray:
    """One row per VM. f = time each VM is busy until. Times are divided by tau (the task's run time on a
    mid-speed VM) so the numbers stay on the same scale for short and long tasks."""
    tau = max(est / 2000.0, 1e-6)
    run = est / mips
    finish = f + run
    x = np.stack([
        mips / 4000.0,                      # how fast the VM is
        run / tau,                          # how long this task would run there
        (f - f.min()) / tau / 5.0,          # queue wait compared with the emptiest VM
        (finish - finish.min()) / tau / 5.0,  # finish time compared with the best option
        power / 200.0,                      # power draw
        (f - f.mean()) / tau / 5.0,         # load compared with the average VM
    ], axis=1)
    return np.clip(x, -5, 5).astype(np.float32)


class SchedEnv:
    def __init__(self, true_len, est_len, n_vms: int):
        self.true = np.asarray(true_len, dtype=float)
        self.est = np.asarray(est_len, dtype=float)
        self.n_vms = n_vms
        self.mips, self.power = vm_specs(n_vms)
        self.f = np.zeros(n_vms)
        self.i = 0

    @property
    def done(self) -> bool:
        return self.i >= len(self.true)

    def features(self) -> np.ndarray:
        return vm_features(self.f, self.est[self.i], self.mips, self.power)

    def step(self, a: int) -> float:
        """Run the current task on VM `a`. Reward: how much the makespan grew, plus a small run-time cost."""
        mk0 = self.f.max()
        run = self.true[self.i] / self.mips[a]
        self.f[a] += run
        self.i += 1
        return -((self.f.max() - mk0) + EXEC_WEIGHT * run) / REWARD_SCALE


class DQNPolicy:
    """Small MLP (6 -> 64 -> 64 -> 1) evaluated with numpy. Weights come from artifacts/layer3.json."""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]]):
        self.layers = layers

    @classmethod
    def from_dict(cls, d: dict) -> "DQNPolicy":
        return cls([(np.asarray(l["W"], dtype=np.float32), np.asarray(l["b"], dtype=np.float32)) for l in d["layers"]])

    @classmethod
    def load(cls, path) -> "DQNPolicy":
        return cls.from_dict(json.loads(Path(path).read_text())["policy"])

    def q(self, feats: np.ndarray) -> np.ndarray:
        h = feats
        for k, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if k < len(self.layers) - 1:
                h = np.maximum(h, 0)
        return h[..., 0]

    def act(self, feats: np.ndarray) -> int:
        return int(np.argmax(self.q(feats)))


def run_schedule(true_len, est_len, n_vms: int, mode: str, policy: DQNPolicy | None = None) -> np.ndarray:
    """Return each VM's total busy time. mode: 'rr' (Round Robin), 'earliest' (greedy baseline), 'dqn'."""
    if mode == "dqn" and policy is None:
        raise ValueError("mode 'dqn' needs a policy")
    env = SchedEnv(true_len, est_len, n_vms)
    while not env.done:
        if mode == "rr":
            a = env.i % n_vms
        elif mode == "earliest":
            a = int(np.argmin(env.features()[:, 3]))
        else:
            a = policy.act(env.features())
        env.step(a)
    return env.f