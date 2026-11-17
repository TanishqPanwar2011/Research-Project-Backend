"""Layer 3: DQN task scheduler (PyTorch training). Inference lives in layer3_env.py (numpy only)."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .layer3_env import (MAX_VMS, N_FEATURES, DQNPolicy, SchedEnv, run_schedule, to_len)


class QNet(nn.Module):
    """Scores one (task, VM) pair. Applied to every VM, so it works for any number of VMs."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEATURES, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def export_policy(net: QNet) -> dict:
    return {"layers": [{"W": m.weight.detach().cpu().numpy().T.tolist(), "b": m.bias.detach().cpu().numpy().tolist()}
                       for m in net.net if isinstance(m, nn.Linear)]}


def make_episode(rng, actual, pred, scale, n_tasks, n_vms):
    """Tasks are real Layer 2 test samples. est = the forecast, or the average when the predictor is off."""
    pick = rng.integers(len(actual), size=n_tasks)
    true = to_len(actual[pick], scale)
    est = to_len(pred[pick], scale) if pred is not None else np.full(n_tasks, true.mean())
    return SchedEnv(true, est, n_vms)


def train_dqn(actual, preds: dict, scale: float, episodes=800, seed=0, gamma=0.9, lr=1e-3, batch=128,
              buffer=60_000, eps_end=0.05, target_every=500, log=print):
    """actual: real CPU values. preds: {name: predicted values} (one per Layer 2 variant).
    Each episode randomly uses one of the forecasts or the 'mean' estimate (predictor off),
    and a random VM count (2-20), so one agent covers every switch combination in the UI."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    actual = np.asarray(actual)
    split = int(len(actual) * 0.75)  # first 75% for training, the rest is held out for evaluation
    tr_actual = actual[:split]
    tr_preds = {k: np.asarray(v)[:split] for k, v in preds.items()}
    sources = list(tr_preds) + [None]

    net, tgt = QNet(), QNet()
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    S = np.zeros((buffer, N_FEATURES), np.float32)
    R = np.zeros(buffer, np.float32)
    NX = np.zeros((buffer, MAX_VMS, N_FEATURES), np.float32)
    NM = np.zeros((buffer, MAX_VMS), bool)
    D = np.zeros(buffer, bool)
    ptr = n = steps = 0
    curve = []

    for ep in range(episodes):
        eps = max(eps_end, 1.0 - ep / (0.6 * episodes))
        vms = int(rng.integers(2, MAX_VMS + 1))
        src = sources[int(rng.integers(len(sources)))]
        env = make_episode(rng, tr_actual, tr_preds[src] if src else None, scale, int(rng.integers(50, 201)), vms)
        ep_reward = 0.0
        while not env.done:
            feats = env.features()
            if rng.random() < eps:
                a = int(rng.integers(vms))
            else:
                with torch.no_grad():
                    a = int(net(torch.from_numpy(feats)).argmax())
            r = env.step(a)
            ep_reward += r
            S[ptr], R[ptr], D[ptr] = feats[a], r, env.done
            NM[ptr] = False
            if not env.done:
                NX[ptr, :vms] = env.features()
                NM[ptr, :vms] = True
            ptr, n, steps = (ptr + 1) % buffer, min(n + 1, buffer), steps + 1

            if n >= batch * 4 and steps % 2 == 0:
                i = rng.integers(n, size=batch)
                s, r_, d = torch.from_numpy(S[i]), torch.from_numpy(R[i]), torch.from_numpy(D[i])
                nx, nm = torch.from_numpy(NX[i]), torch.from_numpy(NM[i])
                with torch.no_grad():
                    nq = tgt(nx).masked_fill(~nm, -1e9).max(1).values
                    y = r_ + gamma * torch.where(d, torch.zeros_like(nq), nq)
                loss = nn.functional.smooth_l1_loss(net(s), y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            if steps % target_every == 0:
                tgt.load_state_dict(net.state_dict())
        curve.append(ep_reward / max(1, env.i))
        if (ep + 1) % 50 == 0:
            log(f"    episode {ep + 1:4d}/{episodes}  eps {eps:.2f}  avg reward/step {np.mean(curve[-50:]):.4f}")

    policy = export_policy(net)
    held = {"actual": actual[split:], "preds": {k: np.asarray(v)[split:] for k, v in preds.items()}}
    return policy, {"curve": [round(float(np.mean(curve[i:i + 20])), 5) for i in range(0, len(curve), 20)],
                    "seconds": round(time.time() - t0, 1), "episodes": episodes, "held_out": held}


def evaluate(policy_dict: dict, held: dict, scale: float, n_tasks=200, episodes=60, vm_counts=(3, 5, 10, 20), seed=1):
    """Mean makespan on HELD-OUT tasks: Round Robin vs earliest-finish baseline vs DQN.
    Same tasks for all three, repeated for each VM count and for each way Layer 2 can be on or off."""
    policy = DQNPolicy.from_dict(policy_dict)
    out = {}
    sources = list(held["preds"]) + ["mean"]
    for src in sources:
        out[src] = {}
        for v in vm_counts:
            rng = np.random.default_rng(seed)
            tot = {"rr": [], "earliest": [], "dqn": []}
            for _ in range(episodes):
                env = make_episode(rng, held["actual"], held["preds"].get(src), scale, n_tasks, v)
                for mode in tot:
                    tot[mode].append(run_schedule(env.true, env.est, v, mode, policy).max())
            out[src][str(v)] = {k: round(float(np.mean(x)), 3) for k, x in tot.items()}
    return out