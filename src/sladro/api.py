"""SLADRO backend: serves the same RunResult JSON as the UI's placeholder pipeline.

All three layers come from pre-computed artifacts trained offline, so this service needs no torch and
starts instantly. Layer 3 is the DQN in artifacts/layer3.json (numpy inference). If that file is missing,
the earliest-finish placeholder is used instead.
Run:  uvicorn --app-dir src sladro.api:app --port 8000
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .layer3_env import POWER, DQNPolicy, run_schedule, to_len

ART = Path(__file__).resolve().parents[2] / "artifacts"

app = FastAPI(title="SLADRO backend")
_cache: dict = {}


def art(name: str) -> dict:
    if name not in _cache:
        p = ART / name
        if not p.exists():
            raise HTTPException(503, f"Missing {p.name}. Run scripts/run_layer2.py first.")
        _cache[name] = json.loads(p.read_text())
    return _cache[name]


def dqn_policy() -> DQNPolicy | None:
    if "policy" not in _cache:
        p = ART / "layer3.json"
        _cache["policy"] = DQNPolicy.load(p) if p.exists() else None
    return _cache["policy"]


class Config(BaseModel):
    featureSelection: bool = True
    predictor: bool = True
    scheduler: bool = True
    seed: int = 42
    numTasks: int = 200
    numVMs: int = 5


@app.get("/health")
def health():
    return {"ok": True, "dqn": dqn_policy() is not None, "artifacts": sorted(p.name for p in ART.glob("*.json"))}


@app.post("/run")
def run(c: Config):
    l1 = art("layer1.json")
    all_feats = l1["all_features"]
    selected = l1["selected"] if c.featureSelection else all_feats
    l2 = art("layer2_selected.json" if c.featureSelection else "layer2_all.json")
    base_pool = art("layer2_all.json")["pool"]  # same real test samples for every combination

    vms, n = max(2, min(20, c.numVMs)), max(10, min(400, c.numTasks))
    rng = np.random.default_rng(c.seed)
    pick = rng.choice(len(base_pool["actual"]), n, replace=n > len(base_pool["actual"]))
    scale = base_pool["scale"]
    true_len = to_len(np.asarray(base_pool["actual"])[pick], scale)
    est_len = to_len(np.asarray(l2["pool"]["predicted"])[pick], scale) if c.predictor else np.full(n, true_len.mean())

    policy = dqn_policy()
    mode = "rr" if not c.scheduler else ("dqn" if policy else "earliest")
    name = {"rr": "Round Robin", "dqn": "DQN", "earliest": "DQN (placeholder: earliest-finish)"}[mode]
    f = run_schedule(true_len, est_len, vms, mode, policy)

    mk = float(f.max())
    u = f / mk
    pw = np.array([POWER[m % 5] for m in range(vms)])
    energy = float((pw * f + 0.4 * pw * (mk - f)).sum() / 1000)
    return {
        "config": c.model_dump(),
        "allFeatures": all_feats,
        "selectedFeatures": selected,
        "predictor": l2["metrics"] if c.predictor else None,
        "schedulerName": name,
        "metrics": {"makespan": mk, "throughput": n / mk, "utilizationSD": float(u.std()), "energyKJ": energy},
        "series": {"actual": l2["series"]["actual"][-80:],
                   "predicted": l2["series"]["predicted"][-80:] if c.predictor else None},
        "vmBusy": f.tolist(),
    }